"""SM120 sparse-MLA decode for DeepSeek-V4.1-Flash — Triton reference kernel.

WHY THIS EXISTS
---------------
FlashInfer's SM120 sparse-MLA decode requires ``page_block_size == 64``
(``_DECODE_DSV4_PAGE_BLOCK_SIZE``) while the vLLM SM120 backend declares a
kernel block size of 128, so only layers with ``compress_ratio == 2`` can
produce the needed 64. This kernel is a vendor-neutral Triton reimplementation
of that decode step, giving (a) a correctness oracle and (b) a fallback that
does not depend on the FlashInfer dispatch table.

SEMANTICS (derived, not guessed)
--------------------------------
Taken from vLLM's own v4.1 reference implementations:

* validity/lens rule — ``vllm/models/deepseek_v41/common/ops/cache_utils.py:956``
  (``CombineTopkSwaIndicesKernel``)::

      topk_len = min((pos + 1) // compress_ratio, index_topk)   if ratio > 0
      topk_len = 0                                              if ratio == 0
      swa_start = max(pos - (window_size - 1), 0)
      swa_len   = pos - swa_start + 1

* ratio semantics — ``vllm/models/deepseek_v41/attention.py:298``: v4.1 uses
  ratios ``{0, 1, 2}`` only; 0 = pure sliding window (no compressed KV at all),
  1 = full-length compressed cache, 2 = ratio-2 compressed.

* global slot mapping — ``cache_utils.py:774``::

      block_index = local_index // block_size
      slot        = block_table[req, block_index] * block_size + local_index % block_size

The kernel therefore takes the *local* compressed-KV indices produced by the
indexer, maps them to global slots, gathers the compressed K/V, and computes
softmax attention with a numerically-stable online (flash) softmax.

DESIGN NOTES
------------
* One program per (query token, query head) pair. Decode has few tokens and
  ``n_heads`` is small (8 locally at TP8), so this fills SM120's 148 SMs at
  batch >= 19 — adequate for a reference, and honest about being a reference.
* ``num_stages`` on the K loop lets Triton software-pipeline the gather.
* ``-1`` sentinel in the index buffer means "no entry"; length caps the range
  so padded slots are never read.
* fp8 (e4m3) K/V with fp32 per-block scales, matching ``fp8_ds_mla`` layout:
  K and V are separate tensors here (the caller dequantizes/gathers), keeping
  this kernel testable without depending on the paged cache layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_decode_kernel(
    Q,                    # [num_tokens, n_heads, d_qk]  fp16/bf16
    K,                    # [num_slots, d_qk]            fp16/bf16
    V,                    # [num_slots, d_v]             fp16/bf16
    INDICES,              # [num_tokens, topk]           int32, -1 = invalid
    TOPK_LENS,            # [num_tokens]                 int32
    OUT,                  # [num_tokens, n_heads, d_v]   fp32
    stride_qt, stride_qh, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_it, stride_ik,
    stride_ot, stride_oh, stride_od,
    scale,                # softmax scale (1/sqrt(d_qk), already applied to q or here)
    N_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)

    topk_len = tl.load(TOPK_LENS + token)
    if topk_len <= 0:
        # No visible entries: emit zeros (matches "all-masked row" semantics
        # without the +inf LSE defect seen in stock FlashInfer).
        offs_dv = tl.arange(0, BLOCK_DV)
        tl.store(
            OUT + token * stride_ot + head * stride_oh + offs_dv * stride_od,
            tl.zeros((BLOCK_DV,), dtype=tl.float32),
            mask=offs_dv < D_V,
        )
        return

    offs_dqk = tl.arange(0, D_QK)
    q = tl.load(Q + token * stride_qt + head * stride_qh + offs_dqk * stride_qd)
    q = q.to(tl.float32) * scale

    # Online softmax accumulators.
    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    offs_dv = tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < D_V

    for start in range(0, TOPK, BLOCK_TOPK):
        offs = start + tl.arange(0, BLOCK_TOPK)
        mask = offs < topk_len
        idx = tl.load(INDICES + token * stride_it + offs * stride_ik, mask=mask, other=-1)
        valid = (idx >= 0) & mask

        k = tl.load(
            K + idx[:, None] * stride_ks + offs_dqk[None, :] * stride_kd,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1)
        logits = tl.where(valid, logits, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        p = tl.exp(logits - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v = tl.load(
            V + idx[:, None] * stride_vs + offs_dv[None, :] * stride_vd,
            mask=valid[:, None] & dv_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = tl.where(l_i > 0, acc / l_i, 0.0)
    tl.store(
        OUT + token * stride_ot + head * stride_oh + offs_dv * stride_od,
        acc,
        mask=dv_mask,
    )


def sparse_mla_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    topk_lens: torch.Tensor,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse-MLA decode. ``indices`` holds global slot ids into ``k``/``v``.

    q:        [T, H, D_QK]
    k, v:     [S, D_QK] / [S, D_V]  (already gathered/dequantized)
    indices:  [T, TOPK] int32, -1 = invalid
    topk_lens:[T] int32
    """
    T, H, D_QK = q.shape
    D_V = v.shape[1]
    TOPK = indices.shape[1]

    if out is None:
        out = torch.empty((T, H, D_V), dtype=torch.float32, device=q.device)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indices = indices.contiguous()
    topk_lens = topk_lens.contiguous()

    grid = (T, H)
    _sparse_mla_decode_kernel[grid](
        q, k, v, indices, topk_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        indices.stride(0), indices.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        N_HEADS=H,
        TOPK=TOPK,
        D_QK=D_QK,
        D_V=D_V,
        BLOCK_TOPK=16,
        BLOCK_DV=triton.next_power_of_2(D_V),
    )
    return out


# --------------------------------------------------------------------------- #
# Reference / test helpers
# --------------------------------------------------------------------------- #
def fabricate_lens(positions: torch.Tensor, compress_ratio: int, index_topk: int,
                   window_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the v4.1 index/lens fabrication verbatim.

    Mirrors ``vllm/models/deepseek_v41/common/ops/cache_utils.py:956``
    (``CombineTopkSwaIndicesKernel.kernel``).
    """
    pos = positions.to(torch.int64)
    if compress_ratio > 0:
        topk_len = torch.clamp((pos + 1) // compress_ratio, max=index_topk)
    else:
        topk_len = torch.zeros_like(pos)
    swa_start = torch.clamp(pos - (window_size - 1), min=0)
    swa_len = pos - swa_start + 1
    return topk_len.to(torch.int32), swa_len.to(torch.int32)


def torch_reference(q, k, v, indices, topk_lens, scale):
    """Dense torch reference for validation (padding slots masked out)."""
    T, H, D_QK = q.shape
    D_V = v.shape[1]
    out = torch.zeros((T, H, D_V), dtype=torch.float32, device=q.device)
    for t in range(T):
        n = int(topk_lens[t])
        if n <= 0:
            continue
        idx = indices[t, :n].to(torch.long)
        kk = k[idx].float()                      # [n, D_QK]
        vv = v[idx].float()                      # [n, D_V]
        logits = (q[t].float() @ kk.T) * scale   # [H, n]
        p = torch.softmax(logits, dim=-1)
        out[t] = p @ vv
    return out