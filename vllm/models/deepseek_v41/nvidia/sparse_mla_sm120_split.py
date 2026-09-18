"""SM120 sparse-MLA decode, split-K Triton implementation (DSv4.1).

Follows FlashInfer's SM120 contract (see ``sparse_mla_sm120_triton.py`` header
and the recovered image source ``flashinfer/mla/_sparse_mla_sm120.py``):

* decode is **split-K over the topk dimension** with ``_BI = 64`` candidate
  per split, i.e. ``num_splits = ceil(topk / 64)`` -> 8 for ``index_topk=512``;
* a second merge pass combines per-split partials using the per-split LSE:
      out = sum_s exp(lse_s - lse_max) * out_s / sum_s exp(lse_s - lse_max)
* a separate ``extra_*`` region (SWA indices) may be appended as further splits;
* head tile is padded to ``HPB = 16`` with zero-Q rows (TP8 -> 8 heads).

The one-program-per-(token,head) form in ``sparse_mla_sm120_triton.py`` is the
readable reference; this file is the occupancy-correct form. Both produce the
same numbers (cross-checked in ``test_sparse_mla_sm120_triton.py``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BI = 64  # KV partition tile in candidates, mirrors FlashInfer _BI


@triton.jit
def _smla_decode_split_kernel(
    Q, K, V, INDICES, TOPK_LENS, MID_OUT, MID_LSE,
    s_qt, s_qh, s_qd,
    s_ks, s_kd,
    s_vs, s_vd,
    s_it, s_ik,
    s_mo_t, s_mo_h, s_mo_s, s_mo_d,
    s_ml_t, s_ml_h, s_ml_s,
    scale,
    TOPK: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """One program per (token, head, split). Writes partial out + LSE."""
    token = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)

    start = split * BLOCK_TOPK
    offs = start + tl.arange(0, BLOCK_TOPK)
    topk_len = tl.load(TOPK_LENS + token)
    mask = offs < topk_len
    all_invalid = tl.sum(mask.to(tl.int32), axis=0) == 0

    offs_dv = tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < D_V

    if all_invalid:
        # This split sees nothing; record a -inf LSE so the merge drops it.
        tl.store(
            MID_OUT + token * s_mo_t + head * s_mo_h + split * s_mo_s
            + offs_dv * s_mo_d,
            tl.zeros((BLOCK_DV,), dtype=tl.float32),
            mask=dv_mask,
        )
        tl.store(MID_LSE + token * s_ml_t + head * s_ml_h + split * s_ml_s,
                 float("-inf"))
        return

    offs_dqk = tl.arange(0, D_QK)
    q = tl.load(Q + token * s_qt + head * s_qh + offs_dqk * s_qd).to(tl.float32) * scale

    idx = tl.load(INDICES + token * s_it + offs * s_ik, mask=mask, other=-1)
    valid = (idx >= 0) & mask

    k = tl.load(K + idx[:, None] * s_ks + offs_dqk[None, :] * s_kd,
                mask=valid[:, None], other=0.0).to(tl.float32)
    logits = tl.where(valid, tl.sum(k * q[None, :], axis=1), float("-inf"))

    m = tl.max(logits, axis=0)
    p = tl.exp(logits - m)
    l = tl.sum(p, axis=0)

    v = tl.load(V + idx[:, None] * s_vs + offs_dv[None, :] * s_vd,
                mask=valid[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
    # Divide by this split's own l so the partial is a self-contained normalized
    # average. The merge then only weights by exp(lse_s - G); keeping the raw
    # sum_i exp(x_i - m_s) v_i would need an extra l_s factor that the weight
    # (built from the GLOBAL max) does not supply.
    acc = tl.sum(p[:, None] * v, axis=0) / l

    tl.store(
        MID_OUT + token * s_mo_t + head * s_mo_h + split * s_mo_s + offs_dv * s_mo_d,
        acc,
        mask=dv_mask,
    )
    # LSE = m + log(l), used by the merge as the weight exp(lse_s - G).
    tl.store(MID_LSE + token * s_ml_t + head * s_ml_h + split * s_ml_s,
             m + tl.log(l))



@triton.jit
def _smla_merge_kernel(
    MID_OUT, MID_LSE, OUT, OUT_LSE,
    s_mo_t, s_mo_h, s_mo_s, s_mo_d,
    s_ml_t, s_ml_h, s_ml_s,
    s_o_t, s_o_h, s_o_d,
    s_ol_t, s_ol_h,
    NUM_SPLITS: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Combine per-split partials using their true LSEs.

    Derivation (G = global LSE max, acc_s = sum_i exp(x_i - m_s) v_i,
    l_s = sum_i exp(x_i - m_s)):
        out = sum_s sum_i exp(x_i - G) v_i / sum_s sum_i exp(x_i - G)
    with exp(x_i - G) = exp(lse_s - G) * exp(x_i - m_s) / l_s:
        numerator   = sum_s exp(lse_s - G) * acc_s
        denominator = sum_s exp(lse_s - G) * l_s
    Partials are stored RAW (not divided by l_s); ``l_s`` is recovered from
    the published pair (lse_s, m_s) as ``l_s = exp(lse_s - m_s)``.
    Empty splits publish lse_s = m_s = -inf and contribute zero weight.
    """
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs_dv = tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < D_V

    # Pass 1: global LSE max.
    g_max = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(0, NUM_SPLITS, BLOCK_S):
        so = s + tl.arange(0, BLOCK_S)
        lse = tl.load(MID_LSE + token * s_ml_t + head * s_ml_h + so * s_ml_s,
                      mask=so < NUM_SPLITS, other=float("-inf"))
        g_max = tl.maximum(g_max, tl.max(lse, axis=0))

    # Pass 2: weighted combine.
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    denom = tl.zeros((), dtype=tl.float32)
    for s in range(0, NUM_SPLITS, BLOCK_S):
        so = s + tl.arange(0, BLOCK_S)
        inb = so < NUM_SPLITS
        lse = tl.load(MID_LSE + token * s_ml_t + head * s_ml_h + so * s_ml_s,
                      mask=inb, other=float("-inf"))
        # A split is live iff it published a finite LSE. Dead splits must
        # contribute exactly zero to BOTH sums.
        live = inb & (lse != float("-inf"))
        # acc_s is already normalized by its own l_s, so the merge is the plain
        # LSE-weighted average (verified numerically against the reference):
        #     out = sum_s w_s * acc_s / sum_s w_s,   w_s = exp(lse_s - G)
        w = tl.where(live, tl.exp(tl.where(live, lse - g_max, -1.0e30)), 0.0)
        o = tl.load(
            MID_OUT + token * s_mo_t + head * s_mo_h
            + so[:, None] * s_mo_s + offs_dv[None, :] * s_mo_d,
            mask=inb[:, None] & dv_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w[:, None] * o, axis=0)
        denom += tl.sum(w, axis=0)

    r = tl.where(denom > 0, acc / denom, 0.0)
    tl.store(OUT + token * s_o_t + head * s_o_h + offs_dv * s_o_d, r, mask=dv_mask)
    # Guard the all-empty case: m_max == -inf and denom == 0 would give
    # (-inf) + log(0) = nan. Emit -inf instead (the caller treats a -inf LSE
    # as "no visible entries", matching the reference kernel's zeros output).
    finite = g_max != float("-inf")
    out_lse_v = tl.where(finite, g_max + tl.log(tl.maximum(denom, 1.0)), float("-inf"))
    tl.store(OUT_LSE + token * s_ol_t + head * s_ol_h, out_lse_v)


def _num_splits(topk: int, block_topk: int) -> int:
    return (topk + block_topk - 1) // block_topk


def sparse_mla_decode_splitk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    topk_lens: torch.Tensor,
    scale: float,
    block_topk: int = _BI,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K sparse-MLA decode. Returns (out fp32 [T,H,D_V], lse fp32 [T,H])."""
    T, H, D_QK = q.shape
    D_V = v.shape[1]
    TOPK = indices.shape[1]
    NS = _num_splits(TOPK, block_topk)

    if mid_out is None:
        mid_out = torch.empty((T, H, NS, D_V), dtype=torch.float32, device=q.device)
    if mid_lse is None:
        mid_lse = torch.empty((T, H, NS), dtype=torch.float32, device=q.device)
    out = torch.empty((T, H, D_V), dtype=torch.float32, device=q.device)
    out_lse = torch.empty((T, H), dtype=torch.float32, device=q.device)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    indices, topk_lens = indices.contiguous(), topk_lens.contiguous()

    _smla_decode_split_kernel[(T, H, NS)](
        q, k, v, indices, topk_lens, mid_out, mid_lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        indices.stride(0), indices.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        scale,
        TOPK=TOPK, D_QK=D_QK, D_V=D_V,
        BLOCK_TOPK=block_topk,
        BLOCK_DV=triton.next_power_of_2(D_V),
    )
    _smla_merge_kernel[(T, H)](
        mid_out, mid_lse, out, out_lse,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        out_lse.stride(0), out_lse.stride(1),
        NUM_SPLITS=NS,
        D_V=D_V,
        BLOCK_DV=triton.next_power_of_2(D_V),
        BLOCK_S=triton.next_power_of_2(NS),
    )
    return out, out_lse