#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated check of the SM120 FP8 paged-MQA indexer with a 32-row page.

Mirrors the authoritative reference in DeepGEMM's own
``tests/test_attention.py`` (``ref_paged_mqa_logits`` and
``kv_cache_cast_to_fp8``) rather than re-deriving the semantics:

* the cache page is ``[block_size * head_dim]`` FP8 values followed by
  ``block_size * 4`` bytes holding one FP32 scale per token;
* ``context_lens[i]`` is the request's length and every one of its ``next_n``
  queries attends the *same* final position, so ``q_offsets == context_len``;
* the result is masked to ``k_offsets < context_len``.

The only deviation is the comparison basis: the reference recomputes logits
from the *pre-quantization* values, which for FP8 hides real roundoff. This
harness instead runs the oracle over the values decoded back from the packed
cache, so a geometry mistake cannot be absorbed by the tolerance.
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import torch

import vllm.utils.deep_gemm as dg

HEADS = 32
DIM = 128
# Store blocks are 64 tokens for both ratios; a ratio-2 layer keeps 32 states
# per store block, which the kernel addresses as `group` 32-token regions.
STORE_BLOCK = 64
# Padded physical block strides are bank-aligned, so keep the stride a multiple
# of the used bytes (page_size * (DIM + 4)).
ALIGN_UNIT = 64 * (DIM + 4)
TOL = 2e-2


def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


def pack_fp8_cache(values, scales, page_size):
    """Pack per-token values/scales into the kernel's page layout.

    Mirrors ``kv_cache_cast_to_fp8``: values for the whole page first, then the
    per-token scales in one contiguous region.
    """
    num_pages = values.shape[0]
    row = DIM + 4
    stride = (page_size * row + ALIGN_UNIT - 1) // ALIGN_UNIT * ALIGN_UNIT
    backing = torch.zeros(num_pages * stride, dtype=torch.uint8, device="cuda")
    flat = backing.view(num_pages, stride)
    quant = (values / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    flat[:, : page_size * DIM] = quant.view(torch.uint8).reshape(
        num_pages, page_size * DIM
    )
    flat[:, page_size * DIM : page_size * row] = (
        scales.contiguous().view(torch.uint8).reshape(num_pages, page_size * 4)
    )
    return backing, stride, quant


def decode_page(backing, stride, page_size, page, drop_scale=False):
    """Decode one page into FP32 [page_size, DIM] values from packed bytes."""
    flat = backing.view(-1, stride)[page]
    values = flat[: page_size * DIM].contiguous().view(torch.float8_e4m3fn).float()
    values = values.view(page_size, DIM)
    if drop_scale:
        return values
    scale = (
        flat[page_size * DIM : page_size * (DIM + 4)]
        .contiguous()
        .view(torch.float32)[:page_size]
    )
    return values * scale[:, None]


def reference(q, weights, backing, stride, page_size, block_table, context_lens):
    """FP32 oracle, structurally identical to the DeepGEMM reference.

    ``q`` is [B, next_n, H, D] already quantized to FP8; the KV side is decoded
    from the packed cache. Every query of request ``i`` attends the request's
    own final position, so the whole context is scored at that offset.
    """
    batch, next_n = q.shape[:2]
    device = q.device
    out = torch.full((batch * next_n, len(range(0, 1))), float("-inf"), device=device)
    rows = []
    for i in range(batch):
        ctx = int(context_lens[i])
        num_blocks = math.ceil(ctx / page_size)
        pages = block_table[i, :num_blocks].long()
        kv = torch.cat(
            [decode_page(backing, stride, page_size, p) for p in pages.tolist()], 0
        )
        qx = q[i].float().transpose(0, 1)  # [H, next_n, D]
        scores = torch.matmul(qx, kv.T)  # [H, next_n, total_tokens]
        k_off = torch.arange(0, kv.shape[0], device=device)
        mask = (k_off[None, :] < ctx) & (k_off[None, :] <= ctx)
        scores = torch.where(mask[None, :, :], scores, float("-inf"))
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].T.contiguous()
        scores = torch.relu(scores) * weight_slice[..., None]
        reduced = scores.sum(dim=0)  # [next_n, total_tokens]
        rows.append(reduced)
    width = max(r.shape[-1] for r in rows)
    out = torch.zeros(batch * next_n, width, device=device, dtype=torch.float32)
    for i, r in enumerate(rows):
        out[i * next_n : (i + 1) * next_n, : r.shape[-1]] = r
    return out


def make_generator(seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def run_case(name, group, batch, context_len, next_n, generator):
    """group=1 -> one 64-token page; group=2 -> a 32-token page region."""
    page_size = STORE_BLOCK // group
    num_blocks_per_req = math.ceil(context_len / page_size)
    num_blocks = batch * num_blocks_per_req

    values = torch.randn(
        num_blocks,
        page_size,
        DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    ).clamp(-3, 3)
    scales = torch.pow(
        2.0,
        torch.randint(
            -4,
            5,
            (num_blocks, page_size),
            generator=generator,
            device="cuda",
        ).float(),
    )
    backing, stride, _ = pack_fp8_cache(values, scales, page_size)
    cache = backing.as_strided(
        (num_blocks, page_size, 1, DIM + 4), (stride, DIM + 4, DIM + 4, 1)
    )

    pool = torch.randperm(num_blocks, generator=generator, device="cuda")
    block_table = torch.zeros(
        (batch, num_blocks_per_req), dtype=torch.int32, device="cuda"
    )
    offset = 0
    for i in range(batch):
        block_table[i] = pool[offset : offset + num_blocks_per_req]
        offset += num_blocks_per_req

    # DeepGEMM reference convention: one column per request holding the length,
    # with every next_n query scoring the same final position.
    context_lens = torch.full(
        (batch, next_n), context_len, dtype=torch.int32, device="cuda"
    )
    q = torch.randn(batch, next_n, HEADS, DIM, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    q_quant = (q.float() / 32.0).clamp(-448, 448).to(torch.float8_e4m3fn)
    weights = (
        torch.randn(batch * next_n, HEADS, generator=generator, device="cuda") * 0.5
    ).to(torch.float32)
    assert weights.is_cuda and weights.dtype == torch.float32

    logits = dg.fp8_fp4_paged_mqa_logits(
        (q_quant, None),
        cache,
        weights,
        context_lens,
        block_table,
        dg.get_paged_mqa_logits_metadata(context_lens, page_size, num_sms()),
        max_model_len=context_len,
        clean_logits=False,
    )
    if not torch.isfinite(logits).any():
        raise AssertionError(f"{name}: kernel produced no finite logits")

    ref = reference(
        q_quant, weights, backing, stride, page_size, block_table, context_lens[:, 0]
    )
    got = logits[:, : ref.shape[-1]]
    error = (got - ref).abs().max().item()
    scale_ref = ref.abs().max().item() or 1.0
    normalized = error / scale_ref
    if normalized > TOL:
        raise AssertionError(f"{name}: normalized error {normalized:.6g} > {TOL}")

    return {
        "case": name,
        "group": group,
        "page_size": page_size,
        "batch": batch,
        "context_len": context_len,
        "next_n": next_n,
        "padded_stride": stride,
        "cache_shape": list(cache.shape),
        "cache_strides": list(cache.stride()),
        "logits_shape": list(logits.shape),
        "finite_logits": int(torch.isfinite(logits).sum().item()),
        "max_abs_error": error,
        "normalized_max_error": normalized,
        "tolerance": TOL,
        "distinct_scales": int(scales.unique().numel()),
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    sink = args.jsonl.open("a") if args.jsonl else None

    def emit(record):
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if sink is not None:
            sink.write(line + "\n")
            sink.flush()

    emit(
        {
            "kind": "environment",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "deep_gemm_module": dg.__file__,
            "reference": "DeepGEMM tests/test_attention.py ref_paged_mqa_logits",
            "comparison_basis": "FP32 oracle decoded from packed FP8 bytes",
        }
    )
    generator = make_generator(20260918)
    failures = []
    # group1 = a 64-token store block; group2 = the 32-state region a ratio-2
    # layer presents. page32/g64 are the pre-existing controls.
    for group, batch, context_len, next_n in (
        (1, 1, 512, 1),
        (1, 3, 256, 1),
        (1, 2, 1024, 1),
        (1, 1, 128, 2),
        (1, 2, 512, 3),
        (2, 1, 512, 1),
        (2, 3, 256, 1),
        (2, 2, 1024, 1),
        (2, 1, 128, 2),
        (2, 2, 512, 3),
    ):
        name = f"g{group}-b{batch}-l{context_len}-n{next_n}"
        try:
            emit(run_case(name, group, batch, context_len, next_n, generator))
        except Exception as exc:  # noqa: BLE001 - report every case
            failures.append(name)
            emit(
                {
                    "case": name,
                    "group": group,
                    "batch": batch,
                    "context_len": context_len,
                    "next_n": next_n,
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc).splitlines()[0][:400],
                    "traceback": traceback.format_exc()[-1500:],
                }
            )
    summary = {
        "kind": "summary",
        "status": "fail" if failures else "pass",
        "failures": failures,
    }
    emit(summary)
    if sink is not None:
        sink.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
