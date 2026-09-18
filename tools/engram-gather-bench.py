# SPDX-License-Identifier: Apache-2.0
"""Cost of a paged embedding gather over device vs pinned host memory.

The Engram lookup reads its tables through UVA pointers, so this measures the
access pattern the decode loop actually pays for. Same kernel, same shapes,
only the residency of the source table differs.
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def gather_kernel(out_ptr, src_addr, idx_ptr, N, ROW: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 256 + tl.arange(0, 256)
    m = offs < N
    row_idx = tl.load(idx_ptr + offs, mask=m, other=0).to(tl.int64)
    base = src_addr.to(tl.pointer_type(tl.uint8))
    for r in tl.static_range(ROW):
        v = tl.load(base + row_idx * ROW + r, mask=m, other=0)
        tl.store(out_ptr + offs.to(tl.int64) * ROW + r, v, mask=m)


def bench(label, pin, n=2_000_000, row=256, idx_n=4096, iters=30):
    src = torch.zeros(n, row, dtype=torch.uint8, pin_memory=pin)
    idx = torch.randint(0, n, (idx_n,), dtype=torch.int64)
    out = torch.zeros(idx_n, row, dtype=torch.uint8, device="cuda")
    # The real engram kernel reaches host tables through UVA, so pass the raw
    # address as an integer; Triton refuses a CPU tensor pointer outright.
    src_addr = src.data_ptr()
    grid = (triton.cdiv(idx_n, 256),)
    for _ in range(3):
        gather_kernel[grid](out, src_addr, idx, idx_n, row)
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(iters):
        gather_kernel[grid](out, src_addr, idx, idx_n, row)
    torch.cuda.synchronize()
    per_step = (time.monotonic() - started) / iters
    print(
        f"{label:12} rows={n:,} gather={idx_n} "
        f"per-step {per_step * 1e3:.3f} ms  -> {1 / per_step:.0f} steps/s"
    )
    return per_step


if __name__ == "__main__":
    device = bench("device", False)
    host = bench("pinned-host", True)
    print(f"\nhost/device cost ratio: {host / device:.1f}x")