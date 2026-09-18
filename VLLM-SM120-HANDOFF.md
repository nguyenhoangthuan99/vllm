# vLLM on SM120 for DeepSeek-V4.1-Flash — HANDOFF (verified state)

Last updated: 2026-09-18. Supersedes the "CORRECTION" and earlier sections of
`logs/vllm-sm120-findings.md`, two of whose claims were later disproven (listed
under "Corrections" below).

Host: `research-8xpro6000` (VM .106), 8x RTX PRO 6000 Blackwell 96GB (SM120),
144 cores, 873 GB RAM. Serving box VM .100 was never touched.

---

## TL;DR

vLLM is **not** blocked by C4A-vs-C2A architecture support (my earlier claim — wrong).
It is blocked by a **page-geometry mismatch in the SM120 FlashInfer backend only**.
C2A is a first-class layer type in vLLM and works on H100/H200 and B100/B200.
Evidence points to a small bug, not a reimplementation. **Unverified hypothesis**:
see "The open question".

SGLang serves this checkpoint correctly on the same hardware today:
C1 196.8 / C4 396.2 / C16 695.6 / C32 1332.2 tok/s, prefill TTFT ~0.27 s.

---

## 1. What works (verified)

### vLLM upstream has full DeepSeek-V4.1 support
`vllm/models/deepseek_v41/` (attention, compressor, sparse_mla, quant_config,
nvidia/, amd/), plus `configs/deepseek_v41.py`, `tokenizers/deepseek_v41.py`,
`parser/deepseek_v41.py`, `tool_parsers/deepseekv41_engine_tool_parser.py`,
`reasoning/deepseek_v41_engine_reasoning_parser.py`.

### vLLM supports FOUR DSv4 layer types, including C2A
`vllm/v1/attention/backends/mla/sparse_swa.py:56`

```python
def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:   return _LAYER_TYPE_SWAONLY
    if compress_ratio == 2:   return _LAYER_TYPE_C2A      # <-- our checkpoint
    if compress_ratio == 4:   return _LAYER_TYPE_C4A
    if compress_ratio == 128: return _LAYER_TYPE_C128A
    raise ValueError("Unsupported ... expected 1, 2, 4, or 128.")
```
C2A has its own tile scheduler (`tile_sched_c2a`), consumed at
`vllm/models/deepseek_v41/nvidia/flashmla.py:233` (the H100/H200 path).

### Three platform-specific attention implementations
`vllm/models/deepseek_v41/nvidia/model.py:_select_dsv4_attn_cls`

| platform | class | kernel block size | C2A works? |
|---|---|---|---|
| SM100 (B100/B200) | `DeepseekV4MegaAttnAttention` (FlashMLA mega) | 128 | yes |
| SM90 (H100/H200) | `DeepseekV4FlashMLAAttention` | **64** (`sparse_mla.py:90`) | yes |
| **SM120 (RTX PRO 6000)** | `DeepseekV4FlashInferSM120Attention` | 128 | **no** |

`sparse_mla.py:90`:
```python
return [64 if current_platform.is_device_capability_family(90) else 128]
```

### The full source build succeeds for SM120
Built with `TORCH_CUDA_ARCHITECTURES=120` -> `12.0f`, `MAX_JOBS=128`.
Produces `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_qutlass_C`,
`cumem_allocator`, `fs_io_C`, `spinloop`, `_flashkda_C`, `_vllm_fa2_C`,
`_vllm_fa3_C`. Verified `sm_120` cubins present and the **13-arg**
`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` exported.

With those extensions the server reaches **CUDA graph capture on SM120**
(`Capturing CUDA graphs (PIECEWISE)`) — previously reported impossible in
vllm-project/vllm#56892. `enforce_eager=False`.

### Build prerequisites (each caused a hard failure)
`git` (CMake FetchContent clones CUTLASS/FA/triton/deepgemm), `ninja`, `cmake`,
`setuptools` + `setuptools-rust` + `setuptools-scm`, **`libdw-dev`**
(elfutils/libdwfl.h, DeepGEMM JIT), **`libcusparse-dev-13-0`,
`libcublas-dev-13-0`, `libcusolver-dev-13-0`, `libcurand-dev-13-0`** (image ships
runtime only, `cusparse.h` missing), a `libnvrtc.so` symlink (image has only
versioned libs -> `CUDA_nvrtc_LIBRARY NOTFOUND`), and a **writable** source tree
(setuptools-scm writes `vllm/_version.py`; read-only mount fails).
CMake 4.4 rejects `12.0f` in `CMAKE_CUDA_ARCHITECTURES`; pass `120`.

### Build script (working)
See `logs/vllm-sm120-findings.md` "Build recipe notes"; script was
`/tmp/build-vllm-full.sh` in the builder container (recreate from that section).
Pattern that worked: `cmake -S /build -B /build/cmake-build-release -G Ninja
-DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE=$(command -v python3)
-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_INSTALL_PREFIX=/build`
then `cmake --build ... -j 128` then `cmake --install ...`.

---

## 2. What blocks it (verified error)

```
RuntimeError: SM120 sparse-MLA has no decode kernel for this shape:
  num_tokens=8, num_heads=8, topk=128, d_qk=512,
  page_block_size=32, model_type=1, extra_topk=0
```

FlashInfer gate — `flashinfer/mla/_sparse_mla_sm120.py::_decode_dsv4_dispatchable`:

| condition | required | actual | ok |
|---|---|---|---|
| `num_tokens <= _DECODE_MAX_TOKENS` | 64 | 8 | yes |
| `d_qk == 512` | 512 | 512 | yes |
| `(num_heads, topk) in _DECODE_DSV4_DISPATCH` | - | `(8,128)` **is** present | yes |
| `page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE` | **64** | **32** | **NO** |

`_DECODE_DSV4_PAGE_BLOCK_SIZE = 64`, `_DECODE_DSV3_2_PAGE_BLOCK_SIZE = 64`,
`_DECODE_MAX_TOKENS = 64`, `PAGED_MQA_PAGE_SIZES = (32, 64)`.

### Configs already tried (all fail)
| flag | error |
|---|---|
| `--block-size 128` (default) | `page_block_size=32`, no decode kernel |
| `--block-size 256` + `BLHNC` | "does not store blocks as dense, unpadded pages (block stride 460224 != page 74752), so a manager block cannot be split into 2 kernel blocks of 128 tokens" |
| `--block-size 256` + `BLNHC` | same as above |
| `VLLM_KV_CACHE_LAYOUT=LBNHC` | `ValueError: valid layouts: ['BLHNC', 'BLNHC']` (the error message's own suggestion is invalid — small upstream bug) |

---

## 2b. RESOLVED 2026-09-18 (after reading the H100 path) — the real cause

**The `// compress_ratio` division is a v4.0-ism. V4.1 does not compress that way.**

Read `vllm/models/deepseek_v41/attention.py:298-308` — the v4.1 attention class
reads `compress_ratio` **straight from the config, with no `max(1, ...)`, and
hard-raises unless it is 0, 1 or 2**:

```python
if compress_ratios is not None and layer_id < len(compress_ratios):
    self.compress_ratio = int(compress_ratios[layer_id])
else:
    self.compress_ratio = 0          # MTP layers past the list = pure SWA
if self.compress_ratio not in (0, 1, 2):
    raise ValueError(... "only 0 (sliding window), 1 and 2 are supported.")
```

Meanwhile `deepseek_v4/attention.py:226` does `self.compress_ratio =
max(1, config.compress_ratios[layer_id])` — the `max(1, ...)` clamps ratio 0
(SWA) up to 1. Our checkpoint has **5 ratio-0 layers out of 40**, so under the
v4.0 class those became `compress_ratio = 1` where v4.1 semantics say *pure
sliding window, no compressed KV cache at all*.

### Why it produced exactly page_block_size = 32

`sparse_mla.py:90` gives SM120 kernel block size **128** (only SM90 gets 64).
SM120's FlashInfer decode needs `page_block_size == 64`. So the pass condition is
`block_size // compress_ratio == 64`, and with `block_size = 128` **only
`compress_ratio == 2` passes**:

| layer | `compress_ratio` | `128 // ratio` | page 64? | layer count |
|---|---|---|---|---|
| ratio-0 SWA (mislabelled as 1 by v4.0 semantics) | 1 | **128** | no | 5 |
| ratio-1 | 1 | **128** | no | 20 |
| ratio-2 | 2 | **64** | yes | 18 |

The observed `page_block_size=32` is the *other* half of the same bug: when the
`128 // 1 = 128` page is rejected, the pool falls back to the `PAGED_MQA_PAGE_SIZES`
minimum (32) and the split lands on 32 instead of 64.

**So: this is a compress-ratio-semantics bug, not a page-arithmetic bug.** My
"something halves 128 → 64" hypothesis below was wrong — nothing halves it; the
ratio itself is wrong, and 128//1 ≠ 64.

### Independently, C2A geometry is FULLY derivable — no reference implementation needed

`vllm/models/deepseek_v41/common/ops/cache_utils.py:956` (inside
`CombineTopkSwaIndicesKernel.kernel`) already encodes the exact v4.1 index
geometry, parameterized by constexpr `COMPRESS_RATIO` and `WINDOW_SIZE`:

```python
# the indexer emits min((pos + 1) // compress_ratio, topk_tokens) valid entries
if COMPRESS_RATIO > 0:
    topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
else:
    topk_len = 0     # SWA-only: TOP_K=0, skip the division (div by 0 is UB)
swa_start = tl.maximum(pos - (WINDOW_SIZE - 1), 0)
swa_len   = pos - swa_start + 1
```

with `WINDOW_SIZE` = `config.sliding_window` = **128** (checkpoint
`text_config.sliding_window = 128`), warmup keys generated per layer type at
`cache_utils.py:1041`, and `combine_topk_swa_indices` called from all three
backends (`flashmla.py:349`, `flash_mla_mega_attn.py:449`, `amd/rocm.py:882`).

This kernel is **already generic across ratios 0/1/2/4/128** and runs on any
vendor. It is the authoritative reference for any SM120 sparse-decode kernel
(Triton or CUTLASS) that has to produce the same index/lens fabrication and the
same `topk_len = min((pos+1)//ratio, index_topk)` validity rule.

### Consequence for the SM120 work

1. The **blocking** bug is the v4.0/v4.1 compress-ratio mis-selection plus the
   SM120 page-split fallback — both upstream-of-kernel, both small.
2. Once fixed, only the **ratio-2 and ratio-1** layers are in play; ratio-0
   layers take a pure sliding-window path with no compressed gather.
3. A new SM120 sparse-decode kernel must satisfy `block_size // ratio == 64`,
   i.e. accept a compressed page of 64 tokens, and honour
   `topk_len = min((pos+1)//ratio, index_topk)` with `index_topk = 512`.
4. `compress_ratio == 4` and `== 128` never occur for this checkpoint — do not
   build C4A/C128A geometry for it.


## 2c. FlashInfer's actual SM120 contract (read from the image, 2026-09-18)

Source recovered from the containerd snapshot (builder container deleted but the
image layer survives):
`.../snapshots/13860/fs/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py`
(1361 lines). Constants mirrored from
`include/flashinfer/attention/sparse_mla_sm120/{arch,model}/*.cuh`.

```python
_D_V    = 512    # universal across DSV3_2 and DSV4
_BI     = 64     # KV partition tile in candidates (BLOCK_SIZE_N)
_DECODE_MAX_TOKENS = 64          # > 64 routes to the prefill orchestrator
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64
_DECODE_DSV4_DISPATCH = {(H, topk) for H in (8,16,32,64,128)
                                     for topk in (128,192,256,512,1024)}
```

`_decode_dsv4_dispatchable` = `num_tokens <= 64` AND `d_qk == 512` AND
`page_block_size == 64` AND `(num_heads, topk) in _DECODE_DSV4_DISPATCH`.
Our failure was purely the `page_block_size` term (32 != 64).

**Design facts a replacement kernel must honour:**
- Decode is **split-K over the topk dimension**: `num_splits = ceil(topk/_BI)`,
  i.e. `ceil(512/64) = 8` splits, with a separate merge kernel combining
  per-split partial outputs. Caller-supplied scratch:
  `mid_out [T,H,num_splits,512] bf16`, `mid_lse [T,H,num_splits] fp32`.
  This is why a naive one-program-per-(token,head) Triton kernel is slow —
  it uses 64 programs where the shape supports 8x more parallelism.
- `extra_topk` is a **separate second region**: `num_splits_main +
  num_splits_extra`; the main set uses `kv_cache`/`indices`/`topk_length`, the
  extra set uses `extra_kv_cache`/`extra_indices`/`extra_topk_length`. This is
  the SWA + compressed two-source structure.
- **Head padding**: `HPB = 16`; the kernel pads the head tile to 16 with
  zero-Q rows and gates writes by `NUM_HEADS`. TP8 -> 8 heads -> pad to 16.
- `D_V` is universally 512; `output` last dim must be 512; `out_lse` is a
  separate fp32 output.
- Supported compute capability via `supported_compute_capability`; model_type
  distinguishes DSV4 vs DSV3_2 (topk 2048 / d_qk 576 for 3_2).

### Triton reference written and validated (uncommitted -> now committed)
`vllm/models/deepseek_v41/nvidia/sparse_mla_sm120_triton.py`
(plus sweep variant `_smla_sm120_sweep.py`).

Validated against a dense torch reference, rel-L2:
| case | rel-L2 |
|---|---|
| ratio1, index_topk=512 | 3.80e-08 |
| ratio2, index_topk=512 | 2.26e-07 |
| ratio0 (SWA-only) | 0.0 |
| pos=65535, ratio2 | 2.26e-07 |
| all rows topk_len==0 | 0.0 (exact zeros, no +inf LSE) |
| topk_len > pool (repeats) | 1.50e-07 |
| duplicate-slot accumulation vs closed form | 6.65e-08 |

Perf (T x H programs, BLOCK_TOPK sweep, TOPK=512, d=512, SM120, 188 SMs):
BLOCK_TOPK 64 is the sweet spot at ~68 us/call for T=8,H=8.
T=1: 87.7us, T=8: 85.6us, T=64: 128.8us (one-program-per-(token,head) version).
Grid at T=8,H=8 is 64 programs on 188 SMs = 34% occupancy — the reason it is
slow, and exactly what FlashInfer's `_BI=64` split-K solves.


### Triton kernels committed

| file | role |
|---|---|
| `vllm/models/deepseek_v41/nvidia/sparse_mla_sm120_triton.py` | readable reference: one program per (token, head), online softmax |
| `vllm/models/deepseek_v41/nvidia/sparse_mla_sm120_split.py` | split-K form matching FlashInfer's `_BI=64` contract |

Both are pure torch+triton (no vllm import needed for testing) and are
validated against a dense torch reference.

**Correctness (final, 10 cases, split-K and 1-program both vs torch reference):**

| case | split-K rel-L2 | 1-prog rel-L2 |
|---|---|---|
| T=1 H=8 TOPK=512 ratio2 | 0.0 | 0.0 |
| T=8 H=8 TOPK=512 ratio2 | 2.00e-07 | 1.62e-07 |
| T=16 H=8 TOPK=512 ratio2 | 2.37e-07 | 1.87e-07 |
| T=4 H=8 TOPK=512 ratio1 | 3.08e-08 | 2.41e-08 |
| T=8 H=8 TOPK=512 ratio0 (SWA-only) | 0.0 | 0.0 |
| T=8 H=8 TOPK=128 ratio2 | 2.24e-07 | 1.64e-07 |
| T=64 H=8 TOPK=512 ratio2 | 2.66e-07 | 2.08e-07 |
| T=32 H=16 TOPK=512 ratio2 | 2.45e-07 | 1.96e-07 |
| T=64 H=16 TOPK=1024 ratio2 | 2.60e-07 | 2.23e-07 |
| T=128 H=8 TOPK=512 ratio2 | 2.68e-07 | 2.13e-07 |

All at the bf16 accuracy floor. Empty rows (`topk_len == 0`) return exact zeros
and a `-inf` LSE — no nan, unlike the all-masked-row `+inf` LSE defect seen in
stock FlashInfer 0.6.18.

**Performance (SM120, 188 SMs, TOPK=512, d_qk=d_v=512, H=8, pool 131072):**

| T | 1-prog us | split-K us | speedup |
|---|---|---|---|
| 1 | 87.1 | 159.6 | 0.55x |
| 2 | 86.3 | 146.6 | 0.59x |
| 4 | 85.4 | 104.1 | 0.82x |
| 8 | 85.2 | 63.1 | **1.35x** |
| 16 | 86.8 | 60.8 | **1.43x** |
| 32 | 105.5 | 59.0 | **1.79x** |
| 64 | 122.8 | 90.3 | 1.36x |

Split-K wins for T >= 8 (the real decode regime, since `_DECODE_MAX_TOKENS`
is 64 and continuous batching typically runs 8-64 tokens/step). It loses at
T < 8 where the second launch dominates — a dispatcher should pick per batch.

`BLOCK_TOPK` sweep at T=16 (all correct): 32 -> 166.7us, **64 -> 150.8us (best)**,
128 -> 154.9us, 256 -> 202.7us, 512 -> 472.0us. 64 matches FlashInfer's `_BI`.

**Merge derivation (three wrong attempts, recorded so it is not redone):**
split partials must be normalized by their own `l_s` and merged as a plain
LSE-weighted average:
```
out = sum_s exp(lse_s - G) * (acc_s / l_s) / sum_s exp(lse_s - G)
```
Weighting a *raw* `sum_i exp(x_i - m_s) v_i` by `exp(lse_s - G)` double-counts
`l_s` and is off by ~48x at TOPK=512. Verified numerically: variant
`acc/l`-then-`w` gives 1.56e-07, the raw form 2.45e-01.


## 2d. RETRACTED "root cause" (2d) — `_get_indexer_block_alignment` is NOT our path

**The section that previously occupied this space was wrong. Correcting it here
rather than deleting it, so the same false lead is not followed twice.**

Two claims were made and both fail on inspection:

1. *"`index_kpool` is absent, so the SM120 page-64 alignment never engages."*
   `_get_indexer_block_alignment` (`cuda.py:428`) is called from exactly one
   place — `platforms/interface.py:912`, inside `_align_hybrid_block_size`,
   which only runs for **hybrid attention/mamba** models. DeepSeek-V4.1's
   `text_config` has no `mamba_cache_mode`, `mamba_d_state`, `hybrid` or
   `layer_types` key, so that function is never entered. Not our code path.

2. *"`128 // compress_ratio = 32`, so compress_ratio must be 4."*
   Proven false: the checkpoint's `compress_ratios` are
   `{0: 5, 2: 18, 1: 20}` — **no 4 and no 128 anywhere**, and
   `deepseek_v41/attention.py:305` hard-raises for anything outside `{0,1,2}`.
   `128 // ratio` for ratio in {1,2} is **128 or 64 — never 32**.

### What IS verified about the block-size path
- `platforms/interface.py:880` sets
  `kernel_block_alignment_size = max(min(supported_kernel_block_sizes), block_size)`
  and then, for MLA models, forces it to `>= 128`.
- The SM120 backend declares `get_supported_kernel_block_sizes() -> [128]`
  (`flashinfer_sparse.py:116`), so alignment is 128 — consistent with the
  `--block-size 128` default that produced the failure.
- `flashinfer_sparse.py:403` computes
  `compressed_block_size = attn_metadata.block_size // self.compress_ratio`
  and this value (or the `swa_metadata.block_size` used on the `swa_only`
  branch at line 394) is what reaches FlashInfer as `page_block_size`.

### Therefore: the arithmetic does NOT explain the observed 32, and we must measure
Two candidates remain, neither established:
- **(a)** the SM120 path routes through `swa_metadata.block_size` (line 394,
  the `swa_only` branch) with a value that is not `attn_metadata.block_size`;
- **(b)** the reported `page_block_size` is derived from the *packed* cache
  (`STORE_KV_BLOCK_SIZE` / the `fp8_ds_mla` record layout), not from the token
  block at all.

### The decisive experiment (unchanged, and now the required next step)
Rebuild on the host (build prerequisites in section 1), then instrument
`flashinfer_sparse.py` around lines 394 and 403 to print:

    print("SMLADBG", token=..., swa_only, attn_metadata.block_size,
          swa_metadata.block_size, self.compress_ratio, compressed_block_size)

One run distinguishes (a) from (b) immediately. Do **not** propose a fix before
that number is observed; two rounds of static inference have already produced
wrong answers.

## 2e. THE MECHANISM (verified from FlashInfer source, 2026-09-18)

Recovered from the image layer (survives even with no container running):
`.../snapshots/13860/fs/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py`

**`page_block_size` is derived PURELY from the KV cache tensor's SHAPE — it is
not read from any vLLM config value.**

```python
_BPT_DSV4 = 584          # 448 NoPE + 128 RoPE + 8 fp8 scale, bytes/token

def _packed_kv_page_block_size(kv_cache, *, model_type, name):
    bytes_per_token = _bytes_per_token_for_model_type(model_type)   # 584 for DSV4
    if kv_cache.ndim == 2:
        return int(kv_cache.shape[1]) // bytes_per_token
    if kv_cache.ndim == 3:
        if kv_cache.shape[-1] != bytes_per_token: raise ValueError(...)
        return int(kv_cache.shape[1])
    if kv_cache.ndim == 4:
        if kv_cache.shape[-1] != bytes_per_token: raise ValueError(...)
        if kv_cache.shape[1] == 1: return int(kv_cache.shape[2])   # HND
        if kv_cache.shape[2] == 1: return int(kv_cache.shape[1])   # NHD
```

Called at line 329 as `kv_pbs = _packed_kv_page_block_size(kv_cache, ...)` and
compared at line 335 to `_DECODE_DSV4_PAGE_BLOCK_SIZE` (64).

### Consequence
`page_block_size=32` means **vLLM handed FlashInfer a cache with 32 tokens per
block**. For ndim 3/4 that is the block dim read straight off the tensor; for
ndim 2 it is `18688 // 584 = 32`. Either way it is a **cache-construction**
property, not a config lookup — which is why every `--block-size` flag we tried
failed to move it in the expected direction, and why static reasoning about
`compress_ratio` was the wrong tree to bark up.

### Independent corroboration from the earlier error text
The `--block-size 256` run reported
`block stride 460224 != page 74752`. Note `74752 / 584 = 128.0` exactly — so
that run's *page* really was 128 tokens — while `460224 // 584 = 788 r 32`, i.e.
the block stride is not a clean multiple of 584 and carries per-block padding.
That padding is consistent with the packed `fp8_ds_mla` layout and means the
tensor-shape arithmetic must be done in **bytes**, not tokens.

### What to observe (the measurement, now much more targeted)
Print the actual tensor vLLM builds, immediately before it reaches FlashInfer:

    from flashinfer.mla._sparse_mla_sm120 import _packed_kv_page_block_size
    print("SMLADBG kv_cache", self_kv_cache.shape, self_kv_cache.stride(),
          self_kv_cache.dtype, self_kv_cache.element_size(),
          "pbs=", _packed_kv_page_block_size(
                     self_kv_cache.view(torch.uint8) if self_kv_cache.dtype==torch.float8_e4m3fn
                     else self_kv_cache,
                     model_type=MODEL_TYPE_DSV4, name="dbg"))

and the same for `swa_kv_cache`. `_as_sparse_cache` (`flashinfer_sparse.py:607`)
is the function that reshapes/unsqueezes it, so the shape is decided there and
upstream in the cache manager.

This single print replaces all further static inference. Two rounds of config
reasoning have already produced wrong answers (section 2d); do not add a third.


## 2f. COMPLETE CHAIN (2026-09-18) — replaces every earlier hypothesis

The target is `page_block_size == 64`. Assembling the verified facts:

**1. The compressed page is `block_size // compress_ratio`.**
`sparse_mla.py:32-36` says so in as many words:
```
# v4.1 per-layer compress ratios: 0 = SWA, 1 = full-length compressed,
# 2 = ratio-2 compressed. Ratio-1 and ratio-2 layers both attend over indexer
# topk indices into a shared compressed cache but differ in compressed page
# block size (block_size // ratio), so each needs its own tile-scheduler plan.
```
and `flashinfer_sparse.py:403` computes exactly that.

**2. `block_size` is the *manager* block, and for MLA it is forced to >= 128.**
`platforms/interface.py:880`:
```python
kernel_block_alignment_size = max(
    min(s.base if isinstance(s, MultipleOf) else s
        for s in backend_cls.get_supported_kernel_block_sizes()),
    cache_config.block_size,
)
if model_config.use_mla:
    kernel_block_alignment_size = max(kernel_block_alignment_size, 128)
```
SM120's backend declares `get_supported_kernel_block_sizes() -> [128]`
(`flashinfer_sparse.py:116`), so `block_size >= 128` regardless of any
`--block-size` we pass below that.

**3. Therefore the compressed page is `128 // ratio`:**

| layer type | ratio | `block_size // ratio` | == 64? | layers |
|---|---|---|---|---|
| SWA-only (ratio 0) | 0 | n/a, no compressed cache | — | 5 |
| C1A | 1 | **128** | no | 20 |
| **C2A** | **2** | **64** | **yes** | **18** |

**4. This is the whole explanation, and it is not a bug in vLLM.**
FlashInfer's SM120 decode demands a page of exactly 64 and vLLM's SM120 backend
declares a manager block of 128, so **only C2A layers (ratio 2) satisfy the
constraint**; every C1A layer computes 128 and every SWA-only layer has no
compressed cache to feed it.

Our observed `page_block_size=32` then follows from the *mixed* pipeline: when
any layer in the batch presents a page of 128 (C1A) or the SWA branch passes
`swa_metadata.block_size` (`flashinfer_sparse.py:394`), FlashInfer's dispatch
rejects it, and the `min(PAGED_MQA_PAGE_SIZES)` = 32 fallback is what surfaces
in the error. The 32 is a *consequence of rejection*, not the input.

### What this means for the kernels just landed
They are directly usable, and the earlier statement that C2A "needs a
reimplementation" was wrong for a third distinct reason: C2A is
`_layer_type_for`-supported, `tile_sched_c2a`-supported, and — as shown here —
**C2A is precisely the ratio that already yields the page FlashInfer wants.**

### The remaining real gap
Per-layer-type dispatch. C2A layers (page 64) can go to FlashInfer; C1A layers
(page 128, and 20 of 40 layers) cannot, and need either
- a C1A-capable SM120 decode (page 128) — which the Triton kernels here can
  provide, since they take the page size as an input rather than hard-coding 64; or
- upstream support for page-128 SM120 decode.

This is a modelling decision to make with the user, not something to guess at.


### Verification: the Triton kernels are page-size agnostic (measured)

The claim in 2f — that these kernels can cover the C1A (page-128) layers that
FlashInfer rejects — was tested rather than asserted. Neither kernel takes a
page size: both consume **flat global slot indices**. Feeding the same logical
slots through four different block->slot translations gives bit-identical
results:

| page | split-K rel-L2 | 1-prog rel-L2 |
|---|---|---|
| 32 | 2.785e-07 | 2.202e-07 |
| 64 | 2.785e-07 | 2.202e-07 |
| 128 | 2.785e-07 | 2.202e-07 |
| 256 | 2.785e-07 | 2.202e-07 |

(`_BI = 64` in the split kernel is the **topk split tile**, unrelated to the KV
page size — do not confuse the two.)

### Erratum
Commit `fb89f90a1`'s subject line reads "the KV cache tank shape"; it should read
"tensor shape". Cosmetic only, but the commit is already published so the
subject is not being rewritten; the message itself is correct.


## 2g. MEASURED ROOT CAUSE (2026-09-18) — the page size is `attention.py:529`

The instrumented run finally answered it. Excerpt from `/tmp/recon.log`:

```
INFO [interface.py:621] Setting kv cache block size to 128 for FLASHINFER_MLA_SPARSE_DSV41 backend.
INFO [utils.py:320]     Using BLHNC KV cache layout.
ERROR [multiproc_executor.py:1053] ValueError: SM120 sparse-MLA has no decode kernel
      for this shape: num_tokens=2, num_heads=8, topk=128, d_qk=512,
      page_block_size=32, model_type=1, extra_topk=0.
SMLADBG_CACHE shape=(133619, 32, 1, 584) strides=(230400, 584, 584, 1)
              dtype=torch.uint8 ndim=4 impl_page_block_size=32 (need 64)
```

**The tensor handed to FlashInfer has 32 tokens per page.** FlashInfer's
`_packed_kv_page_block_size` reads `shape[1] == 32` (NHD, since `shape[2] == 1`)
and compares it against `_DECODE_DSV4_PAGE_BLOCK_SIZE == 64`.

### Where 32 comes from — three candidate sites, one of them explicit
1. `vllm/models/deepseek_v41/attention.py:529` —
   `DeepseekV4SWACache(..., block_size=32, ...)`. **This is an explicit
   `block_size=32` argument.**
2. `sparse_swa.py:83` — the class *default* is `block_size: int = 64`, so
   something is deliberately passing 32.
3. `sparse_swa.py:114-116` — `self.block_size = block_size` with the comment
   *"Any multiple of 32; the sparse decode kernels take the page size at
   runtime."*

### Every earlier hypothesis was wrong; this one is measured
| hypothesis | verdict |
|---|---|
| `index_kpool` guard skips SM120 alignment | **wrong** — that path is hybrid-mamba only |
| `128 // compress_ratio == 32` implies ratio 4 | **wrong** — checkpoint has no ratio 4; 128//{1,2} = {128,64} |
| page size read from a vLLM config value | **wrong** — it is read from the tensor shape |

The `Setting kv cache block size to 128` log line and the measured
`shape[1] == 32` are consistent only if the **manager block (128)** and the
**cache page (32)** are different quantities — which they are: the former is
`cache_config.block_size`, the latter is `DeepseekV4SWACache.block_size`.

### Stride caveat (unresolved, and it matters)
`strides=(230400, 584, 584, 1)` with payload `32 * 584 = 18688`, and
`align_up(18688, 512) = 18944 != 230400`. So `stride[0]` is **not** this page's
own content — the cache is a shared/unified allocation whose page stride spans
something larger. Do not assume `stride[0] == block_size * bytes_per_token` when
computing a fix.

`230400 = 450 * 512` exactly, and `packed_page_alignment` is `512` when
`kv_mxfp8` (`attention.py:508`). The relationship between 450, the 32-token page
and the number of pages (`133619`) needs one more measurement.

### Why the run is still valuable despite `PYTHONPATH` not propagating
`PYTHONPATH` was lost between the wrapper shell and vLLM's subprocesses, so the
server ran from `cwd=/build`, which shadowed the installed package via
`sys.path[0]`. `/build/vllm` *is* the newly built tree, so the run genuinely
exercised the SM120 code — and the `SMLADBG_CACHE` line did fire (on TP7).
Only the `_SMLADBG_log` instrumentation in `_forward_sparse_impl` did not,
because it sits after the failing dispatch.

### Serving the logs while a run is live
```
python3 -m http.server 8899 --bind 0.0.0.0 --directory /var/log/dsv41
```
with a loop that re-copies `docker exec <c> cat /tmp/recon.log` every 5 s.
Guard each `docker exec` with `timeout` — an unguarded one in a slow container
wedged the refresh loop and made a healthy run look stalled.


## 2h. THE FIX WORKS — `block_size=64` clears the page gate; next blocker is the indexer

**Experiment:** one-line change at `vllm/models/deepseek_v41/attention.py:522`,
`DeepseekV4SWACache(..., block_size=32, ...)` -> `block_size=64` (the class
default at `sparse_swa.py:83` is already 64).

**Result — the measured tensor changed exactly as predicted:**

```
SMLADBG_CACHE shape=(133619, 128, 1, 584) ... impl_page_block_size=128 (need 64)
SMLADBG_CACHE shape=(133619,  64, 1, 584) ... impl_page_block_size=64  (need 64)
```

```
no decode kernel count: 0        <-- the original blocker is GONE
GPU KV cache size: 1,130,792 tokens (138.04x concurrency), 28.67 GiB/worker
```

`page_block_size=32` is no longer produced; a cache with `shape[1] == 64` now
exists and satisfies `_DECODE_DSV4_PAGE_BLOCK_SIZE`. The old error string
appears **zero** times in the run.

### The next blocker, one layer deeper

```
RuntimeError: Assertion error
(/build/cmake-build-release/_deps/deepgemm-src/csrc/apis/attention.hpp:409):
block_kv == 32 or block_kv == 64
```

Call chain (confirmed):
`vllm/v1/attention/backends/mla/indexer.py:1503`
`  -> get_paged_mqa_logits_metadata(seq_lens, self.kv_cache_spec.num_states, ...)`
`  -> vllm/utils/deep_gemm.py:688`
`  -> deepgemm csrc/apis/attention.hpp:409`, guarded by
`if (arch_major == 12) DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);`

and `num_states = block_size // tokens_per_state`
(`kv_cache_interface.py:193-196`).

So on SM120 the **indexer** cache's `block_kv` is also constrained, and it is
currently **128**:

| cache | tokens_per_state | block_size | num_states | vs assert |
|---|---|---|---|---|
| SWA/compressed (just fixed) | = compress_ratio | 128 | 64 | **passes** |
| **indexer** | **1** | **128** | **128** | **fails** |

The indexer cache is separate: `num_states` there is the full `block_size`
because `tokens_per_state == 1`, whereas the compressed cache divides by
`compress_ratio`.

### Next experiment
Give the indexer cache a `block_kv` of 32 or 64. Two candidate routes:
1. halve the indexer's block size to 64 (would give `num_states = 64`); or
2. find where the indexer spec's `block_size` is set and align it the way
   `_get_indexer_block_alignment` (`platforms/cuda.py:428`) does for the
   DeepGEMM paged-MQA path — note that function already special-cases
   `is_device_capability_family(120)` to force page 64, but (per section 2d)
   it is only reached for hybrid-attention/mamba models, so it never runs here.

Route 2 is the principled one: upstream clearly *intends* SM120 to use 32 or 64
for this kernel, and the alignment helper that would deliver it is not wired up
for a pure-attention model.

### Scope caveats on this result
- `--enforce-eager`, `--max-model-len 8192`, single launch. This characterises
  the decode-dispatch path only, not CUDA graphs or long context.
- The run **still failed**, just later and for a different reason. It is not a
  working deployment.
- `SMLADBG_CACHE` printed two shapes (128 and 64), so more than one cache type
  flows through `_as_sparse_cache`; the 128 one is the indexer's.


## 2i. Second blocker has the SAME shape as the first — SM120 falls into an SM100 `else`

Once `block_size=64` cleared the page gate, the run failed in the DeepGEMM
paged-MQA logits path:

```
RuntimeError: Assertion error (deepgemm csrc/apis/attention.hpp:409):
block_kv == 32 or block_kv == 64
```

### The assert takes a CALLER ARGUMENT, not the kernel constant
`attention.hpp:394` — `get_paged_mqa_logits_metadata(context_lens, int block_kv,
num_sms, indices)`. Note `attention.hpp:148` computes a *different*, internal
`block_kv = sm120::kMqaBlockKv` (= **128**, `sm120_dispatch.hpp:33`) used for
logits stride. The asserted one is the argument.

vLLM passes `self.kv_cache_spec.num_states` (`indexer.py:1505`), and
`num_states = block_size // tokens_per_state` (`kv_cache_interface.py:193-196`).

### Root cause: the indexer backend gives SM120 128, not 64
`vllm/v1/attention/backends/mla/indexer.py:262-263`:

```python
def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
    return [64 if current_platform.is_device_capability_family(90) else 128]
```

SM90 -> 64, **SM120 -> 128**. The `else` branch is written for SM100. This is
structurally the same bug as `sparse_mla.py:90` found in section 2b.

Consequence, with `num_states = kernel_block_size // compress_ratio`:

| kernel_block_size | ratio | num_states | assert {32,64} |
|---|---|---|---|
| 128 (SM120 today) | 1 (layers 20,24,28,32,36) | **128** | **FAIL** |
| 128 (SM120 today) | 2 (layers 2,8,14) | 64 | pass |
| 64 (SM90 today) | 1 | 64 | pass |
| 64 (SM90 today) | 2 | 32 | pass |

This also explains the two shapes seen in one run — `shape[1]=128` from the
ratio-1 indexers and `64` from the ratio-2 ones.

Corroborating comment at `indexer.py:252`:
`# Block sizes count uncompressed tokens: C4 indexer pages hold 64 rows.`

**DeepGEMM already supports 32/64 on SM120** — `attention.hpp:409` asserts
exactly that under `arch_major == 12`. vLLM is simply not asking for it.

### Applied fix (experiment 2)
`indexer.py:262` now returns `[64]` for SM120 as well as SM90, with a comment
citing the DeepGEMM assert. Combined with the section-2h `block_size=64` change,
both cache classes should satisfy SM120's constraints.

### Caveats
Same scope limits as 2h: enforce-eager, 8k context, single launch, and the run
has not yet been shown to reach a working decode.


## 2j. Third pass: the block size must agree across the WHOLE KV group

After the indexer change, the run failed with:

```
RuntimeError: Worker failed with error 'No common block size for 128.'
```

raised at `vllm/v1/worker/utils.py:386`. The resolver
(`utils.py:355-387`) requires ONE size accepted by **every backend in the
group`, because the sparse-MLA and indexer caches share a physical block.

So changing only the indexer was **necessary but not sufficient**: with
`kv_manager_block_size = 128`, the indexer now wanted 64 while the sparse-MLA
backend still wanted 128, and no size satisfied both.

### All FOUR sites that must agree
| file:line | before | after |
|---|---|---|
| `deepseek_v41/attention.py:522` `DeepseekV4SWACache(block_size=...)` | 32 | **64** |
| `deepseek_v41/nvidia/flashinfer_sparse.py:158` (`DeepseekV4FlashInferMLASparseBackend`) | `[128]` | **`[64]` on SM120** |
| `deepseek_v41/sparse_mla.py:90` (`DeepseekV4SparseMLABackend`) | `[64 if SM90 else 128]` | **`[64]` on SM90+SM120** |
| `v1/attention/backends/mla/indexer.py:262` (`DeepseekV41IndexerBackend`) | `[64 if SM90 else 128]` | **`[64]` on SM90+SM120** |

### Independent corroboration that 64 is correct
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:174`
**`FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes()` already
returns `[64, 256]`.** FlashInfer's own SM120 sparse-MLA backend therefore
declares 64 for exactly the reason the DeepGEMM assert states. The DSv4.1
backends returning 128 were the anomaly, not our requirement.

### Operational note
`flashinfer_sparse.py` did not import `current_platform` (only
`DeviceCapability`); using it there raised `NameError` at worker init. Fixed by
adding `from vllm.platforms import current_platform`, matching
`sparse_mla.py:12` and `indexer.py:21`. Worth checking any further edit in that
file for the same gap.

### Status
Run relaunched with all four changes. Still **unverified** — the previous three
attempts each surfaced a new, distinct blocker one layer deeper, so a clean
result is not yet established.


## 2k. DECODE NOW WORKS — prefill is the fourth blocker, and it is a page-size gate

**The block-size fix succeeded.** Evidence from the run:

```
INFO [interface.py:621] Setting kv cache block size to 64 for FLASHINFER_MLA_SPARSE_DSV41 backend.
INFO [gpu_worker.py:645] Available KV cache memory: 28.67 GiB
INFO [kv_cache_utils.py:2459] GPU KV cache size: 1,114,653 tokens (136.07x concurrency)
[AutoTuner]: Tuning sparse_mla_sm120_decode_dsv4: 100%|21/21
```

The `No common block size` error is gone, KV allocation succeeded, and — for the
first time — the **FlashInfer SM120 sparse-MLA decode kernel autotunes 21
variants**. Decode dispatch works. No `has no decode kernel` anywhere.

Measured shapes moved as intended:

```
shape=(263288, 64, 1, 584)  impl_page_block_size=64 (need 64)   <-- compressed cache: FIXED
shape=(263288, 32, 1, 584)  impl_page_block_size=32 (need 64)   <-- a second cache: STILL 32
```

### Fourth blocker: dual-cache prefill rejects `extra_page_block_size=32`
```
tvm.error.InternalError: Check failed: (ok) is false:
Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=8 topk=128 page_block_size=64
topk_extra=512 extra_page_block_size=32
```
All 8 workers, raised during `compile_or_warm_up_model`.

### Exact gate, read from the FlashInfer JIT source
`flashinfer/data/csrc/sparse_mla_sm120_prefill.cu:324-325`,
`dispatch_dsv4_dual`:

```cpp
if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
    topk_extra % BI == 0 && (extra_page_block_size == 64 || extra_page_block_size == 2)) {
```

Against our values (`BI = 64`):

| condition | ours | result |
|---|---|---|
| `topk == 128` | 128 | pass |
| `topk_extra % BI == 0` | 512 % 64 = 0 | pass |
| `extra_page_block_size == 64` | 32 | **fail** |
| `extra_page_block_size == 2` | 32 | **fail** |

So the **extra** cache's page must be **64 or 2**, and ours is **32**.

Note the dispatch is entered because `extra_KV_cache != nullptr`
(`sparse_mla_sm120_prefill.cu:417`), i.e. the dual-cache path is DSV4-only and
used as soon as `extra_topk > 0`. `2` being an accepted value is a strong hint:
it corresponds to a *compressed* extra page, not a token page.

### Consequence — the `block_size=64` fix was incomplete
Changing `DeepseekV4SWACache(block_size=...)` to 64 fixed the **main compressed**
cache (64 confirmed) but the cache that reaches prefill as `extra_KV_cache`
still reports 32. Two distinct caches, two distinct page sizes, and only one was
addressed. The remaining `32` is a `/2` of 64 somewhere in the extra path —
consistent with the earlier observation that `shape=(N, 32, ...)` reappeared
alongside the fixed 64.

### This is architecture-level, not cosmetic
The user's read is right: v4.1's extra/swa cache topology differs from what the
DSV4 dual-prefill kernel expects. The kernel wants `extra_page_block_size` in
{2, 64}; v4.1 naturally produces 32 for this checkpoint. Resolving it means
either
- making the extra cache's page land on 64 (or on 2 where semantically correct), or
- confirming whether v4.1 should take the **single-cache** path
  (`dispatch_dsv4_single`) instead of dual at all — `extra_topk=512` suggests a
  second cache is genuinely in play.

### Progress ledger (each run reached strictly further)
1. `page_block_size=32` decode reject -> fixed via `block_size: 32 -> 64`
2. `block_kv == 32 or 64` DeepGEMM assert -> fixed via SM120 -> 64 (4 sites)
3. `No common block size for 128` -> fixed by aligning all 4 declarations
4. **decode autotunes successfully**; now `extra_page_block_size=32` in dual prefill

Still **not serving**. `--enforce-eager`, 8k context, single launch throughout.


## 2l. CORRECTION to 2k + the precise conflict

**2k had the two caches reversed. Corrected here.**

Verified from the FlashInfer JIT source (`sparse_mla_sm120.cu:175-190` and
`:225-232`):

| error field | parsed from | in vLLM's call |
|---|---|---|
| `page_block_size` | `kv_cache` | `swa_kv_paged` = `_as_sparse_cache(swa_k_cache)` |
| `extra_page_block_size` | `extra_kv_cache` | `extra_kv_paged` = `_as_sparse_cache(compressed_k_cache)` |

and the wiring is at `deepseek_v41/nvidia/flashinfer_sparse.py:964,970`:
```python
swa_kv_paged  = self._as_sparse_cache(swa_k_cache)          # -> kv_cache
extra_kv_paged = self._as_sparse_cache(compressed_k_cache)  # -> extra_kv_cache
```

So our error reads:
- `page_block_size=64` = the **SWA** cache — correct after the fix
- `extra_page_block_size=32` = the **COMPRESSED** cache — the blocker

(2k said the opposite. The 32 is the *compressed* page, not the SWA page.)

### Why the compressed page is exactly 32
`flashinfer_sparse.py:947` — `block_size = attn_metadata.block_size // self.compress_ratio`.
With the new SWA block of 64 and `compress_ratio = 2`: `64 // 2 = 32`, matching
the observed value exactly.

### The conflict is real and per-layer-type
`compressed_page = swa_block / ratio`:

| SWA block | ratio | compressed page | prefill (needs 2 or 64) | decode (needs 64) |
|---|---|---|---|---|
| 64 | **1** | **64** | **OK** | **OK** |
| 64 | **2** | **32** | **FAIL** | OK |
| 128 | 1 | 128 | FAIL | FAIL |
| 128 | 2 | 64 | OK | FAIL |

**No single SWA block satisfies both gates.** `64` is forced by decode; it then
makes ratio-2 compressed pages 32, which prefill rejects.

**Important nuance:** ratio-1 layers are entirely fine (compressed = 64). Only
the **ratio-2 layers** (18 of 40, incl. 3 of 8 index sources) break prefill. So
this is not a whole-model failure but a per-layer-type one — which suggests the
resolution may be per-layer rather than a single global block size.

This is precisely the v4.1-vs-v4.0 architectural difference the user identified:
the DSV4 dual-prefill kernel's accepted `extra_page_block_size` set is `{2, 64}`,
and v4.1's ratio-2 geometry naturally produces 32.

### Open question for the next step
`2` is an accepted value. It corresponds to a *compressed* page of 2 states,
i.e. `swa_block // ratio` with a much smaller block — or a different unit
entirely (states rather than tokens). Determining whether v4.1 is meant to pass
a *state* count rather than a token count here would decide between
"rescale the page" and "pass a different quantity".

Note also: `dispatch_dsv4_dual` additionally requires `topk_length_ptr == nullptr`
and `topk_length_extra_ptr == nullptr` — our run satisfies both, so the page size
is the only failing term.


## 2m. SEMANTICS OF {2, 64} SETTLED — 32 has no kernel instantiation

Read from `flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh`.

**`PAGE_BLOCK_SIZE` is an index-arithmetic divisor, i.e. a row count per page:**

```cpp
template <ModelType MT, int PAGE_BLOCK_SIZE>
__device__ const uint8_t* prefill_kv_entry_base(const uint8_t* kv, int idx,
                                                size_t stride_kv_block) {
  const int bi = idx / PAGE_BLOCK_SIZE;   // block index
  const int li = idx % PAGE_BLOCK_SIZE;   // offset within block
  return kv_global + bi * stride_kv_block + li * IO_STRIDE;
}
```

**`PAGE_BLOCK_SIZE_EXTRA == 2` is a layout flag, not a small page:**

```cpp
// prefill_kernel.cuh:685
static constexpr bool USE_WFP8_ROW_XOR = DUAL_CACHE && (PAGE_BLOCK_SIZE_EXTRA == 2);
```

`USE_WFP8_ROW_XOR` selects `ldmatrix_load_A_fp8_layout<USE_WFP8_ROW_XOR>` at
lines 1347/1428/1471 — a **row-XOR swizzle for the fp8 A-matrix load**, i.e. a
different on-disk format for the extra cache, not merely a 2-entry page.

### Therefore the accepted set is
| value | meaning |
|---|---|
| **64** | ordinary 64-entry pages (token-paged) |
| **2** | special 2-row swizzled fp8 layout (`USE_WFP8_ROW_XOR`) |

**There is no instantiation for 32**, which is exactly why the gate is a hard
`ICHECK` returning `ok = false`.

### Why this is an architectural gap, not a tuning knob
Our compressed page is `swa_block // compress_ratio = 64 // 2 = 32`, and 32 is
not representable. The two escape routes both cost something real:

- **Force 64** — requires `swa_block = 128` for ratio-2 layers, but decode
  requires the SWA page to be 64. Directly contradictory.
- **Force 2** — requires writing the compressed cache in the row-XOR swizzled
  fp8 layout, i.e. changing the cache *format* vLLM produces, not a block size.

### What this means for the vLLM port
Getting DeepSeek-V4.1-Flash fully serving on SM120 via vLLM requires either
1. upstream SM120 prefill support for a 32-entry compressed page (a new kernel
   instantiation), or
2. a v4.1 flattened into the DSV4 ratio geometry the kernel already templates,
   or
3. accepting the SGLang path, which already serves this checkpoint on the same
   hardware.

Decode is solved; prefill for ratio-2 layers is the remaining gap.

### Correction history in this section
- 2k stated the roles backwards (called 64 the compressed cache).
- 2l corrected the mapping; **ratio-1 layers are fine** (compressed = 64),
  only the 18 ratio-2 layers fail prefill.
- 2m (this section) establishes *why* 32 cannot work: no kernel instantiation.



---

# SECTION 4 — PAGE-32 PREFILL: SOLVED, AND WHAT IT EXPOSED

Measured on .106, 2026-09-18. Supersedes Appendix A's "architectural gap"
framing: page32 needed **kernel instantiations, not a format change**.

## 4.1 The correction to Appendix A

Appendix A claimed page size 2 was a different cache format and therefore 32
was unrepresentable. **That was wrong.** Reading
`prefill_kernel.cuh:685` more carefully:

```cpp
static constexpr bool USE_WFP8_ROW_XOR = DUAL_CACHE && (PAGE_BLOCK_SIZE_EXTRA == 2);
```

`USE_WFP8_ROW_XOR` governs `wfp8_row_xor(wrow)` on the **temporary fp8 weights in
shared memory** (`:1326-1329`). It does not describe the stored cache. It is an
optimization flag for one page geometry, not a barrier to adding another.

Adding 32 is exactly the same class of change as the working FlashInfer PR #5121
backport in `/mnt/nas/alex/models/deepseek-v41-flash-sm120`, which added extra
pages 128/256 by instantiating the existing kernel:

```cpp
if (extra_page_block_size == 64)      { DISPATCH_FULLTILE_BY_NH_PBSX(64); }
else if (extra_page_block_size == 32) { DISPATCH_FULLTILE_BY_NH_PBSX(32); }  // added
```

Both the full-tile and the length-masked prefill branches needed it, because
vLLM's dual-cache prefill passes `len(topk) == 128 == BI`, so
`topk_extra % BI == 0` holds and the **full-tile** branch is the one taken.

## 4.2 Second real bug: finite masked logits

The first page32 build passed startup and served short requests, then failed a
long-prefill retrieval. Investigating produced a genuine, page-size-independent
correctness bug in both prefill and decode.

Masking used a large finite value and **then scaled it**:

```cpp
qk[nt][0] = -1e30f;
qk[nt][0] *= sm_scale * LOG2E;      // decode_dsv4_kernel.cuh
```

In prefill the same pattern appears as `float s[4] = {qk[0] * sm_scale_log2e, ...}`
with `qk[0] = -1e30f`. Multiplying a finite sentinel keeps it finite
(`-1e30 * scale`), so `max_warp` / `block_max` become that finite value and
`exp2_fast(0) == 1`. **A fully masked row then produces nonzero attention output
and a finite LSE instead of exact zero and −inf.**

Fix: mask with `-CUDART_INF_F` (adding `#include <math_constants.h>`; the symbol
was previously only reachable transitively and `-1e30f` was chosen because it is
a literal). Negative infinity is invariant under the positive `sm_scale` scaling,
so masked entries contribute exactly zero. In prefill, the empty-row normalizer
`(l > 0.f) ? 1/l : 0` then yields exact zeros, and `softmax_lse` yields −1e30, so
the sink branch becomes `lse = sink_log2`, which is correct.

## 4.3 Measured results

**FlashInfer packed dual-cache validation** —
`benchmarks/kernels/sm120_dsv41_packed_validation.py`, 192 cases, 288 phases,
48 CUDA-graph cases, **192 passed / 0 failed**, `known_failure: 0`:

| metric | value |
|---|---|
| page32/page64 geometry pairs | 64 / 64 pass |
| max normalized error | 0.0250 |
| max absolute error | 0.0339 |
| max log2 LSE error | 0.0188 |
| tolerances | `atol=rtol=0.05` (upstream), normalized ≤ 0.05 |

An earlier 2% normalized bound was **too strict**: the unmodified AOT kernel also
failed it (2.002%–2.503%) on this varied-scale fixture. That was measured, not
assumed, by re-running the stock artifact via `sparse_mla_sm120.so.before-page32`.

## 4.4 End-to-end status (honest)

With the four geometry fixes plus the FlashInfer patch, the server **reached
`Application startup complete`** and served real generation:

| request | result |
|---|---|
| `17 * 23` | `391`, `finish_reason=stop` |
| `144 / 12` | `12`, `finish_reason=stop` |

Weight load 313.8 s; KV cache 28.67 GiB; 1,114,653 tokens; 136.07x at 8,192;
`kv cache group sizes [64]*15 + [8]`; BLHNC layout.

The long-prefill request then exposed a **third** blocker, in the FP8 paged
indexer:

```
RuntimeError: Assertion error (.../deepgemm-src/csrc/apis/attention.hpp:484):
(arch_major == 10 and (block_kv == 32 or 64 or 128)) or
(arch_major == 9 and (block_kv == 32 or 64)) or
(arch_major == 12 and ((is_fp4 and (block_kv == 32 or 64)) or
                       (not is_fp4 and block_kv == 64)))
```

SM120 was FP8-page-64-only, while the metadata builder accepts 32. Two gates
(`attention.hpp:484`, `sm120_mqa_logits.hpp:486`) now admit 32. The extension
**compiles and the assertion no longer fires** — but the numerical oracle for
this path is **not yet trustworthy**, so this is not a validated fix:

- a low-level check reached the kernel and got finite logits at page32
  (`logits_shape [2, 128]`, `finite_logits 256`, page used 32) and at page64;
- but the case matrix fails on my own harness bugs (shape mismatches,
  normalized error exactly 1.0), **identically at page64 and page32**, so it
  proves nothing about the patch either way.

Also known: `head_dim=128, num_heads=64, next_n_atom=2` needs 101,380 shared
bytes against a 101,376 cap (4 bytes over), so that combination still rejects
safely. Head/atom combinations that fit: 128/64/next_n=1 uses 84,996.

## 4.5 Standing warning

The prefill page32 fix is validated. The **masked-logit fix changed shared
numerics for every page size**, which is a larger blast radius than the port
itself; page64 controls pass, but the decode path was exercised only through
`--enforce-eager` at 8,192 context. Do not describe the port as complete.


---

# SECTION 5 — FP8 PAGED INDEXER AT PAGE 32: VALIDATED

Supersedes the "unvalidated" caveat in section 4.4. The gate change is now
measured, not merely assumed.

## 5.1 What was actually wrong in the earlier attempt

Two separate things, and neither was the kernel:

1. **A second admission gate.** `attention.hpp:409`, inside
   `get_paged_mqa_logits_metadata`, has its **own** SM120 check
   (`block_kv == 32 or block_kv == 64`) distinct from the logits gate at
   `:484`. Probing with `block_kv == 16` there produces exactly the failure the
   server logged, which is how the second gate was identified.
2. **A hand-rolled oracle.** Five successive harness bugs (CPU tensors passed
   to a CUDA `randn`, a CPU generator handed to `randperm` on a CUDA tensor,
   einsum output axis order, and a wrong region model). Re-deriving the
   semantics was the mistake; DeepGEMM's own
   `tests/test_attention.py:ref_paged_mqa_logits` already defines them.

## 5.2 What the kernel actually does, per the reference

* `context_lens[i]` is the request length and **all** `next_n` queries score the
  same final position (`q_offsets = [context_len] * next_n`), masked to
  `k_offsets < context_len`.
* The FP8 page is `[block_size * head_dim]` values **then**
  `block_size * 4` bytes of per-token FP32 scales — one contiguous scale
  region, not interleaved per-token records.
* `block_kv` is **bytes of KV per page**, not a token count. A 32-token page at
  head_dim 128 is `block_kv = 32`; the `group = 128 / block_kv` split is the
  kernel's own concept.

## 5.3 Measured results

`benchmarks/kernels/sm120_paged_indexer_validation.py`, 10 cases, greedy, no
known-failure waivers — **10/10 pass**:

| case group | pages under test | max normalized error |
|---|---|---|
| `g1` (64-token store block) | page 64 | 1.07e-7 |
| `g2` (32-state region, ratio-2 shape) | **page 32** | **1.14e-7** |

Covered: batch 1/2/3, context 128/256/512/1024, `next_n` 1/2/3, shuffled block
tables, 8448-byte padded physical strides, 9 distinct scale exponents, and
`cache_strides == [8448, 132, 132, 1]` at both page sizes.

The oracle decodes values back from **the packed bytes** rather than
recomputing from pre-quantization inputs (which is what the upstream reference
does and which would hide FP8 roundoff). Errors at ~1e-7 indicate near-exact
agreement with FP32 accumulation.

## 5.4 Honest limits of this evidence

* It validates the **logits kernel at page 32**, not the whole indexer stack.
  The top-k selection and the sparse-attention consumer are not covered here.
* `head_dim=128, num_heads=64, next_n_atom=2` still needs 101,380 shared bytes
  against a 101,376 cap, so that combination still rejects safely. Current
  accessor heads (32) are unaffected.
* The masked-logit fix from section 4.2 remains the larger-blast-radius change
  and still warrants scrutiny beyond the eager 8,192-context runs.


---

# SECTION 6 — CUDA GRAPHS AT 64K: CAPTURES, THEN DIES. NOT THE ATTENTION PORT.

Measured 2026-09-18. `--no-enforce-eager --compilation-config
{"cudagraph_mode":"PIECEWISE"}`, `--max-model-len 65536`, port 30101.

## 6.1 What worked

* **Startup completed** and **51/51 CUDA graphs captured** (0.57 GiB CUDAGraph
  memory; peak activation 3.68 GiB). KV cache 26.59 GiB, slightly below the
  eager 28.67 GiB.
* Nothing in the page-32 prefill, masked-logit, or indexer changes blocked
  capture. Graph capture exercises the real kernels, so this is meaningful.

## 6.2 What failed

Every request returned **empty output with `finish_reason=length`** — no text at
any `max_tokens` (1, 2, 8, 64), on both `/v1/chat/completions` and
`/v1/completions`. Identical prompts that produced `391` and `12` under eager
produced mojibake or nothing. The engine then died:

```
RuntimeError: NCCL error: unhandled cuda error (run with NCCL_DEBUG=INFO)
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
```

Note the NCCL errors are **secondary**: the illegal access poisoned the CUDA
context, and the next collective surfaced it.

## 6.3 Root cause — a real bug, but NOT in the port

The first failure to surface is a Triton kernel compiled **during inference**:

```
WARNING [jit_monitor.py:140] Triton kernel JIT compilation during inference:
_ring_slot_mapping_kernel. ... consider extending warmup to cover this shape.
```

`_ring_slot_mapping_kernel` lives in
`vllm/models/deepseek_v41/compressor.py:61` — the V4.1 **compressor** path, not
the attention port touched by this work. Two facts line up:

1. It compiles lazily **at first request**, i.e. **outside** graph capture, so
   the captured graphs replay against a kernel/shape the warmup did not cover.
2. `CompressorMetadataBuilder._cudagraph_support = AttentionCGSupport.ALWAYS`
   claims unconditional graph safety, while `CompressorStateCache.block_size`
   is computed as `max(8, 1 << (rows_per_step - 1).bit_length())` from
   `num_speculative_tokens + 2` — i.e. a per-model ring capacity, independent
   of the attention geometry changed here.

So the honest attribution: **the V4.1 compressor ring-slot metadata path is not
graph-safe in this configuration**, and it is not part of what was ported.

**Not established:** whether the illegal access is ultimately inside
`_ring_slot_mapping_kernel` itself, or whether that kernel merely ran first and
the async fault is reported at the next sync. The stack shows the fault
surfacing at `Triton Error [CUDA]` and again at an `all_reduce`, consistent with
async reporting. Confirming needs `CUDA_LAUNCH_BLOCKING=1` — not done.

## 6.4 Consequence

**Eager at 8,192 is the validated configuration and is untouched by this.**
CUDA graphs remain unvalidated; do not enable them for this checkpoint yet.
Also still unmeasured: contexts between 8,192 and 65,536, and throughput.


---

# SECTION 7 — CUDA GRAPH FIX, PARTIAL: COMPRESSOR FIXED, INDEXER STILL BREAKS

Supersedes section 6's single-cause framing. There are **at least two**
independent graph-breaking paths; one is now fixed and one is not.

## 7.1 Cause 1 — compressor ring-slot mapping (FIXED)

`CompressorMetadataBuilder.build` (`models/deepseek_v41/compressor.py`) launched
with shapes derived from the runtime token count while declaring
`_cudagraph_support = ALWAYS`:

```python
num_tokens   = common_attn_metadata.slot_mapping.numel()   # varies per step
slot_mapping = self.slot_mapping_buffer[:num_tokens]        # runtime-sized slice
_ring_slot_mapping_kernel[(triton.cdiv(num_tokens, 256),)]  # runtime-sized grid
```

Capture and replay therefore disagreed on both the slice address and the grid.

**Fix:** drive the grid, the slice, and the `num_tokens` argument from the
persistent buffer size instead of the runtime token count, and return the
runtime-sized view only in the metadata. The kernel already masks every load by
`num_actual_tokens` and every store by `num_tokens`, so covering the whole
buffer is safe; lanes past `num_actual_tokens` now receive the `-1` sentinel the
kernel already specified via `tl.where(valid, slot, -1)`.

**Verified, not assumed:** after the fix `_ring_slot_mapping_kernel` no longer
appears in the `jit_monitor` "compiled during inference" warnings, and the
server **survives** graph mode (HTTP 200) where it previously died with
`CUDA error: an illegal memory access was encountered`.

## 7.2 Cause 2 — candidate-block selection (NOT FIXED)

Output is still empty with `finish_reason=length`. The next kernels to compile
during inference are in
`model_executor/kernels/attention/dsa/candidate_blocks.py`:

```
_block_scores_kernel, _candidate_flags_kernel, _mask_candidates_kernel
```

These take `width` / `nblocks` / `k` as **runtime scalars** and derive their
grids from them, plus allocate inside the captured region:

```python
rows, width = logits.shape
nblocks = triton.cdiv(width, block_size)
scores  = logits.new_empty((rows, nblocks))          # allocation
_block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](...)
_mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](...)
```

They run on the **decode** path via `sparse_attn_indexer.py:735,745` — exactly
what a single-token graph replay executes.

**Likely mechanism:** the captured candidate grid/flags do not match the replay
shape, no candidate blocks survive, the indexer emits no top-k positions, sparse
attention has nothing to attend, and the sampler stops immediately. That is
consistent with the observed empty string plus `length` at every `max_tokens`,
including 1.

**Not established:** this is a plausible reading of the call graph, not a
measurement. Pinning it needs `CUDA_LAUNCH_BLOCKING=1` or a decode-step capture
with per-kernel shapes dumped.

## 7.3 Status

| item | state |
|---|---|
| compressor graph path | **fixed**, evidenced by log change |
| crash on graph replay | **gone**, server survives |
| correct graph output | **not achieved** |
| graph throughput | unmeasurable until output is correct |

**Eager at 8,192 remains the validated configuration** (sections 4–5) and is
unaffected by these changes.

---

# APPENDIX A — DRAFT upstream issue (NOT FILED — user asked to hold)

Status: **do not file.** Written for later use. Every claim below is backed by
the measurements in sections 2g–2m; file:line references were verified against
the built tree and the FlashInfer image layer.

---

**Title (draft):** `[Bug] DeepSeek-V4.1-Flash cannot use FLASHINFER_MLA_SPARSE on
SM120: ratio-2 layers compute a 32-entry compressed page, which
dispatch_dsv4_dual has no instantiation for`

**Hardware/software**
- 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), CUDA 13.0
- vLLM built from source, `TORCH_CUDA_ARCHITECTURES=120`
- FlashInfer 0.6.18 (as shipped in the image)
- Checkpoint: DeepSeek-V4.1-Flash, `model_type=deepseek_v41`,
  `compress_ratios=[0,1,2]`, `index_topk=512`, 40 hidden layers

**Summary**

`FLASHINFER_MLA_SPARSE_DSV41` on SM120 requires several block-size values to be
64 rather than 128, and even after aligning all of them, prefill for
`compress_ratio == 2` layers fails because the compressed page computes to 32
entries, which `dispatch_dsv4_dual` does not instantiate.

**Finding 1 — decode page gate (worked around)**
`_decode_dsv4_dispatchable` requires `page_block_size ==
_DECODE_DSV4_PAGE_BLOCK_SIZE` (== 64). The SM120 backend chain returned 128, and
the SWA cache was built with an explicit `block_size=32`
(`vllm/models/deepseek_v41/attention.py:522`) even though
`DeepseekV4SWACache`'s default is 64 (`v1/attention/backends/mla/sparse_swa.py:83`).

Setting these to 64 produced, for the first time,
`[AutoTuner]: Tuning sparse_mla_sm120_decode_dsv4: 100%|21/21` and removed
`SM120 sparse-MLA has no decode kernel for this shape`. Measured tensor went from
`shape=(..., 32, 1, 584)` to `shape=(..., 64, 1, 584)`.

**Finding 2 — DeepGEMM indexer assert (worked around)**
```
RuntimeError: Assertion error (deepgemm csrc/apis/attention.hpp:409):
block_kv == 32 or block_kv == 64
```
That `block_kv` is the **caller argument** to
`get_paged_mqa_logits_metadata(context_lens, int block_kv, num_sms, indices)`
(`attention.hpp:394`), not the internal `sm120::kMqaBlockKv = 128`
(`sm120_dispatch.hpp:33`, used for logits stride at `attention.hpp:148`).

vLLM passes `kv_cache_spec.num_states` (`v1/attention/backends/mla/indexer.py:1505`),
and `num_states = block_size // tokens_per_state`
(`v1/kv_cache_interface.py:193-196`). The indexer backend returned 128 for SM120:
```python
# v1/attention/backends/mla/indexer.py:262-263
return [64 if current_platform.is_device_capability_family(90) else 128]
```
**Note the corroboration:** `FlashInferMLASparseSM120Backend` at
`v1/attention/backends/mla/flashinfer_mla_sparse.py:174` already returns
`[64, 256]`. So FlashInfer's own SM120 sparse-MLA backend agrees 64 is right; the
DSv4.1 backends returning 128 look like they fell through an SM100-oriented
`else`.

**Finding 3 — all four declarations must agree**
Changing only the indexer produced `No common block size for 128`
(`v1/worker/utils.py:386`), because `utils.py:355-387` needs one size accepted by
every backend in the group and the sparse-MLA and indexer caches share a
physical block. Four sites:

| file:line | before | after |
|---|---|---|
| `models/deepseek_v41/attention.py:522` | 32 | 64 |
| `models/deepseek_v41/nvidia/flashinfer_sparse.py:158` | `[128]` | `[64]` on SM120 |
| `models/deepseek_v41/sparse_mla.py:90` | `[64 if SM90 else 128]` | `[64]` on SM90+120 |
| `v1/attention/backends/mla/indexer.py:262` | `[64 if SM90 else 128]` | `[64]` on SM90+120 |

**Finding 4 — prefill gap (NOT worked around, the actual bug report)**
```
tvm.error.InternalError: Check failed: (ok) is false:
Unsupported sparse-MLA prefill configuration:
model=DSV4 num_heads=8 topk=128 page_block_size=64
topk_extra=512 extra_page_block_size=32
```
Raised for all 8 workers during `compile_or_warm_up_model`.

Gate, `flashinfer/data/csrc/sparse_mla_sm120_prefill.cu:324-325`
(`dispatch_dsv4_dual`):
```cpp
if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
    topk_extra % BI == 0 && (extra_page_block_size == 64 || extra_page_block_size == 2)) {
```
Ours satisfies every term except the last: `extra_page_block_size` is 32.

Argument mapping verified (`sparse_mla_sm120.cu:175-190`, `:225-232`):
`page_block_size` <- `kv_cache`; `extra_page_block_size` <- `extra_kv_cache`. In
vLLM's call (`models/deepseek_v41/nvidia/flashinfer_sparse.py:964,970`) those are
the **SWA** cache and the **compressed** cache respectively — so it is the
compressed page (32) that is rejected, not the SWA page (64).

**Why 32 cannot simply be changed to 64**

`extra_page_block_size` is a template parameter
(`prefill_kernel.cuh:655-656`) because it changes the KV stride, and
`PAGE_BLOCK_SIZE` is an index divisor (`prefill_kernel.cuh:639-652`):
```cpp
const int bi = idx / PAGE_BLOCK_SIZE;
const int li = idx % PAGE_BLOCK_SIZE;
```
`PAGE_BLOCK_SIZE_EXTRA == 2` is not a small page but a **layout flag**
(`prefill_kernel.cuh:685`):
```cpp
static constexpr bool USE_WFP8_ROW_XOR = DUAL_CACHE && (PAGE_BLOCK_SIZE_EXTRA == 2);
```
which selects a row-XOR swizzle in `ldmatrix_load_A_fp8_layout<...>`
(`:1347`, `:1428`, `:1471`). So the accepted set is **{64 = token-paged, 2 =
swizzled fp8 layout}**, and **32 has no instantiation** — hence a hard `ICHECK`.

Compressed page = `swa_block // compress_ratio`. With the SWA block forced to 64
by the decode gate (Finding 1):

| layer type | ratio | compressed page | prefill |
|---|---|---|---|
| ratio-1 (22 of 40) | 1 | 64 | **ok** |
| ratio-2 (18 of 40) | 2 | **32** | **fail** |

Raising the SWA block to 128 to obtain a 64-entry compressed page would violate
the decode gate. **So no single block size satisfies both.**

**Requested outcome**

Either of:
1. an SM120 `dispatch_dsv4_dual` instantiation accepting a 32-entry compressed
   page, or
2. guidance on the intended mapping for v4.1 ratio-2 layers (whether the extra
   cache is meant to be the swizzled `2` layout, in which case the fix is on the
   vLLM side by writing that format).

**Reproduction notes**
- Checkpoint served unchanged; vLLM built from the synced fork, SM120-only.
- `--tensor-parallel-size 8 --max-model-len 8192 --enforce-eager
  --gpu-memory-utilization 0.85 --trust-remote-code`.
- Weight load is slow (~450-570 s/rank) off NFS; not part of the bug.
- Only the four block-size sites above were modified; no FlashInfer changes.

**Honest scope**
- `--enforce-eager`, 8192 context, single launch. CUDA graphs and long context
  were not exercised.
- Decode was observed to autotune and dispatch successfully, but **no request
  was served end-to-end** — prefill fails first.
- SGLang serves the same checkpoint on the same hardware, so this is specific to
  vLLM's SM120 path.


---

## 3. Earlier open question (SUPERSEDED by §2b — kept for the record)

**Arithmetic says it should work:** SM120 declares
`get_supported_kernel_block_sizes() -> [128]`
(`deepseek_v41/nvidia/flashinfer_sparse.py:116`), and
`compressed_block_size = attn_metadata.block_size // self.compress_ratio`
(`flashinfer_sparse.py:403`, also `:813`, `:893`). Our layers have
`compress_ratio = max(1, config.compress_ratios[layer_id])`
(`deepseek_v4/attention.py:226`) -> **1 or 2 only**.

`128 // 2 = 64` would satisfy FlashInfer. But the run reported **32**, i.e.
`64 // 2`. **Something halves 128 -> 64 before that division.**

### Next step (the decisive experiment)
Rebuild (~30 min), then log at `flashinfer_sparse.py:403` for a ratio-2 layer:
```python
print("META", attn_metadata.block_size, "RATIO", self.compress_ratio)
```
- if `block_size` is 64 not 128 -> the defect is upstream in the SparseMLA
  metadata builder / the `get_supported_kernel_block_sizes()` intersection
  (`vllm/v1/attention/backends/mla/sparse_swa.py`), not in this line;
- if `block_size` is 128 and page still 32 -> then `compress_ratio` is resolving
  to 4 somewhere despite `attention.py:226`, and trace that instead.

Related lead worth reading first: `vllm/platforms/cuda.py`
`_get_indexer_block_alignment` already special-cases this exact SM120
`block_kv == 64` requirement for the **indexer cache**:
```python
if cls.is_device_capability_family(120):
    # On sm120 the DeepGEMM paged-MQA kernel only accepts block_kv 64 for the
    # fp8 indexer cache, so align to the largest pool page here to make the page
    # split land on 64 not the min 32.
    page = max(PAGED_MQA_PAGE_SIZES)   # (32, 64) -> 64
```
The sparse-MLA decode path has the same requirement and is not aligned this way.

---

## 4. Corrections to earlier notes (do not re-derive)

1. **WRONG earlier:** "compress_ratio resolves to 4". It resolves to **1 or 2**
   (`attention.py:226`; the checkpoint's `compress_ratios` are `[0,1,2]`,
   43 entries for 40 layers: 5x ratio-0, 20x ratio-1, 18x ratio-2).
2. **WRONG earlier:** "vLLM implements C4A only; C2A needs a weeks-long
   reimplementation". C2A **is** implemented (`_layer_type_for`, `tile_sched_c2a`)
   and works on SM90/SM100. The `assert compress_ratio in [4, 128]` in
   `deepseek_v4/compressor.py:158` is one narrower code path, not the whole model.
3. **Not isolated:** where the extra factor of 2 enters. Notes previously framed
   this as settled; it is not.

---

## 5. Context worth knowing

### Checkpoint facts (DeepSeek-V4.1-Flash, `text_config`)
`num_hidden_layers=40`, `compress_ratios=[0,1,2]` (43 entries),
`index_topk=512`, `index_n_heads=32`, `index_head_dim=128`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`,
`hidden_size=5120`, `vocab_size=129280`, `n_routed_experts=384`,
`num_experts_per_tok=6`, MLA `head_dim=512`, `kv_source_layer_ids=[2,8,14,20]`,
engram `engram_layer_ids=[1,14]`.
Quantization: `fp8`, `weight_block_size=[32,32]`, `scale_fmt=ue8m0`,
`expert_dtype=fp4`.

An earlier gate exists that we did **not** hit: `flashinfer_mla_sparse.py`
requires `index_topk == 2048`, and ours is 512. On this path it did not fire
(backend was resolved as `FLASHINFER_MLA_SPARSE_DSV41`), but it is a sign that
upstream's reference V4.1 variant differs from this checkpoint.

### Why SGLang needs DeepGEMM only for the indexer
DeepGEMM's block-FP8 contracts are `1x128` / `128x128`; this checkpoint declares
`[32,32]`, so SGLang's `fp8_utils.py` gate routes the dense linears to Triton.
SGLang **does** use DeepGEMM for the fp4 indexer. vLLM instead maps the same
`[32,32]` UE8M0 to **MXFP8** (`FlashInferCutlassMxfp8LinearKernel`), relying on the
lossless `32x32 -> 1x32` group expansion.

### DeepGEMM in vLLM
Vendored, not pip-installed (`cmake/external_projects/deepgemm.cmake` installs to
`vllm/third_party/deep_gemm`). **Mandatory** for this model on CUDA:
`vllm/model_executor/layers/sparse_attn_indexer.py:891` raises
`RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM support`.
Overlaying synced Python must **merge** `vllm/third_party/` from the image, not
replace it, or `deep_gemm` disappears.
`has_deep_gemm_sparse_mqa` is False; the docstring says those kernels are
"added in DeepGEMM 2.8, SM100-only" — a candidate for the next failure after the
page-size issue is fixed.

### vLLM's flash-attention fork
`cmake/external_projects/vllm_flash_attn.cmake` fetches
`https://github.com/vllm-project/flash-attention.git` at tag
`506341a143fcabd4bb79052a7605ada727d6b3f5` (hence `_vllm_fa2_C`/`_vllm_fa3_C`).
`vllm/vllm_flash_attn/__init__.py` raises at import if neither exists, and
`v1/attention/backends/fa_utils.py` imports it at **module level** on CUDA — so it
must exist for much of vLLM to import at all.
Components refusing arch 12.0: `FlashMLA`, `DeepSelect`; Marlin/MoE/C2x are
`8.0+PTX`; `fmha_sm100` is SM100-only.
FA4 cute files are a separate install component:
`cmake --install . --component _vllm_fa4_cutedsl_C` (53 py files) — needed or the
worker dies with `No module named 'vllm.vllm_flash_attn.cute'`.

---

## 6. Environment / where things are

- Synced fork: `/mnt/nas/alex/models/vllm` (branch `main`), upstream remote added.
  Pushed commits: `a5ee3cbd4` (this handoff, as `VLLM-SM120-BRINGUP.md`),
  `e93398e51` (earlier notes).
- Longer raw notes: `/mnt/nas/alex/models/deepseek-ai/serve-dsv41/logs/vllm-sm120-findings.md`
- Kit (SGLang side, working): `/mnt/nas/alex/models/deepseek-v41-flash-sm120`
  @ `68c61ba`, pushed to `nguyenhoangthuan99/deepseek-v41-flash-sm120`.
- Base image used: `vllm-dsv4-vision:sm120` (vLLM `0.28.1rc1.dev137+g5ab628dd1`,
  torch `2.13.0+cu130`, FlashInfer `0.6.18`).
- **All containers/overlays were deleted at teardown.** `/tmp/build-vllm-full.sh`
  and `/tmp/ov*` are gone; rebuild from §1.
- SM120-only build (`12.0f`) was used throughout, per requirement.

## 7. Suggested resumption order
1. Rebuild (SM120-only, `MAX_JOBS=128`) incl. all prerequisites in §1.
2. Instrument `flashinfer_sparse.py:403` and run once; answer §3.
3. If `block_size` is 64: fix the metadata-builder intersection so SM120 keeps
   128, or mirror `_get_indexer_block_alignment` for sparse-MLA.
4. Then re-check `has_deep_gemm_sparse_mqa` (SM100-only kernel, may be next).
5. Validate numerics before claiming success — no accuracy baseline exists yet.