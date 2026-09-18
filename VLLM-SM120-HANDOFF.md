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