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

## 3. The open question (UNVERIFIED hypothesis — start here)

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