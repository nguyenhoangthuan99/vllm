# vLLM on SM120 (RTX PRO 6000) for DeepSeek-V4.1-Flash — findings, 2026-09-18

Host: research-8xpro6000 (VM .106), 8x RTX PRO 6000 Blackwell 96GB.
The SGLang deployment on VM .100 was not touched.

## Summary

vLLM is **blocked by compiled-extension version skew**, not by model or hardware
support. Everything needed now exists upstream; the available prebuilt images
predate it.

## What is confirmed working / present

1. **Upstream vLLM now has full DeepSeek-V4.1 support** (synced fork
   `nguyenhoangthuan99/vllm` @ `2bbdfcfcf`):
   - `vllm/models/deepseek_v41/` (attention, sparse_mla, compressor, quant_config,
     amd/, nvidia/)
   - `vllm/transformers_utils/configs/deepseek_v41.py`
   - `vllm/tokenizers/deepseek_v41.py`, `vllm/parser/deepseek_v41.py`
   - `vllm/tool_parsers/deepseekv41_engine_tool_parser.py`
   - `vllm/reasoning/deepseek_v41_engine_reasoning_parser.py`

2. **Upstream vLLM now has native SM120 support for this model**:
   - `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`
     (`FlashInferMLASparseSM120Impl`, handles `pow2_fp32` scales)
   - `vllm/platforms/cuda.py`: for `device_capability.major == 12` the sparse
     backends are exactly `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`
   - `vllm/utils/flashinfer.py`: `has_flashinfer_sparse_mla_sm120()` and
     `has_flashinfer_sparse_mla_sm120_config(num_q_heads, top_k)` which inspects
     the FlashInfer dispatch table rather than just checking a callable exists.

3. **FlashInfer 0.6.18 (already in the local SM120 image) ships the required
   kernel**: `flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH` **contains
   `(8, 128)`** — this model's `(num_heads, topk)`.

4. **`vllm-dsv4-vision:sm120` ships torch 2.13.0+cu130** — exactly the torch
   version synced vLLM pins (`requirements/cuda.txt: torch==2.13.0`).

## What blocks it

| attempt | result |
|---|---|
| `vllm-dsv4-vision:sm120` as-is | `model type deepseek_v41 ... Transformers does not recognize` — its vLLM code predates V4.1 |
| synced vLLM overlaid on that image (Python layer + image's `.so` files) | gets **past** architecture inspection (`DeepseekV41Config` loads) but fails at `ImportError: vllm.vllm_flash_attn requires (_vllm_fa2_C or _vllm_fa3_C)` |
| image's compiled extensions | has `_flashmla_C`, `_flashkda_C`, `_C_stable_libtorch` — but **no** `_vllm_fa2_C` / `_vllm_fa3_C` |

So the model layer is new enough but the **compiled extensions are older than the
Python that wants them**. A matching build is required.

## The remaining work

Build vLLM from the synced source in a CUDA 13.0.3 image. `docker/Dockerfile` has
~169 build steps and compiles FlashAttention (`_vllm_fa2_C`/`_vllm_fa3_C`), vLLM
C++/CUDA kernels, and MoE ops. Expect **multiple hours** of compilation and a
multi-GB CUDA devel base image, and an SM120 (`12.0f`, not `12.0a`) arch setting
to be correct for RTX PRO 6000.

Unknowns that only the build will settle: whether the SM120 sparse-MLA path passes
CUDA-graph capture (the original complaint in vllm#56892), and whether DSpark
speculation is usable on this path.

## Note on the earlier upstream report

vllm-project/vllm#56892 (open, last updated 2026-09-15) reports ~2.3-4.0 tok/s
aggregate on this exact hardware/model in `--enforce-eager` mode. That report is
against the **old** day-0 image; the SM120 backend and dispatch checks above
landed in upstream afterwards, so the report may no longer reflect main. It has
**not** been re-tested here.

For reference, the current SGLang stack on the same host measures C1 196.8 /
C4 396.2 / C16 695.6 / C32 1332.2 tok/s with prefill TTFT ~0.27 s.

---

## Can vLLM use FLASHINFER_MLA_SPARSE_SM120 for this model? NO

The backend exists and is well-built
(`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`), and on SM120
(`device_capability.major == 12`) `platforms/cuda.py` selects exactly
`[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`. But it has hard gates:

| gate | requirement | this checkpoint | pass |
|---|---|---|---|
| `has_flashinfer_sparse_mla_sm120()` | FlashInfer sparse-MLA API | present (0.6.18) | yes |
| `dtype` | bfloat16 | bf16 | yes |
| `kv_cache_dtype` | **`fp8_ds_mla`** (packed layout) | `fp8_e4m3` | **NO** |
| `index_topk` | **exactly 2048** | **512** | **NO** |
| `attn_type` | decoder self-attention | decoder | yes |

Both failures are hard (`NotImplementedError` / `return` a reason string); there is
no CLI override.

The `index_topk=2048` gate is self-contradictory: FlashInfer's own
`_DECODE_DSV4_DISPATCH` **stops at 512** —
`(8,128) (8,256) (8,512) (16,128) (16,256) (16,512) (32,…) (64,…) (128,…)`.
Our model's `(8,512)` / `(32,512)` shape **is** in that table. So vLLM requires a
`topk` that its own dependency cannot serve, while rejecting the `topk` that only
it can. This looks like a gate written for a different V4.1 variant (1M-context
`topk=2048`), not this one.

`TRITON_MLA` is the other SM120 option but supports only **dense** MLA
(`TritonMLABackend` has no sparse `supports_combination` branch), so it is not a
substitute for this sparse-attention model.

## What the vLLM build revealed about SM120 coverage

Components that DO build for `12.0f`: `dsv3_fused_a_gemm`, `fp32_router_gemm`,
`scaled_mm_c3x_sm120`, `moe_data`, **SM12x NVFP4**, `fused KDA decode`,
`fused GDN decode`.

Components that REFUSE arch 12.0:
- `FlashMLA will not compile: unsupported CUDA architecture 12.0`
- `DeepSelect will not compile: unsupported CUDA architecture 12.0`
- Marlin / Marlin-MoE / scaled_mm_c2x: `8.0+PTX` only
- `fmha_sm100`: SM100-only

So vLLM's SM120 path leans on FlashInfer + Triton, not on its own MLA kernels.

## vLLM maintains its own flash-attention fork

`docker/Dockerfile` → `cmake/external_projects/vllm_flash_attn.cmake` fetches
`https://github.com/vllm-project/flash-attention.git` at pinned tag
`506341a143fcabd4bb79052a7605ada727d6b3f5`. Hence the `_vllm_fa2_C` / `_vllm_fa3_C`
extension names. `vllm/vllm_flash_attn/__init__.py` raises `ImportError` at import
time if neither exists, and `v1/attention/backends/fa_utils.py` imports it at
**module level** on CUDA — so the extensions must exist for much of vLLM to import
at all; there is no bypass.

With CUDA >= 13.0 the fork's `CUDA_SUPPORTED_ARCHS` gains `10.0;11.0;12.0`, so
SM120 is buildable (`12.0f` gencode confirmed in the build log).

## Build recipe notes (for whoever finishes this)

Needed beyond the base image: `git` (CMake FetchContent clones CUTLASS / FA /
triton / deepgemm), `ninja`, `setuptools`, `setuptools-rust`, and a
`libnvrtc.so` symlink (the image ships only versioned libnvrtc, so CMake reports
`CUDA_nvrtc_LIBRARY ... NOTFOUND`). Build from a **writable** copy of the tree
(setuptools-scm writes `vllm/_version.py`; `/src` read-only fails), and set
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM` if `.git` is absent.
CMake 4.4 rejects `12.0f` in `CMAKE_CUDA_ARCHITECTURES`; pass `120` and let vLLM's
CMake add the family suffix.

---

## Live bring-up attempt #2: got all the way to model-construction, then an ABI wall

Approach: synced vLLM **overlaid** on `vllm-dsv4-vision:sm120` (which has
torch 2.13.0+cu130, FlashInfer 0.6.18 with the SM120 DSV4 dispatch, and the
image's ABI-matched `.so` files), plus the **freshly built FlashAttention fork**.
This got dramatically further than any previous attempt:

```
quantization=deepseek_v4_fp8
enforce_eager=False                       <-- CUDA graphs NOT forced off
Using DeepSeek's fp8_ds_mla KV cache format
Using FlashInferCutlassMxfp8LinearKernel for MXFP8 GEMM
Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend
Built engram token map (129280 -> 99092 ids) for layers (1, 14)
Loading safetensors checkpoint shards: 100% Completed | 48/48
```

Errors cleared along the way, each fixed:
1. `model type deepseek_v41 ... Transformers does not recognize` -> synced tree has
   `DeepseekV41Config`.
2. `ImportError: vllm.vllm_flash_attn requires (_vllm_fa2_C or _vllm_fa3_C)` ->
   built vLLM's own fork (`vllm-project/flash-attention@506341a1`) for `12.0f`.
3. `Sparse Attention Indexer CUDA op requires DeepGEMM support` -> the overlay's
   `vllm/third_party/` shadowed the image's vendored `deep_gemm`; merging it back
   gave `has_deep_gemm = True` + `DeepGEMM PDL enabled`.
4. `No module named 'vllm.vllm_flash_attn.cute'` ->
   `cmake --install . --component _vllm_fa4_cutedsl_C` (53 Python files).

### The wall

```
RuntimeError: Worker failed with error
'_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert() expected at most 9
 argument(s) but received 13 argument(s).'
```

The synced Python calls the **13-arg** signature
(`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:1134` and
`torch_bindings.cpp:441`: `..., bool apply_q_norm, bool kv_mxfp8, bool apply_q_rope,
bool is_q_interleaved`), while the image's compiled `_C_stable_libtorch.so` only
has the **9-arg** version. New Python + old compiled extensions cannot work.

**Conclusion: `_C_stable_libtorch` must be rebuilt too**, not just FlashAttention.
The kernel source itself is fine for SM120 (`"requires sm_80+"`), so this is a
build-scope problem, not a hardware/capability problem.

### Important: index_topk gate is NOT actually hit

`kv_cache_dtype` resolved to `fp8_ds_mla` automatically, and the run reached model
construction past the sparse-indexer guard. So the earlier `index_topk == 2048`
gate did not fire on this path (it applies to `supports_combination` for backend
autoselection, which resolved differently here). Worth re-checking once the build
is complete.

### Remaining work is now precisely scoped

Build the full vLLM from the synced tree (which compiles `_C_stable_libtorch`,
`_moe_C_stable_libtorch`, `cumem_allocator`, `spinloop`, the FA fork, vendored
DeepGEMM, and the rest), i.e. the multi-hour full build, rather than the
FA-only shortcut. Build environment fixes already established are recorded above
(git, ninja, setuptools/setuptools-rust, libnvrtc.so symlink, writable tree,
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM`, arch `120` not `12.0f` for CMake 4.4).

### Also confirmed about DeepGEMM

vLLM **vendors** DeepGEMM (`cmake/external_projects/deepgemm.cmake` installs it to
`vllm/third_party/deep_gemm`); it is mandatory for this model on CUDA
(`sparse_attn_indexer.py:891` raises without it). `has_deep_gemm_sparse_mqa` is
still False, and its docstring says those kernels are "added in DeepGEMM 2.8,
SM100-only" - a candidate for the next failure.

### Why SGLang does not use DeepGEMM for the dense linears

Not a capability gap: DeepGEMM's block-FP8 contracts are `1x128` / `128x128`, and
this checkpoint declares `weight_block_size=[32,32]`, so SGLang's `fp8_utils.py`
gate routes to Triton. SGLang *does* use DeepGEMM for the fp4 indexer. vLLM
instead maps `[32,32]` UE8M0 to **MXFP8** (`FlashInferCutlassMxfp8LinearKernel`) -
the same lossless `32x32 -> 1x32` expansion measured in the MXFP8 work here.
