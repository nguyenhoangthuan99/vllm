#!/usr/bin/env bash
# Host command: CONTAINER=vllm-instr bash tools/build-deepgemm-sm120-page32.sh
# Builds only DeepGEMM _C; never stages it into /build or launches GPU work.
# vllm-instr retains the original C++20/CUDA development headers and libraries,
# including the unversioned libnvrtc linker name. vllm-recon needs the same deps.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${CONTAINER:-vllm-instr}"
SRC="${SRC:-/build/cmake-build-release/_deps/deepgemm-src}"
WORK="${WORK:-/tmp/deepgemm-page32}"
OUT="${OUT:-/tmp/deepgemm-page32-output}"
VENV="${VENV:-/opt/sm120-venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"
case "$CONTAINER" in
  vllm-instr|vllm-recon) ;;
  *) printf 'Use CONTAINER=vllm-instr or vllm-recon; refusing %s.\n' "$CONTAINER" >&2; exit 1 ;;
esac
# A unique input directory keeps invocations independent without modifying mounts.
INPUT="$(docker exec "$CONTAINER" mktemp -d /tmp/deepgemm-page32-input.XXXXXX)"
trap 'docker exec "$CONTAINER" rm -rf -- "$INPUT"' EXIT
docker cp "$HERE/deepgemm-sm120-page32.patch" "$CONTAINER:$INPUT/page32.patch"
docker exec -i -e CUDA_HOME="$CUDA_HOME" -e CXX="$CXX" "$CONTAINER" \
  bash -s -- "$SRC" "$WORK" "$OUT" "$VENV" "$INPUT/page32.patch" <<'BUILD'
set -euo pipefail
SRC="$1"
WORK="$2"
OUT="$3"
VENV="$4"
PATCH="$5"
# Keep all writes under /tmp even when the retained build mount is writable.
for path in "$WORK" "$OUT"; do
  [[ "$path" == /tmp/* && "$path" != *'/../'* && "$path" != */.. ]] || {
    printf 'Writable build path must remain under /tmp: %s\n' "$path" >&2
    exit 1
  }
done
[[ -f "$SRC/csrc/python_api.cpp" && -d "$SRC/deep_gemm/include" && -d "$SRC/third-party" ]]
[[ -f /build/tools/build_deepgemm_C.py ]]
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --system-site-packages "$VENV"
fi
mkdir -p "$WORK" "$OUT"
# Only csrc is copied. The extension builder reads the original large dependency
# trees through these links; neither patch nor compiler output targets the links.
STAGE="$(mktemp -d "$WORK/source.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT
cp -a "$SRC/csrc" "$STAGE/csrc"
ln -s "$(realpath "$SRC/deep_gemm")" "$STAGE/deep_gemm"
ln -s "$(realpath "$SRC/third-party")" "$STAGE/third-party"
# Both hunks must apply together, with exact context (no fuzz), or already be
# present together. A partial patch or unfamiliar source fails before compiling.
if patch --force --fuzz=0 --dry-run --reverse -p1 -d "$STAGE" < "$PATCH" >/dev/null 2>&1; then
  printf 'DeepGEMM page32 gates already applied.\n'
elif patch --batch --fuzz=0 --dry-run --forward -p1 -d "$STAGE" < "$PATCH"; then
  patch --batch --fuzz=0 --forward -p1 -d "$STAGE" < "$PATCH"
else
  printf 'Unsupported DeepGEMM source revision; refusing partial patch.\n' >&2
  exit 1
fi
"$VENV/bin/python" /build/tools/build_deepgemm_C.py "$STAGE" "$OUT" "$VENV/bin/python"
EXT_SUFFIX="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
[[ -s "$OUT/_C$EXT_SUFFIX" ]]
printf 'Built extension (container path): %s\n' "$OUT/_C$EXT_SUFFIX"
printf 'Stage this file separately into the host build export; /build is unchanged.\n'
BUILD
