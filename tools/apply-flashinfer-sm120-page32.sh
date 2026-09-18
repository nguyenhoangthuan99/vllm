#!/usr/bin/env bash
# Run inside the retained vLLM build/runtime container, with this patch alongside.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/opt/sm120-venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --system-site-packages "$VENV"
fi
CSRC="$("$VENV/bin/python" -c 'from flashinfer.jit.env import FLASHINFER_CSRC_DIR; print(FLASHINFER_CSRC_DIR)')"
INCLUDE="$CSRC/../include/flashinfer/attention/sparse_mla_sm120"
apply_patch() {
  local directory="$1" patch_file="$HERE/$2"
  if patch --dry-run --silent --reverse -p1 -d "$directory" < "$patch_file" >/dev/null 2>&1; then
    printf '%s already applied.\n' "$2"
  elif patch --dry-run --silent --forward -p1 -d "$directory" < "$patch_file"; then
    patch --forward -p1 -d "$directory" < "$patch_file"
  else
    printf 'Unsupported source revision for %s; refusing partial patch.\n' "$2" >&2
    exit 1
  fi
}
apply_patch "$CSRC" flashinfer-sm120-page32.patch
apply_patch "$INCLUDE" flashinfer-sm120-masked-logits.patch
# A packaged AOT module takes precedence over JIT, even in a fresh workspace.
# Preserve it for rollback; only the patched sparse module must rebuild.
AOT="$("$VENV/bin/python" -c 'from flashinfer.jit.mla import gen_sparse_mla_sm120_module; print(gen_sparse_mla_sm120_module().aot_path)')"
if [[ -f "$AOT" ]]; then
  [[ ! -e "$AOT.before-page32" ]] || { echo 'AOT backup already exists; refusing overwrite.' >&2; exit 1; }
  mv "$AOT" "$AOT.before-page32"
fi
printf '%s\n' 'Use a fresh FLASHINFER_WORKSPACE_BASE for the first patched run.'
