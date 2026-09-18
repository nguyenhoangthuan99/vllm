#!/usr/bin/env bash
# Uses the existing SM120-only build; never rebuilds or removes containers.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the local checkpoint directory}"
BUILD_EXPORT="${BUILD_EXPORT:-/tmp/vllm-build-export}"
IMAGE="${IMAGE:-vllm-dsv4-vision:sm120}"
NAME="${NAME:-vllm-sm120-page32}"
PORT="${PORT:-30100}"
RUN_DIR="${RUN_DIR:-/var/log/dsv41/page32-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# Empty by default: eager is the validated configuration. Set GRAPH_ARGS to
# exercise CUDA graphs (e.g. "--enforce-eager=false --cudagraph-mode=piecewise").
GRAPH_ARGS="${GRAPH_ARGS:---enforce-eager}"
[[ -f "$MODEL_PATH/config.json" && -d "$BUILD_EXPORT/vllm" ]]
if docker container inspect "$NAME" >/dev/null 2>&1; then
  printf 'Container %s already exists; use another NAME or explicitly stop/archive it.\n' "$NAME" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
printf 'Log directory: %s\n' "$RUN_DIR"
# Source-only geometry fixes are compatible with the retained compiled extensions.
for file in models/deepseek_v41/attention.py models/deepseek_v41/sparse_mla.py \
  models/deepseek_v41/nvidia/flashinfer_sparse.py v1/attention/backends/mla/indexer.py; do
  cp "$ROOT/vllm/$file" "$BUILD_EXPORT/vllm/$file"
done
# Foreground container belongs to the supervising process. Logs remain on the host.
docker run --name "$NAME" --gpus all --ipc=host --shm-size=64g \
  --network host --entrypoint /bin/bash \
  -v "$MODEL_PATH:/model:ro" -v "$BUILD_EXPORT:/build:ro" \
  -v "$ROOT/tools:/sm120-tools:ro" -v "$RUN_DIR:/work" \
  -e PYTHONPATH=/build -e VLLM_USE_DEEP_GEMM=1 \
  -e FLASHINFER_WORKSPACE_BASE=/work/flashinfer -e MAX_JOBS=128 \
  -e PORT="$PORT" -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GRAPH_ARGS="$GRAPH_ARGS" \
  ${EXTRA_ENV:+-e $EXTRA_ENV} \
  ${PROFILE_MOUNT:+-v $PROFILE_MOUNT} \
  "$IMAGE" -c '
    set -euo pipefail
    bash /sm120-tools/apply-flashinfer-sm120-page32.sh
    cd /build
    # shellcheck disable=SC2086
    exec ${PROFILE_CMD:-} /opt/sm120-venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model /model --served-model-name dsv41 --tensor-parallel-size 8 \
      --max-model-len "$MAX_MODEL_LEN" --block-size 64 \
      --gpu-memory-utilization 0.85 \
      --trust-remote-code --host 0.0.0.0 --port "$PORT" $GRAPH_ARGS
  ' 2>&1 | tee "$RUN_DIR/server.log"
