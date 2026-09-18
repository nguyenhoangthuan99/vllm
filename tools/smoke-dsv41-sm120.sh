#!/usr/bin/env bash
set -euo pipefail
NAME="${NAME:-vllm-sm120-page32}"
PORT="${PORT:-30100}"
docker exec "$NAME" /opt/sm120-venv/bin/python /sm120-tools/smoke-dsv41-sm120.py \
  --base "http://127.0.0.1:$PORT" "$@"
