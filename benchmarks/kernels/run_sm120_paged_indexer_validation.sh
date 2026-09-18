#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Runs the isolated FP8 paged indexer check against the staged DeepGEMM extension.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAME="${NAME:-vllm-recon}"
WORK=/tmp/sm120-paged-indexer
docker exec "$NAME" mkdir -p "$WORK"
docker cp "$HERE/sm120_paged_indexer_validation.py" "$NAME:$WORK/sm120_paged_indexer_validation.py"
docker exec -i -e PYTHONPATH=/build "$NAME" bash -s -- "$WORK" "${@:-}" <<'RUN'
set -euo pipefail
cd -- "$1"
shift
exec /opt/sm120-venv/bin/python sm120_paged_indexer_validation.py "$@"
RUN