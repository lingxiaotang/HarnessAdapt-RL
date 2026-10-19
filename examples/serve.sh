#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${HARNESSADAPT_CHECKPOINT:-${PACKAGE_ROOT}/models/update45}"
CHECKPOINT_DIR="$(cd -- "${CHECKPOINT_DIR}" && pwd)"
SERVER_IMAGE="${HARNESSADAPT_SERVER_IMAGE:-slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342}"
exec docker run --rm --gpus all --ipc host --network host \
  -v "${CHECKPOINT_DIR}:/model:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python "${SERVER_IMAGE}" -m sglang.launch_server \
  --model-path /model --served-model-name harnessadapt-update45 \
  --host 127.0.0.1 --port "${HARNESSADAPT_PORT:-31980}" \
  --tp "${HARNESSADAPT_TP:-4}" \
  --context-length "${HARNESSADAPT_CONTEXT_LENGTH:-131072}" \
  --mem-fraction-static 0.72 --max-running-requests 16 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code "$@"
