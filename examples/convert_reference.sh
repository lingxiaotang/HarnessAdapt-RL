#!/usr/bin/env bash
# Run inside the training image with Slime, the HF model, and output mounted.
set -euo pipefail
: "${WORKSPACE:?Set WORKSPACE to the mounted training workspace}"
: "${MODEL_DIR:?Set MODEL_DIR to the original HF base model}"
: "${REF_MODEL_PATH:?Set REF_MODEL_PATH to a new conversion output directory}"
if [[ -e "$REF_MODEL_PATH" ]]; then
  echo "Reference output already exists; choose a new REF_MODEL_PATH: $REF_MODEL_PATH" >&2
  exit 2
fi
test -f "$MODEL_DIR/config.json"
export PYTHONPATH="/root/Megatron-LM:${WORKSPACE}/slime${PYTHONPATH:+:${PYTHONPATH}}"
cd "${WORKSPACE}/slime"
source scripts/models/qwen3.5-9B.sh
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint "$MODEL_DIR" --save "$REF_MODEL_PATH"
test "$(cat "$REF_MODEL_PATH/latest_checkpointed_iteration.txt")" = release
test -f "$REF_MODEL_PATH/release/common.pt"
