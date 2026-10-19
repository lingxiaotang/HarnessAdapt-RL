#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
WORKSPACE="${WORKSPACE:-$(cd -- "${PACKAGE_ROOT}/.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${WORKSPACE}/slime}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE}/models/Qwen3.5-9B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${MODEL_DIR}_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${PACKAGE_ROOT}/data/all_cached_train.jsonl}"
VALIDATION_DATA="${VALIDATION_DATA:-${PACKAGE_ROOT}/data/verified.jsonl}"

# Hyperparameters (Table 5 & Appendix B.1)
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-65536}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
NUM_ROLLOUT="${NUM_ROLLOUT:-46}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-46}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
R2E_HINT_MODE="${R2E_HINT_MODE:-skills}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUNS_BASE="${RUNS_BASE:-${PACKAGE_ROOT}/outputs/runs}"
RUN_ROOT="${RUN_ROOT:-${RUNS_BASE}/harnessadapt_qwen35_9b_${STAMP}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"
LOG_FILE="${RUN_ROOT}/train.log"
HEALTH_LOG="${HEALTH_LOG:-${LOG_DIR}/health-monitor.jsonl}"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"
touch "${LOG_FILE}"

export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1

export R2E_CASE_CONCURRENCY="${R2E_CASE_CONCURRENCY:-128}"
export R2E_CONTAINER_START_CONCURRENCY="${R2E_CONTAINER_START_CONCURRENCY:-8}"
export R2E_FAST_MODEL_CLIENT="${R2E_FAST_MODEL_CLIENT:-1}"
export R2E_ADAPTER_PROCESS="${R2E_ADAPTER_PROCESS:-0}"
export SLIME_ADAPTER_CPU_WORKERS="${SLIME_ADAPTER_CPU_WORKERS:-8}"
export SLIME_ADAPTER_HTTP_CONNECTIONS="${SLIME_ADAPTER_HTTP_CONNECTIONS:-256}"
export R2E_CONTAINER_LIMITS_ENABLED="${R2E_CONTAINER_LIMITS_ENABLED:-1}"
export R2E_CONTAINER_MEMORY_GIB="${R2E_CONTAINER_MEMORY_GIB:-8}"
export R2E_CONTAINER_PIDS_LIMIT="${R2E_CONTAINER_PIDS_LIMIT:-512}"
export R2E_MAX_CONTEXT_TOKENS="${R2E_MAX_CONTEXT_TOKENS:-65536}"
export R2E_CONTEXT_SAFETY_MARGIN="${R2E_CONTEXT_SAFETY_MARGIN:-128}"
export R2E_MIN_GENERATION_TOKENS="${R2E_MIN_GENERATION_TOKENS:-256}"
export R2E_MAX_ITERATIONS="${R2E_MAX_ITERATIONS:-60}"
export R2E_VERIFIED_MAX_ITERATIONS="${R2E_VERIFIED_MAX_ITERATIONS:-150}"
export R2E_MAX_CONSECUTIVE_FORMAT_ERRORS="${R2E_MAX_CONSECUTIVE_FORMAT_ERRORS:-3}"
export R2E_STEP_TIMEOUT="${R2E_STEP_TIMEOUT:-45}"
export R2E_HARD_STEP_TIMEOUT="${R2E_HARD_STEP_TIMEOUT:-60}"
export R2E_SETUP_TIMEOUT="${R2E_SETUP_TIMEOUT:-300}"
export R2E_PATCH_TIMEOUT="${R2E_PATCH_TIMEOUT:-90}"
export R2E_GRADE_TIMEOUT="${R2E_GRADE_TIMEOUT:-90}"
export R2E_AGENT_WALL_TIME="${R2E_AGENT_WALL_TIME:-1400}"
export R2E_CASE_TIMEOUT="${R2E_CASE_TIMEOUT:-2420}"
export R2E_VERIFIED_GRADE_TIMEOUT="${R2E_VERIFIED_GRADE_TIMEOUT:-1200}"
export R2E_VERIFIED_AGENT_WALL_TIME="${R2E_VERIFIED_AGENT_WALL_TIME:-5400}"
export R2E_VERIFIED_CASE_TIMEOUT="${R2E_VERIFIED_CASE_TIMEOUT:-8100}"
export R2E_MODEL_REQUEST_TIMEOUT="${R2E_MODEL_REQUEST_TIMEOUT:-1200}"
export R2E_REQUIRE_SUBMISSION="${R2E_REQUIRE_SUBMISSION:-1}"
export R2E_ADD_TURN_REMINDER="${R2E_ADD_TURN_REMINDER:-0}"
export R2E_GRADE_UNSUBMITTED_PATCHES="${R2E_GRADE_UNSUBMITTED_PATCHES:-0}"
export R2E_SAVE_TRAJECTORIES="${R2E_SAVE_TRAJECTORIES:-1}"
export R2E_TRAJECTORY_DIR="${R2E_TRAJECTORY_DIR:-${RUN_ROOT}/trajectories}"
export R2E_HINT_MODE
export TENSORBOARD_DIR

mkdir -p "${R2E_TRAJECTORY_DIR}"

echo "Training schedule: optimizer_updates=${NUM_ROLLOUT}, prompts_per_update=${ROLLOUT_BATCH_SIZE}, samples_per_prompt=${N_SAMPLES_PER_PROMPT}, global_batch=${GLOBAL_BATCH_SIZE}, max_turns=${R2E_MAX_ITERATIONS}"
echo "Training memory: context=${MAX_CONTEXT_LEN}, tp=${TP_SIZE}, cp=${CP_SIZE}, pp=${PP_SIZE}, max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"

cd "${SLIME_DIR}"
source scripts/models/qwen3.5-9B.sh

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${REF_MODEL_PATH}"
  --load "${LOAD_CHECKPOINT_DIR:-${CHECKPOINT_DIR}}"
  --save "${CHECKPOINT_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
if [[ "${NUM_ROLLOUT}" == "0" ]]; then
  CKPT_ARGS+=(--no-load-optim --no-load-rng)
fi
if [[ -n "${CKPT_STEP:-}" ]]; then
  CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

ROLLOUT_ARGS=(
  --custom-generate-function-path "${R2E_GENERATE_PATH:-harnessadapt.training.phase_runtime.generate}"
  --custom-reward-post-process-path "${R2E_REWARD_PATH:-harnessadapt.training.runtime.reward_processing.post_process_rewards}"
  --custom-rollout-log-function-path harnessadapt.training.runtime.monitoring.log_rollout_data
  --custom-eval-rollout-log-function-path harnessadapt.training.runtime.monitoring.log_eval_rollout_data
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --rollout-stop-token-ids 248046 248044
  --rollout-shuffle
  --rollout-seed "${ROLLOUT_SEED:-20260720}"
  --num-steps-per-rollout 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --micro-batch-size 1
  --balance-data
)

EVAL_ARGS=(
  --eval-prompt-data r2egym_verified "${VALIDATION_DATA}"
  --skip-eval-before-train
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
  --eval-temperature "${R2E_EVAL_TEMPERATURE:-1.0}"
  --eval-top-p 1.0
  --eval-max-context-len "${MAX_CONTEXT_LEN}"
  --eval-max-response-len "${MAX_CONTEXT_LEN}"
)
if [[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" || "${R2E_DISABLE_EVAL:-0}" == "1" ]]; then
  EVAL_ARGS=()
else
  EVAL_ARGS+=(--eval-interval "${EVAL_INTERVAL}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --drop-zero-loss-samples
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

# GRPO & Adam hyperparameters (Table 5)
GRPO_ARGS=(
  --advantage-estimator grpo
  --disable-grpo-std-normalization
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LEARNING_RATE:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus "${NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
  --router-disable-retries
  --sglang-mem-fraction-static 0.75
  --sglang-enable-cache-report
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
)

FAULT_TOLERANCE_ARGS=(
  --use-fault-tolerance
  --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-60}"
  --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
  --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --colocate
)

TRACKING_ARGS=(
  --use-tensorboard
  --tb-project-name harnessadapt_rl
  --tb-experiment-name "$(basename "${RUN_ROOT}")"
)

if [[ "${R2E_STATIC_MINISWE:-0}" == "1" ]]; then
  ROLLOUT_ARGS+=(--rollout-function-path harnessadapt.training.static_rollout.generate_rollout)
elif [[ -n "${R2E_COEVOLVE_ROOT:-}" ]]; then
  ROLLOUT_ARGS+=(--rollout-function-path harnessadapt.training.rollout.generate_rollout)
fi

export PYTHON_PATH_VALUE="/root/Megatron-LM:${SLIME_DIR}:${PACKAGE_ROOT}/src:${WORKSPACE}/R2E-Gym/src"
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = [
    key
    for key in os.environ
    if key.startswith("R2E_")
    or key.startswith("HARNESSADAPT_")
    or key.startswith("SLIME_ADAPTER_")
    or key in {
        "MASTER_ADDR",
        "MASTER_PORT",
        "DOCKER_API_VERSION",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_NVLS_ENABLE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "PYTORCH_CUDA_ALLOC_CONF",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT",
        "TENSORBOARD_DIR",
    }
]
env = {key: os.environ[key] for key in keys}
env["PYTHONPATH"] = os.environ["PYTHON_PATH_VALUE"]
print(json.dumps({"env_vars": env}))
PY
)"

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${MASTER_PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port 8265

ray job submit \
  --address http://127.0.0.1:8265 \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- \
  python3 -u -m harnessadapt.training.driver \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${FAULT_TOLERANCE_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${TRACKING_ARGS[@]}" \
  2>&1 | tee -a "${LOG_FILE}"
