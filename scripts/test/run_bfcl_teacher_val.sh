#!/usr/bin/env bash
# Run a strong model as BFCL teacher, then build a GCCE teacher cache.
#
# Defaults target the BFCL training split because GCCE consumes teacher scores
# for training task ids. Override OUTPUT_CACHE if you want to replace the active
# cache used by examples/bfcl_gcce.yaml.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/test/run_bfcl_teacher_val.sh [options] [-- hydra.override=value ...]

Options:
  --model-path PATH             Teacher HF/local model path. Default: IcyFish/Qwen3-4B-EnvTuning
  --teacher-model NAME          Name stored in cache metadata. Default: qwen3-4b-envtuning
  --teacher-parquet PATH        Split to validate/cache. Default: data/bfcl_train_400.parquet
  --output-cache PATH           Output cache. Default: data/teacher_scores_bfcl_400_qwen3_4b_envtuning.json
  --experiment-name NAME        Experiment/log name. Default: bfcl_teacher_qwen3_4b_envtuning_train400
  --num-gpus N                  trainer.n_gpus_per_node. Default: 8
  --tp-size N                   rollout tensor parallel size. Default: 8
  --gpu-mem-util FLOAT          vLLM GPU memory utilization. Default: 0.7
  --max-env-worker N            BFCL env workers. Default: 32
  --restart-services            Restart BFCL service through launcher.
  --no-start-services           Do not ask launcher to start BFCL.
  --dry-run                     Print command only.
  -h, --help                    Show this help.

Examples:
  bash scripts/test/run_bfcl_teacher_val.sh --restart-services

  OUTPUT_CACHE=data/teacher_scores_bfcl_400.json \
    bash scripts/test/run_bfcl_teacher_val.sh --restart-services

  bash scripts/test/run_bfcl_teacher_val.sh \
    --model-path /path/to/Qwen3-4B-EnvTuning \
    --output-cache data/teacher_scores_bfcl_400.json
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-IcyFish/Qwen3-4B-EnvTuning}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-4b-envtuning}"
TEACHER_PARQUET="${TEACHER_PARQUET:-data/bfcl_train_400.parquet}"
OUTPUT_CACHE="${OUTPUT_CACHE:-data/teacher_scores_bfcl_400_qwen3_4b_envtuning.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bfcl_teacher_qwen3_4b_envtuning_train400}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
MAX_ENV_WORKER="${MAX_ENV_WORKER:-32}"
VAL_N="${VAL_N:-1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-8192}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
START_SERVICES=1
RESTART_SERVICES=0
DRY_RUN=0
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="${2:?missing value for --model-path}"
      shift 2
      ;;
    --teacher-model)
      TEACHER_MODEL="${2:?missing value for --teacher-model}"
      shift 2
      ;;
    --teacher-parquet)
      TEACHER_PARQUET="${2:?missing value for --teacher-parquet}"
      shift 2
      ;;
    --output-cache)
      OUTPUT_CACHE="${2:?missing value for --output-cache}"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="${2:?missing value for --experiment-name}"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="${2:?missing value for --num-gpus}"
      shift 2
      ;;
    --tp-size)
      TP_SIZE="${2:?missing value for --tp-size}"
      shift 2
      ;;
    --gpu-mem-util)
      GPU_MEM_UTIL="${2:?missing value for --gpu-mem-util}"
      shift 2
      ;;
    --max-env-worker)
      MAX_ENV_WORKER="${2:?missing value for --max-env-worker}"
      shift 2
      ;;
    --restart-services)
      RESTART_SERVICES=1
      START_SERVICES=1
      shift
      ;;
    --no-start-services)
      START_SERVICES=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_OVERRIDES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${TEACHER_PARQUET}" ]]; then
  echo "ERROR: teacher parquet not found: ${TEACHER_PARQUET}" >&2
  exit 1
fi

TIMESTAMP="$(date "+%Y%m%d_%H%M%S")"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/experiments/bfcl_teacher_val/${EXPERIMENT_NAME}_${TIMESTAMP}}"
VAL_DIR="${VAL_DIR:-${RUN_ROOT}/validation_log}"
LOG_FILE="${RUN_ROOT}/teacher_val.log"
mkdir -p "${RUN_ROOT}" "${VAL_DIR}" "$(dirname "${OUTPUT_CACHE}")"

# This script does not use task exploration or LLM judging, but the training
# stack still constructs DashScope clients while wiring TaskManager objects.
# Provide an inert key so val-only BFCL teacher runs do not require a real API
# key. If the caller exports a real key, keep it.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-unused-for-bfcl-teacher-val}"

# Keep Ray away from /var/tmp by default. RAY_TMPDIR must be short because Ray
# creates AF_UNIX socket paths under it, and Linux caps those at 107 bytes.
# Users can still override these before launching if they have a larger scratch
# mount with a short path.
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ae_ray_${USER:-user}}"
export RAY_OBJECT_SPILL_DIR="${RAY_OBJECT_SPILL_DIR:-${RUN_ROOT}/ray_spill}"
export RAY_LOCAL_FS_CAPACITY_THRESHOLD="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${RAY_OBJECT_SPILL_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/launcher.py"
  --conf "${PROJECT_ROOT}/examples/bfcl_grpo.yaml"
)
if [[ "${START_SERVICES}" == "1" ]]; then
  if [[ "${RESTART_SERVICES}" == "1" ]]; then
    CMD+=(--reboot)
  fi
  CMD+=(--with-bfcl)
fi

CMD+=(
  "data.train_files=${TEACHER_PARQUET}"
  "data.val_files=${TEACHER_PARQUET}"
  "data.validation_shuffle=false"
  "data.val_batch_size=256"
  "data.max_prompt_length=${PROMPT_LENGTH}"
  "data.max_response_length=${RESPONSE_LENGTH}"
  "trainer.project_name=bfcl_teacher_val"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.validation_data_dir=${VAL_DIR}"
  "trainer.val_before_train=true"
  "trainer.val_only=true"
  "trainer.save_freq=0"
  "trainer.save_best_checkpoint=false"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "algorithm.use_kl_in_reward=false"
  "task_manager.n=0"
  "task_manager.mixture.synthetic_data_ratio=0.0"
  "task_manager.mixture.use_original_tasks=true"
  "actor_rollout_ref.actor.use_kl_loss=false"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.rollout.use_qwen3=true"
  "actor_rollout_ref.rollout.temperature=0"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_env_worker=${MAX_ENV_WORKER}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN}"
)

if [[ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "[teacher-val] project:       ${PROJECT_ROOT}"
echo "[teacher-val] model:         ${MODEL_PATH}"
echo "[teacher-val] parquet:       ${TEACHER_PARQUET}"
echo "[teacher-val] validation:    ${VAL_DIR}"
echo "[teacher-val] output cache:  ${OUTPUT_CACHE}"
echo "[teacher-val] log:           ${LOG_FILE}"
printf '[teacher-val] command:'
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

set +e
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e
if [[ "${status}" != "0" ]]; then
  echo "[teacher-val] validation failed with status ${status}" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" scripts/build_teacher_cache_from_validation.py \
  --val-dir "${VAL_DIR}" \
  --output "${OUTPUT_CACHE}" \
  --teacher-model "${TEACHER_MODEL}" \
  --source-split train

BFCL_JSONL="${BFCL_JSONL:-env_service/environments/bfcl/bfcl_data/multi_turn_processed.jsonl}"
if [[ -f "${BFCL_JSONL}" ]]; then
  "${PYTHON_BIN}" scripts/stats_validation_bfcl.py \
    --val-dir "${VAL_DIR}" \
    --parquet "${TEACHER_PARQUET}" \
    --bfcl-jsonl "${BFCL_JSONL}" || true
else
  echo "[teacher-val] skipped stats; BFCL jsonl not found: ${BFCL_JSONL}"
fi

echo "[teacher-val] done"
echo "[teacher-val] cache: ${OUTPUT_CACHE}"
