#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples}"
GPU_SET="${GPU_SET:-0,1,2,3}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"

run_exp() {
  local config_name="$1"
  echo ""
  echo "================ ${config_name} ================"
  echo "GPUs: ${GPU_SET}"
  echo "Config path: ${CONFIG_PATH}"
  echo "Train batch size: ${TRAIN_BATCH_SIZE}"
  echo "vLLM gpu_memory_utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    python -m agentevolver.main_ppo \
      --config-path "${CONFIG_PATH}" \
      --config-name "${config_name}" \
      data.train_batch_size="${TRAIN_BATCH_SIZE}" \
      actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}"
}

cd "${REPO_ROOT}"

run_exp bfcl_grpo_apatch_recovery_unavailable
run_exp bfcl_grpo_apatch_recovery_common

echo ""
echo "Finished BFCL env-lite pair1."
