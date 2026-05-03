#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples}"
GPU_SET="${GPU_SET:-0,1,2,3}"

run_exp() {
  local config_name="$1"
  shift || true
  echo ""
  echo "================ ${config_name} ================"
  echo "GPUs: ${GPU_SET}"
  echo "Config path: ${CONFIG_PATH}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    python -m agentevolver.main_ppo \
      --config-path "${CONFIG_PATH}" \
      --config-name "${config_name}" \
      "$@"
}

cd "${REPO_ROOT}"

run_exp bfcl_grpo_qwen25_14b "$@"
run_exp bfcl_grpo_apatch_observation_required_qwen25_14b "$@"

echo ""
echo "Finished Qwen2.5-14B vanilla RL and ours."

