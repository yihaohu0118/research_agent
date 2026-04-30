#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples}"
GPU_SET="${GPU_SET:-0,1,2,3}"

run_exp() {
  local config_name="$1"
  echo ""
  echo "================ ${config_name} ================"
  echo "GPUs: ${GPU_SET}"
  echo "Config path: ${CONFIG_PATH}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    python -m agentevolver.main_ppo \
      --config-path "${CONFIG_PATH}" \
      --config-name "${config_name}"
}

cd "${REPO_ROOT}"

run_exp bfcl_grpo_apatch_recovery_unavailable_all
run_exp bfcl_grpo_apatch_or_targeted

echo ""
echo "Finished BFCL env-lite pair4."
