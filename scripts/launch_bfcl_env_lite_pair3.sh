#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples}"
GPU_SET="${GPU_SET:-8,9,10,11}"

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

run_exp bfcl_grpo_apatch_or_unavailable_required
run_exp bfcl_grpo_apatch_or_common_typed

echo ""
echo "Finished BFCL env-lite pair3."
