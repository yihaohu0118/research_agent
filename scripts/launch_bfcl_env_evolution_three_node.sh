#!/usr/bin/env bash
# Three-node env-side ablation for BFCL.
#
# Slot 1 (insurance, 0-code change vs prev module):
#   bfcl_grpo_apatch_observation_required_nolc
#     A-Patch + observation_lite restricted to base/miss_func/miss_param
#     (i.e. observation_required minus the LC trade-off).
#
# Slot 2 (NEW: env reactive-feedback channel):
#   bfcl_grpo_apatch_tool_feedback
#     A-Patch + Tool Feedback Evolution. The env enriches parse-error
#     and tool-execution-error replies with diagnostic hints. Train-only;
#     never fires on a successful turn.
#
# Slot 3 (NEW: env online evolution from diagnostic signal):
#   bfcl_grpo_apatch_diagnostic_evolution
#     A-Patch + Diagnostic-Driven Schema Evolution. The env reads the
#     trainer's capability_state.json (the same per-tag failure stats
#     A-Patch consumes for advantage scaling) and grows / shrinks a
#     small set of behavioral guidelines in the system prompt online.
#
# Usage on a single multi-GPU node (each slot uses its own 4-GPU set):
#   GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot1
#   GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot2
#   GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot3
#   GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh all
#
# Usage on three separate nodes (one slot per node):
#   On node A:  GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot1
#   On node B:  GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot2
#   On node C:  GPU_SET=0,1,2,3 bash scripts/launch_bfcl_env_evolution_three_node.sh slot3

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

slot="${1:-all}"

case "${slot}" in
  slot1)
    run_exp bfcl_grpo_apatch_observation_required_nolc
    ;;
  slot2)
    run_exp bfcl_grpo_apatch_tool_feedback
    ;;
  slot3)
    run_exp bfcl_grpo_apatch_diagnostic_evolution
    ;;
  all)
    run_exp bfcl_grpo_apatch_observation_required_nolc
    run_exp bfcl_grpo_apatch_tool_feedback
    run_exp bfcl_grpo_apatch_diagnostic_evolution
    ;;
  *)
    echo "Usage: $0 {slot1|slot2|slot3|all}" >&2
    exit 2
    ;;
esac

echo ""
echo "Finished BFCL env-evolution three-node ablation: ${slot}"
