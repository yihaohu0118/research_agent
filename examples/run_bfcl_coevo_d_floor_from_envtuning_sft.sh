#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_DIR="$(pwd)"
CONFIG_PATH="${PROJECT_DIR}/examples"
ENV_URL="${ENV_URL:-http://127.0.0.1:8082}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-ckpts/bfcl_envtuning_sft_base_missfunc_200/latest}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bfcl_coevo_d_floor_envtuning_sft_base_missfunc_200}"
PROJECT_NAME="${PROJECT_NAME:-bfcl_coevo_d_floor_envtuning_sft}"

"${PYTHON_BIN}" -m agentevolver.main_ppo \
  --config-path "${CONFIG_PATH}" \
  --config-name bfcl_coevo_d_floor \
  env_service.env_url="${ENV_URL}" \
  actor_rollout_ref.model.path="${SFT_CKPT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  tocf.stats.dump_dir="experiments/${EXPERIMENT_NAME}/tocf_stats" \
  gcce.oracle_env.dump_dir="experiments/${EXPERIMENT_NAME}/gcce_oracle"
