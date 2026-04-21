#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 of CoEvo-D: Warm-start SFT on the offline demo pool.
#
# Expected walltime on 4xA100-80GB / 4xH800 : ~15-25 minutes for 349 demos
# x 2 epochs at max_seq_length=12288. Much faster if you set --max-seq-length
# lower (demos rarely exceed 4-6k tokens post-template).
#
# After this finishes, point your GRPO configs at the saved checkpoint:
#
#   actor_rollout_ref.model.path=<SFT_OUTPUT_DIR>
#
# (see examples/bfcl_coevo_d_from_sft.yaml for a drop-in variant.)
# ---------------------------------------------------------------------------
set -euo pipefail

# --- knobs (override via env) ------------------------------------------------
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DEMO_POOL="${DEMO_POOL:-data/bfcl_demo_pool_400.json}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/bfcl_sft_warmstart}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEED="${SEED:-0}"
NGPUS="${NGPUS:-4}"

# --- preflight --------------------------------------------------------------
if [[ ! -f "${DEMO_POOL}" ]]; then
  echo "[run_sft_warmstart] demo pool not found at ${DEMO_POOL}." >&2
  echo "Build it first:" >&2
  echo "  python scripts/build_bfcl_demo_pool.py \\" >&2
  echo "    --train-parquet data/bfcl_train_400.parquet \\" >&2
  echo "    --output        ${DEMO_POOL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[run_sft_warmstart] base_model=${BASE_MODEL}"
echo "[run_sft_warmstart] demo_pool=${DEMO_POOL}"
echo "[run_sft_warmstart] output_dir=${OUTPUT_DIR}"
echo "[run_sft_warmstart] epochs=${NUM_EPOCHS}  lr=${LEARNING_RATE}  seq=${MAX_SEQ_LENGTH}  n_gpus=${NGPUS}"

# --- launch -----------------------------------------------------------------
torchrun \
  --nproc_per_node="${NGPUS}" \
  --master_port="${MASTER_PORT:-29500}" \
  scripts/sft_warmstart.py \
    --demo-pool "${DEMO_POOL}" \
    --model-name-or-path "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-train-epochs "${NUM_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --per-device-train-batch-size "${PER_DEVICE_BS}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --seed "${SEED}"

echo "[run_sft_warmstart] done. SFT ckpt at: ${OUTPUT_DIR}"
echo "[run_sft_warmstart] next:"
echo "  python -m agentevolver.main_ppo --config-name bfcl_coevo_d_from_sft"
