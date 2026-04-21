#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SFT_DATA="${TRAIN_SFT_DATA:-data/bfcl_envtuning_sft_train_base_missfunc_200.parquet}"
VAL_SFT_DATA="${VAL_SFT_DATA:-data/bfcl_envtuning_sft_val_base_missfunc_200.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ckpts/bfcl_envtuning_sft_base_missfunc_200}"
NGPUS="${NGPUS:-4}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SAVE_FREQ="${SAVE_FREQ:-1000000}"
TEST_FREQ="${TEST_FREQ:-1000000}"

if ! "${PYTHON_BIN}" -c "import verl" >/dev/null 2>&1; then
  echo "[bfcl_envtuning_sft] verl is not installed in the current Python environment." >&2
  echo "[bfcl_envtuning_sft] Install the repo requirements or a compatible verl build first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${TRAIN_SFT_DATA}" ]]; then
  echo "[bfcl_envtuning_sft] train parquet not found: ${TRAIN_SFT_DATA}" >&2
  exit 1
fi
if [[ ! -f "${VAL_SFT_DATA}" ]]; then
  echo "[bfcl_envtuning_sft] val parquet not found: ${VAL_SFT_DATA}" >&2
  exit 1
fi

echo "[bfcl_envtuning_sft] launching verl fsdp_sft_trainer"
torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_SFT_DATA}" \
  data.val_files="${VAL_SFT_DATA}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  model.partial_pretrain="${BASE_MODEL}" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  optim.lr="${LEARNING_RATE}" \
  trainer.default_local_dir="${OUTPUT_ROOT}" \
  trainer.project_name=bfcl-envtuning-sft \
  trainer.experiment_name=base-missfunc-200 \
  trainer.logger="['console']" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.default_hdfs_dir=null \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  ulysses_sequence_parallel_size=1 \
  use_remove_padding=true

LATEST_CKPT="$("${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("global_step_*"):
    if not path.is_dir():
        continue
    match = re.search(r"global_step_(\d+)$", path.name)
    if match:
        candidates.append((int(match.group(1)), path))

if candidates:
    print(max(candidates, key=lambda item: item[0])[1])
PY
)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "[bfcl_envtuning_sft] failed to find a saved checkpoint under ${OUTPUT_ROOT}" >&2
  exit 1
fi

ln -sfn "$(basename "${LATEST_CKPT}")" "${OUTPUT_ROOT}/latest"

echo "[bfcl_envtuning_sft] done"
echo "[bfcl_envtuning_sft] latest checkpoint: ${OUTPUT_ROOT}/latest"
echo "[bfcl_envtuning_sft] next:"
echo "  bash examples/run_bfcl_coevo_d_floor_from_envtuning_sft.sh"
