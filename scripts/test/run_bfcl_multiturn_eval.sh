#!/usr/bin/env bash
# =============================================================================
# BFCL Multi-Turn Full Evaluation Script
#
# Mirrored from T3RL's scripts/test/run_bfcl_multiturn_eval.sh, with a slightly
# more forgiving BFCL_ROOT auto-discovery so it can be launched from the
# AgentEvolver repo.
#
# Usage:
#   bash scripts/test/run_bfcl_multiturn_eval.sh
#
# Environment overrides:
#   MODEL_PATH      - Path to local model weights
#   MODEL_NAME      - BFCL registry key / result folder name
#   NUM_GPUS        - Number of GPUs for tensor parallelism (default: 2)
#   BACKEND         - Inference backend: sglang or vllm (default: sglang)
#   GPU_MEM_UTIL    - GPU memory utilization fraction (default: 0.9)
#   TEMPERATURE     - Sampling temperature (default: 0.001)
#   RESULT_DIR      - Directory for result files
#   SCORE_DIR       - Directory for score files
#   TEST_CATEGORY   - Test category (default: multi_turn)
#   SKIP_GENERATE   - Set to 1 to skip generation and only evaluate
#   SKIP_EVALUATE   - Set to 1 to skip evaluation and only generate
#   BFCL_ROOT       - Path to berkeley-function-call-leaderboard root
#   SUBSET_PARQUET  - Optional parquet whose BFCL ids define a reporting subset
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_bfcl_root() {
    local candidates=()

    if [[ -n "${BFCL_ROOT:-}" ]]; then
        candidates+=("${BFCL_ROOT}")
    fi

    candidates+=(
        "${PROJECT_ROOT}/env_service/environments/bfcl/gorilla/berkeley-function-call-leaderboard"
        "${PROJECT_ROOT}/../T3RL-main/3rdparty/gorilla/berkeley-function-call-leaderboard"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -d "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! command -v bfcl >/dev/null 2>&1; then
    echo "ERROR: 'bfcl' command not found."
    echo "Please activate the bfcl environment first, for example:"
    echo "  conda activate bfcl"
    exit 1
fi

if ! BFCL_ROOT_RESOLVED="$(resolve_bfcl_root)"; then
    echo "ERROR: Could not locate berkeley-function-call-leaderboard."
    echo "Set BFCL_ROOT manually, e.g.:"
    echo "  BFCL_ROOT=/path/to/berkeley-function-call-leaderboard bash scripts/test/run_bfcl_multiturn_eval.sh"
    exit 1
fi

export BFCL_PROJECT_ROOT="${BFCL_ROOT_RESOLVED}"

# ---- Configuration ----
MODEL_PATH="${MODEL_PATH:-/home/yihao_hyh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct-FC}"
NUM_GPUS="${NUM_GPUS:-2}"
BACKEND="${BACKEND:-sglang}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TEMPERATURE="${TEMPERATURE:-0.001}"
TEST_CATEGORY="${TEST_CATEGORY:-multi_turn}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_EVALUATE="${SKIP_EVALUATE:-0}"
RESULT_DIR="${RESULT_DIR:-${BFCL_ROOT_RESOLVED}/result}"
SCORE_DIR="${SCORE_DIR:-${BFCL_ROOT_RESOLVED}/score}"
SUBSET_PARQUET="${SUBSET_PARQUET:-}"

echo "============================================"
echo "  BFCL Multi-Turn Evaluation"
echo "============================================"
echo "Project root:   ${PROJECT_ROOT}"
echo "BFCL root:      ${BFCL_ROOT_RESOLVED}"
echo "Model path:     ${MODEL_PATH}"
echo "Model name:     ${MODEL_NAME}"
echo "Backend:        ${BACKEND}"
echo "Num GPUs:       ${NUM_GPUS}"
echo "GPU mem util:   ${GPU_MEM_UTIL}"
echo "Temperature:    ${TEMPERATURE}"
echo "Test category:  ${TEST_CATEGORY}"
echo "Result dir:     ${RESULT_DIR}"
echo "Score dir:      ${SCORE_DIR}"
if [[ -n "${SUBSET_PARQUET}" ]]; then
    echo "Subset parquet: ${SUBSET_PARQUET}"
fi
echo "============================================"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

for f in config.json tokenizer_config.json; do
    if [[ ! -f "${MODEL_PATH}/${f}" ]]; then
        echo "ERROR: Required file not found: ${MODEL_PATH}/${f}"
        exit 1
    fi
done

if [[ "${SKIP_GENERATE}" != "1" ]]; then
    echo ""
    echo "[Step 1/2] Generating responses..."
    echo "  bfcl generate --model ${MODEL_NAME} --test-category ${TEST_CATEGORY}"
    echo ""

    bfcl generate \
        --model "${MODEL_NAME}" \
        --test-category "${TEST_CATEGORY}" \
        --backend "${BACKEND}" \
        --num-gpus "${NUM_GPUS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --temperature "${TEMPERATURE}" \
        --local-model-path "${MODEL_PATH}" \
        --result-dir "${RESULT_DIR}" \
        --include-input-log \
        -o

    echo ""
    echo "[Step 1/2] Generation complete."
else
    echo ""
    echo "[Step 1/2] Skipped generation (SKIP_GENERATE=1)."
fi

if [[ "${SKIP_EVALUATE}" != "1" ]]; then
    echo ""
    echo "[Step 2/2] Evaluating results..."
    echo "  bfcl evaluate --model ${MODEL_NAME} --test-category ${TEST_CATEGORY}"
    echo ""

    bfcl evaluate \
        --model "${MODEL_NAME}" \
        --test-category "${TEST_CATEGORY}" \
        --result-dir "${RESULT_DIR}" \
        --score-dir "${SCORE_DIR}"

    echo ""
    echo "[Step 2/2] Evaluation complete."

    MODEL_SCORE_DIR="${SCORE_DIR}/${MODEL_NAME//\//_}"
    if [[ -d "${MODEL_SCORE_DIR}" ]]; then
        echo ""
        echo "============================================"
        echo "  Score Summary"
        echo "============================================"
        for f in "${MODEL_SCORE_DIR}"/*multi_turn*; do
            if [[ -f "$f" ]]; then
                echo ""
                echo "--- $(basename "$f") ---"
                python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
if isinstance(data, dict):
    acc = data.get('accuracy', data)
    if isinstance(acc, dict):
        for k, v in acc.items():
            print(f'  {k}: {v}')
    else:
        print(f'  accuracy: {acc}')
elif isinstance(data, list):
    correct = sum(1 for d in data if d.get('valid', False))
    print(f'  {correct}/{len(data)} correct ({correct/len(data)*100:.1f}%)')
"
            fi
        done
    fi

    if [[ -n "${SUBSET_PARQUET}" ]]; then
        echo ""
        echo "[Step 2/2] Summarizing official scores on subset..."
        python3 "${PROJECT_ROOT}/tools/bfcl_subset_score.py" \
            --parquet "${SUBSET_PARQUET}" \
            --score-dir "${SCORE_DIR}" \
            --model-name "${MODEL_NAME}"
    fi
else
    echo ""
    echo "[Step 2/2] Skipped evaluation (SKIP_EVALUATE=1)."
fi

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
echo "Results: ${RESULT_DIR}"
echo "Scores:  ${SCORE_DIR}"
