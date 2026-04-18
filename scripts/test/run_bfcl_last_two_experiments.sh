#!/usr/bin/env bash
# Run the last two BFCL experiments from run_all_experiments.sh:
#   1. bfcl_tocf_pace
#   2. bfcl_gcce
#
# All normal run_all_experiments.sh options are forwarded, except --suite and
# --only are owned by this wrapper.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/test/run_bfcl_last_two_experiments.sh [run_all options] [-- hydra.override=value ...]

Runs only:
  - bfcl_tocf_pace
  - bfcl_gcce

Examples:
  bash scripts/test/run_bfcl_last_two_experiments.sh --dry-run
  bash scripts/test/run_bfcl_last_two_experiments.sh --mode eval --restart-services
  bash scripts/test/run_bfcl_last_two_experiments.sh --mode train --restart-services
  VAL_N=1 bash scripts/test/run_bfcl_last_two_experiments.sh --mode eval --continue-on-error -- trainer.n_gpus_per_node=4

Notes:
  Do not pass --suite or --only; this wrapper fixes them to the last two BFCL experiments.
  LOG_ROOT defaults to experiments/run_bfcl_last_two_experiments/<timestamp>.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ALL="${SCRIPT_DIR}/run_all_experiments.sh"

if [[ ! -x "${RUN_ALL}" && ! -f "${RUN_ALL}" ]]; then
  echo "ERROR: cannot find ${RUN_ALL}" >&2
  exit 1
fi

CONTINUE_ON_ERROR=0
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --suite|--only)
      echo "ERROR: ${arg} is managed by this wrapper; do not pass it." >&2
      usage >&2
      exit 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      ;;
  esac
done

TIMESTAMP="$(date "+%Y%m%d_%H%M%S")"
export LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/experiments/run_bfcl_last_two_experiments/${TIMESTAMP}}"
mkdir -p "${LOG_ROOT}"

EXPERIMENTS=(
  "bfcl_tocf_pace"
  "bfcl_gcce"
)

PASSED=()
FAILED=()

echo "[run_bfcl_last_two] project: ${PROJECT_ROOT}"
echo "[run_bfcl_last_two] logs:    ${LOG_ROOT}"
echo "[run_bfcl_last_two] experiments: ${EXPERIMENTS[*]}"

for exp_name in "${EXPERIMENTS[@]}"; do
  echo
  echo "############################################################"
  echo "[run_bfcl_last_two] ${exp_name}"
  echo "############################################################"

  set +e
  bash "${RUN_ALL}" --suite bfcl --only "${exp_name}" "$@"
  status="$?"
  set -e

  if [[ "${status}" == "0" ]]; then
    PASSED+=("${exp_name}")
  else
    FAILED+=("${exp_name}:${status}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      break
    fi
  fi
done

echo
echo "============================================================"
echo "[run_bfcl_last_two summary]"
echo "============================================================"
echo "passed: ${PASSED[*]:-none}"
echo "failed: ${FAILED[*]:-none}"
echo "logs:   ${LOG_ROOT}"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  exit 1
fi
