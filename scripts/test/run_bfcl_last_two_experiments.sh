#!/usr/bin/env bash
# Run the last two experiments from the original four-row BFCL ladder.
# In the expanded six-row ladder these are the middle pair:
#   1. bfcl_tocf_pace
#   2. bfcl_gcce
#
# All normal run_all_experiments.sh options are forwarded, except --suite and
# --only/--only-exact are owned by this wrapper.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/test/run_bfcl_last_two_experiments.sh [options] [run_all options] [-- hydra.override=value ...]

Runs only:
  - bfcl_tocf_pace
  - bfcl_gcce

Options:
  --gcce-teacher-cache PATH     Optional. Sets gcce.teacher.cache_path for
                                bfcl_gcce only. Note: with the current
                                bfcl_gcce.yaml gcce.teacher.enable=false, so
                                this path is loaded but never queried unless
                                you also pass +gcce.teacher.enable=true.

Examples:
  bash scripts/test/run_bfcl_last_two_experiments.sh --dry-run
  bash scripts/test/run_bfcl_last_two_experiments.sh --mode eval --start-services
  bash scripts/test/run_bfcl_last_two_experiments.sh --mode train --start-services
  # If you ever do precompute a real teacher cache and want GCCE to use it:
  bash scripts/test/run_bfcl_last_two_experiments.sh --mode train --start-services \
    --gcce-teacher-cache data/teacher_scores_bfcl_400.json \
    -- +gcce.teacher.enable=true
  VAL_N=1 bash scripts/test/run_bfcl_last_two_experiments.sh --mode eval --continue-on-error -- trainer.n_gpus_per_node=4

Notes:
  Do not pass --suite, --only, or --only-exact; this wrapper fixes them to the original four-row ladder's last two BFCL experiments.
  Global Hydra overrides after -- are forwarded to both experiments. Use
  --gcce-teacher-cache for GCCE-only teacher-cache overrides.
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
GCCE_TEACHER_CACHE="${GCCE_TEACHER_CACHE:-}"
FORWARDED_ARGS=()

append_forwarded_arg() {
  case "$1" in
    gcce.teacher.cache_path=*|+gcce.teacher.cache_path=*)
      GCCE_TEACHER_CACHE="${1#*=}"
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      ;;
  esac
}

forwarded_has_hydra_separator() {
  local arg
  for arg in "${FORWARDED_ARGS[@]}"; do
    if [[ "${arg}" == "--" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --suite|--only|--only-exact)
      echo "ERROR: $1 is managed by this wrapper; do not pass it." >&2
      usage >&2
      exit 2
      ;;
    --gcce-teacher-cache)
      GCCE_TEACHER_CACHE="${2:?missing value for --gcce-teacher-cache}"
      shift 2
      ;;
    --restart-services|--reboot-services)
      # Current run_all_experiments.sh supports --start-services. Keep this
      # alias so older commands keep working with the pair wrapper.
      FORWARDED_ARGS+=(--start-services)
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      FORWARDED_ARGS+=("$1")
      shift
      ;;
    --)
      append_forwarded_arg "$1"
      shift
      while [[ $# -gt 0 ]]; do
        append_forwarded_arg "$1"
        shift
      done
      ;;
    *)
      append_forwarded_arg "$1"
      shift
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

  cmd=(bash "${RUN_ALL}" --suite bfcl --only-exact "${exp_name}" "${FORWARDED_ARGS[@]}")
  if [[ "${exp_name}" == "bfcl_gcce" && -n "${GCCE_TEACHER_CACHE}" ]]; then
    if forwarded_has_hydra_separator; then
      cmd+=("gcce.teacher.cache_path=${GCCE_TEACHER_CACHE}")
    else
      cmd+=(-- "gcce.teacher.cache_path=${GCCE_TEACHER_CACHE}")
    fi
  fi

  set +e
  "${cmd[@]}"
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
