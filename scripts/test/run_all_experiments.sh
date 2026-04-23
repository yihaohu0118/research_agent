#!/usr/bin/env bash
# Run selected experiment configs under examples/ (not every YAML file).
#
# Suites:
#   core     — AppWorld (basic, overall) + full BFCL ablation ladder
#   appworld — basic + overall only
#   bfcl     — BFCL baseline + ablations only (see BFCL_EXPERIMENTS below)
#
# Default mode is validation-only, which is the safest way to "test" every
# experiment without launching full PPO training. Use --mode train for full runs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/test/run_all_experiments.sh [options] [-- hydra.override=value ...]

Options:
  --suite core|appworld|bfcl   Which experiment group to run. Default: core.
  --mode eval|train            eval uses trainer.val_only=true. Default: eval.
  --only PATTERN               Run experiment names containing PATTERN.
  --only-exact NAME            Run exactly one experiment name.
  --start-services             Start appworld/bfcl/reme services when needed.
  --keep-services              Do not stop services started by this script.
  --continue-on-error          Keep running after a failed experiment.
  --skip-missing               Skip experiments whose required files are absent.
  --dry-run                    Print commands without executing them.
  --override VALUE             Add one Hydra override. Can be repeated.
  -h, --help                   Show this help.

Environment:
  PYTHON_BIN                   Python executable. Default: python3.
  LOG_ROOT                     Directory for logs. Default: experiments/run_all_experiments/<timestamp>.
  VAL_N                        Optional validation rollout count override, e.g. VAL_N=1.
  APPWORLD_PORT                Default: 8080.
  BFCL_PORT                    Default: 8082.
  REME_PORT                    Default: 8001.

Examples:
  bash scripts/test/run_all_experiments.sh --dry-run
  bash scripts/test/run_all_experiments.sh --suite bfcl --start-services
  VAL_N=1 bash scripts/test/run_all_experiments.sh --continue-on-error -- trainer.n_gpus_per_node=4
  bash scripts/test/run_all_experiments.sh --mode train --suite bfcl --start-services
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
APPWORLD_PORT="${APPWORLD_PORT:-8080}"
BFCL_PORT="${BFCL_PORT:-8082}"
REME_PORT="${REME_PORT:-8001}"

SUITE="core"
MODE="eval"
ONLY_PATTERN=""
ONLY_EXACT=""
START_SERVICES=0
KEEP_SERVICES=0
CONTINUE_ON_ERROR=0
SKIP_MISSING=0
DRY_RUN=0
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="${2:?missing value for --suite}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --only)
      ONLY_PATTERN="${2:?missing value for --only}"
      shift 2
      ;;
    --only-exact)
      ONLY_EXACT="${2:?missing value for --only-exact}"
      shift 2
      ;;
    --start-services|--with-services)
      START_SERVICES=1
      shift
      ;;
    --keep-services)
      KEEP_SERVICES=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --skip-missing)
      SKIP_MISSING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --override)
      EXTRA_OVERRIDES+=("${2:?missing value for --override}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_OVERRIDES+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SUITE}" in
  core|appworld|bfcl) ;;
  *)
    echo "Invalid --suite: ${SUITE}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  eval|train) ;;
  *)
    echo "Invalid --mode: ${MODE}" >&2
    exit 2
    ;;
esac

TIMESTAMP="$(date "+%Y%m%d_%H%M%S")"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/experiments/run_all_experiments/${TIMESTAMP}}"
SERVICE_LOG_DIR="${LOG_ROOT}/services"
mkdir -p "${LOG_ROOT}" "${SERVICE_LOG_DIR}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

STARTED_PIDS=()

cleanup() {
  if [[ "${KEEP_SERVICES}" == "1" || "${#STARTED_PIDS[@]}" == "0" ]]; then
    return
  fi

  echo
  echo "[cleanup] stopping services started by this script..."
  local pid
  for pid in "${STARTED_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

port_is_open() {
  local host="$1"
  local port="$2"
  (echo >/dev/tcp/"${host}"/"${port}") >/dev/null 2>&1
}

wait_for_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_seconds="${4:-180}"
  local start
  start="$(date +%s)"

  while true; do
    if port_is_open "${host}" "${port}"; then
      echo "[service] ${name} is ready at ${host}:${port}"
      return 0
    fi

    if (( "$(date +%s)" - start >= timeout_seconds )); then
      echo "[service] timeout waiting for ${name} at ${host}:${port}" >&2
      return 1
    fi
    sleep 2
  done
}

start_service() {
  local service="$1"
  local host="127.0.0.1"
  local port=""
  local log_file="${SERVICE_LOG_DIR}/${service}.log"
  local -a cmd=()

  case "${service}" in
    appworld)
      port="${APPWORLD_PORT}"
      cmd=(bash "${PROJECT_ROOT}/env_service/launch_script/appworld.sh")
      ;;
    bfcl)
      port="${BFCL_PORT}"
      cmd=(bash "${PROJECT_ROOT}/env_service/launch_script/bfcl.sh")
      ;;
    reme)
      port="${REME_PORT}"
      if command -v reme >/dev/null 2>&1; then
        cmd=(
          reme
          config=default
          backend=http
          thread_pool_max_workers=256
          http.host="${host}"
          http.port="${port}"
          http.limit_concurrency=256
          llm.default.model_name=qwen-max-2025-01-25
          embedding_model.default.model_name=text-embedding-v4
          vector_store.default.backend=local
          op.rerank_memory_op.params.enable_llm_rerank=false
        )
      else
        echo "[service] ReMe is required but 'reme' command was not found." >&2
        echo "[service] Install/start ReMe manually, or skip the overall experiment with --only/--suite." >&2
        return 1
      fi
      ;;
    *)
      echo "Unknown service: ${service}" >&2
      return 1
      ;;
  esac

  if port_is_open "${host}" "${port}"; then
    echo "[service] ${service} already running at ${host}:${port}"
    return 0
  fi

  if [[ "${START_SERVICES}" != "1" ]]; then
    echo "[service] ${service} is not reachable at ${host}:${port}" >&2
    echo "[service] Start it manually or rerun with --start-services." >&2
    return 1
  fi

  echo "[service] starting ${service}; log: ${log_file}"
  "${cmd[@]}" >"${log_file}" 2>&1 &
  STARTED_PIDS+=("$!")
  wait_for_port "${service}" "${host}" "${port}" 180
}

missing_required_files() {
  local csv="$1"
  local -a files=()
  if [[ -n "${csv}" ]]; then
    local old_ifs="${IFS}"
    IFS=','
    read -r -a files <<<"${csv}"
    IFS="${old_ifs}"
  fi
  if [[ "${#files[@]}" -eq 0 ]]; then
    return 0
  fi

  local missing=0
  local path
  for path in "${files[@]}"; do
    [[ -z "${path}" ]] && continue
    if [[ ! -e "${PROJECT_ROOT}/${path}" ]]; then
      echo "  missing: ${path}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

OVERRIDES=()

build_overrides() {
  OVERRIDES=()

  if [[ "${MODE}" == "eval" ]]; then
    OVERRIDES+=(
      "trainer.val_before_train=true"
      "trainer.val_only=true"
      "trainer.save_freq=0"
    )
    if [[ -n "${VAL_N:-}" ]]; then
      OVERRIDES+=("actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}")
    fi
  else
    OVERRIDES+=("trainer.val_only=false")
  fi

  if [[ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]]; then
    OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
  fi
}

run_experiment() {
  local name="$1"
  local services_csv="$2"
  local required_csv="$3"

  if [[ -n "${ONLY_PATTERN}" && "${name}" != *"${ONLY_PATTERN}"* ]]; then
    return 0
  fi
  if [[ -n "${ONLY_EXACT}" && "${name}" != "${ONLY_EXACT}" ]]; then
    return 0
  fi

  echo
  echo "============================================================"
  echo "[experiment] ${name}"
  echo "============================================================"

  if ! missing_required_files "${required_csv}"; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[experiment] skipped ${name}; required files are missing."
      SKIPPED+=("${name}")
      return 0
    fi
    echo "[experiment] required files are missing for ${name}." >&2
    return 1
  fi

  local -a services=()
  if [[ -n "${services_csv}" ]]; then
    local old_ifs="${IFS}"
    IFS=','
    read -r -a services <<<"${services_csv}"
    IFS="${old_ifs}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${#services[@]}" -gt 0 ]]; then
      echo "[service] required: ${services[*]}"
    fi
  else
    local service
    for service in "${services[@]}"; do
      [[ -z "${service}" ]] && continue
      start_service "${service}"
    done
  fi

  build_overrides

  local log_file="${LOG_ROOT}/${name}.log"
  local -a cmd=(
    "${PYTHON_BIN}"
    -m agentevolver.main_ppo
    --config-path "${PROJECT_ROOT}/examples"
    --config-name "${name}"
    "${OVERRIDES[@]}"
  )

  echo "[experiment] log: ${log_file}"
  printf '[experiment] command:'
  printf ' %q' "${cmd[@]}"
  echo

  if [[ "${DRY_RUN}" == "1" ]]; then
    PASSED+=("${name}:dry-run")
    return 0
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "${status}" == "0" ]]; then
    PASSED+=("${name}")
    echo "[experiment] ${name} passed"
    return 0
  fi

  FAILED+=("${name}:${status}")
  echo "[experiment] ${name} failed with status ${status}" >&2
  return "${status}"
}

APPWORLD_EXPERIMENTS=(
  "basic|appworld|"
  "overall|appworld,reme|"
)

# BFCL ablation ladder (same 400-train / 400-test split for all rows; same
# universal hyperparameters lr=1e-6, entropy_coeff=0, total_epochs=10,
# temperature=1.0, strict_tool_parser=true across every row):
#   bfcl_grpo        — pure GRPO, no TOCF patches
#   bfcl_tocf_tpatch — isolated T-Patch (per-category task sampling) ablation
#   bfcl_tocf_fpatch — isolated F-Patch (T3RL-style dense reward) ablation
#   bfcl_tocf_cpatch — isolated C-Patch (per-category cue library) ablation
#   bfcl_tocf_apatch — isolated A-Patch (tag-aware advantage weighting) ablation
#   bfcl_tocf_coevo — GRPO + A + E + S (cold-start co-evolution)
#   bfcl_tocf_ae    — GRPO + A + E only (cold-start; pair with coevo for Δ_S)
BFCL_EXPERIMENTS=(
  "bfcl_grpo|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_tpatch|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_fpatch|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_cpatch|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_apatch|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_coevo|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
  "bfcl_tocf_ae|bfcl|data/bfcl_train_400.parquet,data/bfcl_test_400.parquet"
)

SELECTED_EXPERIMENTS=()
case "${SUITE}" in
  appworld)
    SELECTED_EXPERIMENTS+=("${APPWORLD_EXPERIMENTS[@]}")
    ;;
  bfcl)
    SELECTED_EXPERIMENTS+=("${BFCL_EXPERIMENTS[@]}")
    ;;
  core)
    SELECTED_EXPERIMENTS+=("${APPWORLD_EXPERIMENTS[@]}" "${BFCL_EXPERIMENTS[@]}")
    ;;
esac

PASSED=()
FAILED=()
SKIPPED=()

echo "[run_all] project: ${PROJECT_ROOT}"
echo "[run_all] suite:   ${SUITE}"
echo "[run_all] mode:    ${MODE}"
echo "[run_all] logs:    ${LOG_ROOT}"

for spec in "${SELECTED_EXPERIMENTS[@]}"; do
  IFS='|' read -r exp_name services required_files <<<"${spec}"
  if ! run_experiment "${exp_name}" "${services}" "${required_files}"; then
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

echo
echo "============================================================"
echo "[summary]"
echo "============================================================"
echo "passed:  ${PASSED[*]:-none}"
echo "skipped: ${SKIPPED[*]:-none}"
echo "failed:  ${FAILED[*]:-none}"
echo "logs:    ${LOG_ROOT}"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  exit 1
fi
