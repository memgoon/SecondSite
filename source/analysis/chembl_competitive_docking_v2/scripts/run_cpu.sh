#!/usr/bin/env bash
# CPU-parallel, resumable competitive docking with corrected site/charge contracts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${ROOT}/../.." && pwd)"
ENV_BIN="${ENV_BIN:-${WORKSPACE}/analysis/chembl_candidate_docking_v1/tools/dock_env/bin}"
FPOCKET_BIN="${FPOCKET_BIN:-${WORKSPACE}/analysis/structure_known_candidate_rebuild/tools/fpocket_env/bin/fpocket}"
RUN_STAGE="${RUN_STAGE:-all}"
RUN_PROFILE="${RUN_PROFILE:-full}"
CPU_BUDGET="${CPU_BUDGET:-224}"
CPU_PER_JOB="${CPU_PER_JOB:-2}"
WORKERS="${WORKERS:-$(( CPU_BUDGET / CPU_PER_JOB ))}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-24}"
SEEDS="${SEEDS:-5}"
NUM_MODES="${NUM_MODES:-10}"
DOCKING_PH="${DOCKING_PH:-7.4}"

if [[ ! -x "${ENV_BIN}/python" || ! -x "${ENV_BIN}/vina" || ! -x "${ENV_BIN}/obabel" || ! -x "${FPOCKET_BIN}" ]]; then
  echo "Missing dependency. ENV_BIN=${ENV_BIN}; FPOCKET_BIN=${FPOCKET_BIN}" >&2
  exit 2
fi
if (( CPU_BUDGET < 1 || CPU_PER_JOB < 1 || WORKERS < 1 )); then
  echo "CPU_BUDGET, CPU_PER_JOB, and WORKERS must be positive" >&2
  exit 2
fi
if (( WORKERS * CPU_PER_JOB > CPU_BUDGET )); then
  echo "WORKERS*CPU_PER_JOB exceeds CPU_BUDGET" >&2
  exit 2
fi

mkdir -p "${ROOT}/logs" "${ROOT}/inputs" "${ROOT}/results"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "root=${ROOT} stage=${RUN_STAGE} profile=${RUN_PROFILE} cpu_budget=${CPU_BUDGET} cpu_per_job=${CPU_PER_JOB} workers=${WORKERS} seeds=${SEEDS} exhaustiveness=${EXHAUSTIVENESS} pH=${DOCKING_PH}"

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "prepare" ]]; then
  "${ENV_BIN}/python" "${ROOT}/scripts/prepare_inputs.py" \
    --root "${ROOT}" \
    --env-bin "${ENV_BIN}" \
    --fpocket-bin "${FPOCKET_BIN}" \
    --pH "${DOCKING_PH}"
fi

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "jobs" || "${RUN_STAGE}" == "dock" ]]; then
  "${ENV_BIN}/python" "${ROOT}/scripts/build_jobs.py" \
    --root "${ROOT}" \
    --profile "${RUN_PROFILE}" \
    --seeds "${SEEDS}" \
    --exhaustiveness "${EXHAUSTIVENESS}" \
    --cpu-per-job "${CPU_PER_JOB}" \
    --num-modes "${NUM_MODES}"
fi

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "dock" ]]; then
  "${ENV_BIN}/python" "${ROOT}/scripts/run_docking_parallel.py" \
    --root "${ROOT}" \
    --env-bin "${ENV_BIN}" \
    --workers "${WORKERS}"
fi

if [[ "${RUN_STAGE}" == "all" || "${RUN_STAGE}" == "analyze" ]]; then
  "${ENV_BIN}/python" "${ROOT}/scripts/analyze_results.py" --root "${ROOT}"
fi

echo "Complete. Results: ${ROOT}/results"
