#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(dirname -- "$SCRIPT_DIR")
cd "$PACKAGE_DIR"
mkdir -p gpu_output/logs gpu_output/progress
exec 9>gpu_output/RUN.lock
flock -n 9 || { echo 'Another supervisor is already running.'; exit 1; }
export CPU_THREADS_PER_GPU=${CPU_THREADS_PER_GPU:-6}
export OMP_NUM_THREADS=$CPU_THREADS_PER_GPU
export MKL_NUM_THREADS=$CPU_THREADS_PER_GPU
export OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_GPU
export PYTHONUNBUFFERED=1
PYTHON_BIN=${PYTHON_BIN:-python3}
IFS=',' read -r -a GPUs <<< "${GPU_IDS:-0,1,2,3}"
declare -A SEEN_GPUS=()
for gpu in "${GPUs[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU_IDS must be comma-separated integers'; exit 1; }
    [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo 'Duplicate GPU ID'; exit 1; }
    SEEN_GPUS[$gpu]=1
done
[[ "$CPU_THREADS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || exit 1
echo "Supervisor PID=$$; GPU_IDS=${GPUs[*]}; threads/GPU=$CPU_THREADS_PER_GPU"
echo $$ > gpu_output/supervisor.pid
CUDA_VISIBLE_DEVICES=${GPUs[0]} "$PYTHON_BIN" scripts/gpu_runtime.py preflight
pids=()
stop_workers() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo 'Stopped. Valid completed fits will be reused; interrupted fits restart from epoch 1.'
    exit 130
}
trap stop_workers INT TERM
for worker in "${!GPUs[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUs[$worker]} "$PYTHON_BIN" scripts/gpu_runtime.py worker --worker "$worker" >> "gpu_output/logs/train_worker${worker}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then
    echo 'A worker failed. See worker logs. Restart the same command after resolving the cause.'
    exit 1
fi
"$PYTHON_BIN" scripts/validate_fits.py
"$PYTHON_BIN" scripts/package_return.py
echo 'GPU training complete. Bootstrap is NOT run here; fetch triggers local CPU analysis.'
