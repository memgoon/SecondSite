#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package=$(dirname -- "$script_dir")
python_bin=${PYTHON_BIN:-python3}
stage=${RUN_STAGE:-all}
[[ "$stage" =~ ^(all|preflight|infer|pack)$ ]] || exit 2
IFS=',' read -r -a gpus <<< "${GPU_IDS:-0,1,2,3}"
(( ${#gpus[@]} > 0 )) || exit 2
declare -A seen=()
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z ${seen[$gpu]:-} ]] || exit 2
  seen[$gpu]=1
done
mkdir -p "$package/gpu_output/logs"
exec 9>"$package/gpu_output/RUN.lock"
flock -n 9 || { echo 'Another completion run is active'; exit 3; }
export PYTHONUNBUFFERED=1
export CPU_THREADS=${CPU_THREADS:-6} BATCH_SIZE=${BATCH_SIZE:-16} C3_BATCH_SIZE=${C3_BATCH_SIZE:-4}
export OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=1
pids=()
cleanup() {
  trap - INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM
if [[ "$stage" != pack ]]; then
  CUDA_VISIBLE_DEVICES=${gpus[0]} "$python_bin" "$script_dir/runtime.py" preflight &
  pids+=("$!")
  wait "${pids[0]}"
  pids=()
fi
if [[ "$stage" == all || "$stage" == infer ]]; then
  for rank in "${!gpus[@]}"; do
    echo "Worker $rank GPU ${gpus[$rank]} -> infer_worker${rank}.log"
    CUDA_VISIBLE_DEVICES=${gpus[$rank]} "$python_bin" "$script_dir/runtime.py" infer --rank "$rank" \
      >> "$package/gpu_output/logs/infer_worker${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" == 0 ]] || { echo 'Inference failed. Completed chunks are reusable; inspect logs.'; exit 4; }
fi
if [[ "$stage" == all || "$stage" == infer || "$stage" == pack ]]; then
  "$python_bin" "$script_dir/transfer.py" return
  echo 'Complete. Fetch to run CPU comparisons, bootstrap and web exports locally.'
fi
