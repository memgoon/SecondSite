#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package=$(dirname -- "$script_dir")
python_bin=${PYTHON_BIN:-python3}
stage=${RUN_STAGE:-all}
gpu_csv=${GPU_IDS:-0,1,2,3}
gpu_csv=${gpu_csv//[[:space:]]/}
IFS=',' read -r -a gpus <<< "$gpu_csv"
[[ "$stage" =~ ^(all|preflight|deploy|infer|pack)$ ]] || exit 2
(( ${#gpus[@]} > 0 )) || exit 2
declare -A seen=()
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z ${seen[$gpu]:-} ]] || exit 2
    seen[$gpu]=1
done
mkdir -p "$package/gpu_output/logs"
exec 9>"$package/gpu_output/RUN.lock"
flock -n 9 || { echo 'A run or transfer is already active'; exit 3; }
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CPU_THREADS=${CPU_THREADS:-6}
export OMP_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS
if [[ "$stage" == all || "$stage" == preflight || "$stage" == deploy || "$stage" == infer ]]; then
    CUDA_VISIBLE_DEVICES=${gpus[0]} "$python_bin" "$script_dir/runtime.py" preflight 2>&1 | tee -a "$package/gpu_output/logs/preflight.log"
fi
workers() {
    local phase=$1
    local pids=()
    for rank in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpus[$rank]} "$python_bin" "$script_dir/runtime.py" "$phase" --rank "$rank" --world "${#gpus[@]}" >> "$package/gpu_output/logs/${phase}_worker${rank}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    (( failed == 0 )) || { echo "$phase failed; see worker logs. Same command resumes completed units."; exit 4; }
}
if [[ "$stage" == all || "$stage" == deploy ]]; then workers deploy; fi
if [[ "$stage" == all || "$stage" == infer ]]; then workers infer; fi
if [[ "$stage" == all || "$stage" == infer || "$stage" == pack ]]; then
    "$python_bin" "$script_dir/transfer.py" return
    echo 'GPU complete. Fetch for CPU validation, comparisons and SecondSite exports.'
fi
