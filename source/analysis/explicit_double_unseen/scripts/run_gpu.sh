#!/usr/bin/env bash
set -euo pipefail
package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project_root=$(cd "$package/../.." && pwd)
cd "$project_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
python_bin=${PYTHON_BIN:-python3}
IFS=',' read -r -a gpu_devices <<< "${GPU_IDS:-0}"
declare -A unique_devices=()
for gpu in "${gpu_devices[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU_IDS must contain comma-separated integer IDs'; exit 2; }
    [[ -z ${unique_devices[$gpu]+set} ]] || { echo 'Duplicate GPU ID'; exit 2; }
    unique_devices[$gpu]=1
done
mkdir -p "$package/gpu_output/logs"
exec 9>"$package/gpu_output/RUN.lock"
flock -n 9 || { echo 'Another explicit-double-unseen run is active'; exit 2; }
CUDA_VISIBLE_DEVICES="${gpu_devices[0]}" "$python_bin" "$package/scripts/gpu_preflight.py" --project-root "$project_root"
pids=()
cleanup() {
    trap - INT TERM
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 130
}
trap cleanup INT TERM
for rank in "${!gpu_devices[@]}"; do
    log="$package/gpu_output/logs/train_worker${rank}.log"
    echo "Starting worker $rank on GPU ${gpu_devices[$rank]}; append log: $log"
    CUDA_VISIBLE_DEVICES="${gpu_devices[$rank]}" "$python_bin" -u "$package/scripts/train_gpu.py" \
        --project-root "$project_root" --worker-rank "$rank" --world-size "${#gpu_devices[@]}" >> "$log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ $failed == 0 ]] || { echo 'Worker failed; check logs. Valid completed fits will resume.'; exit 1; }
"$python_bin" "$package/scripts/validate_fits.py"
archive=explicit_double_unseen_gpu_return_v1.tar.gz
tar -czf "$package/$archive.tmp" \
    analysis/explicit_double_unseen/gpu_output/RUN_CONTRACT.json \
    analysis/explicit_double_unseen/gpu_output/FITS_VALIDATION.json \
    analysis/explicit_double_unseen/gpu_output/FIT_SUMMARY.tsv \
    analysis/explicit_double_unseen/gpu_output/benchmark \
    analysis/explicit_double_unseen/gpu_output/logs
mv "$package/$archive.tmp" "$package/$archive"
(cd "$package" && sha256sum "$archive" > "$archive.sha256")
echo "GPU fits complete. Return archive: $package/$archive"
echo 'Aggregation and 10000-replicate bootstrap run on the local CPU server after fetch.'
