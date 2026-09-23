#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/chembl_c3_appendon
upstream=$project_root/analysis/role_complete_pair_matrix
stage=${RUN_STAGE:-pilot}
gpu_csv=${GPU_IDS:-0,1}
gpu_csv=${gpu_csv//[[:space:]]/}
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 4 )); then
    echo "GPU_IDS must list between one and four GPU indices" >&2
    exit 2
fi
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ || -n ${seen_gpu[$gpu]:-} ]]; then
        echo "GPU_IDS must contain distinct numeric indices" >&2
        exit 2
    fi
    seen_gpu[$gpu]=1
done
if [[ $stage != pilot && $stage != full && $stage != all ]]; then
    echo "RUN_STAGE must be pilot, full, or all" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${RUN_USER:-hs0517}
export LOGNAME=$USER
export OMP_NUM_THREADS=${CPU_THREADS:-8}
export MKL_NUM_THREADS=${CPU_THREADS:-8}

chunk_rows=${C3_CHUNK_ROWS:-10000}
max_batch_rows=${C3_MAX_BATCH_ROWS:-8}
max_batch_residues=${C3_MAX_BATCH_RESIDUES:-4096}
pilot_rows=${C3_PILOT_ROWS_PER_SHARD:-2500}
log_dir=$package/gpu_output/logs
mkdir -p "$log_dir"

cd "$project_root"
sha256sum -c "$package/manifests/CPU_INPUTS.sha256" > "$log_dir/cpu_inputs_checksum.log"
python "$package/scripts/validate_cpu_contract.py" \
    --project-root "$project_root" > "$log_dir/cpu_validation.log"

# Revalidate the frozen large OOD caches, selected-chain tensors, deployment
# checkpoints, and source fingerprint with the accepted upstream fail-closed
# preflight.  This performs no network request.
CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$upstream/scripts/gpu_preflight.py" \
    --project-root "$project_root" --scope chembl --device cuda:0 \
    2>&1 | tee "$log_dir/upstream_gpu_preflight.log"
sha256sum -c "$package/manifests/CPU_INPUTS.sha256" > "$log_dir/post_preflight_checksum.log"

run_shards() {
    local mode=$1
    local offset=0
    while (( offset < 4 )); do
        local pids=()
        local shards=()
        local slot=0
        while (( slot < ${#gpu_ids[@]} && offset + slot < 4 )); do
            local shard=$((offset + slot))
            local gpu=${gpu_ids[$slot]}
            (
                CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/infer_c3_appendon.py" \
                    --project-root "$project_root" --mode "$mode" \
                    --shard "$shard" --num-shards 4 --device cuda:0 \
                    --pilot-rows "$pilot_rows" --chunk-rows "$chunk_rows" \
                    --max-batch-rows "$max_batch_rows" \
                    --max-batch-residues "$max_batch_residues" \
                    > "$log_dir/${mode}_shard${shard}.log" 2>&1
            ) &
            pids+=("$!")
            shards+=("$shard")
            slot=$((slot + 1))
        done
        local failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
        if (( failed != 0 )); then
            echo "C3 $mode wave failed. Worker log tails:" >&2
            for shard in "${shards[@]}"; do
                echo "===== ${mode}_shard${shard}.log =====" >&2
                tail -100 "$log_dir/${mode}_shard${shard}.log" >&2 || true
            done
            echo "Rerun the identical command; hash-valid completed chunks are skipped." >&2
            exit 3
        fi
        offset=$((offset + ${#gpu_ids[@]}))
    done
}

if [[ $stage == pilot || $stage == all ]]; then
    run_shards pilot
    python "$package/scripts/aggregate_c3_appendon.py" \
        --project-root "$project_root" --pilot-only --write \
        2>&1 | tee "$log_dir/aggregate_timing.log"
    python "$package/scripts/aggregate_c3_appendon.py" \
        --project-root "$project_root" --pilot-only --validate-only >/dev/null
    echo "Timing pilot PASS. Inspect: $package/gpu_output/timing/C3_TIMING_SUMMARY.json"
fi

if [[ $stage == full || $stage == all ]]; then
    run_shards full
    python "$package/scripts/aggregate_c3_appendon.py" \
        --project-root "$project_root" --write \
        2>&1 | tee "$log_dir/aggregate_full.log"
    sha256sum -c "$package/manifests/GPU_RESULTS_PORTABLE.sha256" \
        > "$log_dir/gpu_results_checksum.log"
    python "$package/scripts/aggregate_c3_appendon.py" \
        --project-root "$project_root" --validate-only >/dev/null
    archive=$package/chembl_c3_appendon_gpu_return_20260901.tar.gz
    archive_tmp=$(mktemp /tmp/chembl_c3_appendon_gpu_return_20260901.XXXXXX.tar.gz)
    cleanup_archive_tmp() {
        rm -f "$archive_tmp"
    }
    trap cleanup_archive_tmp EXIT
    tar --exclude='*/__pycache__' --exclude='*.pyc' \
        --exclude='analysis/chembl_c3_appendon/gpu_output/chunks' \
        --exclude='analysis/chembl_c3_appendon/chembl_c3_appendon_gpu_return_*.tar.gz' \
        -czf "$archive_tmp" analysis/chembl_c3_appendon
    mv -f "$archive_tmp" "$archive"
    trap - EXIT
    echo "PASS: $archive"
    echo "Remote resumable chunks remain under $package/gpu_output/chunks"
fi
