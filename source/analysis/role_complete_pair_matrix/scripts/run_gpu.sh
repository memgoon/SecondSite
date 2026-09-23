#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/role_complete_pair_matrix
stage=${RUN_STAGE:-benchmark}
gpu_csv=${GPU_IDS:-0}
gpu_csv=${gpu_csv//[[:space:]]/}
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"

if [[ $stage != benchmark && $stage != chembl && $stage != all ]]; then
    echo "RUN_STAGE must be benchmark, chembl, or all" >&2
    exit 2
fi
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 2 )); then
    echo "GPU_IDS must contain one or two comma-separated GPU indices" >&2
    exit 2
fi
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ ]] || [[ -n ${seen[$gpu]:-} ]]; then
        echo "Invalid or duplicate GPU id: $gpu" >&2
        exit 2
    fi
    seen[$gpu]=1
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${RUN_USER:-hs0517}
export LOGNAME=$USER
export OMP_NUM_THREADS=${CPU_THREADS:-8}
export MKL_NUM_THREADS=${CPU_THREADS:-8}
log_dir=$package/gpu_output/logs
mkdir -p "$log_dir"

CUDA_VISIBLE_DEVICES=$gpu_csv EXPECTED_GPUS=${#gpu_ids[@]} python - <<'PY' 2>&1 | tee "$log_dir/environment.log"
import os
import sys
import numpy
import pandas
import torch
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
if torch.cuda.device_count() != int(os.environ["EXPECTED_GPUS"]):
    raise SystemExit("visible GPU count differs from GPU_IDS")
for index in range(torch.cuda.device_count()):
    print("GPU {}: {}".format(index, torch.cuda.get_device_name(index)))
PY

required=(
    "$package/validation/CPU_CONTRACT.json"
    "$package/data/EVERY_PAIR.tsv.gz"
    "$package/data/PROTEIN_ANCHORED.tsv.gz"
    "$package/data/PROTEIN_LIGAND_ROLE_COMPLETE.tsv.gz"
    "$package/scripts/model_definitions.py"
    "$package/scripts/train_matrix.py"
    "$project_root/analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py"
    "$project_root/analysis/allosteric_pair_benchmark_broad_superset/data/POCKET_INDICES.json"
)
for path in "${required[@]}"; do
    if [[ ! -f $path ]]; then
        echo "Missing required artifact: $path" >&2
        exit 2
    fi
done

run_benchmark() {
    CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/gpu_preflight.py" \
        --project-root "$project_root" --scope benchmark --device cuda:0 \
        2>&1 | tee "$log_dir/preflight_benchmark.log"
    local world=${#gpu_ids[@]}
    local pids=()
    for rank in "${!gpu_ids[@]}"; do
        local gpu=${gpu_ids[$rank]}
        (
            CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/train_matrix.py" \
                --project-root "$project_root" \
                --worker-rank "$rank" --world-size "$world" --device cuda:0 \
                --epochs 25 --patience 5 --min-epochs 1 \
                --batch-size 6 --eval-batch-size 8 \
                --full-bidirectional-batch-size 1 \
                --full-bidirectional-eval-batch-size 2 \
                --hidden-dim 256 --heads 4 --dropout 0.30 \
                --lr 0.0001 --weight-decay 0.0001 \
                --max-atoms 120 --max-protein-residues 4096 \
                > "$log_dir/train_worker${rank}.log" 2>&1
        ) &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if (( failed != 0 )); then
        echo "One or more benchmark workers failed; rerun the same command to resume" >&2
        exit 3
    fi
    python "$package/scripts/aggregate_matrix.py" --project-root "$project_root" \
        2>&1 | tee "$log_dir/aggregate_matrix.log"
    python "$package/scripts/analyze_matrix_patterns.py" --project-root "$project_root" \
        2>&1 | tee "$log_dir/analyze_matrix_patterns.log"
    python "$package/scripts/bundle_fit_audit.py" --project-root "$project_root" \
        2>&1 | tee "$log_dir/bundle_fit_audit.log"
}

run_sharded_scope() {
    local scope=$1
    local pids=()
    for shard in 0 1 2 3; do
        local slot=$((shard % ${#gpu_ids[@]}))
        local gpu=${gpu_ids[$slot]}
        (
            CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/infer_chembl.py" \
                --project-root "$project_root" --scope "$scope" \
                --shard "$shard" --num-shards 4 --device cuda:0 \
                --batch-size "${FULL_BATCH_SIZE:-128}" \
                --biochemical-batch-size "${BIOCHEMICAL_BATCH_SIZE:-2}" \
                > "$log_dir/infer_${scope}_shard${shard}.log" 2>&1
        ) &
        pids+=("$!")
        if (( ${#pids[@]} == ${#gpu_ids[@]} )); then
            local failed=0
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then failed=1; fi
            done
            if (( failed != 0 )); then
                echo "$scope inference failed; rerun to resume" >&2
                exit 4
            fi
            pids=()
        fi
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if (( failed != 0 )); then
        echo "$scope inference failed; rerun to resume" >&2
        exit 4
    fi
}

run_chembl() {
    python - \
        "$package/gpu_output/benchmark/aggregate/AGGREGATE_VALIDATION.json" \
        "$package/gpu_output/benchmark/aggregate/PATTERN_ANALYSIS_VALIDATION.json" \
        "$package/gpu_output/benchmark/FIT_AUDIT_BUNDLE.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Complete RUN_STAGE=benchmark before ChEMBL deployment")
value = json.loads(path.read_text(encoding="utf-8"))
required = {
    "status": "validated",
    "contract_id": "role_complete_pair_matrix_v2",
    "model_version": "role_complete_matrix_v2",
    "observed_fits": 1080,
    "common_role_complete_rows_per_cell": 395,
    "common_general_double_unseen_rows_per_cell": 2721,
    "common_general_hard_row_comparison_total_rows": 87072,
    "primary_hard_generalization_endpoint": "protein_macro_fold_restricted.auroc",
    "primary_hard_generalization_effective_rows": 1805,
    "primary_hard_generalization_effective_groups": 112,
    "checkpoint_selection_model_aware": True,
    "checkpoint_selection_minimum_valid_groups": 8,
    "checkpoint_selection_expected_fallback_fits": 132,
    "checkpoint_selection_observed_fallback_fits": 132,
    "structurally_constant_checkpoint_selection_fits": 0,
    "single_input_numerical_audit_failed_rows": 0,
}
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit("aggregate contract mismatch for {}: {} != {}".format(key, value.get(key), expected))
patterns = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    patterns.get("status") != "validated"
    or not patterns.get("pooled_metrics_are_secondary")
    or patterns.get("primary_family_endpoint")
    != "fold-restricted within-ligand AUROC"
    or patterns.get("hard_primary_endpoint")
    != "fold-restricted within-protein AUROC on 1,805/2,721 rows and 112 groups"
    or patterns.get("hard_primary_comparator") != "ligand-only"
    or patterns.get("hard_pooled_metric_role")
    != "secondary identical-2,721-row split-change sensitivity"
    or patterns.get("primary_family_comparator")
    != "protein-only (ligand-only is a structural 0.5 check)"
    or patterns.get("primary_ligand_comparator")
    != "ligand-only (protein-only is a structural 0.5 check)"
):
    raise SystemExit("pattern-analysis contract is invalid")
audit_path = Path(sys.argv[3])
if not audit_path.is_file():
    raise SystemExit("fit audit bundle is missing")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
archive = audit_path.parent / "FIT_AUDIT_REPORTS.tar.gz"
actual = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None
if (
    audit.get("status") != "validated"
    or audit.get("fit_directories") != 1080
    or not archive.is_file()
    or audit.get("archive_sha256") != actual
):
    raise SystemExit("fit audit bundle is invalid")
PY
    if [[ ! -f $package/gpu_output/benchmark/FIT_AUDIT_REPORTS.tar.gz ]]; then
        python "$package/scripts/bundle_fit_audit.py" --project-root "$project_root" \
            2>&1 | tee "$log_dir/bundle_fit_audit.log"
    fi
    CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/gpu_preflight.py" \
        --project-root "$project_root" --scope chembl --device cuda:0 \
        2>&1 | tee "$log_dir/preflight_chembl.log"
    CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/train_deploy_models.py" \
        --project-root "$project_root" --device cuda:0 \
        --batch-size 6 --full-bidirectional-batch-size 1 \
        2>&1 | tee "$log_dir/train_deploy_models.log"
    CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/infer_chembl.py" \
        --project-root "$project_root" --scope reference --device cuda:0 \
        --biochemical-batch-size "${BIOCHEMICAL_BATCH_SIZE:-2}" \
        2>&1 | tee "$log_dir/infer_reference.log"
    run_sharded_scope full
    run_sharded_scope biochemical
    python "$package/scripts/aggregate_chembl.py" --project-root "$project_root" \
        2>&1 | tee "$log_dir/aggregate_chembl.log"
}

if [[ $stage == benchmark || $stage == all ]]; then run_benchmark; fi
if [[ $stage == chembl || $stage == all ]]; then run_chembl; fi

cd "$project_root"
archive=$package/role_complete_pair_matrix_gpu_return_20260822.tar.gz
items=(
    analysis/role_complete_pair_matrix/README.md
    analysis/role_complete_pair_matrix/EXPERIMENT_CONTRACT.md
    analysis/role_complete_pair_matrix/CODEX_GPU_TASK.md
    analysis/role_complete_pair_matrix/methods
    analysis/role_complete_pair_matrix/requirements-cpu.txt
    analysis/role_complete_pair_matrix/requirements-gpu.txt
    analysis/role_complete_pair_matrix/scripts
    analysis/role_complete_pair_matrix/data
    analysis/role_complete_pair_matrix/validation
    analysis/role_complete_pair_matrix/gpu_output/logs
)
if [[ -d analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate ]]; then
    items+=(analysis/role_complete_pair_matrix/gpu_output/benchmark/aggregate)
    items+=(analysis/role_complete_pair_matrix/gpu_output/benchmark/workers)
    items+=(analysis/role_complete_pair_matrix/gpu_output/benchmark/FIT_AUDIT_REPORTS.tar.gz)
    items+=(analysis/role_complete_pair_matrix/gpu_output/benchmark/FIT_AUDIT_BUNDLE.json)
fi
if [[ -d analysis/role_complete_pair_matrix/gpu_output/deploy ]]; then
    items+=(analysis/role_complete_pair_matrix/gpu_output/deploy)
fi
if [[ -d analysis/role_complete_pair_matrix/gpu_output/chembl ]]; then
    items+=(analysis/role_complete_pair_matrix/gpu_output/chembl)
fi
tar --exclude='*/__pycache__' --exclude='*.pyc' \
    -czf "$archive.tmp" "${items[@]}"
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
