#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/property_balanced_chembl
ligand_package=$project_root/analysis/ligand_chemistry_balancing
chembl_package=$project_root/analysis/chembl_external_prioritization
main_package=$project_root/analysis/allosteric_pair_benchmark_main
scope=${RUN_SCOPE:-reference}
gpu_csv=${GPU_IDS:-0}
gpu_csv=${gpu_csv//[[:space:]]/}
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
log_dir=$package/gpu_output/logs
mkdir -p "$log_dir"

if [[ $scope != reference && $scope != full ]]; then
    echo "RUN_SCOPE must be reference or full" >&2
    exit 2
fi
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 2 )); then
    echo "GPU_IDS must contain one or two comma-separated GPU indices" >&2
    exit 2
fi
declare -A seen_gpu_ids=()
for gpu in "${gpu_ids[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ ]] || [[ -n ${seen_gpu_ids[$gpu]:-} ]]; then
        echo "Invalid or duplicate GPU index: $gpu" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu]=1
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${USER:-hs0517}
export LOGNAME=${LOGNAME:-$USER}
export EXPECTED_GPU_COUNT=${#gpu_ids[@]}

CUDA_VISIBLE_DEVICES=$gpu_csv python - <<'PY' 2>&1 | tee "$log_dir/environment.log"
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
visible = torch.cuda.device_count()
expected = int(os.environ["EXPECTED_GPU_COUNT"])
if visible != expected:
    raise SystemExit("Expected {} visible GPUs, found {}".format(expected, visible))
for index in range(visible):
    print("GPU {}: {}".format(index, torch.cuda.get_device_name(index)))
PY

required=(
    "$package/config.json"
    "$package/data/UNIPROT_PFAM_CACHE.json"
    "$package/data/FAMILY_NOVELTY_AUDIT.json"
    "$package/data/C3_DEPLOY_EPOCH_AUDIT.json"
    "$package/data/chembl_pocket_extension/TARGET_CHAIN_SEQUENCES.tsv"
    "$package/data/chembl_pocket_extension/POCKET_INDICES.json"
    "$package/data/chembl_pocket_extension/POCKET_ALIGNMENT_AUDIT.tsv"
    "$package/data/chembl_pocket_extension/POCKET_COVERAGE.json"
    "$package/C3_POCKET_EXTENSION_CHECKSUMS.sha256"
    "$ligand_package/PROTEIN_ANCHORED_CHEMISTRY_BALANCED.tsv.gz"
    "$ligand_package/PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260824.tsv.gz"
    "$ligand_package/PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260825.tsv.gz"
    "$ligand_package/gpu_output/common_evaluation/ALL_COMMON_EVALUATION_PREDICTIONS.tsv.gz"
    "$ligand_package/gpu_output/random_sampling_sensitivity/ALL_PREDICTIONS.tsv.gz"
    "$main_package/scripts/train_main_benchmark.py"
    "$main_package/gpu_cache/MODEL_READY.tsv.gz"
    "$main_package/data/POCKET_INDICES.json"
    "$chembl_package/config.json"
    "$chembl_package/scripts/infer_chembl.py"
    "$chembl_package/scripts/train_deploy_models.py"
    "$chembl_package/scripts/prepare_reference_gpu.py"
    "$chembl_package/data/CURRENT_BENCHMARK_PAIR_BLACKLIST.tsv.gz"
    "$chembl_package/gpu_cache/REFERENCE_MODEL_READY.tsv.gz"
)
for path in "${required[@]}"; do
    if [[ ! -f $path ]]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
done

(cd "$project_root" && sha256sum -c \
    analysis/property_balanced_chembl/C3_POCKET_EXTENSION_CHECKSUMS.sha256) \
    2>&1 | tee "$log_dir/validate_c3_readiness_checksums.log"

python "$package/scripts/test_c3_subset_logic.py" \
    2>&1 | tee "$log_dir/test_c3_subset_logic.log"

python "$package/scripts/build_pfam_novelty_cache.py" \
    --project-root "$project_root" --validate-only \
    2>&1 | tee "$log_dir/validate_pfam_cache.log"

python "$package/scripts/validate_family_novelty.py" \
    --project-root "$project_root" --no-write \
    2>&1 | tee "$log_dir/validate_family_novelty.log"

python "$package/scripts/freeze_c3_deploy_epoch.py" \
    --project-root "$project_root" --validate-only \
    2>&1 | tee "$log_dir/validate_c3_deploy_epoch.log"

python "$package/scripts/build_chembl_pocket_indices.py" \
    --project-root "$project_root" --validate-only \
    2>&1 | tee "$log_dir/validate_c3_pocket_build.log"

python "$package/scripts/validate_chembl_c3_readiness.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/validate_c3_readiness.log"

printf '%s  %s\n' \
    101d235bb29b1c61fcce77b4e7785b30f02e6e78bcaa86fdaf0446c1b332e26b \
    "$ligand_package/PROTEIN_ANCHORED_CHEMISTRY_BALANCED.tsv.gz" \
    346e6dcc1da7125a7ca6bb9abdc1e139cc2c1773a24c3a2ea909360c31f11d0f \
    "$ligand_package/PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260824.tsv.gz" \
    c19e9b82a92f46cbe0b5a9cec57a37acd5ab87c1b8ba4e649a0164dcf54281d9 \
    "$ligand_package/PROTEIN_ANCHORED_COUNT_MATCHED_RANDOM_SEED_20260825.tsv.gz" \
  | sha256sum -c - 2>&1 | tee "$log_dir/cohort_hashes.log"

train_extension() {
    local gpu=$1
    local selection_seeds=$2
    local index_name=$3
    local log_name=$4
    CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/train_random_extension.py" \
        --project-root "$project_root" \
        --selection-seeds "$selection_seeds" \
        --overall-index-name "$index_name" \
        --device cuda:0 \
        --batch-size "${BALANCED_BATCH_SIZE:-6}" \
        --eval-batch-size "${BALANCED_EVAL_BATCH_SIZE:-8}" \
        2>&1 | tee "$log_dir/$log_name"
}

# Ensure the frozen source comparator checkpoints exist. This call is
# resumable and performs no inference.
CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$chembl_package/scripts/train_deploy_models.py" \
    --project-root "$project_root" --device cuda:0 \
    --batch-size "${DEPLOY_BATCH_SIZE:-6}" \
    2>&1 | tee "$log_dir/validate_source_deploy.log"

train_property() {
    local gpu=$1
    local models=$2
    local index_name=$3
    local log_name=$4
    CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/train_property_deploy.py" \
        --project-root "$project_root" --models "$models" \
        --index-name "$index_name" --device cuda:0 \
        --batch-size "${DEPLOY_BATCH_SIZE:-6}" \
        2>&1 | tee "$log_dir/$log_name"
}

# C3 must use embeddings generated from the exact selected-chain FASTAs.  The
# legacy full-screen tensors are deliberately not reused for pocket indexing.
# A completed 808/808 cache is validated and skipped on resume.
if python "$package/scripts/validate_chembl_target_chain_embeddings.py" \
    --project-root "$project_root" --summary-only \
    2>&1 | tee "$log_dir/validate_chembl_target_chain_embeddings_preflight.log"
then
    echo "SKIP validated selected-chain embedding cache"
else
    ESM3_GPU_DEVICE=${gpu_ids[0]} bash \
        "$package/scripts/embed_chembl_target_chains_gpu.sh" \
        2>&1 | tee "$log_dir/embed_chembl_target_chains.log"
    python "$package/scripts/validate_chembl_target_chain_embeddings.py" \
        --project-root "$project_root" --summary-only \
        2>&1 | tee "$log_dir/validate_chembl_target_chain_embeddings.log"
fi

if (( ${#gpu_ids[@]} == 2 )); then
    train_property "${gpu_ids[0]}" ligand SHARD_DEPLOY_LIGAND.json \
        train_property_ligand.log &
    pid_a=$!
    train_property "${gpu_ids[1]}" c2 SHARD_DEPLOY_C2.json \
        train_property_c2.log &
    pid_b=$!
    failed=0
    if ! wait "$pid_a"; then failed=1; fi
    if ! wait "$pid_b"; then failed=1; fi
    if (( failed != 0 )); then
        echo "Property deploy training failed" >&2
        exit 3
    fi
    train_property "${gpu_ids[0]}" c3 SHARD_DEPLOY_C3.json \
        train_property_c3.log
    train_property "${gpu_ids[0]}" ligand,c2,c3 DEPLOY_INDEX.json \
        consolidate_property_deploy.log
else
    train_property "${gpu_ids[0]}" ligand,c2,c3 DEPLOY_INDEX.json \
        train_property_deploy.log
fi

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/infer_chembl.py" \
    --project-root "$project_root" --scope reference --device cuda:0 \
    --batch-size-reference "${REFERENCE_BATCH_SIZE:-64}" \
    2>&1 | tee "$log_dir/infer_reference.log"

python "$package/scripts/aggregate_results.py" \
    --project-root "$project_root" --scope reference --bootstrap 10000 \
    2>&1 | tee "$log_dir/aggregate_reference.log"

if [[ $scope == full ]]; then
    pids=()
    for shard in 0 1 2 3; do
        slot=$((shard % ${#gpu_ids[@]}))
        gpu=${gpu_ids[$slot]}
        CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/infer_chembl.py" \
            --project-root "$project_root" --scope full --shard "$shard" \
            --num-shards 4 --device cuda:0 \
            --batch-size-full "${FULL_BATCH_SIZE:-256}" \
            > "$log_dir/infer_full_shard${shard}.log" 2>&1 &
        pids+=("$!")
        if (( ${#pids[@]} == ${#gpu_ids[@]} )); then
            failed=0
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then failed=1; fi
            done
            if (( failed != 0 )); then
                echo "One or more full inference shards failed" >&2
                exit 3
            fi
            pids=()
        fi
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if (( failed != 0 )); then
        echo "One or more full inference shards failed" >&2
        exit 3
    fi
    python "$package/scripts/aggregate_results.py" \
        --project-root "$project_root" --scope full --bootstrap 10000 \
        2>&1 | tee "$log_dir/aggregate_full.log"
fi

# Secondary/SI stability extension.  It remains frozen and mandatory for the
# completed return package, but runs after the primary ChEMBL deployment so a
# time-limited interruption preserves the higher-priority outputs first.
if (( ${#gpu_ids[@]} == 2 )); then
    train_extension "${gpu_ids[0]}" 20260824 SHARD_EXTENSION_20260824.json \
        train_extension_20260824.log &
    pid_a=$!
    train_extension "${gpu_ids[1]}" 20260825 SHARD_EXTENSION_20260825.json \
        train_extension_20260825.log &
    pid_b=$!
    failed=0
    if ! wait "$pid_a"; then failed=1; fi
    if ! wait "$pid_b"; then failed=1; fi
    if (( failed != 0 )); then
        echo "Random extension training failed" >&2
        exit 3
    fi
else
    train_extension "${gpu_ids[0]}" 20260824,20260825 RANDOM_EXTENSION_SHARD.json \
        train_extension.log
fi

python "$package/scripts/validate_random_extension.py" --project-root "$project_root" \
    2>&1 | tee "$log_dir/validate_random_extension.log"

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/score_random_extension.py" \
    --project-root "$project_root" --device cuda:0 \
    --eval-batch-size "${BALANCED_EVAL_BATCH_SIZE:-8}" \
    2>&1 | tee "$log_dir/score_random_extension.log"

python "$package/scripts/analyze_random_extension.py" --project-root "$project_root" \
    2>&1 | tee "$log_dir/analyze_random_extension.log"

python - "$package" "$scope" <<'PY'
import json
import sys
from pathlib import Path
package = Path(sys.argv[1])
scope = sys.argv[2]
paths = [
    package / "gpu_output/random_extension/RANDOM_EXTENSION_INDEX.json",
    package / "gpu_output/random_extension_scoring/VALIDATION.json",
    package / "gpu_output/random_extension_analysis/PREREGISTERED_EXTENSION_RESULT.json",
    package / "gpu_output/deploy_models/DEPLOY_INDEX.json",
    package / "gpu_output/reference/REFERENCE_INFERENCE.json",
    package / "gpu_output/aggregate/AGGREGATE_REFERENCE.json",
]
if scope == "full":
    paths.extend(
        package / "gpu_output/full/FULL_INFERENCE_SHARD{:02d}.json".format(index)
        for index in range(4)
    )
    paths.append(package / "gpu_output/aggregate/AGGREGATE_FULL.json")
for path in paths:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "validated":
        raise SystemExit("validation failed: {}".format(path))
deploy = json.loads(paths[3].read_text(encoding="utf-8"))
if deploy.get("n_completed") != 9 or deploy.get("training_rows") != 3022:
    raise SystemExit("property deploy contract failed")
reference = json.loads(paths[4].read_text(encoding="utf-8"))
if reference.get("probability_conversion") != "sigmoid applied after FP32 logit cast":
    raise SystemExit("FP32 reference inference contract failed")
if reference.get("family_novelty_amendment_id") != "property_balanced_chembl_pfam_novelty_v1":
    raise SystemExit("Pfam family-novelty amendment missing")
if reference.get("family_novelty_counts", {}).get("pfam_family_unseen_rows", 0) <= 0:
    raise SystemExit("Pfam family-unseen reference stratum is empty")
if reference.get("c3_metric_scope") != "pocket_available_rows_only":
    raise SystemExit("C3 subset-only inference contract failed")
if reference.get("c3_pocket_available_rows", 0) <= 0:
    raise SystemExit("C3 reference subset is empty")
aggregate = json.loads(paths[5].read_text(encoding="utf-8"))
c3_metrics = [
    row for row in aggregate.get("primary_metrics", [])
    if row.get("model") == "property_c3"
]
if (
    len(c3_metrics) != 1
    or c3_metrics[0].get("evaluation_subset") != "pocket_available_matched"
):
    raise SystemExit("aggregate C3 metrics are not restricted to the matched subset")
c2_subsets = {
    row.get("evaluation_subset")
    for row in aggregate.get("primary_metrics", [])
    if row.get("model") == "property_c2"
}
if c2_subsets != {"all_rows", "pocket_available_matched"}:
    raise SystemExit("aggregate C2 metrics lack full and C3-matched scopes")
if scope == "full":
    full_aggregate = json.loads(paths[-1].read_text(encoding="utf-8"))
    accounting = full_aggregate.get("full_c3_pocket_accounting", {})
    if (
        accounting.get("pocket_available_rows", 0) <= 0
        or accounting.get("pocket_unavailable_rows", 0) <= 0
    ):
        raise SystemExit("full aggregate lacks explicit C3 pocket accounting")
audit = json.loads(
    (package / "data/FAMILY_NOVELTY_AUDIT.json").read_text(encoding="utf-8")
)
expected = audit["reference_all"]
observed = reference["family_novelty_counts"]
expected_counts = {
    "pfam_annotated_rows": expected["rows"] - expected["pfam_annotation_unavailable_rows"],
    "pfam_family_seen_rows": expected["pfam_seen_rows"],
    "pfam_family_unseen_rows": expected["pfam_unseen_rows"],
    "pfam_annotation_unavailable_rows": expected["pfam_annotation_unavailable_rows"],
}
if observed != expected_counts:
    raise SystemExit(
        "reference Pfam accounting mismatch: observed={} expected={}".format(
            observed, expected_counts
        )
    )
print(json.dumps({"status": "validated", "scope": scope, "reports": len(paths)}, indent=2))
PY

cd "$project_root"
archive=$package/property_balanced_chembl_gpu_return_20260821.tar.gz
items=(
    analysis/property_balanced_chembl/README.md
    analysis/property_balanced_chembl/EXPERIMENT_CONTRACT.md
    analysis/property_balanced_chembl/CODEX_GPU_TASK.md
    analysis/property_balanced_chembl/REVIEW_AMENDMENT.md
    analysis/property_balanced_chembl/C3_POCKET_EXTENSION_AUDIT.md
    analysis/property_balanced_chembl/C3_POCKET_EXTENSION_CHECKSUMS.sha256
    analysis/property_balanced_chembl/config.json
    analysis/property_balanced_chembl/data
    analysis/property_balanced_chembl/scripts
    analysis/property_balanced_chembl/gpu_output
)
tar --exclude='*/__pycache__' --exclude='*.pyc' \
    --exclude='*/gpu_output/full/full_predictions_*.tsv.gz' \
    --exclude='*/best.pt' \
    -czf "$archive.tmp" "${items[@]}"
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
(cd "$package" && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
cat "$archive.sha256"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive (with $archive.sha256)"
