#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/chembl_external_prioritization
scope=${RUN_SCOPE:-reference}
gpu_csv=${GPU_IDS:-0}
log_dir=$package/gpu_output/logs
mkdir -p "$log_dir"

if [ "$scope" != reference ] && [ "$scope" != full ]; then
    echo "RUN_SCOPE must be reference or full" >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if [ "${#gpu_ids[@]}" -lt 1 ] || [ "${#gpu_ids[@]}" -gt 2 ]; then
    echo "GPU_IDS must list one or two GPUs, for example 0 or 0,1" >&2
    exit 2
fi
for gpu in "${gpu_ids[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU identifier: $gpu" >&2
        exit 2
    fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${USER:-hs0517}
export LOGNAME=${LOGNAME:-$USER}

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python - <<'PY' 2>&1 | tee "$log_dir/environment.log"
import sys
import numpy
import pandas
import torch
from rdkit import rdBase
import unimol_tools
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
print("RDKit:", rdBase.rdkitVersion)
print("Uni-Mol:", getattr(unimol_tools, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("Selected GPU:", torch.cuda.get_device_name(0))
PY

for required in \
    "$package/validation/CPU_INPUT_VALIDATION.json" \
    "$package/data/CHEMBL_DIRECT_REFERENCE_PAIRS.tsv.gz" \
    "$package/data/CURRENT_BENCHMARK_PAIR_BLACKLIST.tsv.gz" \
    "$project_root/analysis/allosteric_pair_benchmark_main/scripts/train_main_benchmark.py" \
    "$project_root/analysis/allosteric_pair_benchmark_main/scripts/prepare_exact_gpu_inputs.py" \
    "$project_root/analysis/allosteric_pair_benchmark_main/gpu_cache/MODEL_READY.tsv.gz" \
    "$project_root/17.paired_dataset/legacy/0.ligand_path_index.pkl"
do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 2
    fi
done

cd "$project_root"
sha256sum -c analysis/chembl_external_prioritization/MANIFEST.sha256 \
    2>&1 | tee "$log_dir/manifest_validation.log"
python "$package/scripts/validate_cpu_bundle.py" --project-root "$project_root" \
    2>&1 | tee "$log_dir/validate_cpu_bundle.log"

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/prepare_reference_gpu.py" \
    --project-root "$project_root" --batch-size "${UNIMOL_BATCH_SIZE:-64}" \
    2>&1 | tee "$log_dir/prepare_reference_gpu.log"

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/train_deploy_models.py" \
    --project-root "$project_root" --device cuda:0 \
    --batch-size "${DEPLOY_BATCH_SIZE:-6}" \
    2>&1 | tee "$log_dir/train_deploy_models.log"

CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python "$package/scripts/infer_chembl.py" \
    --project-root "$project_root" --scope reference --device cuda:0 \
    --batch-size-reference "${REFERENCE_BATCH_SIZE:-64}" \
    2>&1 | tee "$log_dir/infer_reference.log"

python "$package/scripts/aggregate_chembl_results.py" \
    --project-root "$project_root" --scope reference --bootstrap 10000 \
    2>&1 | tee "$log_dir/aggregate_reference.log"

if [ "$scope" = full ]; then
    pids=()
    for shard in 0 1 2 3; do
        slot=$((shard % ${#gpu_ids[@]}))
        gpu=${gpu_ids[$slot]}
        (
            CUDA_VISIBLE_DEVICES=$gpu python "$package/scripts/infer_chembl.py" \
                --project-root "$project_root" --scope full --shard "$shard" \
                --num-shards 4 --device cuda:0 \
                --batch-size-full "${FULL_BATCH_SIZE:-256}" \
                > "$log_dir/infer_full_shard${shard}.log" 2>&1
        ) &
        pids+=("$!")
        if [ "${#pids[@]}" -eq "${#gpu_ids[@]}" ]; then
            failed=0
            for pid in "${pids[@]}"; do
                if ! wait "$pid"; then
                    failed=1
                fi
            done
            if [ "$failed" -ne 0 ]; then
                echo "One or more full-inference shards failed" >&2
                exit 3
            fi
            pids=()
        fi
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "One or more full-inference shards failed" >&2
        exit 3
    fi
    python "$package/scripts/aggregate_chembl_results.py" \
        --project-root "$project_root" --scope full --bootstrap 10000 \
        2>&1 | tee "$log_dir/aggregate_full.log"
fi

python - "$package" "$scope" <<'PY'
import json
import sys
from pathlib import Path
package = Path(sys.argv[1])
scope = sys.argv[2]
paths = [
    package / "validation/CPU_INPUT_VALIDATION.json",
    package / "validation/GPU_REFERENCE_INPUT_VALIDATION.json",
    package / "gpu_output/deploy_models/DEPLOY_INDEX.json",
    package / "gpu_output/reference/REFERENCE_INFERENCE.json",
    package / "gpu_output/aggregate/AGGREGATE_REFERENCE.json",
]
if scope == "full":
    paths.extend(package / "gpu_output/full/FULL_INFERENCE_SHARD{:02d}.json".format(i) for i in range(4))
    paths.append(package / "gpu_output/aggregate/AGGREGATE_FULL.json")
for path in paths:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "validated":
        raise SystemExit("validation failed: {}".format(path))
    if (
        path.name == "REFERENCE_INFERENCE.json"
        or path.name.startswith("FULL_INFERENCE_SHARD")
    ) and report.get("probability_conversion") != "sigmoid applied after FP32 logit cast":
        raise SystemExit("legacy probability conversion remains: {}".format(path))
print(json.dumps({"status": "validated", "scope": scope, "reports": len(paths)}, indent=2))
PY

cd "$project_root"
archive=$package/chembl_external_prioritization_gpu_return_20260818.tar.gz
archive_items=(
    analysis/chembl_external_prioritization/README.md
    analysis/chembl_external_prioritization/EXPERIMENT_CONTRACT.md
    analysis/chembl_external_prioritization/CODEX_GPU_TASK.md
    analysis/chembl_external_prioritization/config.json
    analysis/chembl_external_prioritization/MANIFEST.sha256
    analysis/chembl_external_prioritization/methods
    analysis/chembl_external_prioritization/scripts
    analysis/chembl_external_prioritization/data
    analysis/chembl_external_prioritization/validation
    analysis/chembl_external_prioritization/gpu_cache/REFERENCE_MODEL_READY.tsv.gz
    analysis/chembl_external_prioritization/gpu_cache/REFERENCE_LIGAND_MANIFEST.tsv.gz
    analysis/chembl_external_prioritization/gpu_output/deploy_models
    analysis/chembl_external_prioritization/gpu_output/reference
    analysis/chembl_external_prioritization/gpu_output/aggregate
    analysis/chembl_external_prioritization/gpu_output/logs
)
if [ -d analysis/chembl_external_prioritization/gpu_output/full ]; then
    archive_items+=(analysis/chembl_external_prioritization/gpu_output/full)
fi
tar --exclude='*/gpu_output/full/full_predictions_*.tsv.gz' \
    --exclude='*/gpu_cache/exact_reference_ligands' \
    -czf "$archive.tmp" "${archive_items[@]}"
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
