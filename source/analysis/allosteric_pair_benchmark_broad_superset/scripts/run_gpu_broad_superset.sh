#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/allosteric_pair_benchmark_broad_superset
main_package=$project_root/analysis/allosteric_pair_benchmark_main
output=$package/gpu_output/broad_superset
log_dir=$output/logs
mkdir -p "$log_dir"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${USER:-hs0517}
export LOGNAME=${LOGNAME:-$USER}

python - <<'PY' 2>&1 | tee "$log_dir/environment.log"
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
print("unimol_tools:", getattr(unimol_tools, "__version__", "unknown"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

for required in \
    "$main_package/scripts/train_main_benchmark.py" \
    "$main_package/scripts/prepare_exact_gpu_inputs.py" \
    "$main_package/scripts/aggregate_main_results.py" \
    "$main_package/scripts/bootstrap_oof.py" \
    "$main_package/gpu_cache/MODEL_READY.tsv.gz" \
    "$project_root/17.paired_dataset/legacy/0.ligand_path_index.pkl"
do
    if [ ! -f "$required" ]; then
        echo "Missing frozen dependency: $required" >&2
        exit 2
    fi
done

cd "$project_root"
sha256sum -c analysis/allosteric_pair_benchmark_broad_superset/MANIFEST.sha256 \
    2>&1 | tee "$log_dir/manifest_validation.log"

python "$package/scripts/validate_extra_target_embeddings.py" \
    --project-root "$project_root" --allow-incomplete --summary-only \
    2>&1 | tee "$log_dir/validate_extra_embeddings_before.log"
if ! python - "$package/validation/EXTRA_TARGET_EMBEDDING_VALIDATION.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "validated" else 1)
PY
then
    bash "$package/scripts/embed_extra_target_chains_gpu.sh" \
        2>&1 | tee "$log_dir/embed_extra_target_chains.log"
fi

python "$package/scripts/validate_extra_target_embeddings.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/validate_extra_embeddings_after.log"

python "$package/scripts/prepare_broad_gpu_inputs.py" \
    --project-root "$project_root" \
    --batch-size "${UNIMOL_BATCH_SIZE:-64}" \
    2>&1 | tee "$log_dir/prepare_broad_gpu_inputs.log"

python "$package/scripts/train_broad_superset.py" \
    --project-root "$project_root" \
    --models ligand,protein,c1,c2,c3,d1 \
    --regimes row_random,unseen_family \
    --folds 0,1,2,3,4 \
    --seeds 20260817,20260818,20260819 \
    --epochs 25 --patience 5 \
    --batch-size "${BROAD_BATCH_SIZE:-6}" \
    --eval-batch-size "${BROAD_EVAL_BATCH_SIZE:-8}" \
    --device cuda:0 \
    2>&1 | tee "$log_dir/train_broad_superset.log"

python "$package/scripts/aggregate_pool_comparison.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/aggregate_pool_comparison.log"

python "$package/scripts/bootstrap_pool_delta.py" \
    --project-root "$project_root" --replicates 10000 \
    2>&1 | tee "$log_dir/bootstrap_pool_delta.log"

python - "$package/validation/GPU_INPUT_VALIDATION.json" \
         "$package/validation/SPLIT_VALIDATION.json" \
         "$output/TRAINING_INDEX.json" \
         "$output/aggregate/AGGREGATE_VALIDATION.json" \
         "$output/aggregate/BOOTSTRAP_VALIDATION.json" <<'PY'
import json
import sys
gpu, split, training, aggregate, bootstrap = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
if gpu.get("status") != "validated" or not gpu.get("gates", {}).get("arm_b_is_strict_superset_of_arm_a"):
    raise SystemExit("Arm B model-ready superset gate failed")
if split.get("status") != "validated":
    raise SystemExit("split validation failed")
if training.get("status") != "validated" or training.get("n_completed_fits") != 180:
    raise SystemExit("180 Arm B fits were not completed")
if aggregate.get("status") != "validated" or bootstrap.get("status") != "validated":
    raise SystemExit("paired aggregation/bootstrap failed")
print(json.dumps({
    "status": "validated",
    "arm_a_rows": gpu["arm_a_rows_retained"],
    "arm_b_rows": gpu["model_ready_rows"],
    "arm_b_proteins": gpu["model_ready_proteins"],
    "additional_rows": gpu["additional_rows_retained"],
    "completed_fits": training["n_completed_fits"],
}, indent=2, sort_keys=True))
PY

cd "$project_root"
archive=$package/allosteric_pair_benchmark_broad_superset_gpu_return_20260818.tar.gz
tar --exclude='*/best.pt' -czf "$archive.tmp" \
    analysis/allosteric_pair_benchmark_broad_superset/README.md \
    analysis/allosteric_pair_benchmark_broad_superset/EXPERIMENT_CONTRACT.md \
    analysis/allosteric_pair_benchmark_broad_superset/CODEX_GPU_TASK.md \
    analysis/allosteric_pair_benchmark_broad_superset/requirements-cpu.txt \
    analysis/allosteric_pair_benchmark_broad_superset/requirements-gpu.txt \
    analysis/allosteric_pair_benchmark_broad_superset/MANIFEST.sha256 \
    analysis/allosteric_pair_benchmark_broad_superset/methods \
    analysis/allosteric_pair_benchmark_broad_superset/docs \
    analysis/allosteric_pair_benchmark_broad_superset/scripts \
    analysis/allosteric_pair_benchmark_broad_superset/data \
    analysis/allosteric_pair_benchmark_broad_superset/validation \
    analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/BROAD_MODEL_READY.tsv.gz \
    analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/LIGAND_EMBEDDING_MANIFEST.tsv.gz \
    analysis/allosteric_pair_benchmark_broad_superset/gpu_cache/PUBCHEM_EXACT_SMILES.tsv.gz \
    analysis/allosteric_pair_benchmark_broad_superset/gpu_output/broad_superset
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
