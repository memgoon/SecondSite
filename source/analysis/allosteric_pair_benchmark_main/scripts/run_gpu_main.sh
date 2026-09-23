#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/allosteric_pair_benchmark_main
output=$package/gpu_output/main_benchmark
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

python "$package/scripts/validate_target_chain_embeddings.py" \
    --project-root "$project_root" --allow-incomplete --summary-only \
    2>&1 | tee "$log_dir/validate_target_chain_embeddings_before.log"

if ! python - "$package/validation/TARGET_CHAIN_EMBEDDING_VALIDATION.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "validated" else 1)
PY
then
    bash "$package/scripts/embed_target_chains_gpu.sh" \
        2>&1 | tee "$log_dir/embed_target_chains.log"
fi

python "$package/scripts/validate_target_chain_embeddings.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/validate_target_chain_embeddings_after.log"

python "$package/scripts/prepare_exact_gpu_inputs.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/prepare_exact_gpu_inputs.log"

python "$package/scripts/train_main_benchmark.py" \
    --project-root "$project_root" \
    --models ligand,protein,c1,c2,c3,d1 \
    --regimes row_random,unseen_family \
    --folds 0,1,2,3,4 \
    --seeds 20260817,20260818,20260819 \
    --epochs 25 --patience 5 \
    --batch-size "${MAIN_BATCH_SIZE:-6}" \
    --eval-batch-size "${MAIN_EVAL_BATCH_SIZE:-8}" \
    --device cuda:0 \
    2>&1 | tee "$log_dir/train_main_benchmark.log"

python "$package/scripts/aggregate_main_results.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/aggregate_main_results.log"

python - "$package/validation/GPU_INPUT_VALIDATION.json" \
         "$output/TRAINING_INDEX.json" \
         "$output/aggregate/AGGREGATE_VALIDATION.json" <<'PY'
import json
import sys

gpu = json.load(open(sys.argv[1], encoding="utf-8"))
training = json.load(open(sys.argv[2], encoding="utf-8"))
aggregate = json.load(open(sys.argv[3], encoding="utf-8"))
if gpu.get("status") != "validated":
    raise SystemExit("exact GPU input validation failed")
if training.get("status") != "validated" or training.get("n_completed_fits") != 180:
    raise SystemExit("180-fit training contract was not completed")
if aggregate.get("status") != "validated":
    raise SystemExit("OOF aggregate validation failed")
if "No connectivity-key tensor fallback" not in gpu.get("exact_identity_policy", ""):
    raise SystemExit("exact ligand policy changed")
print(json.dumps({
    "status": "validated",
    "model_ready_rows": gpu["model_ready_rows"],
    "model_ready_proteins": gpu["model_ready_proteins"],
    "completed_fits": training["n_completed_fits"],
    "maximum_epoch_hits": training["maximum_epoch_hits"],
}, indent=2, sort_keys=True))
PY

cd "$project_root"
archive=$package/allosteric_pair_benchmark_main_gpu_return_20260817.tar.gz
tar --exclude='*/best.pt' -czf "$archive.tmp" \
    analysis/allosteric_pair_benchmark_main/README.md \
    analysis/allosteric_pair_benchmark_main/EXPERIMENT_CONTRACT.md \
    analysis/allosteric_pair_benchmark_main/CODEX_GPU_TASK.md \
    analysis/allosteric_pair_benchmark_main/MANIFEST.sha256 \
    analysis/allosteric_pair_benchmark_main/requirements-cpu.txt \
    analysis/allosteric_pair_benchmark_main/config \
    analysis/allosteric_pair_benchmark_main/methods \
    analysis/allosteric_pair_benchmark_main/docs \
    analysis/allosteric_pair_benchmark_main/scripts \
    analysis/allosteric_pair_benchmark_main/tests \
    analysis/allosteric_pair_benchmark_main/data \
    analysis/allosteric_pair_benchmark_main/validation \
    analysis/allosteric_pair_benchmark_main/gpu_cache/MODEL_READY.tsv.gz \
    analysis/allosteric_pair_benchmark_main/gpu_cache/LIGAND_EMBEDDING_MANIFEST.tsv.gz \
    analysis/allosteric_pair_benchmark_main/gpu_cache/PUBCHEM_EXACT_SMILES.tsv.gz \
    analysis/allosteric_pair_benchmark_main/gpu_output/main_benchmark
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
