#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package_dir=$project_root/analysis/new_pairing_control_ladder
log_dir=$package_dir/gpu_output/logs
mkdir -p "$log_dir"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${USER:-hs0517}
export LOGNAME=${LOGNAME:-$USER}

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("PyTorch:", torch.__version__, flush=True)
PY

python "$package_dir/scripts/prepare_gpu_control_ladder.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/prepare_gpu_control_ladder.log"

python "$package_dir/scripts/build_control_stage_datasets.py" \
    --project-root "$project_root" \
    2>&1 | tee "$log_dir/build_control_stage_datasets.log"

python "$package_dir/scripts/train_control_ladder.py" \
    --project-root "$project_root" \
    --stages uncontrolled,protein_controlled,fully_controlled \
    --models c1,c2,c3,d1,graph_cross_pair \
    --seed 20260817 \
    --epochs 12 --patience 3 \
    --batch-size "${PILOT_BATCH_SIZE:-6}" \
    --eval-batch-size "${PILOT_EVAL_BATCH_SIZE:-8}" \
    --device cuda:0 \
    2>&1 | tee "$log_dir/train_control_ladder.log"

python - "$package_dir/gpu_cache/CONTROL_STAGE_VALIDATION.json" "$package_dir/gpu_output/CONTROL_LADDER_REPORT.json" <<'PY'
import json
import sys

stages = json.load(open(sys.argv[1], encoding="utf-8"))
report = json.load(open(sys.argv[2], encoding="utf-8"))
if stages.get("status") != "validated" or report.get("status") != "validated":
    raise SystemExit("control ladder did not validate")
expected_stages = {"uncontrolled", "protein_controlled", "fully_controlled"}
expected_models = {"c1", "c2", "c3", "d1", "graph_cross_pair"}
if set(report.get("stages", {})) != expected_stages:
    raise SystemExit("stage set mismatch")
for name, value in report["stages"].items():
    if set(value.get("models", {})) != expected_models:
        raise SystemExit("model set mismatch for " + name)
if not stages.get("protein_template_exactly_reproduced"):
    raise SystemExit("Pfam-only template was not reproduced")
fully = {row["split"]: row for row in stages["stage_counts"]["fully_controlled"]}
expected = {"train": 2177, "val": 321, "test": 364}
if {key: int(fully[key]["n_rows"]) for key in expected} != expected:
    raise SystemExit("fully controlled counts changed")
print(json.dumps({
    "status": "validated",
    "base_rows": stages["base_rows"],
    "stage_counts": stages["stage_counts"],
    "models": sorted(expected_models),
}, indent=2, sort_keys=True))
PY

cd "$project_root"
archive=$package_dir/new_pairing_control_ladder_gpu_return_20260817.tar.gz
items=(
    analysis/new_pairing_control_ladder/EXPERIMENT_CONTRACT.md
    analysis/new_pairing_control_ladder/validation
    analysis/new_pairing_control_ladder/gpu_cache/BASE_EMBEDDING_VALIDATION.json
    analysis/new_pairing_control_ladder/gpu_cache/CONTROL_STAGE_VALIDATION.json
    analysis/new_pairing_control_ladder/gpu_cache/BASE_MODEL_READY.tsv.gz
    analysis/new_pairing_control_ladder/gpu_cache/stages
    analysis/new_pairing_control_ladder/gpu_output/logs
    analysis/new_pairing_control_ladder/gpu_output/CONTROL_LADDER_REPORT.json
    analysis/new_pairing_control_ladder/gpu_output/uncontrolled/metrics
    analysis/new_pairing_control_ladder/gpu_output/uncontrolled/predictions
    analysis/new_pairing_control_ladder/gpu_output/uncontrolled/STAGE_REPORT.json
    analysis/new_pairing_control_ladder/gpu_output/protein_controlled/metrics
    analysis/new_pairing_control_ladder/gpu_output/protein_controlled/predictions
    analysis/new_pairing_control_ladder/gpu_output/protein_controlled/STAGE_REPORT.json
    analysis/new_pairing_control_ladder/gpu_output/fully_controlled/metrics
    analysis/new_pairing_control_ladder/gpu_output/fully_controlled/predictions
    analysis/new_pairing_control_ladder/gpu_output/fully_controlled/STAGE_REPORT.json
)
tar -czf "$archive.tmp" "${items[@]}"
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
