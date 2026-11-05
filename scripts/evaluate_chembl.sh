#!/usr/bin/env bash
set -euo pipefail
data_root=${1:?Usage: bash scripts/evaluate_chembl.sh DATA_ROOT OUTPUT_DIR}
output_dir=${2:?Output directory required}
python -m secondsite chembl-metrics --data-root "$data_root" --output "$output_dir"
