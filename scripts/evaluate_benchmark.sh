#!/usr/bin/env bash
set -euo pipefail
data_root=${1:?Usage: bash scripts/evaluate_benchmark.sh DATA_ROOT OUTPUT_DIR}
output_dir=${2:?Output directory required}
python -m secondsite.evaluate_models --predictions "$data_root/benchmark/predictions/main_five_fold_all_seeds.tsv.gz" --output "$output_dir"
