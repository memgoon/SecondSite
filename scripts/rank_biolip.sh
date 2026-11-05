#!/usr/bin/env bash
set -euo pipefail
data_root=${1:?Usage: bash scripts/rank_biolip.sh DATA_ROOT OUTPUT_DIR}
output_dir=${2:?Output directory required}
python -m secondsite biolip-rank --data-root "$data_root" --output "$output_dir" --evaluate
