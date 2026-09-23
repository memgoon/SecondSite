#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/allosteric_pair_benchmark_broad_superset
input_dir=$package/data/extra_target_chain_fasta
output_dir=$package/gpu_cache/extra_target_chain_embeddings
gpu_device=${ESM3_GPU_DEVICE:-0}
workers=${ESM3_WORKERS:-1}
token=${ESM3_HF_TOKEN:-${HF_TOKEN:-}}
token_source=environment
legacy_runner=$project_root/14.Organized_input/10.docker_run.sh

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for esm3-embedder:latest." >&2
    exit 2
fi
if ! docker image inspect esm3-embedder:latest >/dev/null 2>&1; then
    echo "Missing Docker image esm3-embedder:latest." >&2
    exit 2
fi
if [ ! -d "$input_dir" ]; then
    echo "Missing additional target-chain FASTA directory: $input_dir" >&2
    exit 2
fi

mkdir -p "$output_dir"
if [ -z "$token" ] && [ -f "$legacy_runner" ]; then
    token=$(python3 - "$legacy_runner" <<'PY'
import re
import sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"--hf_token\s+(['\"])([^'\"]+)\1", text)
print(match.group(2) if match else "")
PY
)
    token_source="existing local ESM3 runner"
fi
if [ -z "$token" ]; then
    echo "Set ESM3_HF_TOKEN/HF_TOKEN or retain the existing local ESM3 runner." >&2
    exit 2
fi
echo "ESM3 credential source: $token_source (value hidden)"

docker run --rm --gpus "device=$gpu_device" \
    -e USER="${USER:-hs0517}" \
    -e LOGNAME="${LOGNAME:-${USER:-hs0517}}" \
    -v "$input_dir:/data/input:ro" \
    -v "$output_dir:/data/output" \
    esm3-embedder:latest \
    --input_dir /data/input \
    --output_dir /data/output \
    --model esm3_sm_open_v1 \
    --workers "$workers" \
    --hf_token "$token"
