#!/usr/bin/env bash
set -euo pipefail

# Generate one ESM3 residue tensor for each frozen structure-matched target
# chain.  Credentials are intentionally supplied only at runtime through the
# environment; no token is stored in this release package.
project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/allosteric_pair_benchmark_main
input_dir=$package/data/target_chain_fasta_v2
output_dir=$package/gpu_cache/target_chain_embeddings_v2
gpu_device=${ESM3_GPU_DEVICE:-0}
workers=${ESM3_WORKERS:-1}
token=${ESM3_HF_TOKEN:-${HF_TOKEN:-}}
token_source="environment"
legacy_runner=$project_root/14.Organized_input/10.docker_run.sh

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for the local esm3-embedder image." >&2
    exit 2
fi
if ! docker image inspect esm3-embedder:latest >/dev/null 2>&1; then
    echo "Missing Docker image esm3-embedder:latest." >&2
    exit 2
fi
if [ ! -d "$input_dir" ]; then
    echo "Missing frozen target-chain FASTA directory: $input_dir" >&2
    exit 2
fi

mkdir -p "$output_dir"
args=(
    --input_dir /data/input
    --output_dir /data/output
    --model esm3_sm_open_v1
    --workers "$workers"
)
# The established GPU-server workflow stores the required token in its local
# 10.docker_run.sh.  Reuse that private server configuration only when no
# environment token was supplied.  The value is never printed or copied into
# this release package.
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
    echo "ESM3 requires a Hugging Face token. Set ESM3_HF_TOKEN or HF_TOKEN; no token was found in $legacy_runner." >&2
    exit 2
fi
args+=(--hf_token "$token")
echo "ESM3 credential source: $token_source (value hidden)"

docker run --rm --gpus "device=$gpu_device" \
    -v "$input_dir:/data/input:ro" \
    -v "$output_dir:/data/output" \
    esm3-embedder:latest "${args[@]}"
