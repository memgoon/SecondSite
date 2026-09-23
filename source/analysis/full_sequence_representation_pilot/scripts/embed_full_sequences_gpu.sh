#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/full_sequence_representation_pilot
representation=${REPRESENTATION:-full_canonical_uniprot}
case $representation in
  full_canonical_uniprot)
    input_dir=$package/data/full_sequence_chunks
    output_dir=$package/gpu_cache/chunk_embeddings
    validation_name=FULL_SEQUENCE_EMBEDDING_VALIDATION.json
    files_name=FULL_SEQUENCE_EMBEDDING_FILES.tsv
    expected_residues=289481
    ;;
  rechunked_selected_structure_chain)
    input_dir=$package/data/rechunked_selected_chain_chunks
    output_dir=$package/gpu_cache/rechunked_selected_chain_chunk_embeddings
    validation_name=RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json
    files_name=RECHUNKED_SELECTED_CHAIN_EMBEDDING_FILES.tsv
    expected_residues=169680
    ;;
  *)
    echo "Unknown REPRESENTATION: $representation" >&2
    exit 2
    ;;
esac
gpu_device=${ESM3_GPU_DEVICE:-0}
workers=${ESM3_WORKERS:-1}
token=${ESM3_HF_TOKEN:-${HF_TOKEN:-}}
token_source=environment
legacy_runner=$project_root/14.Organized_input/10.docker_run.sh

if [[ ! -d $input_dir ]]; then
  echo "Missing frozen chunk FASTAs: $input_dir" >&2
  exit 2
fi
mkdir -p "$output_dir" "$package/gpu_output"
if python3 - "$package/gpu_output/$validation_name" "$package/gpu_output/$files_name" "$expected_residues" "$representation" <<'PY'
import json, sys
from pathlib import Path
report, table = map(Path, sys.argv[1:3])
expected_residues = int(sys.argv[3])
representation = sys.argv[4]
if not report.is_file() or not table.is_file():
    raise SystemExit(1)
value = json.loads(report.read_text())
if value.get("status") != "validated" or value.get("representation") != representation or value.get("proteins") != 426 or value.get("total_residues") != expected_residues:
    raise SystemExit(1)
if sum(1 for _ in table.open()) != 427:
    raise SystemExit(1)
PY
then
  echo "Validated $representation embeddings already exist; skipping ESM3 generation."
  exit 0
fi

missing_input=$(mktemp -d /tmp/fullseq_esm_inputs.XXXXXX)
cleanup() { rm -rf -- "$missing_input"; }
trap cleanup EXIT
missing=0
for fasta in "$input_dir"/*.fasta; do
  stem=${fasta##*/}
  stem=${stem%.fasta}
  if [[ ! -f $output_dir/$stem.pt && ! -f $output_dir/$stem.pth ]]; then
    cp -- "$fasta" "$missing_input/"
    missing=$((missing + 1))
  fi
done

if (( missing > 0 )); then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required for esm3-embedder:latest" >&2
    exit 2
  fi
  if ! docker image inspect esm3-embedder:latest >/dev/null 2>&1; then
    echo "Missing Docker image esm3-embedder:latest" >&2
    exit 2
  fi
  if [[ -z $token && -f $legacy_runner ]]; then
    token=$(python3 - "$legacy_runner" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"--hf_token\s+(['\"])([^'\"]+)\1", text)
print(match.group(2) if match else "")
PY
    )
    token_source="existing local ESM3 runner"
  fi
  if [[ -z $token ]]; then
    echo "Set ESM3_HF_TOKEN or HF_TOKEN; no credential was found" >&2
    exit 2
  fi
  echo "Embedding $missing missing $representation chunks; existing chunk tensors are retained."
  echo "ESM3 credential source: $token_source (value hidden)"
  docker run --rm --gpus "device=$gpu_device" \
    -v "$missing_input:/data/input:ro" \
    -v "$output_dir:/data/output" \
    esm3-embedder:latest \
    --input_dir /data/input \
    --output_dir /data/output \
    --model esm3_sm_open_v1 \
    --workers "$workers" \
    --hf_token "$token"
else
  echo "All raw $representation chunk tensors already exist; proceeding to deterministic stitching."
fi

python_bin=${PYTHON_BIN:-python3}
"$python_bin" "$package/scripts/stitch_full_sequence_embeddings.py" \
  --project-root "$project_root" --representation "$representation"
