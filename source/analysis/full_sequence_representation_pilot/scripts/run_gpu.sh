#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/disk1/11.HS_allostery}
package=$project_root/analysis/full_sequence_representation_pilot
stage=${RUN_STAGE:-all}
gpu_csv=${GPU_IDS:-0}
gpu_csv=${gpu_csv//[[:space:]]/}
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"

if [[ $stage != embed && $stage != train && $stage != aggregate && $stage != all ]]; then
  echo "RUN_STAGE must be embed, train, aggregate, or all" >&2
  exit 2
fi
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 2 )); then
  echo "GPU_IDS must contain one or two comma-separated GPU indices" >&2
  exit 2
fi
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! $gpu =~ ^[0-9]+$ ]] || [[ -n ${seen[$gpu]:-} ]]; then
    echo "Invalid or duplicate GPU id: $gpu" >&2
    exit 2
  fi
  seen[$gpu]=1
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export USER=${RUN_USER:-hs0517}
export LOGNAME=$USER
export OMP_NUM_THREADS=${CPU_THREADS:-8}
export MKL_NUM_THREADS=${CPU_THREADS:-8}
log_dir=$package/gpu_output/logs
mkdir -p "$log_dir"

CUDA_VISIBLE_DEVICES=$gpu_csv EXPECTED_GPUS=${#gpu_ids[@]} python3 - <<'PY' 2>&1 | tee "$log_dir/environment.log"
import os, sys
import numpy, pandas, torch
print("Python:", sys.version.replace("\n", " "))
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
if torch.cuda.device_count() != int(os.environ["EXPECTED_GPUS"]):
    raise SystemExit("visible GPU count differs from GPU_IDS")
for index in range(torch.cuda.device_count()):
    print("GPU {}: {}".format(index, torch.cuda.get_device_name(index)))
PY

python3 "$package/scripts/gpu_preflight.py" \
  --project-root "$project_root" --scope cpu --device cpu \
  2>&1 | tee "$log_dir/preflight_cpu.log"

run_embed() {
  if (( ${#gpu_ids[@]} == 1 )); then
    PROJECT_ROOT=$project_root ESM3_GPU_DEVICE=${gpu_ids[0]} REPRESENTATION=rechunked_selected_structure_chain \
      bash "$package/scripts/embed_full_sequences_gpu.sh" \
      2>&1 | tee "$log_dir/embed_rechunked_selected_chain.log"
    PROJECT_ROOT=$project_root ESM3_GPU_DEVICE=${gpu_ids[0]} REPRESENTATION=full_canonical_uniprot \
      bash "$package/scripts/embed_full_sequences_gpu.sh" \
      2>&1 | tee "$log_dir/embed_full_sequences.log"
  else
    (
      PROJECT_ROOT=$project_root ESM3_GPU_DEVICE=${gpu_ids[0]} REPRESENTATION=rechunked_selected_structure_chain \
        bash "$package/scripts/embed_full_sequences_gpu.sh" \
        > "$log_dir/embed_rechunked_selected_chain.log" 2>&1
    ) &
    local selected_pid=$!
    (
      PROJECT_ROOT=$project_root ESM3_GPU_DEVICE=${gpu_ids[1]} REPRESENTATION=full_canonical_uniprot \
        bash "$package/scripts/embed_full_sequences_gpu.sh" \
        > "$log_dir/embed_full_sequences.log" 2>&1
    ) &
    local full_pid=$!
    local failed=0
    if ! wait "$selected_pid"; then failed=1; fi
    if ! wait "$full_pid"; then failed=1; fi
    if (( failed != 0 )); then
      echo "One or more ESM3 representation embeddings failed; rerun the same command to resume" >&2
      exit 3
    fi
  fi
}

run_train() {
  CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python3 "$package/scripts/gpu_preflight.py" \
    --project-root "$project_root" --scope train --device cuda:0 \
    2>&1 | tee "$log_dir/preflight_train.log"
  local world=${#gpu_ids[@]}
  local pids=()
  for rank in "${!gpu_ids[@]}"; do
    local gpu=${gpu_ids[$rank]}
    (
      CUDA_VISIBLE_DEVICES=$gpu python3 "$package/scripts/train_full_sequence.py" \
        --project-root "$project_root" \
        --worker-rank "$rank" --world-size "$world" --device cuda:0 \
        --models protein,c1,c2,c3 \
        --representations rechunked_selected_structure_chain,full_canonical_uniprot \
        --seeds 20260817,20260818,20260819 --folds 0,1,2,3,4 \
        --epochs 25 --patience 5 --min-epochs 1 \
        --batch-size 6 --eval-batch-size 8 \
        --c3-batch-size 1 --c3-eval-batch-size 2 \
        --hidden-dim 256 --heads 4 --dropout 0.30 \
        --lr 0.0001 --weight-decay 0.0001 --max-atoms 120 \
        > "$log_dir/train_worker${rank}.log" 2>&1
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed != 0 )); then
    echo "One or more workers failed; rerun the identical command to resume completed fits" >&2
    exit 3
  fi
}

run_aggregate() {
  CUDA_VISIBLE_DEVICES=${gpu_ids[0]} python3 "$package/scripts/gpu_preflight.py" \
    --project-root "$project_root" --scope aggregate --device cuda:0 \
    2>&1 | tee "$log_dir/preflight_aggregate.log"
  python3 "$package/scripts/aggregate_pilot.py" \
    --project-root "$project_root" \
    --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-10000}" \
    --workers "${BOOTSTRAP_WORKERS:-16}" \
    2>&1 | tee "$log_dir/aggregate.log"
}

if [[ $stage == embed || $stage == all ]]; then run_embed; fi
if [[ $stage == train || $stage == all ]]; then run_train; fi
if [[ $stage == aggregate || $stage == all ]]; then run_aggregate; fi

cd "$project_root"
archive=$package/full_sequence_representation_pilot_gpu_return_20260827.tar.gz
items=(
  analysis/full_sequence_representation_pilot/README.md
  analysis/full_sequence_representation_pilot/EXPERIMENT_CONTRACT.md
  analysis/full_sequence_representation_pilot/CODEX_GPU_TASK.md
  analysis/full_sequence_representation_pilot/scripts
  analysis/full_sequence_representation_pilot/data/FULL_SEQUENCE_MANIFEST.tsv
  analysis/full_sequence_representation_pilot/data/CHUNK_MANIFEST.tsv
  analysis/full_sequence_representation_pilot/data/RECHUNKED_SELECTED_CHAIN_MANIFEST.tsv
  analysis/full_sequence_representation_pilot/data/RECHUNKED_SELECTED_CHAIN_CHUNK_MANIFEST.tsv
  analysis/full_sequence_representation_pilot/data/PILOT_COHORT.tsv.gz
  analysis/full_sequence_representation_pilot/validation
  analysis/full_sequence_representation_pilot/gpu_output/logs
)
if [[ -f analysis/full_sequence_representation_pilot/gpu_output/FULL_SEQUENCE_EMBEDDING_VALIDATION.json ]]; then
  items+=(analysis/full_sequence_representation_pilot/gpu_output/FULL_SEQUENCE_EMBEDDING_VALIDATION.json)
  items+=(analysis/full_sequence_representation_pilot/gpu_output/FULL_SEQUENCE_EMBEDDING_FILES.tsv)
fi
if [[ -f analysis/full_sequence_representation_pilot/gpu_output/RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json ]]; then
  items+=(analysis/full_sequence_representation_pilot/gpu_output/RECHUNKED_SELECTED_CHAIN_EMBEDDING_VALIDATION.json)
  items+=(analysis/full_sequence_representation_pilot/gpu_output/RECHUNKED_SELECTED_CHAIN_EMBEDDING_FILES.tsv)
fi
if [[ -d analysis/full_sequence_representation_pilot/gpu_output/benchmark/fits ]]; then
  items+=(analysis/full_sequence_representation_pilot/gpu_output/benchmark/fits)
fi
if [[ -d analysis/full_sequence_representation_pilot/gpu_output/aggregate ]]; then
  items+=(analysis/full_sequence_representation_pilot/gpu_output/aggregate)
fi
tar --exclude='*/__pycache__' --exclude='*.pyc' --exclude='*/best.pt' \
  -czf "$archive.tmp" "${items[@]}"
mv -f "$archive.tmp" "$archive"
gzip -t "$archive"
sha256sum "$archive"
stat -c 'size_bytes=%s' "$archive"
echo "Complete. Return archive: $archive"
