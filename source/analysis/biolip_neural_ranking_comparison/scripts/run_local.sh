#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package=$(dirname -- "$script_dir")
python_bin=${PYTHON_BIN:-python3}
workers=${BOOTSTRAP_WORKERS:-8}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$package/cpu_output"
exec 9>"$package/cpu_output/RUN.lock"
flock -n 9 || { echo 'Another comparison run holds the lock.' >&2; exit 1; }
"$python_bin" "$script_dir/test_statistics.py"
"$python_bin" "$script_dir/prepare.py"
"$python_bin" -u "$script_dir/run_comparison.py" --workers "$workers"
