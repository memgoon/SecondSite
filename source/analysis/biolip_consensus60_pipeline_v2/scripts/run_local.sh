#!/usr/bin/env bash
set -euo pipefail
PACKAGE=/disk9/13.Heesu_Allostery/analysis/biolip_consensus60_pipeline_v2
PYTHON=/disk9/13.Heesu_Allostery/miniconda3/bin/python
export PYTHONDONTWRITEBYTECODE=1
# One native math thread PER process prevents workers x BLAS oversubscription.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/biolip60_v2_matplotlib_${UID}"
cd "$PACKAGE"
sha256sum --check --quiet PACKAGE_CHECKSUMS.sha256
exec "$PYTHON" scripts/run_pipeline.py "$@"
