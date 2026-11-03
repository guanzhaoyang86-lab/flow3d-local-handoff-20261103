#!/bin/bash
# Sourced only inside a Slurm allocation on Delta.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'Slurm allocation required.\n' >&2; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { printf 'Taichi 1.7.4 environment requires x86_64.\n' >&2; exit 1; }
: "${LBM500K_PACKAGE_ROOT:?Submit with submit_delta.sh}"
: "${LBM500K_OUTPUT_DIR:?Submit with submit_delta.sh}"
export LBM500K_ENV_PREFIX="${LBM500K_ENV_PREFIX:-$HOME/envs/flow3d-lbm3d}"
if command -v module >/dev/null; then
    module load "${LBM500K_PYTHON_MODULE:-miniforge3-python}"
fi
command -v conda >/dev/null || { printf 'Cannot activate the existing Delta conda environment.\n' >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LBM500K_ENV_PREFIX"
export LBM500K_PYTHON="$LBM500K_ENV_PREFIX/bin/python"
[[ -x "$LBM500K_PYTHON" ]] || { printf 'Missing generation Python: %s\n' "$LBM500K_PYTHON" >&2; exit 1; }
export SLURM_EXPORT_ENV=ALL
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd -- "$LBM500K_PACKAGE_ROOT"
