#!/bin/bash -l
set -euo pipefail
: "${BASELINE_JOB_KEY:?Missing frozen job key}"
: "${BASELINE_FROZEN_SHA256:?Missing submission-time frozen fingerprint}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/u/zguan2/projects/flow3d_runs/experiments/baselines_n12_20261027/source/src
exec srun --ntasks=1 --gpu-bind=none \
  /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B \
  /u/zguan2/projects/flow3d_runs/experiments/baselines_n12_20261027/launch/batch_runtime.py \
  --job-key "$BASELINE_JOB_KEY"
