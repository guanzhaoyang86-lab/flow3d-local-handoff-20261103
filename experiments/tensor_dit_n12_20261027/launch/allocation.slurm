#!/bin/bash -l
set -euo pipefail
: "${TENSOR_JOB_KEY:?Missing Tensor job key}"
: "${TENSOR_FROZEN_SHA256:?Missing frozen input fingerprint}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/u/zguan2/projects/flow3d_runs/experiments/tensor_dit_n12_20261027/source/src
exec srun --ntasks=1 --gpu-bind=none \
  /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B \
  /u/zguan2/projects/flow3d_runs/experiments/tensor_dit_n12_20261027/launch/runtime.py \
  --job-key "$TENSOR_JOB_KEY"
