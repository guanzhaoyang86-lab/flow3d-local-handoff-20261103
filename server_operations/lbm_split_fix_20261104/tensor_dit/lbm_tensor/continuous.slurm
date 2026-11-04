#!/bin/bash -l
#SBATCH --job-name=own3_lbm_tensor_dit
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21:00:00
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --open-mode=append
#SBATCH --chdir=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/tensor_dit/lbm_tensor
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/tensor_dit/lbm_tensor/logs/%x_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/tensor_dit/lbm_tensor/logs/%x_%j.err
set -euo pipefail
: "${1:?Pass reviewed frozen_manifest.json SHA256 as first argument}"
export TENSOR_FROZEN_SHA256="$1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/tensor_dit/lbm_tensor/source/src
for key in lbm_tensor_prepare lbm_pilot lbm_tensor_dit; do
    export TENSOR_JOB_KEY="$key"
    srun --ntasks=1 --gpu-bind=none /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/tensor_dit/lbm_tensor/launch/runtime.py --job-key "$key"
done
