#!/bin/bash -l
#SBATCH --job-name=own3_lbm_adapter
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterok:3515292
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/lbm_adapter
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/lbm_adapter/logs/%x_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/lbm_adapter/logs/%x_%j.err
set -euo pipefail
: "${1:?Pass the reviewed adapter operation_manifest.json SHA256}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec srun --ntasks=1 /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/lbm_adapter/prepare.py --execute --operation-fingerprint "$1"
