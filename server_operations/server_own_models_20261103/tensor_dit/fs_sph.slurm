#!/bin/bash -l
#SBATCH --job-name=own_tensor_fs_sph_s31
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=13:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --chdir=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/logs/%x_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/logs/%x_%j.err
set -euo pipefail
: "${1:?Pass the reviewed operation_manifest.json SHA256 as the first argument}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/u/zguan2/projects/flow3d_runs/experiments/tensor_dit_n12_20261027/source/src
exec srun --ntasks=1 --gpu-bind=none /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/tensor_dit/continuous.py run --dataset fs_sph --operation-fingerprint "$1"
