#!/bin/bash -l
#SBATCH --job-name=own3_lbm_dit3d
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dit3d_lbm
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dit3d_lbm/logs/%x_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dit3d_lbm/logs/%x_%j.err
set -euo pipefail
: "${DIT_LBM_FROZEN_SHA256:?Submission must bind the reviewed frozen manifest}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
exec srun --ntasks=1 --gpu-bind=none /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dit3d_lbm/run_lbm.py
