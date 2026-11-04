#!/bin/bash
#SBATCH --job-name=own3_lbm_dps
#SBATCH --partition=ghx4
#SBATCH --account=biup-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/dps/lbm/logs/train_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/dps/lbm/logs/train_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/lbm_split_fix_20261104/dps/lbm/runtime.py train
