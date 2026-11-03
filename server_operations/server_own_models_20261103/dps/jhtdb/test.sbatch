#!/bin/bash
#SBATCH --job-name=own3_jhtdb_dps_evaluate
#SBATCH --partition=ghx4
#SBATCH --account=biup-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dps/jhtdb/logs/test_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dps/jhtdb/logs/test_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /sw/user/python/miniforge3-pytorch-2.10.0/bin/python -B /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dps/jhtdb/runtime.py test
