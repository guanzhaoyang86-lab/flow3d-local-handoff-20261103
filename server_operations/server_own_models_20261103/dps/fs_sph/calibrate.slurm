#!/bin/bash
#SBATCH --job-name=own3_fs_sph_dps_calibration
#SBATCH --partition=ghx4
#SBATCH --account=biup-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dps/fs_sph/reports/calibrate_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
exec /sw/user/python/miniforge3-pytorch-2.10.0/bin/python /u/zguan2/projects/flow3d_runs/operations/server_own_models_20261103/dps/fs_sph/runtime.py calibrate
