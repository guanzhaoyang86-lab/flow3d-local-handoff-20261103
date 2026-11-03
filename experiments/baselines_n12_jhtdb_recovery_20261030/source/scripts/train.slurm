#!/bin/bash -l
#SBATCH --job-name=flow3d_baseline
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --exclude=gh131
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=flow3d_baseline_%j.out
#SBATCH --error=flow3d_baseline_%j.err
set -euo pipefail
: "${FLOW3D_BASELINE_ROOT:?Set baseline source directory}"
: "${FLOW3D_BASELINE_MANIFEST:?Set split manifest}"
: "${FLOW3D_BASELINE_OUTPUT:?Set a fresh training directory}"
: "${FLOW3D_BASELINE_MODEL:?Set mean/pod/unet/fno/cnn/deeponet/unet_diffusion}"
FLOW3D_BASELINE_PYTHON=${FLOW3D_BASELINE_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
srun --ntasks=1 "$FLOW3D_BASELINE_PYTHON" "$FLOW3D_BASELINE_ROOT/scripts/run_baseline.py" train \
  --manifest "$FLOW3D_BASELINE_MANIFEST" --model "$FLOW3D_BASELINE_MODEL" \
  --config "${FLOW3D_BASELINE_CONFIG:-$FLOW3D_BASELINE_ROOT/configs/defaults.json}" \
  --num-particles "${FLOW3D_BASELINE_N:-2}" --seed "${FLOW3D_BASELINE_SEED:-31}" \
  --output-dir "$FLOW3D_BASELINE_OUTPUT" --device cuda
"$FLOW3D_BASELINE_PYTHON" "$FLOW3D_BASELINE_ROOT/scripts/run_baseline.py" verify "$FLOW3D_BASELINE_OUTPUT"
