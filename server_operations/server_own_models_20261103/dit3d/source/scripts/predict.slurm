#!/bin/bash -l
#SBATCH --job-name=flow3d_base_eval
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --exclude=gh131
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=flow3d_base_eval_%j.out
#SBATCH --error=flow3d_base_eval_%j.err
set -euo pipefail
: "${FLOW3D_BASELINE_ROOT:?Set baseline source directory}"
: "${FLOW3D_BASELINE_MANIFEST:?Set split manifest}"
: "${FLOW3D_BASELINE_OUTPUT_FILE:?Set a fresh NPZ output path}"
FLOW3D_BASELINE_PYTHON=${FLOW3D_BASELINE_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}
FLOW3D_BASELINE_EXTRA=()
if [[ -n ${FLOW3D_BASELINE_CHECKPOINT:-} ]]; then
  FLOW3D_BASELINE_EXTRA+=(--checkpoint "$FLOW3D_BASELINE_CHECKPOINT")
else
  : "${FLOW3D_BASELINE_MODEL:?Set pinn or rbf when no checkpoint is supplied}"
  FLOW3D_BASELINE_EXTRA+=(--model "$FLOW3D_BASELINE_MODEL")
fi
if [[ -n ${FLOW3D_BASELINE_VISCOSITY:-} ]]; then
  FLOW3D_BASELINE_EXTRA+=(--viscosity "$FLOW3D_BASELINE_VISCOSITY")
fi
if [[ -n ${FLOW3D_BASELINE_CONFIG:-} ]]; then
  FLOW3D_BASELINE_EXTRA+=(--config "$FLOW3D_BASELINE_CONFIG")
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
srun --ntasks=1 "$FLOW3D_BASELINE_PYTHON" "$FLOW3D_BASELINE_ROOT/scripts/run_baseline.py" predict \
  --manifest "$FLOW3D_BASELINE_MANIFEST" --output "$FLOW3D_BASELINE_OUTPUT_FILE" \
  --num-particles "${FLOW3D_BASELINE_N:-2}" --seed "${FLOW3D_BASELINE_SEED:-31}" \
  --case-index "${FLOW3D_BASELINE_CASE_INDEX:-0}" --split "${FLOW3D_BASELINE_SPLIT:-validation}" \
  --device cuda --evaluate "${FLOW3D_BASELINE_EXTRA[@]}"
"$FLOW3D_BASELINE_PYTHON" "$FLOW3D_BASELINE_ROOT/scripts/run_baseline.py" verify "$FLOW3D_BASELINE_OUTPUT_FILE"
