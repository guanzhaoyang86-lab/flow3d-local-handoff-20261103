#!/bin/bash -l
#SBATCH --job-name=flow3d_dps_prior
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --exclude=gh131
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --no-requeue
#SBATCH --output=flow3d_dps_prior_%A_%a.out
#SBATCH --error=flow3d_dps_prior_%A_%a.err
set -euo pipefail
: "${FLOW3D_DPS_ROOT:?Set this to the isolated flow3d-dps source directory}"
: "${FLOW3D_DPS_MANIFEST:?Set the original train/validation/test manifest path}"
: "${FLOW3D_DPS_OUTPUT:?Set a new output root for the unconditional priors}"
FLOW3D_DPS_PYTHON=${FLOW3D_DPS_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}
FLOW3D_DPS_SEED=$((31 + SLURM_ARRAY_TASK_ID))
if [[ -e "$FLOW3D_DPS_OUTPUT/prior_s$FLOW3D_DPS_SEED/latest.pt" ]]; then
    echo 'Prior output already exists; choose a fresh batch or resume explicitly.' >&2
    exit 2
fi
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
srun --ntasks=1 "$FLOW3D_DPS_PYTHON" "$FLOW3D_DPS_ROOT/scripts/train_flow_prior.py" \
    --manifest "$FLOW3D_DPS_MANIFEST" \
    --output-dir "$FLOW3D_DPS_OUTPUT/prior_s$FLOW3D_DPS_SEED" \
    --seed "$FLOW3D_DPS_SEED" --architecture dit3d --epochs 100 --device cuda --amp
"$FLOW3D_DPS_PYTHON" "$FLOW3D_DPS_ROOT/scripts/verify_artifact.py" \
    --prior-dir "$FLOW3D_DPS_OUTPUT/prior_s$FLOW3D_DPS_SEED" --epochs 100
