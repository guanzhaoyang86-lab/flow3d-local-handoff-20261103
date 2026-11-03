#!/bin/bash -l
#SBATCH --job-name=flow3d_dps_sample
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --exclude=gh131
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=flow3d_dps_sample_%j.out
#SBATCH --error=flow3d_dps_sample_%j.err
set -euo pipefail
: "${FLOW3D_DPS_ROOT:?Set the isolated flow3d-dps source directory}"
: "${FLOW3D_DPS_MANIFEST:?Set the original split manifest path}"
: "${FLOW3D_DPS_CHECKPOINT:?Set an unconditional prior checkpoint}"
: "${FLOW3D_DPS_OUTPUT_FILE:?Set a new posterior NPZ path}"
: "${FLOW3D_DPS_SIGMA:?Set the displacement-residual scale selected on validation data}"
FLOW3D_DPS_PYTHON=${FLOW3D_DPS_PYTHON:-/sw/user/python/miniforge3-pytorch-2.10.0/bin/python}
if [[ -e "$FLOW3D_DPS_OUTPUT_FILE" ]]; then
    echo 'Posterior output already exists; choose a fresh output file.' >&2
    exit 2
fi
FLOW3D_DPS_EXTRA=()
if [[ -n ${FLOW3D_DPS_VIEWS:-} ]]; then
    FLOW3D_DPS_EXTRA+=(--views "$FLOW3D_DPS_VIEWS")
fi
if [[ ${FLOW3D_DPS_EVALUATE:-0} == 1 ]]; then
    FLOW3D_DPS_EXTRA+=(--evaluate)
fi
if [[ ${FLOW3D_DPS_REQUIRE_VALID_REPLAY:-0} == 1 ]]; then
    FLOW3D_DPS_EXTRA+=(--require-valid-replay)
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
srun --ntasks=1 "$FLOW3D_DPS_PYTHON" "$FLOW3D_DPS_ROOT/scripts/sample_flow_dps.py" \
    --checkpoint "$FLOW3D_DPS_CHECKPOINT" --manifest "$FLOW3D_DPS_MANIFEST" \
    --output "$FLOW3D_DPS_OUTPUT_FILE" --sigma "$FLOW3D_DPS_SIGMA" \
    --split "${FLOW3D_DPS_SPLIT:-validation}" --case-index "${FLOW3D_DPS_CASE_INDEX:-0}" \
    --num-particles "${FLOW3D_DPS_N:-2}" --num-probes "${FLOW3D_DPS_NUM_PROBES:-16}" \
    --num-samples "${FLOW3D_DPS_NUM_SAMPLES:-2}" --seed "${FLOW3D_DPS_SEED:-47}" \
    --observation-type "${FLOW3D_DPS_OBSERVATION_TYPE:-projected}" \
    --boundary-projection "${FLOW3D_DPS_BOUNDARY_PROJECTION:-none}" \
    --sampler "${FLOW3D_DPS_SAMPLER:-ddpm}" --device cuda "${FLOW3D_DPS_EXTRA[@]}"
"$FLOW3D_DPS_PYTHON" "$FLOW3D_DPS_ROOT/scripts/verify_artifact.py" \
    --posterior "$FLOW3D_DPS_OUTPUT_FILE" --num-particles "${FLOW3D_DPS_N:-2}" \
    --num-samples "${FLOW3D_DPS_NUM_SAMPLES:-2}"
