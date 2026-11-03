#!/bin/bash -l
set -euo pipefail
source "${LBM500K_PACKAGE_ROOT:?}/job_env.sh"
lbm500k_chunk="${SLURM_ARRAY_TASK_ID:-0}"
lbm500k_resume=()
if [[ -f "$LBM500K_OUTPUT_DIR/generation.json" ]]; then
    lbm500k_resume=(--resume)
elif [[ "$lbm500k_chunk" != 0 ]]; then
    printf 'The initial chunk must initialize the dataset first.\n' >&2
    exit 1
fi
exec srun --ntasks=1 --gpu-bind=none "$LBM500K_PYTHON" -B \
    "$LBM500K_PACKAGE_ROOT/run_generation.py" \
    --source-root "$LBM500K_PACKAGE_ROOT/source" \
    --upstream-repo "$LBM500K_PACKAGE_ROOT/upstream/taichi_LBM3D" \
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 \
    --chunk-index "$lbm500k_chunk" "${lbm500k_resume[@]}"
