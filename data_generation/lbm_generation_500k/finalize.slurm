#!/bin/bash -l
set -euo pipefail
source "${LBM500K_PACKAGE_ROOT:?}/job_env.sh"
exec srun --ntasks=1 --gpu-bind=none "$LBM500K_PYTHON" -B \
    "$LBM500K_PACKAGE_ROOT/run_generation.py" \
    --source-root "$LBM500K_PACKAGE_ROOT/source" \
    --upstream-repo "$LBM500K_PACKAGE_ROOT/upstream/taichi_LBM3D" \
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 --resume --finalize
