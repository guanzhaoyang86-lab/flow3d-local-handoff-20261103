#!/bin/bash -l
#SBATCH --job-name=lbm-arm-finalize
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
: "${LBM500K_ARM_DIR:?}"
: "${LBM500K_OUTPUT_DIR:?}"
source "$LBM500K_ARM_DIR/arm_runtime.sh"
exec srun --ntasks=1 --gpu-bind=none python -B "$LBM500K_PACKAGE_ROOT/run_generation.py" \
    --source-root "$LBM500K_PACKAGE_ROOT/source" \
    --upstream-repo "$LBM500K_PACKAGE_ROOT/upstream/taichi_LBM3D" \
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 --resume --finalize \
    --runtime-profile deltaai-arm-llvm20 --environment-manifest "$LBM500K_ARM_ROOT/validated_env.json"
