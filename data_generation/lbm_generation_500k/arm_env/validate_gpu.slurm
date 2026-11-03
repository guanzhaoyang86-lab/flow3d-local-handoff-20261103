#!/bin/bash -l
#SBATCH --job-name=lbm-arm-validate
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${LBM500K_ARM_DIR:?}"
: "${LBM500K_OUTPUT_DIR:?}"
source "$LBM500K_ARM_DIR/arm_runtime.sh"
cd -- "$LBM500K_ARM_ROOT"
srun --ntasks=1 --gpu-bind=none python "$LBM500K_ARM_ROOT/validate_gpu.py"
lbm500k_resume=()
[[ ! -f "$LBM500K_OUTPUT_DIR/generation.json" ]] || lbm500k_resume=(--resume)
exec srun --ntasks=1 --gpu-bind=none python -B "$LBM500K_PACKAGE_ROOT/run_generation.py" \
    --source-root "$LBM500K_PACKAGE_ROOT/source" \
    --upstream-repo "$LBM500K_PACKAGE_ROOT/upstream/taichi_LBM3D" \
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 --chunk-index 0 \
    --runtime-profile deltaai-arm-llvm20 --environment-manifest "$LBM500K_ARM_ROOT/validated_env.json" \
    "${lbm500k_resume[@]}"
