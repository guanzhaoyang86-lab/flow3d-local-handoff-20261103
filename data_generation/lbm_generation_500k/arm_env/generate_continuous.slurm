#!/bin/bash -l
# Resume all unfinished chunks and finalize within one GPU allocation.
#SBATCH --job-name=lbm500k-arm-continuous
#SBATCH --account=biup-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/u/zguan2/projects/flow3d_runs/lbm_generation_500k/arm_env
#SBATCH --output=/u/zguan2/projects/flow3d_runs/lbm_generation_500k/arm_env/logs/%x_%j.out
#SBATCH --error=/u/zguan2/projects/flow3d_runs/lbm_generation_500k/arm_env/logs/%x_%j.err
set -euo pipefail
export LBM500K_ARM_DIR=/u/zguan2/projects/flow3d_runs/lbm_generation_500k/arm_env
export LBM500K_OUTPUT_DIR=/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023
source "$LBM500K_ARM_DIR/arm_runtime.sh"

lbm500k_refresh() {
    python -B "$LBM500K_PACKAGE_ROOT/refresh_progress.py" --write >/dev/null ||
        printf 'Progress refresh failed; production chunk records remain authoritative.\n' >&2
}
trap lbm500k_refresh EXIT

# A completed chunk has already validated all its cases. Preserve its original
# record; the finalizer below rechecks every archive, including skipped chunks.
lbm500k_pending="$(python -B - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ['LBM500K_PACKAGE_ROOT'])
from refresh_progress import validated_chunk_counts
output = Path(os.environ['LBM500K_OUTPUT_DIR'])
if (output / 'COMPLETE.json').exists():
    raise SystemExit('Dataset already has a completion marker; inspect before resubmitting.')
_, records, errors = validated_chunk_counts(output)
if errors:
    raise SystemExit('\n'.join(errors))
done = {record['chunk_index'] for record in records if record['status'] == 'completed'}
print(' '.join(str(index) for index in range(25) if index not in done))
PY
)"
lbm500k_args=("$LBM500K_PACKAGE_ROOT/run_generation.py"
    --source-root "$LBM500K_PACKAGE_ROOT/source"
    --upstream-repo "$LBM500K_PACKAGE_ROOT/upstream/taichi_LBM3D"
    --output-dir "$LBM500K_OUTPUT_DIR" --num-chunks 25 --resume
    --runtime-profile deltaai-arm-llvm20
    --environment-manifest "$LBM500K_ARM_ROOT/validated_env.json")

lbm500k_refresh
for lbm500k_chunk in $lbm500k_pending; do
    printf 'Starting chunk %s in continuous allocation %s\n' "$lbm500k_chunk" "$SLURM_JOB_ID"
    srun --ntasks=1 --gpu-bind=none python -B "${lbm500k_args[@]}" --chunk-index "$lbm500k_chunk"
    lbm500k_refresh
done
printf 'All chunks generated; validating the complete dataset.\n'
srun --ntasks=1 --gpu-bind=none python -B "${lbm500k_args[@]}" --finalize
