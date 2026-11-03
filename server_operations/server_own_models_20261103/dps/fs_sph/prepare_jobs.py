from pathlib import Path
here = Path(__file__).resolve().parent
(here / 'reports').mkdir(exist_ok=True)
template = '''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=ghx4
#SBATCH --account=biup-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={here}/reports/{stage}_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
exec /sw/user/python/miniforge3-pytorch-2.10.0/bin/python {here}/runtime.py {stage}
'''
for stage, suffix, hours in [('train', '', 8), ('calibrate', '_calibration', 8), ('test', '_evaluate', 16)]:
    (here / (stage + '.slurm')).write_text(template.format(
        name='own3_fs_sph_dps' + suffix, hours=hours, here=here, stage=stage))
print('Prepared FS train, calibration and heldout test Slurm stages')
