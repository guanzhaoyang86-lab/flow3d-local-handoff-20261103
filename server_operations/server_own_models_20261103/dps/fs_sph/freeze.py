"""Freeze the independent transient DPS implementation after CPU checks."""
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
package = here.parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
config = json.loads((here / 'config.json').read_text())
if sha(Path(config['manifest'])) != config['manifest_sha256']:
    raise ValueError('FS dataset manifest changed')
files = list((package / 'fs_source').rglob('*.py'))
files += list(here.glob('*.py')) + list(here.glob('*.slurm')) + list(here.glob('*.sbatch')) + [here / 'config.json', here / 'README.md']
files = sorted(set(path for path in files if '__pycache__' not in path.parts))
lock = {'schema': 'fs_dps_source_lock_v1', 'dataset': 'fs_sph',
        'manifest_sha256': config['manifest_sha256'],
        'files': [{'path': str(path.relative_to(package)), 'sha256': sha(path)} for path in files]}
(here / 'frozen_manifest.json').write_text(json.dumps(lock, indent=2) + '\n')
print(sha(here / 'frozen_manifest.json'))
