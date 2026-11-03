from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
package = here.parent
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = sorted((package / 'lbm_source').rglob('*.py'))
files += [here / name for name in ('config.json', 'runtime.py', 'freeze.py', 'train.sbatch', 'calibrate.sbatch', 'test.sbatch')]
config = json.loads((here / 'config.json').read_text())
manifest_sha = None if config.get('deferred_dataset_adoption') else digest(Path(config['manifest']))
lock = {'schema': 'dps_source_lock_v1', 'dataset': config['dataset'], 'manifest_sha256': manifest_sha,
        'files': {str(p.relative_to(package)): digest(p) for p in files}}
(here / 'frozen_manifest.json').write_text(json.dumps(lock, indent=2) + '\n')
print(digest(here / 'frozen_manifest.json'))
