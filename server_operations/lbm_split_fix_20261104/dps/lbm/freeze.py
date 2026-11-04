from pathlib import Path
import hashlib
import json
import re

here = Path(__file__).resolve().parent
package = here.parent
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = sorted((package / 'lbm_source').rglob('*.py'))
files += list(here.glob('*.py')) + list(here.glob('*.sbatch')) + [here / 'config.json', here / 'README.txt']
files = sorted(set(files))
config = json.loads((here / 'config.json').read_text())
for name in ('expected_adapter_sha256', 'expected_raw_manifest_sha256', 'expected_raw_complete_sha256', 'expected_raw_dataset_version'):
    if re.fullmatch('[0-9a-f]{64}', config.get(name, '')) is None:
        raise ValueError('Missing reviewed fingerprint: ' + name)
manifest_sha = None if config.get('deferred_dataset_adoption') else digest(Path(config['manifest']))
lock = {'schema': 'dps_source_lock_v1', 'dataset': config['dataset'], 'manifest_sha256': manifest_sha,
        'files': {str(p.relative_to(package)): digest(p) for p in files}}
(here / 'frozen_manifest.json').write_text(json.dumps(lock, indent=2) + '\n')
print(digest(here / 'frozen_manifest.json'))
