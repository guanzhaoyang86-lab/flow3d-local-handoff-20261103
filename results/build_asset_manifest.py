#!/usr/bin/env python3
"""Build a download index from exported records; do not transfer any assets."""
from datetime import datetime, timezone
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    inventory = json.loads((HERE / 'results_inventory.json').read_text())
    assets = []
    dataset_paths = {
        'fs_sph': '/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027',
        'jhtdb': '/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32',
        'lbm': '/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023',
    }
    for name, source in dataset_paths.items():
        completion = inventory.get('datasets', {}).get(name, {}).get('adapter_completion')
        assets.append({
            'id': 'data-' + name, 'source_path': source,
            'destination': 'data/' + name, 'kind': 'directory', 'size_bytes': None,
            'description': 'Prepared dataset directory; size not scanned.',
            'completion_marker_present_at_inventory': completion.get('status') == 'completed' if completion else None,
            'note': ('LBM generation continues on the server; a download before COMPLETE.json is a partial snapshot.'
                     if name == 'lbm' else 'Includes adapted fields/windows and their metadata; no raw simulation rerun is needed.'),
        })
    for row in inventory['current_baseline_runs']:
        if row['status'] != 'completed':
            continue
        weight = next((item for item in row.get('checkpoints', []) if item['filename'] == 'latest.pt'), None)
        if weight is None:
            continue
        destination = Path('results') / Path(row['exported_run_record']).parent / 'latest.pt'
        run_name = Path(row['run_directory']).name
        assets.append({
            'id': 'checkpoint-' + row['dataset'] + '-' + run_name,
            'source_path': weight['source_path'], 'destination': destination.as_posix(),
            'kind': 'file', 'size_bytes': weight['bytes'],
            'recorded_sha256': weight.get('recorded_sha256'),
            'description': 'Completed baseline latest.pt; includes training state, not an inference-only stripped weight.',
            'current_single_seed_policy': row['current_single_seed_policy'],
            'dataset_asset': 'data-' + row['dataset'], 'model': row['model'], 'seed': row['seed'],
        })
    for row in inventory['legacy_conditional_runs']:
        if not row['status'].startswith('completed'):
            continue
        weight = next((item for item in row.get('checkpoints', []) if item['filename'] == 'latest.pt'), None)
        if weight is None:
            continue
        if row['status'] == 'completed_verified':
            relative = Path('legacy_conditional/completed_training') / row['key'] / 'training/latest.pt'
        else:
            relative = Path('legacy_conditional/reused_N2_training') / row['key'] / 'latest.pt'
        assets.append({
            'id': 'checkpoint-legacy-' + row['key'], 'source_path': weight['source_path'],
            'destination': (Path('results') / relative).as_posix(), 'kind': 'file',
            'size_bytes': weight['bytes'], 'recorded_sha256': weight.get('recorded_sha256'),
            'description': 'Completed historical conditional model on the old LBM dataset.',
            'model': row['model'], 'seed': row['seed'], 'N': row['N'],
            'dataset_note': 'Old LBM data/protocol, distinct from the new data/lbm 500k collection.',
        })
    payload = {
        'schema': 'flow3d_external_assets_v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'destination_root': 'The handoff_20261103 directory containing models/, experiments/, results/ and local_support/.',
        'weights_policy': 'One recommended latest.pt per completed run; weights are indexed, not stored in Git.',
        'directory_size_policy': 'Null means not measured; no estimate was substituted for a disk scan.',
        'assets': assets,
    }
    target = HERE / 'asset_manifest.json'
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(target)
    weights = [item for item in assets if item['kind'] == 'file']
    print(json.dumps({'dataset_assets': len(assets) - len(weights), 'recommended_weights': len(weights),
                      'recommended_weight_bytes': sum(item['size_bytes'] for item in weights)}, indent=2))


if __name__ == '__main__':
    main()
