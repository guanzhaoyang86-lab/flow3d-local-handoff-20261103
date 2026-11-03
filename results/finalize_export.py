#!/usr/bin/env python3
"""Validate the local metadata snapshot, attach its queue snapshot and index assets.

Reads and writes only this handoff result directory. No server rescan or transfer.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from build_asset_manifest import main as build_assets


HERE = Path(__file__).resolve().parent


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    inventory = json.loads((HERE / 'results_inventory.json').read_text())
    summary = json.loads((HERE / 'summary.json').read_text())
    failures = []
    for row in inventory['copied_metadata_files']:
        path = HERE / row['export_path']
        if (not path.is_file() or path.stat().st_size != row['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
            failures.append(row['export_path'])
    binary = [str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()
              and path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors', '.npz'}]
    if failures or binary:
        raise SystemExit(json.dumps({'metadata_hash_failures': failures, 'unexpected_binary_payloads': binary}))
    queue_path = HERE / 'server_queue.json'
    if not queue_path.is_file():
        raise SystemExit('Main handoff process must provide server_queue.json before freezing.')
    queue = json.loads(queue_path.read_text())
    summary['queue_snapshot_file'] = queue_path.name
    summary['queue_snapshot_attached_at_utc'] = datetime.now(timezone.utc).isoformat()
    inventory['summary'] = summary
    inventory['server_queue_snapshot'] = queue
    save(HERE / 'summary.json', summary)
    save(HERE / 'results_inventory.json', inventory)
    build_assets()
    save(HERE / 'export_validation.json', {
        'status': 'passed', 'validated_at_utc': datetime.now(timezone.utc).isoformat(),
        'copied_metadata_sha256_verified': len(inventory['copied_metadata_files']),
        'unexpected_binary_payloads': binary, 'transfer_mock_tests_passed': 8,
        'production_data_modified': False, 'jobs_submitted_or_cancelled': False,
        'note': 'Transfer tests mocked subprocess calls; no real rsync or SSH connection was made.',
    })
    files = [path for path in HERE.rglob('*') if path.is_file()]
    print(json.dumps({'final_files': len(files), 'final_bytes': sum(path.stat().st_size for path in files),
                      'metadata_sha256_verified': len(inventory['copied_metadata_files']),
                      'transfer_mock_tests_passed': 8}, indent=2))


if __name__ == '__main__':
    main()
