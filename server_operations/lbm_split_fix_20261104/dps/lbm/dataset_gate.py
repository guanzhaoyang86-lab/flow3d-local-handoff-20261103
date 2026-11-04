"""Validate the published LBM split and adapter before every production stage."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re

COUNTS = {'train': 2499, 'validation': 313, 'test': 313}
TOTAL = 3125


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def split_rows(manifest, adapted=False):
    require(manifest.get('mode') == 'scientific' and manifest.get('split_unit') == 'flow_group_id'
            and manifest.get('allow_overlap_for_smoke_test') is False, 'Scientific flow-group split required')
    splits = manifest.get('splits', {})
    require(set(splits) == set(COUNTS), 'Exactly train/validation/test splits required')
    require({name: len(splits[name]) for name in COUNTS} == COUNTS, 'Published LBM split must be 2499/313/313')
    ids, groups, paths = set(), set(), set()
    for name in COUNTS:
        for row in splits[name]:
            case, group = row.get('case_id'), row.get('flow_group_id')
            require(isinstance(case, str) and bool(case) and isinstance(group, str) and bool(group), 'Case/group identity missing')
            path = PurePosixPath(row['path'])
            expected = PurePosixPath('fields') / (case + '.npz') if adapted else PurePosixPath(case + '.npz')
            require(path == expected, 'Case path must preserve the original case identity')
            require(case not in ids and group not in groups and str(path) not in paths, 'Case/path/flow-group overlap')
            ids.add(case); groups.add(group); paths.add(str(path))
    return splits


def verify_raw(config):
    require(config['dataset'] == 'lbm' and config['expected_split_counts'] == COUNTS,
            'Frozen DPS configuration must declare the actual LBM split')
    root = Path(config['raw_data_root'])
    marker_path, manifest_path = root / 'COMPLETE.json', root / 'manifest.json'
    require(sha(marker_path) == config['expected_raw_complete_sha256'], 'Raw completion certificate changed')
    require(sha(manifest_path) == config['expected_raw_manifest_sha256'], 'Raw scientific manifest changed')
    marker, manifest = read(marker_path), read(manifest_path)
    require(marker.get('status') == 'completed', 'Raw LBM collection is incomplete')
    require(marker.get('num_cases') == TOTAL and marker.get('num_independent_flow_groups') == TOTAL,
            'All3125 independent raw flows are required')
    require(marker.get('physical_trajectories') == 500000 and marker.get('single_view_trajectories') == 500000
            and marker.get('projection_names') == ['xz'], 'Complete500k single-XZ release required')
    require(marker.get('split_counts') == COUNTS, 'Raw completion split counts differ')
    require(Path(marker['manifest']).resolve() == manifest_path.resolve(), 'Raw marker points to another manifest')
    version = sha(root / 'SHA256SUMS')
    require(version == config['expected_raw_dataset_version'] == marker.get('dataset_version'), 'Raw dataset version changed')
    checksums = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if name.startswith('./'):
            name = name[2:]
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None and name not in checksums, 'Invalid raw checksum listing')
        checksums[name] = digest
    require(checksums.get('manifest.json') == sha(manifest_path), 'Raw checksum index does not certify this manifest')
    splits = split_rows(manifest)
    require(all(row['path'] in checksums for rows in splits.values() for row in rows), 'Raw cases lack checksums')
    return {'root': root, 'manifest': manifest, 'marker': marker, 'splits': splits, 'checksums': checksums,
            'summary': {'status': 'verified_complete_raw_release', 'split_counts': COUNTS,
                        'manifest_sha256': sha(manifest_path), 'complete_sha256': sha(marker_path),
                        'dataset_version': version, 'independent_flow_groups': TOTAL}}


def verify_adapter(config, raw=None):
    raw = verify_raw(config) if raw is None else raw
    marker_path, manifest_path = Path(config['completion_marker']), Path(config['manifest'])
    marker, manifest = read(marker_path), read(manifest_path)
    require(marker.get('status') == 'completed' and marker.get('dataset_id') == config['dataset_id'], 'LBM adapter is incomplete or misidentified')
    require(marker.get('manifest_sha256') == sha(manifest_path), 'Adapted manifest checksum mismatch')
    require(marker.get('adapter_sha256') == config['expected_adapter_sha256'], 'Reviewed repaired adapter code is required')
    require(marker.get('source_complete_sha256') == config['expected_raw_complete_sha256'], 'Adapter refers to another raw certificate')
    require(marker.get('source_dataset_version') == config['expected_raw_dataset_version']
            == manifest.get('source_dataset_version'), 'Adapter source version differs')
    require(marker.get('split_counts') == COUNTS, 'Adapted completion split counts differ')
    require(marker.get('verified_fields') == TOTAL and marker.get('physical_trajectories') == 500000,
            'Adapter has not verified the complete3125case/500k release')
    for key in ('arrays_modified', 'source_data_mutated', 'training_or_rank_selection_performed'):
        require(marker.get(key) is False, 'Adapter must preserve arrays/data/split: ' + key)
    require(manifest.get('schema') == 'flow3d_lbm_frozen_manifest_v1' and manifest.get('dataset_id') == config['dataset_id']
            and manifest.get('dataset_kind') == 'sparse_projected_particle_flow', 'Explicit LBM adapter manifest required')
    require(Path(manifest['source_dataset_root']).resolve() == raw['root'].resolve(), 'Adapter source directory differs')
    require(manifest.get('velocity_units') == 'lattice cells per LBM step' and manifest.get('time_units') == 'LBM steps'
            and manifest.get('unit_scale') == [1., 1., 1.], 'Native LBM lattice units must be preserved')
    protocol = manifest.get('observation_protocol', {})
    require(all(protocol.get(key) == value for key, value in {'N':12, 'probe_count':32, 'observation_seed':47,
            'view':'xz', 'training_seed':31}.items()), 'N12/XZ/seed47/seed31 protocol differs')
    splits = split_rows(manifest, adapted=True)
    fields = manifest_path.parent / 'fields'
    available = {path.name for path in fields.iterdir() if path.is_file()}
    for name in COUNTS:
        for original, row in zip(raw['splits'][name], splits[name]):
            require(row['case_id'] == original['case_id'] == row.get('source_case_id')
                    and row['flow_group_id'] == original['flow_group_id'], 'Adapter changed case order or flow-group membership')
            require(row.get('sha256') == raw['checksums'][original['path']], 'Adapter case checksum differs from raw release')
            require(PurePosixPath(row['path']).name in available, 'Adapted case file is missing')
    return {'status': 'verified_complete_adapter', 'split_counts': COUNTS,
            'manifest_sha256': sha(manifest_path), 'adapter_sha256': marker['adapter_sha256'],
            'source_dataset_version': marker['source_dataset_version'], 'verified_fields': TOTAL}
