#!/usr/bin/env python3
"""Publish a model-neutral LBM release only after complete generation checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023')
OUTPUT = Path('/work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32')
COUNTS = {'train': 2500, 'validation': 312, 'test': 313}
EXPECTED_CASES = 3125


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_operation(expected):
    require(sha(ROOT / 'operation_manifest.json') == expected, 'Adapter submission fingerprint changed')
    for name, digest in read(ROOT / 'operation_manifest.json')['files'].items():
        require(sha(ROOT / name) == digest, 'Adapter source changed: ' + name)
    plan = read(ROOT / 'plan.json')
    require(sha(SOURCE / 'generation.json') == plan['source_generation_sha256'], 'Raw generation configuration changed')
    return plan


def parse_checksums(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split('  ', 1)
        require(len(fields) == 2 and re.fullmatch('[a-f0-9]{64}', fields[0]), 'Malformed SHA256SUMS entry')
        digest, name = fields
        require(Path(name).name == name and name not in ('.', '..') and name not in entries, 'Unsafe/duplicate checksum filename')
        entries[name] = digest
    return entries


def source_ready(source=SOURCE):
    source = Path(source)
    marker = read(source / 'COMPLETE.json')
    require(marker.get('status') == 'completed', 'LBM final validation is incomplete')
    require(marker.get('num_cases') == EXPECTED_CASES and marker.get('num_independent_flow_groups') == EXPECTED_CASES, 'Exactly 3125 completed independent cases required')
    require(marker.get('physical_trajectories') == 500000 and marker.get('single_view_trajectories') == 500000, 'Exactly 500000 validated trajectories required')
    require(marker.get('projection_names') == ['xz'] and marker.get('split_counts') == COUNTS, 'Final LBM projection/split counts changed')
    require(marker.get('dataset_version') == sha(source / 'SHA256SUMS'), 'Final completion does not bind checksum list')
    checksums = parse_checksums(source / 'SHA256SUMS')
    expected_names = {f'case_{index:05d}.npz' for index in range(EXPECTED_CASES)}
    require({name for name in checksums if name.endswith('.npz')} == expected_names, 'Checksum list is not the full planned archive set')
    require({p.name for p in source.glob('*.npz')} == expected_names, 'Raw archive set is incomplete or unexpected')
    for name in ('manifest.json', 'collection.json', 'generation.json', 'plan.json'):
        require(name in checksums and sha(source / name) == checksums[name], 'Final metadata checksum mismatch: ' + name)
    collection = read(source / 'collection.json')
    require(collection.get('complete') is True and collection.get('num_cases') == EXPECTED_CASES, 'Collection has not completed')
    manifest = read(source / 'manifest.json')
    require(manifest.get('mode') == 'scientific' and manifest.get('split_unit') == 'flow_group_id', 'LBM requires a scientific flow-group split')
    require(manifest.get('allow_overlap_for_smoke_test') is False, 'Overlapping smoke splits cannot train a formal model')
    ids, paths, group_sets = set(), set(), {}
    for split, count in COUNTS.items():
        rows = manifest.get('splits', {}).get(split, [])
        require(len(rows) == count, 'Unexpected split count: ' + split)
        group_sets[split] = set()
        for row in rows:
            path = (source / row['path']).resolve()
            require(path.parent == source.resolve() and path.name in expected_names, 'Manifest archive path escapes source')
            require(row['case_id'] == path.stem and row['case_id'] not in ids and path.name not in paths, 'Duplicate/mismatched case identity')
            require(isinstance(row['flow_group_id'], str) and row['flow_group_id'], 'Missing independent flow identity')
            ids.add(row['case_id']); paths.add(path.name); group_sets[split].add(row['flow_group_id'])
    require(not (group_sets['train'] & group_sets['validation'] or group_sets['train'] & group_sets['test'] or group_sets['validation'] & group_sets['test']), 'LBM flow groups leak across splits')
    require(sum(map(len, group_sets.values())) == EXPECTED_CASES, 'LBM groups are not one per physical flow')
    return marker, manifest, checksums


def check_archive(path, row):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        require(metadata.get('case_id') == row['case_id'] and metadata.get('flow_group_id') == row['flow_group_id'], 'NPZ identity differs from manifest')
        require(metadata.get('field') == 'taichi_lbm3d_lid_driven_cavity' and metadata.get('frozen_snapshot_used_for_advection') is True, 'Archive is not a frozen cavity snapshot')
        require(metadata.get('velocity_units') == 'lattice cells per LBM step' and metadata.get('time_units') == 'LBM steps', 'LBM unit declarations changed')
        field, solid = archive['flow_field'], archive['solid_mask']
        require(field.shape == (1, 3, 32, 32, 32) and bool(np.isfinite(field).all()), 'Invalid finite 32-cubed field')
        require(solid.shape == (32, 32, 32) and solid.dtype == np.bool_ and bool(solid.any()) and not bool(solid[1:-1, 1:-1, 1:-1].any()), 'Expected public exterior cavity walls without internal obstacles')
        require(archive['trajectories_2d'].shape == (1, 1, 160, 20, 2) and archive['observation_mask'].shape == (1, 1, 160, 20), 'Expected 160 XZ trajectories at 20 times')
        require(archive['particle_counts'].reshape(-1).tolist() == [160], 'Particle count changed')
        require(np.array_equal(archive['projection_matrix'], np.array([[[1., 0., 0.], [0., 0., 1.]]])), 'Only XZ camera allowed')
        times = archive['observation_times']
        require(times.shape == (20,) and bool(np.isfinite(times).all()) and bool((np.diff(times) > 0).all()), 'Invalid observation times')
        require(bool(np.isfinite(archive['trajectories_2d'][archive['observation_mask']]).all()), 'Nonfinite visible tracks')


def prepare(source=SOURCE, output=OUTPUT):
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Complete adaptation must run in a real Slurm allocation')
    source, output = Path(source), Path(output)
    marker, raw_manifest, checksums = source_ready(source)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'COMPLETE.json').exists():
            require(read(output / 'COMPLETE.json').get('source_dataset_version') == marker['dataset_version'], 'An incompatible adapted release exists')
        fields = output / 'fields'; fields.mkdir(exist_ok=True)
        splits = {}; finished = 0
        for split in ('train', 'validation', 'test'):
            splits[split] = []
            for row in raw_manifest['splits'][split]:
                raw = source / row['path']; target = fields / raw.name; expected = checksums[raw.name]
                require(sha(raw) == expected, 'Raw archive checksum mismatch: ' + raw.name)
                check_archive(raw, row)
                if target.exists():
                    require(sha(target) == expected, 'Existing adapted archive differs; refusing overwrite: ' + str(target))
                else:
                    temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
                    shutil.copyfile(raw, temporary)
                    require(sha(temporary) == expected, 'Archive changed while copying: ' + raw.name)
                    temporary.replace(target)
                splits[split].append({**row, 'path': 'fields/' + raw.name, 'sha256': expected, 'source_case_id': row['case_id']})
                finished += 1
                if finished % 125 == 0:
                    print(json.dumps({'validated_and_copied_cases': finished, 'target': EXPECTED_CASES}), flush=True)
        require(sha(source / 'SHA256SUMS') == marker['dataset_version'], 'Raw checksums changed during adaptation')
        provenance = output / 'provenance'; provenance.mkdir(exist_ok=True)
        for name in ('COMPLETE.json', 'SHA256SUMS', 'manifest.json', 'collection.json', 'generation.json', 'plan.json'):
            shutil.copyfile(source / name, provenance / name)
        manifest = {**raw_manifest, 'schema': 'flow3d_lbm_frozen_manifest_v1', 'dataset_id': 'lbm32_xz_500k_20261103',
                    'scientific_scope': 'Frozen lid-driven cavity snapshots; not validated turbulence.',
                    'source_dataset_root': str(source), 'source_dataset_version': marker['dataset_version'],
                    'velocity_units': 'lattice cells per LBM step', 'time_units': 'LBM steps', 'unit_scale': [1., 1., 1.],
                    'unit_scale_semantics': 'Identity in native lattice units; no conversion to SI units is asserted.',
                    'geometry_policy': 'Preserve true public exterior solid_mask; per-case lid velocity is not an inference condition.',
                    'observation_protocol': {'N': 12, 'probe_count': 32, 'observation_seed': 47, 'view': 'xz', 'training_seed': 31,
                      'selection': 'Pinned sparse_dataset._stable_seed(47,0,split,case_id,0); draw source sample then randperm(160); last32 probes, first12 remaining candidates; training may reshuffle only remaining128 using epoch observations_only seed.'},
                    'splits': splits}
        write(output / 'manifest.json', manifest)
        result = {'status': 'completed', 'dataset_id': manifest['dataset_id'], 'verified_fields': finished, 'split_counts': COUNTS,
                  'physical_trajectories': 500000, 'single_view_trajectories': 500000,
                  'manifest_sha256': sha(output / 'manifest.json'), 'source_dataset_version': marker['dataset_version'],
                  'source_complete_sha256': sha(source / 'COMPLETE.json'), 'adapter_sha256': sha(__file__),
                  'slurm_job_id': os.environ['SLURM_JOB_ID'], 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'arrays_modified': False, 'source_data_mutated': False, 'training_or_rank_selection_performed': False}
        write(output / 'COMPLETE.json', result)
        print(json.dumps(result, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--operation-fingerprint')
    args = parser.parse_args()
    if args.execute:
        require(args.operation_fingerprint, 'Execution requires a submission-time fingerprint')
        verify_operation(args.operation_fingerprint)
        prepare()
    else:
        print(json.dumps({'status': 'waiting_for_final_release' if not (SOURCE / 'COMPLETE.json').exists() else 'final_marker_present_requires_full_validation',
                          'source': str(SOURCE), 'output': str(OUTPUT), 'required_dependency': 'afterok:3515292',
                          'expected_cases': EXPECTED_CASES, 'expected_trajectories': 500000, 'source_or_output_mutated': False}, indent=2))


if __name__ == '__main__':
    main()
