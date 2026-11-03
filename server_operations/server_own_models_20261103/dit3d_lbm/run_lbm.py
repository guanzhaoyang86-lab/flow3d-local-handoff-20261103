#!/usr/bin/env python3
"""Explicit LBM binding for the immutable full-volume DiT3D allocation controller."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / 'dit3d'
MODEL_SHA = '400e72abd126bbd48555e3fbad55967191795a036c3ac9bf19c615dabe101084'
DATA = Path('/work/hdd/biup/zguan2/datasets/experiment_adapters_20261103/lbm_frozen32')
RAW = Path('/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023')
OUTPUT = Path('/work/hdd/biup/zguan2/results/server_own_models_20261103/dit3d/lbm_s31')
COUNTS = {'train': 2500, 'validation': 312, 'test': 313}
DATASET_ID = 'lbm32_xz_500k_20261103'

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

def source_check(require_submission=False):
    manifest = ROOT / 'frozen_manifest.json'
    expected = os.environ.get('DIT_LBM_FROZEN_SHA256')
    require(not require_submission or bool(expected) and sha(manifest) == expected,
            'Submission-time LBM wrapper SHA missing or changed')
    local_files = read(manifest)['files']
    for name, digest in local_files.items():
        require(sha(ROOT / name) == digest, 'LBM wrapper input changed: ' + name)
    model_manifest = MODEL_ROOT / 'frozen_manifest.json'
    require(sha(model_manifest) == MODEL_SHA, 'Reviewed DiT3D source manifest changed')
    model_files = read(model_manifest)['files']
    for name, digest in model_files.items():
        require(sha(MODEL_ROOT / name) == digest, 'Frozen DiT3D input changed: ' + name)
    return {'wrapper_manifest_sha256': sha(manifest), 'wrapper_files': len(local_files),
            'model_manifest_sha256': MODEL_SHA, 'model_files': len(model_files)}

def data_check(dataset='lbm'):
    require(dataset == 'lbm', 'This controller only accepts LBM')
    marker = read(DATA / 'COMPLETE.json')
    manifest_path = DATA / 'manifest.json'
    manifest = read(manifest_path)
    require(marker.get('status') == 'completed' and marker.get('dataset_id') == DATASET_ID,
            'LBM adapter is not complete or dataset identity differs')
    require(manifest.get('schema') == 'flow3d_lbm_frozen_manifest_v1'
            and manifest.get('dataset_id') == DATASET_ID, 'Explicit LBM manifest required')
    require(marker.get('arrays_modified') is False and marker.get('source_data_mutated') is False
            and marker.get('training_or_rank_selection_performed') is False
            and marker.get('verified_fields') == sum(COUNTS.values()), 'LBM adapter validation is incomplete')
    manifest_digest = sha(manifest_path)
    require(marker.get('manifest_sha256') == manifest_digest, 'Adapted manifest checksum mismatch')
    raw_marker = read(RAW / 'COMPLETE.json')
    require(raw_marker.get('status') == 'completed' and raw_marker.get('num_cases') == sum(COUNTS.values())
            and raw_marker.get('num_independent_flow_groups') == sum(COUNTS.values())
            and raw_marker.get('physical_trajectories') == 500000
            and raw_marker.get('single_view_trajectories') == 500000, 'Raw 500k LBM release is incomplete')
    version = sha(RAW / 'SHA256SUMS')
    require(raw_marker.get('dataset_version') == version == marker.get('source_dataset_version')
            == manifest.get('source_dataset_version'), 'LBM source dataset versions differ')
    require(marker.get('source_complete_sha256') == sha(RAW / 'COMPLETE.json'),
            'Raw COMPLETE marker differs from adapter provenance')
    require(manifest.get('mode') == 'scientific' and manifest.get('split_unit') == 'flow_group_id'
            and manifest.get('allow_overlap_for_smoke_test') is False, 'Scientific disjoint LBM split required')
    require(manifest.get('velocity_units') == 'lattice cells per LBM step'
            and manifest.get('time_units') == 'LBM steps' and manifest.get('unit_scale') == [1., 1., 1.],
            'Native LBM lattice units must be preserved')
    protocol = manifest.get('observation_protocol', {})
    require(all(protocol.get(key) == value for key, value in
                {'N': 12, 'probe_count': 32, 'observation_seed': 47, 'view': 'xz', 'training_seed': 31}.items()),
            'LBM adapter observation protocol differs')
    counts = {name: len(manifest.get('splits', {}).get(name, [])) for name in COUNTS}
    require(counts == COUNTS == marker.get('split_counts') == raw_marker.get('split_counts'),
            'LBM split counts differ')
    expected, ids, groups = set(), set(), set()
    for split in COUNTS:
        for row in manifest['splits'][split]:
            path = PurePosixPath(row['path'])
            require(path.parent == PurePosixPath('fields') and path.suffix == '.npz',
                    'LBM field path must stay in fields/')
            require(path.name not in expected and row['case_id'] not in ids
                    and row['flow_group_id'] not in groups, 'LBM case/flow group overlaps')
            require(row['case_id'] == path.stem == row.get('source_case_id'), 'LBM source case identity differs')
            require(isinstance(row.get('sha256'), str) and len(row['sha256']) == 64,
                    'LBM field checksum absent from verified adapter')
            expected.add(path.name); ids.add(row['case_id']); groups.add(row['flow_group_id'])
    available = {entry.name for entry in os.scandir(DATA / 'fields') if entry.is_file()}
    require(not expected - available, 'LBM adapted field files are missing')
    return {'dataset': 'lbm', 'dataset_id': DATASET_ID, 'manifest': str(manifest_path),
            'manifest_sha256': manifest_digest, 'adapter_complete_sha256': sha(DATA / 'COMPLETE.json'),
            'source_dataset_version': version, 'split_counts': counts,
            'velocity_units': manifest['velocity_units'], 'time_units': manifest['time_units'],
            'scientific_scope': manifest['scientific_scope']}

def command(dataset, manifest, output, epochs, resume=False):
    require(dataset == 'lbm', 'This controller only accepts LBM')
    cmd = [sys.executable, '-B', str(MODEL_ROOT / 'source/scripts/run_baseline.py'),
           'train', '--manifest', str(manifest), '--model', 'dit3d',
           '--config', str(ROOT / 'configs/lbm_s31.json'), '--num-particles', '12', '--seed', '31',
           '--epochs', str(epochs), '--output-dir', str(output), '--device', 'cuda']
    if resume:
        cmd += ['--resume', str(output / 'latest.pt')]
    return cmd

def allocation_controller():
    # Source is verified before importing it. Changes below affect this module instance
    # only: no frozen file is rewritten. Static training entry is explicitly overridden,
    # because the original controller routes any non-JHTDB name to its FS temporal entry.
    spec = importlib.util.spec_from_file_location('immutable_dit_allocation', MODEL_ROOT / 'launch/run_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.RESULT_ROOT = OUTPUT.parent
    module.source_check = source_check
    module.data_check = data_check
    module.command = command
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    source = source_check(require_submission=not args.preflight_only)
    if args.preflight_only:
        if not (DATA / 'COMPLETE.json').exists():
            print(json.dumps({'status': 'waiting_for_complete_lbm_adapter', 'source': source,
                              'data_root': str(DATA), 'training_started': False}, indent=2))
        else:
            print(json.dumps({'status': 'source_and_data_preflight_passed', 'source': source,
                              'data': data_check(), 'gpu_preflight': 'deferred to Slurm allocation'}, indent=2))
    else:
        allocation_controller().run('lbm')

if __name__ == '__main__':
    main()
