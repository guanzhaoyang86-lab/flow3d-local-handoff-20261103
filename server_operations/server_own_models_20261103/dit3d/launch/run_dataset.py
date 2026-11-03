#!/usr/bin/env python3
"""Run one verified DiT3D dataset allocation: epoch 1 gate, then resume to 100."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    'jhtdb': Path('/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32'),
    'fs_sph': Path('/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027'),
}
COUNTS = {'jhtdb': (2500, 312, 313), 'fs_sph': (126, 18, 18)}
RESULT_ROOT = Path('/work/hdd/biup/zguan2/results/server_own_models_20261103/dit3d')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_check(require_submission=False):
    path = ROOT/'frozen_manifest.json'
    expected = os.environ.get('DIT_FROZEN_SHA256')
    if require_submission and (not expected or sha(path) != expected):
        raise ValueError('Submission-time DiT source manifest SHA missing or changed')
    frozen = read(path)
    for name, digest in frozen['files'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen DiT input changed: ' + name)
    return {'frozen_manifest_sha256': sha(path), 'checked_files': len(frozen['files'])}


def data_check(dataset):
    root = DATA[dataset]
    manifest = root/'manifest.json'
    value = read(manifest)
    complete = read(root/'COMPLETE.json')
    if complete.get('status') not in ('completed', 'validated') or complete.get('manifest_sha256') != sha(manifest):
        raise ValueError('Adapted dataset completion/manifest does not verify')
    counts = tuple(len(value.get('splits', {}).get(split, [])) for split in ('train', 'validation', 'test'))
    if counts != COUNTS[dataset]:
        raise ValueError('Unexpected scientific split counts: ' + str(counts))
    frozen_data = read(ROOT/'data_bindings.json')[dataset]
    if frozen_data['manifest_sha256'] != sha(manifest):
        raise ValueError('Manifest changed since this operation was prepared')
    if dataset == 'fs_sph' and (value.get('schema') != 'fs_transient_baseline_manifest_v1' or value.get('mode') != 'within_case_temporal'):
        raise ValueError('FS requires the temporal-window adapter')
    if dataset == 'jhtdb' and value.get('mode') == 'smoke_test':
        raise ValueError('A scientific JHTDB manifest is required')
    # Only path availability is checked here. The frozen dataset readers validate
    # arrays on use; no test arrays are opened for training/preprocessing.
    directories = {}
    for split in ('train', 'validation', 'test'):
        for row in value['splits'][split]:
            path = root/row['path']
            directories.setdefault(path.parent, set()).add(path.name)
    for directory, expected in directories.items():
        available = {entry.name for entry in os.scandir(directory) if entry.is_file()}
        if expected - available:
            raise FileNotFoundError('Missing dataset archives: ' + str(sorted(expected - available)[:5]))
    return {'dataset': dataset, 'manifest': str(manifest), 'manifest_sha256': sha(manifest),
            'split_counts': dict(zip(('train', 'validation', 'test'), counts))}


def gpu_check():
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Training must run in a Slurm GPU allocation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one working visible CUDA GPU required')
    cap = torch.cuda.get_device_capability(0)
    arch = 'sm_' + ''.join(map(str, cap))
    if arch not in torch.cuda.get_arch_list():
        raise ValueError('PyTorch lacks this GPU architecture: ' + arch)
    x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    x.square().sum().backward()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(x.grad).all()):
        raise ValueError('CUDA finite-gradient check failed')
    return {'python': platform.python_version(), 'machine': platform.machine(),
            'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'capability': list(cap),
            'compiled_architectures': torch.cuda.get_arch_list()}


def command(dataset, manifest, output, epochs, resume=False):
    script = ROOT/('source/scripts/run_baseline.py' if dataset == 'jhtdb' else 'launch/fs_dit3d.py')
    cmd = [sys.executable, '-B', str(script), 'train', '--manifest', str(manifest),
           '--model', 'dit3d', '--config', str(ROOT/'configs'/f'{dataset}_s31.json'),
           '--num-particles', '12', '--seed', '31', '--epochs', str(epochs),
           '--output-dir', str(output), '--device', 'cuda']
    if resume:
        cmd += ['--resume', str(output/'latest.pt')]
    return cmd


def checkpoint_check(dataset, output, data, minimum_epoch=1):
    import torch
    state = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
    if state.get('model') != 'dit3d' or state.get('manifest_sha256') != data['manifest_sha256']:
        raise ValueError('Checkpoint model/dataset provenance mismatch')
    cfg = read(ROOT/'configs'/f'{dataset}_s31.json')
    if {k:v for k,v in state['config'].items() if k != 'epochs'} != {k:v for k,v in cfg.items() if k != 'epochs'}:
        raise ValueError('Checkpoint configuration differs from frozen protocol')
    epoch = state.get('epoch')
    if not isinstance(epoch, int) or not minimum_epoch <= epoch <= 100:
        raise ValueError('Checkpoint epoch outside valid continuation range')
    if state.get('optimizer_steps', 0) < 1 or not math.isfinite(state.get('best_validation_loss', float('nan'))):
        raise ValueError('Checkpoint has no verified finite learning progress')
    for group in ('system', 'ema_system'):
        if not all(bool(torch.isfinite(tensor).all()) for tensor in state[group].values()):
            raise ValueError('Nonfinite checkpoint state')
    status = read(output/'run.json')
    if epoch == 100 and status.get('status') != 'completed':
        raise ValueError('Expected completed epoch-stage marker')
    if dataset == 'fs_sph' and state.get('runtime_sha256') != sha(ROOT/'launch/fs_dit3d.py'):
        raise ValueError('FS temporal runtime changed')
    return {'epoch': epoch, 'optimizer_steps': state['optimizer_steps'],
            'best_validation_loss': state['best_validation_loss'],
            'checkpoint_sha256': sha(output/'latest.pt')}


def run(dataset):
    source = source_check(require_submission=True)
    runtime = gpu_check()
    data = data_check(dataset)
    output = RESULT_ROOT/(dataset+'_s31')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'model': 'dit3d', 'dataset': dataset, 'seed': 31,
                    'source': source, 'manifest_sha256': data['manifest_sha256'],
                    'method': 'full_volume_DiT3D_with_frozen_baseline_observation_protocol'}
        binding = output/'operation_binding.json'
        if binding.exists() and read(binding) != identity:
            raise ValueError('Output belongs to a different frozen operation')
        write(binding, identity)
        status = {'status': 'running', **identity, 'runtime': runtime,
                  'slurm_job_id': os.environ['SLURM_JOB_ID'], 'started_unix': time.time()}
        write(output/'allocation_status.json', status)
        try:
            if not (output/'latest.pt').exists():
                subprocess.run(command(dataset, data['manifest'], output, 1), check=True)
            checked = checkpoint_check(dataset, output, data)
            write(output/'PILOT_COMPLETE.json', {'status': 'completed', 'scope': 'at_least_one_completed_formal_epoch_verified_before_continuation_to_100',
                  'continued_without_repeating_epoch': True, 'runtime': runtime, 'checkpoint': checked})
            if checked['epoch'] < 100:
                subprocess.run(command(dataset, data['manifest'], output, 100, resume=True), check=True)
            final = checkpoint_check(dataset, output, data, minimum_epoch=100)
            status.update(status='completed', finished_unix=time.time(), final=final)
        except BaseException as error:
            status.update(status='failed', finished_unix=time.time(), error=str(error))
            write(output/'allocation_status.json', status)
            raise
        write(output/'allocation_status.json', status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=tuple(DATA), required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if args.preflight_only:
        print(json.dumps({'source': source_check(), 'data': data_check(args.dataset),
                          'gpu_preflight': 'deferred to Slurm allocation; no login-node training'}, indent=2))
    else:
        run(args.dataset)


if __name__ == '__main__':
    main()
