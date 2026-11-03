#!/usr/bin/env python3
"""Portable baseline listing, preflight, isolated smoke, and guarded local training.

Never submits/cancels Slurm jobs, changes frozen sources, or writes into imported results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    'fs_sph': 'baselines_n12_20261027',
    'jhtdb': 'baselines_n12_jhtdb_recovery_20261030',
}
MODELS = ('mean', 'pod', 'cnn', 'deeponet', 'unet', 'fno', 'unet_diffusion', 'rbf')
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'REQUEUED'}
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def experiment(dataset):
    return ROOT / 'experiments' / EXPERIMENTS[dataset]


def result_dir(dataset, model):
    name = model if model in ('mean', 'pod', 'rbf') else model + '_s31'
    return ROOT / 'results' / 'current_baselines' / dataset / name


def snapshot(path=None):
    path = Path(path) if path else ROOT / 'results/server_queue.json'
    value = read(path)
    jobs = value.get('jobs')
    if not isinstance(jobs, list):
        raise ValueError('Server snapshot needs a jobs list; update results/server_queue.json.')
    for job in jobs:
        if not isinstance(job, dict) or not all(key in job for key in ('dataset', 'model', 'stage', 'state')):
            raise ValueError('Every snapshot job needs dataset/model/stage/state; formal local training is disabled.')
        state = str(job['state']).split()[0].rstrip('+')
        if state not in ACTIVE | TERMINAL:
            raise ValueError('Unrecognized server state: ' + state)
    return value


def matching_jobs(value, dataset, model):
    return [job for job in value['jobs'] if job['dataset'] == dataset and job['model'] == model]


def assert_training_unassigned(dataset, model, path=None):
    value = snapshot(path)
    assigned = [job for job in matching_jobs(value, dataset, model)
                if job['stage'] in ('train', 'fit') and str(job['state']).split()[0].rstrip('+') in ACTIVE]
    if assigned:
        raise ValueError('Formal training is assigned to the server: ' + json.dumps(assigned) +
                         '. Keep that job running; choose an isolated smoke or a different task.')
    history = value.get('previously_assigned_jobs', [])
    unresolved = [job for job in history if job.get('dataset') == dataset and job.get('model') == model
                  and job.get('stage') in ('train', 'fit')
                  and not any(str(current.get('job_id')) == str(job.get('job_id'))
                              and str(current['state']).split()[0].rstrip('+') in TERMINAL
                              for current in value['jobs'])]
    if unresolved:
        raise ValueError('A previously assigned server job disappeared from squeue; absence is not proof of failure or no work. '
                         'Inspect accounting and resulting artifacts before assigning this task locally.')
    stamp = value.get('captured_at_utc')
    if not stamp:
        raise ValueError('Server snapshot lacks captured_at_utc; refresh it before formal local training.')
    captured = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    if captured.tzinfo is None:
        raise ValueError('Server snapshot time must include a timezone.')
    age = (datetime.now(timezone.utc) - captured).total_seconds()
    if age < -300 or age > 86400:
        raise ValueError('Server snapshot is older than 24 hours or has an invalid future time. Refresh it before formal training.')
    if read_completed_run(dataset, model):
        raise ValueError('This model already has a completed imported run. Use smoke/evaluation with its checkpoint.')
    return value


def read_completed_run(dataset, model):
    path = result_dir(dataset, model) / 'run.json'
    return path.exists() and read(path).get('status') == 'completed'


def verify_source(dataset):
    root = experiment(dataset)
    frozen = read(root / 'frozen_manifest.json')
    checked = 0
    for name, expected in frozen['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Frozen manifest path escapes its experiment: ' + name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Frozen source missing/changed (check Git CRLF conversion): ' + name)
        checked += 1
    return {'experiment': str(root), 'checked_frozen_files': checked,
            'frozen_manifest_sha256': sha(root / 'frozen_manifest.json')}


def verify_data(data_root):
    root = Path(data_root).expanduser().resolve()
    manifest = root / 'manifest.json'
    data = read(manifest)
    marker = read(root / 'COMPLETE.json')
    if marker.get('status') not in ('completed', 'validated'):
        raise ValueError('Dataset COMPLETE.json does not certify a completed/validated adapter.')
    digest = sha(manifest)
    if marker.get('manifest_sha256') and marker['manifest_sha256'] != digest:
        raise ValueError('Dataset completion marker and manifest SHA256 differ.')
    counts = {}
    for split in ('train', 'validation', 'test'):
        rows = data.get('splits', {}).get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError('Missing nonempty dataset split: ' + split)
        for row in rows:
            raw = Path(row['path'])
            if raw.is_absolute() or re.match(r'^[A-Za-z]:[\\/]', row['path']):
                raise ValueError('Expected the original portable relative-path manifest; do not rewrite checkpoint manifests.')
            archive = (root / raw).resolve()
            if not archive.is_relative_to(root) or not archive.is_file():
                raise ValueError('Dataset archive missing or outside root: ' + str(archive))
        counts[split] = len(rows)
    return {'manifest': str(manifest), 'manifest_sha256': digest, 'split_counts': counts,
            'archive_check': 'All referenced paths exist; array checks are performed by the original model runtime.'}


def verify_environment(device):
    if sys.version_info < (3, 10):
        raise ValueError('Python >= 3.10 is required; Python 3.12 is recommended.')
    import numpy
    import torch
    info = {'python': platform.python_version(), 'machine': platform.machine(),
            'torch': str(torch.__version__), 'numpy': numpy.__version__, 'device': device,
            'cuda_runtime': torch.version.cuda}
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable. Install a current PyTorch CUDA build that supports RTX 5090.')
        if torch.cuda.device_count() != 1:
            raise ValueError('Select exactly one GPU with CUDA_VISIBLE_DEVICES before running this launcher.')
        info.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                    vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                    compiled_architectures=torch.cuda.get_arch_list())
        # Real forward/backward execution catches unsupported GPU wheel binaries.
        x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
        x.square().sum().backward()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(x.grad).all()):
            raise ValueError('CUDA forward/backward preflight failed.')
    return info


def checkpoint_path(args):
    if args.model == 'rbf':
        if args.checkpoint:
            raise ValueError('RBF fits each observation case and does not use a trained checkpoint.')
        return None
    path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else result_dir(args.dataset, args.model) / 'latest.pt'
    if not path.is_file():
        raise ValueError('Checkpoint not downloaded: ' + str(path) + '. Download release artifacts or pass --checkpoint.')
    with path.open('rb') as stream:
        if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError('Checkpoint is only a Git LFS pointer; fetch the binary artifact first.')
    return path


def validate_checkpoint(path, args, data):
    if path is None:
        return None
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('model') != args.model or state.get('manifest_sha256') != data['manifest_sha256']:
        raise ValueError('Checkpoint model or manifest SHA256 differs from this request.')
    if args.dataset == 'fs_sph' and state.get('runtime_sha256') != sha(experiment(args.dataset) / 'launch/fs_baseline.py'):
        raise ValueError('FS checkpoint runtime hash differs; preserve original source bytes/LF line endings.')
    return {'path': str(path), 'sha256': sha(path), 'epoch': state.get('epoch')}


def parse_queue_output(output):
    jobs = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split('|')]
        if len(fields) != 5 or not re.fullmatch(r'[0-9]+(?:_[0-9,:%\[\]\-]+)?', fields[0]):
            raise ValueError('Unrecognized squeue output; previous snapshot was preserved.')
        job_id, name, state, reason, start = fields
        if state not in ACTIVE | TERMINAL:
            raise ValueError('Unknown squeue state; previous snapshot was preserved: ' + state)
        dataset = model = None
        stage = 'other'
        match = re.fullmatch(r'b12_(fs_sph|jhtdb)_(.+)', name)
        if match:
            dataset, suffix = match.groups()
            if suffix.endswith('_calibration'):
                model, stage = suffix[:-len('_calibration')], 'calibration'
            elif suffix in MODELS:
                model, stage = suffix, 'train'
            elif suffix in ('pilot', 'prepare'):
                stage = suffix
            else:
                raise ValueError('Unrecognized baseline job name; previous snapshot was preserved: ' + name)
            if model is not None and model not in MODELS:
                raise ValueError('Unrecognized baseline model: ' + str(model))
        elif name.startswith('lbm'):
            dataset, stage = 'lbm', 'generation'
        jobs.append({'job_id': job_id, 'name': name, 'dataset': dataset, 'model': model,
                     'stage': stage, 'state': state, 'reason': reason,
                     'estimated_start_server_local': start})
    return jobs


def refresh_queue(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', args.host):
        raise ValueError('--host must be an SSH config alias or user@hostname without shell syntax.')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
               "squeue -h -u zguan2 -o '%i|%j|%T|%r|%S'"]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            check=True, timeout=60)
    jobs = parse_queue_output(result.stdout)
    value = {'captured_at_utc': datetime.now(timezone.utc).isoformat(),
             'server_timezone': 'America/Chicago', 'source': 'read-only SSH squeue via ' + args.host,
             'jobs': jobs}
    target = ROOT/'results/server_queue.json'
    if target.exists():
        previous = snapshot(target)
        history = {str(job.get('job_id')): job for job in previous.get('previously_assigned_jobs', [])}
        for job in previous['jobs']:
            if job.get('dataset') in EXPERIMENTS and job.get('stage') in ('train', 'fit'):
                history[str(job.get('job_id'))] = job
        value['previously_assigned_jobs'] = list(history.values())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + str(os.getpid()) + '.tmp')
    try:
        write(temporary, value)
        snapshot(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({'updated': str(target), 'job_count': len(jobs),
                      'captured_at_utc': value['captured_at_utc']}, indent=2))


def list_tasks(args):
    try:
        server = snapshot(args.server_snapshot)
        error = None
    except (OSError, ValueError, KeyError) as exc:
        server, error = {'jobs': []}, str(exc)
    rows = []
    for dataset in EXPERIMENTS:
        for model in MODELS:
            jobs = matching_jobs(server, dataset, model)
            rows.append({'dataset': dataset, 'model': model,
                         'formal_train': 'per_case_fit_only' if model == 'rbf' else
                             ('blocked_unknown_snapshot' if error else
                              'server_assigned' if any(str(j['state']).split()[0].rstrip('+') in ACTIVE and j['stage'] in ('train', 'fit') for j in jobs) else
                              'already_completed' if read_completed_run(dataset, model) else 'check_before_start'),
                         'checkpoint_present': (result_dir(dataset, model)/'latest.pt').is_file(),
                         'server_jobs': jobs, 'isolated_validation_smoke': 'available_with_data_and_checkpoint' if model != 'rbf' else 'available_with_data'})
    print(json.dumps({'snapshot_error': error, 'baselines': rows,
                      'other_models': {'Tensor-DiT': 'experiments/tensor_dit_n12_20261027/launch/tensor_model.py',
                                       'DiT3D': 'models/flow3d-current/scripts/train_sparse_track_diffusion.py',
                                       'DPS': 'models/flow3d-dps/scripts/train_flow_prior.py'},
                      'notes': 'Archived queue snapshot is not live. Read README.txt for additional-model preparation.'}, indent=2))


def preflight(args):
    result = {'source': verify_source(args.dataset), 'data': verify_data(args.data_root),
              'environment': verify_environment(args.device)}
    print(json.dumps(result, indent=2))
    return result


def build_command(args, output, data, config_path, checkpoint=None):
    root = experiment(args.dataset)
    common = ['--manifest', data['manifest'], '--model', args.model, '--config', str(config_path),
              '--num-particles', '12', '--seed', '31', '--device', args.device]
    if args.command == 'train':
        script = root / ('source/scripts/run_baseline.py' if args.dataset == 'jhtdb' else 'launch/fs_baseline.py')
        command = [sys.executable, '-B', str(script), 'train', *common,
                   '--epochs', '100', '--output-dir', str(output/'artifacts')]
        if args.resume:
            command += ['--resume', str(Path(args.resume).expanduser().resolve())]
    elif args.dataset == 'jhtdb':
        command = [sys.executable, '-B', str(root/'source/scripts/run_baseline.py'), 'predict', *common,
                   '--split', 'validation', '--case-index', str(args.case_index),
                   '--output', str(output/'predictions'/'case.npz'), '--evaluate']
    else:
        command = [sys.executable, '-B', str(root/'launch/fs_baseline.py'), 'evaluate', *common,
                   '--split', 'validation', '--case-indices', str(args.case_index),
                   '--output-dir', str(output/'predictions')]
    if checkpoint is not None:
        command += ['--checkpoint', str(checkpoint)]
    return command


def run_local(args):
    if args.command == 'train':
        if args.model == 'rbf':
            raise ValueError('RBF has no global training stage; use a per-case smoke instead.')
        server = assert_training_unassigned(args.dataset, args.model, args.server_snapshot)
    else:
        server = None
    checked = preflight(args)
    data = checked['data']
    path = checkpoint_path(args) if args.command == 'smoke' else None
    checkpoint = validate_checkpoint(path, args, data) if path else None
    if args.command == 'smoke' and not 0 <= args.case_index < data['split_counts']['validation']:
        raise ValueError('Invalid validation case index.')
    if args.command == 'train' and args.resume:
        validate_checkpoint(Path(args.resume).expanduser().resolve(), args, data)
    cfg = read(experiment(args.dataset)/'configs'/f'{args.model}_s31.json')
    if args.command == 'smoke':
        cfg.update(fit_steps=args.fit_steps, sampling_steps=args.sampling_steps, num_samples=1)
    name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('--run-name must contain only letters, digits, underscores, and hyphens.')
    output = ROOT/'local_runs'/args.dataset/args.model/(args.command+'_'+name)
    command = build_command(args, output, data, output/'request_config.json', path)
    report = {'command': command, 'output_dir': str(output), 'execution_requested': args.execute,
              'scope': 'isolated_validation_smoke_not_scientific_result' if args.command == 'smoke' else 'local_training',
              'checks': checked, 'checkpoint': checkpoint,
              'server_snapshot_timestamp': server.get('captured_at_utc') if server else None}
    print(json.dumps(report, indent=2))
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    write(output/'request_config.json', cfg)
    write(output/'local_request.json', report)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    write(output/'local_status.json', {'status': 'completed' if result.returncode == 0 else 'failed',
                                     'returncode': result.returncode, 'scope': report['scope']})
    if result.returncode:
        raise SystemExit(result.returncode)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('list'); s.add_argument('--server-snapshot', type=Path)
    s = sub.add_parser('refresh-queue'); s.add_argument('--host', required=True)
    for name in ('preflight', 'smoke', 'train'):
        s = sub.add_parser(name)
        s.add_argument('--dataset', choices=tuple(EXPERIMENTS), required=True)
        s.add_argument('--data-root', type=Path, required=True)
        s.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if name != 'preflight':
            s.add_argument('--model', choices=MODELS, required=True)
            s.add_argument('--run-name')
            s.add_argument('--execute', action='store_true', help='Run after successful checks; otherwise print the command only.')
        if name == 'smoke':
            s.add_argument('--checkpoint', type=Path)
            s.add_argument('--case-index', type=int, default=0)
            s.add_argument('--fit-steps', type=int, default=10)
            s.add_argument('--sampling-steps', type=int, default=10)
        if name == 'train':
            s.add_argument('--server-snapshot', type=Path)
            s.add_argument('--resume', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'list':
            list_tasks(args)
        elif args.command == 'refresh-queue':
            refresh_queue(args)
        elif args.command == 'preflight':
            preflight(args)
        else:
            if args.command == 'smoke' and (args.fit_steps < 1 or args.sampling_steps < 1):
                raise ValueError('Smoke optimization/sampling steps must be positive.')
            run_local(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
