#!/usr/bin/env python3
"""Execute one frozen, dependency-guarded baseline allocation; never submit jobs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/src'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(command):
    print(json.dumps({'command': [str(v) for v in command]}, allow_nan=False), flush=True)
    subprocess.run([str(v) for v in command], check=True)


def verify_inputs():
    lock_path = ROOT / 'frozen_manifest.json'
    expected = os.environ.get('BASELINE_FROZEN_SHA256')
    if not expected or sha(lock_path) != expected:
        raise RuntimeError('Submission-time frozen manifest fingerprint missing or changed')
    for name, digest in read(lock_path)['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen experiment input changed: {name}')


def gpu_check():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A Slurm allocation is required')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one working CUDA GPU must be visible')
    x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    x.square().sum().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('CUDA finite-gradient check failed')
    return {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'cuda_math_verified': True}


def data_ready(spec):
    root = Path(spec['derived_root'])
    marker = read(root / 'COMPLETE.json')
    if marker.get('status') not in ('completed', 'validated'):
        raise RuntimeError('Dataset adapter has not published completion')
    manifest = root / 'manifest.json'
    if marker.get('manifest_sha256') and sha(manifest) != marker['manifest_sha256']:
        raise RuntimeError('Published completion marker does not match the adapted manifest')
    data = read(manifest)
    if not all(data.get('splits', {}).get(k) for k in ('train', 'validation', 'test')):
        raise RuntimeError('Adapted manifest needs three nonempty declared splits')
    return manifest


def completed_checkpoint(output, model, epoch, manifest):
    import torch
    output = Path(output)
    state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
    if state.get('model') != model or state.get('epoch') != epoch:
        raise RuntimeError(f'Wrong final checkpoint model/epoch for {output}')
    status = read(output / 'run.json')
    if status.get('status') != 'completed':
        raise RuntimeError(f'Training did not publish completion: {output}')
    digest = state.get('manifest_sha256')
    if digest is not None and digest != sha(manifest):
        raise RuntimeError('Checkpoint manifest differs from requested data')
    if epoch and state.get('optimizer_steps', state.get('updates', 0)) < 1:
        raise RuntimeError('Checkpoint has no successful optimization update')
    return state


def train_command(dataset, model, config, output, manifest, seed, epochs, resume=True):
    python = sys.executable
    if dataset == 'jhtdb':
        command = [python, '-B', ROOT / 'source/scripts/run_baseline.py', 'train']
    else:
        command = [python, '-B', ROOT / 'launch/fs_baseline.py', 'train']
    command += ['--manifest', manifest, '--model', model, '--config', config,
                '--num-particles', 12, '--seed', seed, '--epochs', epochs,
                '--output-dir', output, '--device', 'cuda']
    checkpoint = Path(output) / 'latest.pt'
    if resume and checkpoint.exists():
        command += ['--resume', checkpoint]
    return command


def pilot_manifest(dataset, manifest, output):
    data = read(manifest)
    out = json.loads(json.dumps(data))
    out['pilot_only'] = True
    out['pilot_parent_manifest_sha256'] = sha(manifest)
    out['pilot_note'] = 'Training input smoke and resource benchmark; never a main scientific checkpoint.'
    limits = {'train': 32, 'validation': 8, 'test': 1}
    for split, rows in data['splits'].items():
        selected = rows[:limits[split]]
        for row in selected:
            path = Path(row['path'])
            row['path'] = str(path if path.is_absolute() else manifest.parent / path)
        out['splits'][split] = selected
    if dataset == 'jhtdb':
        # This remains group-disjoint; no one-case overlap smoke mode is used.
        out.pop('case_count', None)
    write(output, out)
    return Path(output)


def predict_command(dataset, model, config, output, manifest, seed, case_index, checkpoint=None):
    if dataset == 'jhtdb':
        existing = Path(output).with_suffix('.npz')
        if existing.exists():
            metadata = read(existing.with_suffix('.json'))
            requested = read(config)
            if (metadata.get('model') != model or metadata.get('config') != requested
                    or metadata.get('manifest_sha256') != sha(manifest)
                    or metadata.get('split') != 'validation'
                    or metadata.get('case_id') != read(manifest)['splits']['validation'][case_index]['case_id']
                    or metadata.get('output_sha256') != sha(existing)):
                raise RuntimeError('Existing prediction does not match this validation request')
            if checkpoint and metadata.get('checkpoint_sha256') != sha(checkpoint):
                raise RuntimeError('Existing prediction was generated by a different checkpoint')
            return [sys.executable, '-B', ROOT / 'source/scripts/run_baseline.py', 'verify', existing]
        command = [sys.executable, '-B', ROOT / 'source/scripts/run_baseline.py', 'predict',
                   '--case-index', case_index, '--output', Path(output).with_suffix('.npz'), '--evaluate']
    else:
        command = [sys.executable, '-B', ROOT / 'launch/fs_baseline.py', 'evaluate',
                   '--case-indices', str(case_index), '--output-dir', output, '--resume']
    command += ['--manifest', manifest, '--model', model, '--config', config,
                '--num-particles', 12, '--seed', seed, '--split', 'validation', '--device', 'cuda']
    if checkpoint:
        command += ['--checkpoint', checkpoint]
    return command


def run_pilot(plan, dataset, manifest, output):
    output = Path(output)
    manifest = pilot_manifest(dataset, manifest, output / 'pilot_manifest.json')
    results = []
    for model in plan['learned_models']:
        cfg = read(ROOT / 'configs' / f'{model}_s31.json')
        cfg['epochs'] = 1
        cfg_path = output / f'{model}_pilot.json'
        write(cfg_path, cfg)
        destination = output / model
        began = time.monotonic()
        if (destination / 'latest.pt').exists():
            import torch
            old = torch.load(destination / 'latest.pt', map_location='cpu', weights_only=True)
            already_complete = old.get('epoch') == 1 and (destination / 'run.json').exists() and read(destination / 'run.json').get('status') == 'completed'
        else:
            already_complete = False
        if not already_complete:
            run(train_command(dataset, model, cfg_path, destination, manifest, 31, 1))
        completed_checkpoint(destination, model, 1, manifest)
        results.append({'model': model, 'elapsed_seconds': time.monotonic() - began,
                        'checkpoint_reused': already_complete,
                        'elapsed_scope': 'checkpoint_verification' if already_complete else 'training_and_verification'})
    for model in ('mean', 'pod'):
        config = ROOT / 'configs' / f'{model}_s31.json'
        destination = output / model
        if not (destination / 'latest.pt').exists():
            run(train_command(dataset, model, config, destination, manifest, 31, 100, resume=False))
        completed_checkpoint(destination, model, 0, manifest)
    for model in ('pod', 'rbf'):
        cfg = read(ROOT / 'configs' / f'{model}_s31.json')
        cfg['fit_steps'] = 2
        config = output / 'smoke_configs' / f'{model}.json'
        write(config, cfg)
        checkpoint = output / 'pod/latest.pt' if model == 'pod' else None
        run(predict_command(dataset, model, config, output / 'smoke_predictions' / model, manifest, 31, 0, checkpoint))
    result = {'status': 'completed', 'dataset': dataset, 'N': 12, 'test_used': False,
              'input_view': 'xz', 'benchmark': results,
              'scope': 'Five learned forward/backward and two fit paths; no scientific results.'}
    write(output / 'PILOT_COMPLETE.json', result)
    return result


def calibrate(plan, task, manifest):
    dataset, model = task['dataset'], task['model']
    output = Path(task['output_dir'])
    rows = []
    scores = {}
    checkpoint = Path(plan['result_root']) / dataset / 'pod/latest.pt' if model == 'pod' else None
    if checkpoint:
        completed_checkpoint(checkpoint.parent, model, 0, manifest)
    for sigma in plan['sigma_candidates']:
        cfg = read(ROOT / 'configs' / f'{model}_s31.json')
        cfg['sigma'] = sigma
        cfg_path = output / f'config_sigma_{sigma}.json'
        write(cfg_path, cfg)
        for index in range(min(plan['calibration_cases'], len(read(manifest)['splits']['validation']))):
            case_out = output / f'sigma_{sigma}' / f'case_{index:04d}'
            run(predict_command(dataset, model, cfg_path, case_out, manifest, 31, index, checkpoint))
            if dataset == 'jhtdb':
                value = read(case_out.with_suffix('.json'))['evaluation']['posterior_mean_field_relative_l2_unknown_interior']
            else:
                value = read(case_out / 'summary.json')['summary']['physical_velocity_relative_l2']
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError('Calibration has no finite validation field error')
            scores.setdefault(sigma, []).append(value)
            rows.append({'sigma': sigma, 'case_index': index, 'output': str(case_out), 'validation_relative_l2': value})
    means = {sigma: sum(values) / len(values) for sigma, values in scores.items()}
    selected = min(means, key=lambda sigma: (means[sigma], sigma))
    selected_config = read(ROOT / 'configs' / f'{model}_s31.json')
    selected_config['sigma'] = selected
    write(output / 'selected_config.json', selected_config)
    write(output / 'CALIBRATION_COMPLETE.json', {'status': 'completed', 'model': model,
          'split': 'validation', 'seed': 31, 'N': 12, 'sigma_candidates': plan['sigma_candidates'],
          'outputs': rows, 'mean_validation_errors': means, 'selected_sigma': selected,
          'selection_status': 'frozen_from_predeclared_validation_subset', 'test_used': False})


def execute(plan, job, index):
    dataset = job['dataset']
    spec = plan['datasets'][dataset]
    output = Path(plan['result_root']) / dataset
    if job['stage'] == 'prepare':
        if dataset == 'jhtdb':
            run([sys.executable, '-B', ROOT / 'launch/prepare_jhtdb.py', '--source-root', spec['raw_root'],
                 '--output-root', spec['derived_root'], '--device', 'cuda', '--resume'])
        else:
            run([sys.executable, '-B', ROOT / 'launch/prepare_fs.py', '--release-root', spec['raw_root'],
                 '--run-dir', spec['raw_run'], '--output-root', spec['derived_root'],
                 '--protocol-root', ROOT / 'protocol', '--device', 'cpu', '--workers', 16, '--resume'])
        data_ready(spec)
        return
    manifest = data_ready(spec)
    if job['stage'] == 'pilot':
        run_pilot(plan, dataset, manifest, output / 'pilot')
        return
    pilot = read(output / 'pilot/PILOT_COMPLETE.json')
    if pilot.get('status') != 'completed' or pilot.get('N') != 12:
        raise RuntimeError('Real-data CUDA pilot has not completed')
    task = job['tasks'][index]
    if task['stage'] == 'calibration':
        calibrate(plan, task, manifest)
        return
    model, destination = task['model'], Path(task['output_dir'])
    epoch = 0 if model in ('mean', 'pod') else 100
    checkpoint = destination / 'latest.pt'
    if checkpoint.exists():
        import torch
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('epoch') == epoch:
            completed_checkpoint(destination, model, epoch, manifest)
            print('Verified existing completed training; no duplicate optimization.', flush=True)
            return
        if epoch == 0:
            raise RuntimeError('Existing partial statistical checkpoint needs inspection')
    config = ROOT / 'configs' / f'{model}_s{task["seed"]}.json'
    run(train_command(dataset, model, config, destination, manifest, task['seed'], 100))
    completed_checkpoint(destination, model, epoch, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-key', required=True)
    args = parser.parse_args()
    verify_inputs()
    plan = read(ROOT / 'plan.json')
    job = next(j for j in plan['jobs'] if j['key'] == args.job_key)
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    if index < 0 or index >= len(job.get('tasks', [None])):
        raise RuntimeError('Unexpected Slurm array task index')
    status = Path(plan['result_root']) / 'job_status' / f'{job["key"]}_{index}.json'
    lock_path = status.with_suffix('.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_lock = lock_path.open('a')
    fcntl.flock(allocation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {'status': 'running', 'job_key': job['key'], 'index': index,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'started_unix': time.time()}
    try:
        record['runtime'] = gpu_check()
        write(status, record)
        execute(plan, job, index)
        record.update(status='completed', finished_unix=time.time())
        write(status, record)
    except BaseException as error:
        record.update(status='failed', error=str(error), traceback=traceback.format_exc(), finished_unix=time.time())
        write(status, record)
        raise


if __name__ == '__main__':
    main()
