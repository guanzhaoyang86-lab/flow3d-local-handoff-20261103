#!/usr/bin/env python3
"""Run a frozen Tensor preparation, real-CUDA pilot, or formal training task."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_inputs():
    expected = os.environ.get('TENSOR_FROZEN_SHA256')
    if not expected or expected != sha(ROOT / 'frozen_manifest.json'):
        raise RuntimeError('Missing or changed submission-time Tensor fingerprint')
    for name, digest in read(ROOT / 'frozen_manifest.json')['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen input changed: ' + name)
    plan = read(ROOT / 'plan.json')
    if sha(Path(plan['baseline_root']) / 'frozen_manifest.json') != plan['baseline_frozen_sha256']:
        raise RuntimeError('The reused baseline data pipeline fingerprint changed')


def gpu_check():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run inside the submitted Slurm allocation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one working CUDA GPU required')
    x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    x.square().sum().backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError('CUDA gradient check failed')
    return {'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'cuda_gradient_verified': True}


def data_ready(spec):
    root = Path(spec['derived_root'])
    marker = read(root / 'COMPLETE.json')
    manifest = root / 'manifest.json'
    if marker.get('status') not in ('completed', 'validated'):
        raise RuntimeError('Dataset preparation has not completed')
    if not marker.get('manifest_sha256') or sha(manifest) != marker['manifest_sha256']:
        raise RuntimeError('Dataset manifest is not bound to its completion marker')
    rows = read(manifest).get('splits', {})
    if not all(rows.get(split) for split in ('train', 'validation', 'test')):
        raise RuntimeError('Three nonempty data splits required')
    return manifest


def artifact_ready(cache, manifest, dataset):
    import tensor_data
    cache = Path(cache)
    marker = read(cache / 'COMPLETE.json')
    artifact = cache / 'selected.pt'
    if marker.get('status') != 'completed' or marker.get('manifest_sha256') != sha(manifest):
        raise RuntimeError('Tensor preparation has not completed for this manifest')
    if marker.get('artifact_sha256', marker.get('selected_artifact_sha256')) != sha(artifact):
        raise RuntimeError('Selected Tucker artifact differs from its completion marker')
    tensor_data.load_artifact(artifact, manifest, dataset)
    return artifact


def run(args):
    args = [str(v) for v in args]
    print(json.dumps({'command': args}), flush=True)
    subprocess.run(args, check=True)


def train_command(dataset, manifest, artifact, config, output, seed, epochs, *, pilot=False):
    command = [sys.executable, '-B', ROOT / 'launch/tensor_model.py', 'train',
               '--dataset', dataset, '--manifest', manifest, '--artifact', artifact,
               '--config', config, '--output-dir', output, '--seed', seed, '--epochs', epochs, '--device', 'cuda']
    if pilot:
        command += ['--max-train-cases', 8, '--max-validation-cases', 2]
    checkpoint = Path(output) / 'latest.pt'
    if checkpoint.exists():
        command += ['--resume', checkpoint]
    return command


def completed_checkpoint(output, dataset, epoch, manifest, artifact, config=None):
    import torch
    state = torch.load(Path(output) / 'latest.pt', map_location='cpu', weights_only=True)
    if state.get('model') != 'tensor_dit' or state.get('dataset') != dataset or state.get('epoch') != epoch:
        raise RuntimeError('Wrong model, dataset, or final training epoch')
    if state.get('manifest_sha256') != sha(manifest) or state.get('artifact_sha256') != sha(artifact):
        raise RuntimeError('Checkpoint manifest/Tucker artifact mismatch')
    if state.get('optimizer_steps', 0) < 1:
        raise RuntimeError('No successful optimizer step')
    if read(Path(output) / 'run.json').get('status') != 'completed':
        raise RuntimeError('Training has not published completion')
    if config is not None and state.get('config') != read(config):
        raise RuntimeError('Checkpoint config differs from requested settings')
    return state


def run_pilot(plan, dataset, manifest, artifact, output):
    output = Path(output)
    cfg = read(ROOT / 'configs' / f'{dataset}_s31.json')
    cfg['epochs'] = 1
    config = output / 'pilot_config.json'
    write(config, cfg)
    destination = output / 'train'
    already_complete = False
    if (destination / 'latest.pt').exists() and (destination / 'run.json').exists():
        import torch
        state = torch.load(destination / 'latest.pt', map_location='cpu', weights_only=True)
        already_complete = state.get('epoch') == 1 and read(destination / 'run.json').get('status') == 'completed'
    start = time.monotonic()
    if not already_complete:
        run(train_command(dataset, manifest, artifact, config, destination, 31, 1, pilot=True))
    state = completed_checkpoint(destination, dataset, 1, manifest, artifact, config)
    training_seconds = time.monotonic() - start
    # Exercise full configured DDIM sampling and posthoc masked physical scoring
    # on a validation case, using only XZ observations to construct prediction.
    prediction = output / 'sample'
    start = time.monotonic()
    run([sys.executable, '-B', ROOT / 'launch/tensor_model.py', 'evaluate',
         '--dataset', dataset, '--manifest', manifest, '--checkpoint', destination / 'latest.pt',
         '--output-dir', prediction, '--split', 'validation', '--case-indices', '0', '--device', 'cuda', '--resume'])
    sampling_seconds = time.monotonic() - start
    status = {'status': 'completed', 'dataset': dataset, 'model': 'tensor_dit', 'N': 12,
              'epochs': 1, 'train_case_limit': 8, 'validation_case_limit': 2, 'test_used': False,
              'optimizer_steps': state['optimizer_steps'], 'manifest_sha256': sha(manifest),
              'artifact_sha256': sha(artifact), 'config_sha256': sha(config),
              'training_elapsed_seconds': training_seconds, 'sampling_elapsed_seconds': sampling_seconds,
              'scope': 'Data, forward/backward, EMA, checkpoint, DDIM and validation scoring smoke; no main scientific results.'}
    write(output / 'PILOT_COMPLETE.json', status)
    return status


def pilot_ready(directory, manifest, artifact):
    marker = read(Path(directory) / 'PILOT_COMPLETE.json')
    if marker.get('status') != 'completed' or marker.get('model') != 'tensor_dit':
        raise RuntimeError('Tensor GPU pilot incomplete')
    if marker.get('manifest_sha256') != sha(manifest) or marker.get('artifact_sha256') != sha(artifact):
        raise RuntimeError('Pilot belongs to a different dataset or tensor artifact')
    if marker.get('optimizer_steps', 0) < 1:
        raise RuntimeError('Pilot has no actual optimization')
    return marker


def formal_scope(state, manifest):
    if (state.get('pilot_subset') or state.get('max_train_cases') is not None
            or state.get('max_validation_cases') is not None
            or any(state.get('limits', {}).values())):
        raise RuntimeError('A limited smoke checkpoint cannot be a formal result')
    splits = read(manifest)['splits']
    for split, key in (('train', 'training_case_ids'), ('validation', 'validation_case_ids')):
        if state.get(key) != [row['case_id'] for row in splits[split]]:
            raise RuntimeError('Formal checkpoint does not cover the full ' + split + ' split')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-key', required=True)
    key = parser.parse_args().job_key
    verify_inputs()
    plan = read(ROOT / 'plan.json')
    job = next((job for job in plan['jobs'] if job['key'] == key), None)
    if job is None or os.environ.get('TENSOR_JOB_KEY') != key:
        raise RuntimeError('Unknown/mismatched frozen task key')
    dataset = job['dataset']
    dataset_output = Path(plan['result_root']) / dataset
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    count = len(job.get('tasks', [None]))
    if not 0 <= index < count:
        raise RuntimeError('Array index outside frozen task plan')
    task = job.get('tasks', [None])[index]
    stage_output = dataset_output / ('tensor_space' if job['stage'] == 'tensor_prepare' else 'pilot' if job['stage'] == 'pilot' else f'tensor_dit_s{task["seed"]}')
    stage_output.mkdir(parents=True, exist_ok=True)
    lock = (stage_output / '.runtime.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = stage_output / 'allocation_status.json'
    status = {'status': 'running', 'key': key, 'dataset': dataset, 'stage': job['stage'],
              'array_index': index, 'job_id': os.environ.get('SLURM_JOB_ID'), 'started_unix': time.time()}
    write(status_path, status)
    try:
        status['gpu'] = gpu_check()
        manifest = data_ready(plan['datasets'][dataset])
        cache = dataset_output / 'tensor_space'
        if job['stage'] == 'tensor_prepare':
            run([sys.executable, '-B', ROOT / 'launch/tensor_data.py', 'prepare',
                 '--dataset', dataset, '--manifest', manifest, '--output-dir', cache, '--device', 'cuda',
                 '--ranks', ','.join(map(str, plan['rank_candidates'])),
                 '--hooi-iterations', plan['hooi_iterations'], '--resume'])
            artifact_ready(cache, manifest, dataset)
        elif job['stage'] == 'pilot':
            artifact = artifact_ready(cache, manifest, dataset)
            run_pilot(plan, dataset, manifest, artifact, stage_output)
        else:
            artifact = artifact_ready(cache, manifest, dataset)
            pilot_ready(dataset_output / 'pilot', manifest, artifact)
            config = ROOT / 'configs' / f'{dataset}_s{task["seed"]}.json'
            already_complete = False
            if (stage_output / 'latest.pt').exists() and (stage_output / 'run.json').exists():
                import torch
                old = torch.load(stage_output / 'latest.pt', map_location='cpu', weights_only=True)
                already_complete = old.get('epoch') == task['epochs'] and read(stage_output / 'run.json').get('status') == 'completed'
            if not already_complete:
                run(train_command(dataset, manifest, artifact, config, stage_output, task['seed'], task['epochs']))
            state = completed_checkpoint(stage_output, dataset, task['epochs'], manifest, artifact, config)
            # Formal outputs must cover the full training/validation split.
            formal_scope(state, manifest)
        status.update(status='completed', completed_unix=time.time())
        write(status_path, status)
    except BaseException as exc:
        status.update(status='failed', error=str(exc), failed_unix=time.time(), traceback=traceback.format_exc())
        write(status_path, status)
        raise


if __name__ == '__main__':
    main()
