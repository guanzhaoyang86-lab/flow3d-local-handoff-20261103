#!/usr/bin/env python3
"""Run one frozen static-DPS stage; never submit/cancel jobs or alter old runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE = PACKAGE / 'static_source'
sys.path.insert(0, str(SOURCE / 'src'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_inputs(require_fingerprint=True):
    lock_path = HERE / 'frozen_manifest.json'
    expected = os.environ.get('FLOW3D_DPS_FROZEN_SHA256')
    if require_fingerprint and (not expected or expected != sha(lock_path)):
        raise ValueError('Submission-time DPS frozen fingerprint is missing or changed')
    lock = read(lock_path)
    for name, digest in lock['files'].items():
        if sha(PACKAGE / name) != digest:
            raise ValueError('Frozen DPS input changed: ' + name)
    config = read(HERE / 'config.json')
    if sha(config['manifest']) != lock['manifest_sha256']:
        raise ValueError('Prepared dataset manifest changed')
    marker = read(config['completion_marker'])
    if marker.get('status') != 'completed' or marker.get('manifest_sha256') != lock['manifest_sha256']:
        raise ValueError('Dataset completion does not certify this manifest')
    return config


def cuda_check():
    import torch
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Production stages require a Slurm allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one working CUDA GPU is required')
    probe = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    probe.square().sum().backward()
    if not bool(torch.isfinite(probe.grad).all()):
        raise ValueError('CUDA gradient check failed')
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'cuda_arithmetic_verified': True}


def run(command):
    print(json.dumps({'command': [str(item) for item in command]}), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def training_command(config, output, resume=None):
    command = [sys.executable, '-B', SOURCE / 'scripts/train_flow_prior.py',
               '--manifest', config['manifest'], '--output-dir', output, '--device', 'cuda']
    for name, value in config['training'].items():
        if name == 'amp':
            if value:
                command.append('--amp')
        else:
            command.extend(('--' + name.replace('_', '-'), value))
    if resume is not None:
        command.extend(('--resume', resume))
    return command


def verify_prior(output, config):
    from flow_observation.dps.audit import verify_prior as audit
    import torch
    audit(output, config['training']['epochs'])
    state = torch.load(Path(output) / 'latest.pt', map_location='cpu', weights_only=True)
    if state['manifest_sha256'] != sha(config['manifest']) or state['train_config']['seed'] != 31:
        raise ValueError('Prior manifest/seed differs from this experiment')
    if state['conditioning'] != 'none' or state['normalization_fit_split'] != 'train':
        raise ValueError('DPS requires a separate flow-only, train-normalized prior')
    return state


def restore_committed_history(path, checkpoint_epoch):
    """An append may precede the atomic checkpoint; preserve only committed rows."""
    path = Path(path)
    if not path.exists():
        return
    original = path.read_text()
    committed = {}
    for line in original.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = row.get('epoch')
        if type(epoch) is int and 1 <= epoch <= checkpoint_epoch:
            committed[epoch] = row
    backup = path.with_name('history_before_resume_' + os.environ.get('SLURM_JOB_ID', 'local') + '.jsonl')
    if not backup.exists():
        backup.write_text(original)
    temporary = path.with_name(path.name + '.resume.tmp')
    temporary.write_text(''.join(json.dumps(committed[index], allow_nan=False) + '\n' for index in sorted(committed)))
    temporary.replace(path)


def train(config):
    import torch
    output = Path(config['result_root']) / 'prior_s31'
    checkpoint = output / 'latest.pt'
    resume = None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('method') != 'unconditional-flow-prior' or state.get('manifest_sha256') != sha(config['manifest']):
            raise ValueError('Existing checkpoint is not this unconditional prior')
        if state.get('epoch') == config['training']['epochs']:
            verify_prior(output, config)
            print('Verified existing completed prior; no duplicate training.', flush=True)
            return
        restore_committed_history(output / 'history.jsonl', state['epoch'])
        resume = checkpoint
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Prior directory has artifacts but no committed checkpoint; inspect before restart')
    run(training_command(config, output, resume))
    verify_prior(output, config)


def sampling_command(config, split, index, sigma, output):
    checkpoint = Path(config['result_root']) / 'prior_s31/latest.pt'
    command = [sys.executable, '-B', SOURCE / 'scripts/sample_flow_dps.py', '--checkpoint', checkpoint,
               '--manifest', config['manifest'], '--output', output, '--split', split,
               '--case-index', index, '--sigma', sigma, '--device', 'cuda', '--evaluate']
    for name, value in config['sampling'].items():
        command.extend(('--' + name.replace('_', '-'), value))
    return command


def verify_prediction(config, split, index, sigma, output):
    from flow_observation.dps.audit import verify_posterior
    verify_posterior(output, config['sampling']['num_particles'], config['sampling']['num_samples'])
    row = read(Path(output).with_suffix('.json'))
    expected = dict(config['sampling'], split=split, case_index=index, sigma=sigma)
    saved = row['config']
    if (row.get('manifest_sha256') != sha(config['manifest'])
            or row.get('prior_checkpoint_sha256') != sha(Path(config['result_root']) / 'prior_s31/latest.pt')
            or any(saved.get(key) != value for key, value in expected.items())
            or row.get('reference_used_during_inference') is not False):
        raise ValueError('Existing posterior does not match the frozen request')
    return row


def predict(config, split, index, sigma, output):
    output = Path(output)
    if output.exists():
        return verify_prediction(config, split, index, sigma, output)
    if output.with_suffix('.json').exists():
        raise ValueError('Orphan output metadata exists; inspect rather than overwrite')
    run(sampling_command(config, split, index, sigma, output))
    return verify_prediction(config, split, index, sigma, output)


def calibrate(config):
    verify_prior(Path(config['result_root']) / 'prior_s31', config)
    root = Path(config['result_root']) / 'validation_calibration'
    checkpoint = Path(config['result_root']) / 'prior_s31/latest.pt'
    scores, records = {}, []
    for sigma in config['calibration']['sigma_candidates']:
        values = []
        for index in config['calibration']['case_indices']:
            output = root / f'sigma_{sigma}' / f'case_{index:04d}.npz'
            row = predict(config, 'validation', index, sigma, output)
            value = row['posthoc_evaluation'][config['calibration']['selection_metric']]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Validation sigma selection requires finite declared field error')
            values.append(value)
            records.append({'sigma': sigma, 'case_index': index, 'case_id': row['case_id'],
                            'output': str(output), 'output_sha256': row['output_sha256'], 'score': value})
        scores[sigma] = sum(values) / len(values)
    selected = min(scores, key=lambda value: (scores[value], value))
    selection = {'dataset': config['dataset'], 'sampling': config['sampling'], 'sigma': selected,
                 'manifest_sha256': sha(config['manifest']), 'checkpoint_sha256': sha(checkpoint),
                 'selection_split': 'validation', 'test_used': False,
                 'metric': config['calibration']['selection_metric'], 'mean_errors': scores, 'records': records}
    write(root / 'selected_config.json', selection)
    write(root / 'CALIBRATION_COMPLETE.json', {'status': 'completed', 'test_used': False,
          'selected_config_sha256': sha(root / 'selected_config.json'), 'selected_sigma': selected,
          'validation_cases_per_sigma': len(config['calibration']['case_indices'])})


def test(config):
    from flow_observation.sparse_dataset import load_sparse_flow_manifest
    verify_prior(Path(config['result_root']) / 'prior_s31', config)
    root = Path(config['result_root'])
    selected_path = root / 'validation_calibration/selected_config.json'
    marker = read(root / 'validation_calibration/CALIBRATION_COMPLETE.json')
    selected = read(selected_path)
    if (marker.get('status') != 'completed' or marker.get('test_used') is not False
            or marker.get('selected_config_sha256') != sha(selected_path)
            or selected['manifest_sha256'] != sha(config['manifest'])
            or selected['checkpoint_sha256'] != sha(root / 'prior_s31/latest.pt')
            or selected['sampling'] != config['sampling']):
        raise ValueError('Held-out testing requires the matching frozen validation selection')
    _, splits = load_sparse_flow_manifest(config['manifest'])
    records, errors = [], []
    for index in range(len(splits['test'])):
        output = root / 'heldout_test' / f'case_{index:04d}.npz'
        row = predict(config, 'test', index, selected['sigma'], output)
        value = row['posthoc_evaluation'][config['calibration']['selection_metric']]
        if isinstance(value, (int, float)) and math.isfinite(value):
            errors.append(value)
        records.append({'case_index': index, 'case_id': row['case_id'], 'field_error': value,
                        'valid_samples': row['valid_sample_count'], 'samples': config['sampling']['num_samples'],
                        'output_sha256': row['output_sha256']})
    write(root / 'heldout_test/TEST_COMPLETE.json', {'status': 'completed', 'dataset': config['dataset'],
          'sigma_selected_on': 'validation', 'selected_config_sha256': sha(selected_path),
          'cases': len(records), 'finite_error_cases': len(errors),
          'mean_field_error': sum(errors) / len(errors) if len(errors) == len(records) else None,
          'metric_coverage_complete': len(errors) == len(records), 'records': records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'train', 'calibrate', 'test'))
    args = parser.parse_args()
    config = verify_inputs(require_fingerprint=args.stage != 'preflight')
    if args.stage == 'preflight':
        from flow_observation.sparse_dataset import load_sparse_flow_manifest
        from flow_observation.dps.data import read_prior_field, load_observation_case
        _, splits = load_sparse_flow_manifest(config['manifest'])
        counts = {name: len(rows) for name, rows in splits.items()}
        if counts != config['expected_split_counts']:
            raise ValueError('Prepared split counts differ from declared protocol')
        field, _ = read_prior_field(splits['train'][0].path)
        observation = load_observation_case(splits['validation'][0], 'validation', 12,
                                           observation_seed=47, num_probes=32, views=[0])
        if set(observation['particle_indices'].tolist()) & set(observation['probe_particle_indices'].tolist()):
            raise ValueError('Observed/probe particle identity overlap')
        print(json.dumps({'status': 'preflight_passed', 'dataset': config['dataset'], 'split_counts': counts,
              'field_shape': list(field.shape), 'observation_shape': list(observation['tracks'].shape),
              'prior_training': 'not_run', 'real_cuda_verification': 'pending Slurm allocation'}))
        return
    runtime = cuda_check()
    status = Path(config['result_root']) / 'job_status' / f'{args.stage}_{os.environ["SLURM_JOB_ID"]}.json'
    record = {'status': 'running', 'stage': args.stage, 'dataset': config['dataset'], 'runtime': runtime,
              'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write(status, record)
    try:
        {'train': train, 'calibrate': calibrate, 'test': test}[args.stage](config)
    except BaseException as error:
        write(status, dict(record, status='failed', error=str(error)))
        raise
    write(status, dict(record, status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
