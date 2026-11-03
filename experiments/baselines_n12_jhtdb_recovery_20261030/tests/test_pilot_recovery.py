"""Regression coverage for recovery from the real POD config/output collision."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _runtime():
    spec = importlib.util.spec_from_file_location(
        'pilot_recovery_runtime', ROOT / 'launch/batch_runtime.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def test_pilot_recovers_legacy_collision_and_verifies_on_repeat(tmp_path, monkeypatch):
    runtime = _runtime()
    from flow_baselines.common import no_output_collision
    from flow_baselines.reporting import verify

    plan = _read(ROOT / 'plan.json')
    manifest = tmp_path / 'manifest.json'
    runtime.write(manifest, {
        'case_count': 3,
        'splits': {
            split: [{'case_id': split + '_0', 'flow_group_id': split,
                     'path': split + '.npz'}]
            for split in ('train', 'validation', 'test')
        },
    })
    output = tmp_path / 'pilot'
    pilot_manifest = runtime.pilot_manifest(
        'jhtdb', manifest, output / 'pilot_manifest.json')

    # These small completed checkpoints exercise the real reuse and provenance
    # checks. No network construction, model training, or GPU is needed.
    preserved = [manifest, pilot_manifest]
    for model in (*plan['learned_models'], 'mean', 'pod'):
        destination = output / model
        destination.mkdir()
        epoch = 1 if model in plan['learned_models'] else 0
        config = _read(ROOT / 'configs' / f'{model}_s31.json')
        if epoch:
            config['epochs'] = epoch
        checkpoint = destination / 'latest.pt'
        torch.save({
            'format': 'flow3d-baseline-v1', 'model': model, 'epoch': epoch,
            'config': config, 'manifest_sha256': _sha(pilot_manifest),
            'optimizer_steps': 1 if epoch else 0,
        }, checkpoint)
        completion = destination / 'run.json'
        runtime.write(completion, {
            'status': 'completed', 'model': model, 'epoch': epoch,
            'config': config, 'checkpoint_sha256': _sha(checkpoint),
        })
        preserved.extend((checkpoint, completion))

    legacy = output / 'pod_fit_smoke.json'
    runtime.write(legacy, {'seed': 31, 'fit_steps': 2,
                           'note': 'Original conflicting configuration; preserve it.'})
    preserved.append(legacy)
    before = {path: _sha(path) for path in preserved}
    calls = []

    def simulate_cli(command, *, check):
        # Keep runtime.run and predict_command real; intercept only the process
        # boundary where the expensive CUDA fit would otherwise execute.
        assert check is True
        assert Path(command[2]) == ROOT / 'source/scripts/run_baseline.py'
        operation = command[3]
        if operation == 'verify':
            target = Path(command[4])
            result = verify(target)
            assert result['verified'] is True
            calls.append((operation, result['model']))
            return subprocess.CompletedProcess(command, 0)

        assert operation == 'predict', 'Completed pilot checkpoints must not retrain'
        args = command[4:]
        value = lambda flag: args[args.index(flag) + 1]
        model = value('--model')
        assert model in ('pod', 'rbf')
        target = Path(value('--output'))
        config_path = Path(value('--config'))
        assert config_path != target.with_suffix('.json')
        assert config_path.parent == output / 'smoke_configs'
        assert target.parent == output / 'smoke_predictions'
        assert value('--split') == 'validation'
        assert value('--case-index') == '0'
        assert value('--seed') == '31'
        assert value('--num-particles') == '12'
        assert '--evaluate' in args
        assert Path(value('--manifest')) == pilot_manifest
        assert (legacy in before and _sha(legacy) == before[legacy])
        # This is the actual guard that rejected the old pilot, before any
        # inference work. Merely checking strings would miss this regression.
        no_output_collision(target)
        config = _read(config_path)
        assert config['fit_steps'] == 2
        assert config['seed'] == 31
        checkpoint = Path(value('--checkpoint')) if '--checkpoint' in args else None
        assert checkpoint == (output / 'pod/latest.pt' if model == 'pod' else None)
        metadata = {
            'format': 'flow3d-baseline-prediction-v1', 'status': 'completed',
            'model': model, 'config': config, 'actual_predictions': 1,
            'reference_used_during_inference': False,
            'manifest_sha256': _sha(pilot_manifest), 'split': 'validation',
            'case_id': 'validation_0',
            'checkpoint_sha256': _sha(checkpoint) if checkpoint else None,
        }
        predictions = np.zeros((1, 3, 1, 1, 1), dtype=np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, predictions=predictions,
                                prediction_mean=predictions.mean(0),
                                metadata=np.array(json.dumps(metadata)))
        temporary.replace(target)
        runtime.write(target.with_suffix('.json'),
                      dict(metadata, output_sha256=_sha(target)))
        assert verify(target)['verified'] is True
        calls.append((operation, model))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, 'run', simulate_cli)
    first = runtime.run_pilot(plan, 'jhtdb', manifest, output)
    assert calls == [('predict', 'pod'), ('predict', 'rbf')]
    predictions = sorted((output / 'smoke_predictions').iterdir())
    assert len(predictions) == 4
    published = {path: _sha(path) for path in predictions}
    second = runtime.run_pilot(plan, 'jhtdb', manifest, output)
    assert calls == [('predict', 'pod'), ('predict', 'rbf'),
                     ('verify', 'pod'), ('verify', 'rbf')]
    assert {path: _sha(path) for path in preserved} == before
    assert {path: _sha(path) for path in predictions} == published
    assert not (output / 'pod_fit_smoke.npz').exists()
    assert _read(output / 'PILOT_COMPLETE.json') == second
    for result in (first, second):
        assert result['status'] == 'completed'
        assert result['test_used'] is False
        assert {row['model'] for row in result['benchmark']} == set(plan['learned_models'])
        assert all(row['checkpoint_reused'] is True for row in result['benchmark'])
        assert all(row['elapsed_scope'] == 'checkpoint_verification'
                   for row in result['benchmark'])
        assert all(row['elapsed_seconds'] >= 0 for row in result['benchmark'])


def test_prepared_recovery_plan_keeps_seed31_and_original_provenance():
    plan = _read(ROOT / 'plan.json')
    original_root = Path(plan['recovery']['original_experiment'])
    original_path = original_root / 'plan.json'
    original = _read(original_path)
    frozen = _read(original_root / 'frozen_manifest.json')
    assert original_root != ROOT
    assert _sha(original_path) == frozen['files']['plan.json']
    assert original['training_seeds'] == [31, 32, 33]
    assert plan['training_seeds'] == [31]
    assert set(plan['datasets']) == {'jhtdb'}
    assert plan['result_root'] == original['result_root']
    assert plan['datasets']['jhtdb'] == original['datasets']['jhtdb']
    assert plan['recovery']['reuse_completed_pilot_checkpoints'] is True
    assert plan['recovery']['completed_preparation_job_id'] == '3477662'
    assert plan['recovery']['failed_pilot_job_id'] == '3487230'

    jobs = plan['jobs']
    assert len(jobs) == plan['sbatch_submission_count'] == 10
    assert jobs[0]['key'] == 'jhtdb_pilot'
    assert jobs[0]['stage'] == 'pilot'
    assert jobs[0]['dependencies'] == []
    assert not any(job['stage'] == 'prepare' for job in jobs)
    original_jobs = {job['key']: job for job in original['jobs']}
    for previous, job in zip(jobs, jobs[1:]):
        assert ['afterok', 'jhtdb_pilot'] in job['dependencies']
        assert any(key == previous['key'] for _, key in job['dependencies'])
        assert job['tasks'] == [task for task in original_jobs[job['key']]['tasks']
                                if task['seed'] == 31]
        assert len(job['tasks']) == 1
    tasks = [task for job in jobs for task in job.get('tasks', [])]
    assert all(task['seed'] == 31 for task in tasks)
    assert {task['model'] for task in tasks if task['stage'] == 'train'} == set(plan['learned_models'])
    assert sum(task['stage'] == 'train' for task in tasks) == 5
