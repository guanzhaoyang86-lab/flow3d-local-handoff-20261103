import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'launch' / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_n12_task_counts_roles_and_budget():
    launch = module('submission_counts', 'submit_batch.py')
    plan = launch.build_plan()
    assert plan['logical_allocations'] == 42
    assert plan['sbatch_submission_count'] == 22
    assert plan['requested_gpu_hour_upper_bound'] == 274
    assert plan['requested_gpu_hour_upper_bound'] < plan['initial_account_balance_gpu_hours']
    tasks = [task for job in plan['jobs'] for task in job.get('tasks', [])]
    assert sum(task['stage'] == 'train' for task in tasks) == 30
    assert sum(task['stage'] == 'fit_statistics' for task in tasks) == 4
    assert sum(task['stage'] == 'calibration' for task in tasks) == 4
    assert all(t['N'] == 12 and t['model'] not in ('pinn', 'dit3d', 'tensor_dit') for t in tasks)
    assert all(t['epochs'] == 0 for t in tasks if t['stage'] == 'fit_statistics')


def test_dependency_chain_limits_each_dataset_to_one_allocation():
    launch = module('submission_dependencies', 'submit_batch.py')
    plan = launch.build_plan()
    for dataset in plan['datasets']:
        previous = None
        for job in [j for j in plan['jobs'] if j['dataset'] == dataset]:
            deps = job['dependencies']
            if previous is not None:
                assert any(key == previous for _, key in deps)
            if job['stage'] not in ('prepare', 'pilot'):
                assert ['afterok', dataset + '_pilot'] in deps
            previous = job['key']
    accepted = {j['key']: str(10000 + i) for i, j in enumerate(plan['jobs'])}
    for dataset in plan['datasets']:
        rbf = next(j for j in plan['jobs'] if j['key'] == dataset + '_rbf_calibration')
        assert ['afterany', dataset + '_unet_diffusion'] in rbf['dependencies']
    for job in plan['jobs']:
        command = launch.command(job, accepted)
        assert '--gpus-per-node=1' in command
        if len(job.get('tasks', [])) == 3:
            assert '--array=0-2%1' in command
        if job['dependencies']:
            assert any(arg.startswith('--dependency=') for arg in command)
        assert not any(arg.startswith('--dependency=') for arg in launch.command(job, test_only=True))


def test_checkpoints_require_real_completed_epoch_and_manifest(tmp_path):
    runtime = module('runtime_checkpoints', 'batch_runtime.py')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    output = tmp_path / 'run'
    output.mkdir()
    state = {'model': 'cnn', 'epoch': 100, 'manifest_sha256': runtime.sha(manifest), 'optimizer_steps': 1}
    torch.save(state, output / 'latest.pt')
    (output / 'run.json').write_text(json.dumps({'status': 'completed'}))
    assert runtime.completed_checkpoint(output, 'cnn', 100, manifest)['epoch'] == 100
    state['epoch'] = 99
    torch.save(state, output / 'latest.pt')
    with pytest.raises(RuntimeError, match='epoch'):
        runtime.completed_checkpoint(output, 'cnn', 100, manifest)
    state.update(epoch=100, optimizer_steps=0)
    torch.save(state, output / 'latest.pt')
    with pytest.raises(RuntimeError, match='optimization'):
        runtime.completed_checkpoint(output, 'cnn', 100, manifest)


def test_prediction_resume_validates_case_and_provenance(tmp_path):
    runtime = module('runtime_resume', 'batch_runtime.py')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'splits': {'validation': [{'case_id': 'v0'}]}}))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'sigma': .05, 'num_particles': 12}))
    target = tmp_path / 'prediction.npz'
    target.write_bytes(b'prediction-artifact')
    metadata = {'model': 'rbf', 'config': json.loads(config.read_text()), 'manifest_sha256': runtime.sha(manifest),
                'split': 'validation', 'case_id': 'v0', 'output_sha256': runtime.sha(target)}
    target.with_suffix('.json').write_text(json.dumps(metadata))
    command = runtime.predict_command('jhtdb', 'rbf', config, target, manifest, 31, 0)
    assert 'verify' in command
    metadata['case_id'] = 'another-case'
    target.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='does not match'):
        runtime.predict_command('jhtdb', 'rbf', config, target, manifest, 31, 0)


def test_fs_evaluation_resume_and_truth_independent_task_command(tmp_path):
    runtime = module('runtime_fs_command', 'batch_runtime.py')
    command = runtime.predict_command('fs_sph', 'rbf', tmp_path / 'config.json', tmp_path / 'output',
                                      tmp_path / 'manifest.json', 31, 0)
    assert '--resume' in command and '--case-indices' in command
    assert '--split' in command and command[command.index('--split') + 1] == 'validation'
    assert not any('probe' in str(arg) or 'trajectories_3d' in str(arg) for arg in command)
