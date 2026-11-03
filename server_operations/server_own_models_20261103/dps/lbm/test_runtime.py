import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('dps_runtime', Path(__file__).with_name('runtime.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)

def test_lbm_adoption_requires_complete_counts_and_reviewed_adapter(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    marker = tmp_path / 'COMPLETE.json'
    config = {'dataset': 'lbm', 'manifest': str(manifest), 'completion_marker': str(marker),
              'deferred_dataset_adoption': True, 'expected_adapter_sha256': 'reviewed',
              'completion_requirements': {'verified_fields': 3125, 'physical_trajectories': 500000, 'arrays_modified': False}}
    runtime.write(tmp_path / 'config.json', config)
    runtime.write(tmp_path / 'frozen_manifest.json', {'files': {}, 'manifest_sha256': None})
    completed = {'status': 'completed', 'manifest_sha256': runtime.sha(manifest), 'adapter_sha256': 'reviewed',
                 'verified_fields': 3125, 'physical_trajectories': 500000, 'arrays_modified': False}
    monkeypatch.setattr(runtime, 'HERE', tmp_path)
    runtime.write(marker, completed)
    assert runtime.verify_inputs(False)['dataset'] == 'lbm'
    for key, bad in [('status', 'running'), ('adapter_sha256', 'other'), ('verified_fields', 1000),
                     ('physical_trajectories', 100000), ('arrays_modified', True), ('manifest_sha256', 'wrong')]:
        runtime.write(marker, dict(completed, **{key: bad}))
        with pytest.raises(ValueError):
            runtime.verify_inputs(False)

def test_lbm_does_not_add_lid_or_boundary_conditions():
    config = runtime.read(Path(__file__).with_name('config.json'))
    assert config['dataset'] == 'lbm' and config['sampling']['boundary_projection'] == 'none'
    assert (runtime.SOURCE / 'src/flow_observation/dps/inference.py').read_bytes() == (runtime.PACKAGE / 'static_source/src/flow_observation/dps/inference.py').read_bytes()

def test_single_xz_posthoc_needs_no_3d_mask_and_reports_depth_unidentified(tmp_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from flow_observation.dps.inference import _posthoc_evaluate
    path = tmp_path / 'case.npz'
    np.savez(path, flow_field=np.full((1,3,4,4,4), .1, dtype=np.float32),
             solid_mask=np.zeros((4,4,4), dtype=bool),
             trajectories_2d=np.ones((1,1,44,3,2), dtype=np.float32),
             observation_mask=np.ones((1,1,44,3), dtype=bool),
             observation_times=np.array([0.,.1,.2], dtype=np.float64),
             projection_matrix=np.array([[[1,0,0],[0,0,1]]], dtype=np.float32),
             domain_bounds=np.array([[0,3],[0,3],[0,3]], dtype=np.float32),
             metadata=np.array('{}'))
    args = SimpleNamespace(split='test', num_particles=12, num_probes=32, observation_seed=47,
                           observation_type='projected', trajectory_substeps=4)
    result = _posthoc_evaluate(torch.full((2,3,4,4,4), .1), SimpleNamespace(path=path, case_id='fixture', flow_group_id='flow_fixture'), args)
    assert result['posterior_mean_field_relative_l2_unknown_interior'] == 0
    assert len(result['probe_metrics']) == 2
    assert all(row['probe_track_rmse_cells'] is None and 'depth' in row['reason'] for row in result['probe_metrics'])

def test_real_cli_accepts_commands(tmp_path):
    from flow_observation.dps.training import parser as training_parser
    from flow_observation.dps.inference import parser as sampling_parser
    config = runtime.read(Path(__file__).with_name('config.json'))
    train = training_parser().parse_args([str(x) for x in runtime.training_command(config, tmp_path)[3:]])
    sample = sampling_parser().parse_args([str(x) for x in runtime.sampling_command(config, 'validation', 0, .05, tmp_path/'case.npz')[3:]])
    assert train.seed == 31 and train.epochs == 100 and train.architecture == 'dit3d' and train.amp
    assert sample.num_particles == 12 and sample.num_probes == 32 and sample.views == '0'
    assert sample.num_samples == 2 and sample.sampling_steps == 50 and sample.seed == 47 and sample.evaluate

def test_resume_history_keeps_only_committed_rows(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'fixture')
    path = tmp_path / 'history.jsonl'
    original = '\n'.join(json.dumps({'epoch': i}) for i in (1, 2, 3)) + '\n{"epoch":'
    path.write_text(original)
    runtime.restore_committed_history(path, 2)
    assert [json.loads(row)['epoch'] for row in path.read_text().splitlines()] == [1, 2]
    assert (tmp_path / 'history_before_resume_fixture.jsonl').read_text() == original

def test_existing_posterior_is_verified_not_resampled(tmp_path, monkeypatch):
    output = tmp_path / 'case.npz'; output.write_bytes(b'fixture')
    monkeypatch.setattr(runtime, 'verify_prediction', lambda *args: 'verified')
    monkeypatch.setattr(runtime, 'run', lambda *args: pytest.fail('resampled existing posterior'))
    assert runtime.predict({}, 'validation', 0, .05, output) == 'verified'

def test_orphan_metadata_is_not_overwritten(tmp_path):
    output = tmp_path / 'case.npz'; output.with_suffix('.json').write_text('{}')
    with pytest.raises(ValueError, match='Orphan'):
        runtime.predict({}, 'validation', 0, .05, output)

def test_sigma_calibration_never_uses_test_and_ties_choose_smallest(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    (tmp_path / 'prior_s31').mkdir(); (tmp_path / 'prior_s31/latest.pt').write_bytes(b'fixture')
    config = {'dataset': 'jhtdb', 'manifest': str(manifest), 'result_root': str(tmp_path), 'sampling': {'num_samples': 2},
              'calibration': {'sigma_candidates': [.1, .02, .05], 'case_indices': [0, 1], 'selection_metric': 'error'}}
    monkeypatch.setattr(runtime, 'verify_prior', lambda *args: None)
    calls = []
    def predict(config, split, index, sigma, output):
        calls.append(split)
        return {'posthoc_evaluation': {'error': .3}, 'case_id': str(index), 'output_sha256': 'fixture'}
    monkeypatch.setattr(runtime, 'predict', predict)
    runtime.calibrate(config)
    path = tmp_path / 'validation_calibration/selected_config.json'
    selected = runtime.read(path)
    assert selected['sigma'] == .02 and selected['test_used'] is False and set(calls) == {'validation'}
    assert runtime.read(path.parent / 'CALIBRATION_COMPLETE.json')['selected_config_sha256'] == runtime.sha(path)

def test_nonfinite_validation_metric_does_not_publish_completion(tmp_path, monkeypatch):
    config = {'result_root': str(tmp_path), 'calibration': {'sigma_candidates': [.05], 'case_indices': [0], 'selection_metric': 'error'}}
    monkeypatch.setattr(runtime, 'verify_prior', lambda *args: None)
    monkeypatch.setattr(runtime, 'predict', lambda *args: {'posthoc_evaluation': {'error': float('nan')}})
    with pytest.raises(ValueError, match='finite'):
        runtime.calibrate(config)
    assert not (tmp_path / 'validation_calibration/CALIBRATION_COMPLETE.json').exists()
