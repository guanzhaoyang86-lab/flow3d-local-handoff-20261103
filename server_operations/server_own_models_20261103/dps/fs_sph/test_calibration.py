import pytest
import runtime as r

def test_validation_only_selection_and_test_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'HERE', tmp_path)
    (tmp_path / 'frozen_manifest.json').write_text('{}')
    checkpoint = tmp_path / 'prior.pt'; checkpoint.write_bytes(b'synthetic')
    config = {'result_root': str(tmp_path / 'results'), 'manifest_sha256': 'manifest', 'sampling': {'seed': 47}, 'calibration': {'case_indices': [0, 1], 'sigma_candidates': [.02, .05, .1], 'split': 'validation', 'test_used': False}}
    monkeypatch.setattr(r, 'load_model', lambda *a: (checkpoint, {}, None, None))
    calls = []
    def predict(config, state, digest, splits, split, index, sigma, output, model, diffusion):
        calls.append((split, index, sigma))
        return {'output_sha256': 'synthetic', 'posthoc_evaluation': {'posterior_mean': {'physical_velocity_relative_l2': 1.0}}}
    monkeypatch.setattr(r, 'predict', predict)
    r.calibrate(config, {}, {})
    assert len(calls) == 6 and all(x[0] == 'validation' for x in calls)
    path = tmp_path / 'results/selected_config.json'; selection = r.read_json(path)
    assert selection['status'] == 'frozen' and selection['sigma'] == .02 and selection['test_used'] is False
    r.calibrate(config, {}, {}); assert len(calls) == 6
    selection['sampling']['seed'] = 99; r.write_json(path, selection)
    r.write_json(path.parent / 'CALIBRATION_COMPLETE.json', {'selected_config_sha256': r.sha256(path)})
    with pytest.raises(RuntimeError, match='unchanged validation-only selection'):
        r.evaluate_test(config, {}, {'test': []})
    assert len(calls) == 6
