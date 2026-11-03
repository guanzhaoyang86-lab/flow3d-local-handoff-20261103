"""Regression checks for geometry scope and inference provenance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow_baselines.common import check_rectangular_geometry
from flow_baselines.inference import predict
from flow_baselines.training import train
from flow_observation.sparse_dataset import load_sparse_flow_manifest
from test_pipeline import tiny


@pytest.mark.parametrize('mask', [torch.zeros(2, 3, dtype=torch.bool),
    torch.zeros(0, 3, 4, dtype=torch.bool), torch.zeros(3, 3, 3),
    torch.ones(2, 2, 2, dtype=torch.bool)])
def test_geometry_rejects_invalid_masks(mask):
    with pytest.raises(ValueError):
        check_rectangular_geometry(mask)


def _add_internal_obstacle(path):
    with np.load(path, allow_pickle=False) as data:
        values = {key:data[key] for key in data.files
                  if key not in ('initial_hidden_depth','trajectories_3d')}
    values['solid_mask'][3, 3, 3] = True
    np.savez(path, **values)


@pytest.mark.parametrize('entry', ['train', 'predict', 'fit_case'])
def test_entry_points_reject_internal_obstacles(entry, small_manifest, tmp_path):
    _, splits = load_sparse_flow_manifest(small_manifest)
    if entry == 'fit_case':
        from flow_baselines.case_fit import fit_case
        mask = torch.zeros(8, 8, 8, dtype=torch.bool)
        mask[3, 3, 3] = True
        with pytest.raises(ValueError, match='Internal solid obstacles'):
            fit_case('rbf', {'solid_mask':mask}, tiny(), SimpleNamespace())
    elif entry == 'train':
        _add_internal_obstacle(splits['train'][0].path)
        with pytest.raises(ValueError, match='Internal solid obstacles'):
            train('mean', small_manifest, tmp_path/'mean', tiny(), device='cpu')
    else:
        _add_internal_obstacle(splits['validation'][0].path)
        with pytest.raises(ValueError, match='Internal solid obstacles'):
            predict(small_manifest, tmp_path/'rbf.npz', kind='rbf', config=tiny(), device='cpu')


@pytest.mark.parametrize('name,epoch,selection', [
    ('best.pt', 1, 'best_validation'),
    ('epoch_0001.pt', 1, 'explicit_epoch'),
    ('latest.pt', 2, 'final_epoch')])
def test_checkpoint_selection_records_actual_epoch(name, epoch, selection, small_manifest, tmp_path):
    checkpoint = train('cnn', small_manifest, tmp_path/'cnn', tiny(), device='cpu')
    state = torch.load(checkpoint, weights_only=True)
    state['config']['epochs'], state['epoch'] = 2, epoch
    selected = tmp_path/name
    torch.save(state, selected)
    output = predict(small_manifest, tmp_path/'out.npz', checkpoint=selected, device='cpu')
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['checkpoint_epoch'] == epoch
    assert metadata['checkpoint_selection'] == selection


def test_fixed_field_depth_fit_uses_requested_learning_rate(small_manifest, tmp_path, monkeypatch):
    from flow_baselines import inference
    captured = {}
    def diagnostic(field, observation, **kwargs):
        captured.update(kwargs)
        return observation.initialize_depth(torch.Generator().manual_seed(kwargs['seed']))
    monkeypatch.setattr(inference, 'fit_depth_for_fixed_field', diagnostic)
    cfg = tiny(depth_learning_rate=.037)
    checkpoint = train('mean', small_manifest, tmp_path/'mean', cfg, device='cpu')
    output = predict(small_manifest, tmp_path/'out.npz', checkpoint=checkpoint, device='cpu')
    assert captured == {'steps':100, 'learning_rate':.037, 'seed':cfg['sampling_seed']}
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['fixed_field_depth_fit'] == {
        'max_steps':100, 'learning_rate':.037, 'seed':cfg['sampling_seed']}
