"""Independent regressions for primary-only training and interrupted resume."""
import json

import pytest
import torch
from torch import nn

from flow_baselines.config import resolve_config
from flow_baselines.training import _loss, train


def _tiny(epochs=1):
    return resolve_config(overrides={
        'epochs': epochs, 'amp': False, 'condition_dim': 8, 'num_probes': 1,
        'condition_flow_grid': 2, 'cnn': {'width': 4, 'depth': 1},
    })


class _PrimaryOnlyEncoder(nn.Module):
    def forward(self, tracks, mask, projection, times, bounds):
        return tracks.flatten(1)[:, :1]

    def predict_condition_flow(self, condition):
        raise AssertionError('Disabled auxiliary target was evaluated')


class _PrimaryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.))

    def forward(self, condition):
        return self.value.expand(len(condition), 3, 2, 2, 2)


def test_zero_auxiliary_weight_keeps_primary_objective_and_gradients():
    system = nn.ModuleDict({'encoder': _PrimaryOnlyEncoder(), 'decoder': _PrimaryDecoder()})
    batch = {
        'tracks': torch.zeros(1, 1, 1, 2, 2),
        'mask': torch.ones(1, 1, 1, 2, dtype=torch.bool),
        'projection': torch.zeros(1, 1, 2, 3),
        'times': torch.tensor([[0., 1.]]),
        'bounds': torch.tensor([[[0., 1.]]*3]),
        'flow': torch.zeros(1, 3, 2, 2, 2),
        'solid_mask': torch.zeros(1, 2, 2, 2, dtype=torch.bool),
        # There is no unknown region for the unused auxiliary flow target.
        'boundary_mask': torch.ones(1, 2, 2, 2, dtype=torch.bool),
    }
    loss = _loss(system, batch, 'cnn', {'condition_flow_weight': 0.})
    assert loss.item() == 1.
    loss.backward()
    assert system['decoder'].value.grad.item() == 2.


@pytest.mark.parametrize('partial_tail', [False, True])
def test_resume_discards_uncommitted_history_without_changing_weights(
        small_manifest, tmp_path, partial_tail):
    complete = train('cnn', small_manifest, tmp_path/'complete', _tiny(2), device='cpu')
    partial = train('cnn', small_manifest, tmp_path/'partial', _tiny(), device='cpu')
    history = partial.parent/'history.jsonl'
    if partial_tail:
        extra = '{"epoch": 2, "train_loss": '
    else:
        extra = json.dumps({'epoch': 2, 'train_loss': 999., 'validation_loss': 999.})+'\n'
    with history.open('a') as stream:
        stream.write(extra)
    resumed = train('cnn', small_manifest, partial.parent, _tiny(2), device='cpu', resume=partial)
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row['epoch'] for row in rows] == [1, 2]
    assert rows[-1]['train_loss'] != 999.
    uninterrupted = torch.load(complete, map_location='cpu', weights_only=True)
    restored = torch.load(resumed, map_location='cpu', weights_only=True)
    for key in ('system', 'ema_system'):
        assert all(torch.equal(uninterrupted[key][name], restored[key][name])
                   for name in uninterrupted[key])


def test_fresh_training_refuses_to_mix_with_existing_history(small_manifest, tmp_path):
    directory = tmp_path/'stale'
    directory.mkdir()
    (directory/'history.jsonl').write_text('{"epoch": 37}\n')
    with pytest.raises(ValueError, match='artifacts exist'):
        train('cnn', small_manifest, directory, _tiny(), device='cpu')
    assert (directory/'history.jsonl').read_text() == '{"epoch": 37}\n'
