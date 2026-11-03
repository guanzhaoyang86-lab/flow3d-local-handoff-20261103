"""Loss-scale recovery must not turn an overflowing step into an EMA update."""
import json

import pytest
import torch

from flow_observation.dps import training


def _args(manifest, output, epochs=1, resume=None):
    values = ['--manifest', str(manifest), '--output-dir', str(output), '--device', 'cpu',
        '--architecture', 'dit3d', '--epochs', str(epochs), '--dit-hidden-dim', '24',
        '--dit-depth', '1', '--dit-heads', '3', '--patch-size', '2', '--diffusion-steps', '8']
    if resume is not None:
        values += ['--resume', str(resume)]
    return training.parser().parse_args(values)


def test_amp_overflow_skips_step_backs_off_and_next_step_recovers():
    # The CPU GradScaler exercises the same overflow bookkeeping without CUDA.
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.)
    with torch.autocast('cpu', dtype=torch.float16):
        loss = model(torch.ones(1, 1)).float().square().sum()
    assert torch.isfinite(loss)
    assert not training._optimizer_step(loss, model, optimizer, scaler, 1.)
    assert model.weight.item() == 1.
    assert scaler.get_scale() == 32768.
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.float16):
        loss = model(torch.ones(1, 1)).float().square().sum() / 4
    assert training._optimizer_step(loss, model, optimizer, scaler, 1.)
    assert model.weight.item() < 1.


def test_non_amp_nonfinite_gradient_remains_fatal():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    # A finite forward value can have a nonfinite derivative at zero.
    loss = (model.weight * 0).sqrt().sum()
    assert torch.isfinite(loss)
    with pytest.raises(RuntimeError, match='non-finite'):
        training._optimizer_step(loss, model, optimizer, scaler, 1.)


def test_skipped_updates_leave_ema_unchanged_and_cannot_complete(small_manifest, tmp_path, monkeypatch):
    output = tmp_path / 'skipped'
    monkeypatch.setattr(training, '_optimizer_step', lambda *args: False)
    with pytest.raises(RuntimeError, match='no successful optimizer updates'):
        training.train(_args(small_manifest, output))
    checkpoint = torch.load(output / 'latest.pt', weights_only=True)
    assert checkpoint['successful_optimizer_steps'] == 0
    assert checkpoint['skipped_amp_steps'] == 2
    assert all(torch.equal(checkpoint['model'][key], checkpoint['ema_model'][key])
               for key in checkpoint['model'])
    assert json.loads((output / 'run.json').read_text())['status'] != 'completed'


def test_optimizer_step_counts_survive_resume(small_manifest, tmp_path):
    output = tmp_path / 'resume'
    training.train(_args(small_manifest, output))
    training.train(_args(small_manifest, output, epochs=2, resume=output / 'latest.pt'))
    checkpoint = torch.load(output / 'latest.pt', weights_only=True)
    assert checkpoint['successful_optimizer_steps'] == 4
    assert checkpoint['skipped_amp_steps'] == 0
    record = json.loads((output / 'run.json').read_text())
    assert record['successful_optimizer_steps'] == 4
    assert record['skipped_amp_steps'] == 0
