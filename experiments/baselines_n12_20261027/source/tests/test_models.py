import math
import pytest
import torch
from flow_baselines.models import SpectralConv3D, build_decoder, coordinate_grid
torch.set_num_threads(1)

CONFIG = {"unet": {"base_channels": 4, "channel_multipliers": (1, 2)},
          "cnn": {"width": 4, "depth": 2},
          "fno": {"width": 4, "modes": 2, "depth": 2, "padding": 1},
          "deeponet": {"latent": 6, "hidden": 8, "layers": 2}}

@pytest.mark.parametrize("kind", CONFIG)
def test_finite_condition_sensitive_forward_backward(kind):
    torch.manual_seed(18)
    model = build_decoder(kind, (5, 6, 7), condition_dim=5, **CONFIG[kind])
    condition = torch.randn(2, 5, requires_grad=True)
    prediction = model(condition)
    assert prediction.shape == (2, 3, 5, 6, 7)
    assert torch.isfinite(prediction).all()
    assert not torch.allclose(prediction[0], prediction[1])
    (prediction - torch.randn_like(prediction)).square().mean().backward()
    assert torch.isfinite(condition.grad).all()
    assert condition.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_xyz_channels_and_zyx_storage():
    grid = coordinate_grid((3, 4, 5))
    torch.testing.assert_close(grid[0, 0, 1, 2], torch.linspace(-1, 1, 5))
    torch.testing.assert_close(grid[0, 1, 1, :, 2], torch.linspace(-1, 1, 4))
    torch.testing.assert_close(grid[0, 2, :, 2, 2], torch.linspace(-1, 1, 3))

@pytest.mark.parametrize("shape", [(3, 5, 7), (4, 6, 8), (1, 2, 3)])
def test_full_spectrum_identity_odd_even_small(shape):
    model = SpectralConv3D(1, 1, modes=5)
    with torch.no_grad():
        model.weights.zero_()
        model.weights[..., 0].fill_(1)
    values = torch.randn(2, 1, *shape)
    torch.testing.assert_close(model(values), values, rtol=1e-5, atol=1e-6)

def test_all_spectral_quadrants_have_gradients():
    model = SpectralConv3D(2, 3, modes=(2, 2, 3))
    x = torch.randn(2, 2, 7, 8, 9, requires_grad=True)
    (model(x) * torch.randn(2, 3, 7, 8, 9)).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for bank in model.weights.grad:
        assert torch.isfinite(bank).all() and bank.abs().sum() > 0

def test_fno_cpu_amp_fft_backward():
    model = build_decoder("fno", (5, 6, 7), condition_dim=5, **CONFIG["fno"])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prediction = model(torch.randn(2, 5))
        loss = prediction.float().square().mean()
    assert prediction.dtype == torch.bfloat16
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

def test_truncated_spectrum_filters_high_frequency_on_each_spatial_axis():
    model = SpectralConv3D(1, 1, modes=2)
    with torch.no_grad():
        model.weights.zero_()
        model.weights[..., 0].fill_(1)
    shape = (8, 10, 12)
    ones = torch.ones(1, 1, *shape)
    torch.testing.assert_close(model(ones), ones)
    for axis, size in enumerate(shape):
        signal = torch.cos(2 * math.pi * 3 * torch.arange(size) / size)
        dimensions = [1, 1, 1, 1, 1]
        dimensions[axis + 2] = size
        assert model(signal.reshape(dimensions).expand_as(ones)).abs().max() < 2e-6

@pytest.mark.parametrize("kind", CONFIG)
def test_optimizer_step_improves_fixed_synthetic_task(kind):
    torch.manual_seed(23)
    model = build_decoder(kind, (3, 3, 4), condition_dim=5, **CONFIG[kind])
    condition = torch.randn(2, 5)
    target = coordinate_grid((3, 3, 4)).expand(2, -1, -1, -1, -1) * condition[:, 0, None, None, None, None]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = (model(condition) - target).square().mean()
    before.backward()
    optimizer.step()
    assert (model(condition) - target).square().mean() < before.detach()

def test_configuration_and_condition_errors_are_explicit():
    with pytest.raises(ValueError, match="unsupported"):
        build_decoder("fno", (4, 4, 4), modse=4)
    with pytest.raises(ValueError, match="positive integer"):
        build_decoder("cnn", (4, 4, 4), depth=0)
    model = build_decoder("cnn", (3, 3, 3), condition_dim=5, **CONFIG["cnn"])
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2, 6))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 5), float("nan")))
