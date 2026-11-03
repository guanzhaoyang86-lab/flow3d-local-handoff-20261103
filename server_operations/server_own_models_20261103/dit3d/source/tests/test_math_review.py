"""Independent invariants used in the baseline mathematical code review."""

import pytest
import torch
from torch import nn

from flow_baselines.linear import PODField, fit_pod
from flow_baselines.models import SpectralConv3D
from flow_baselines.pinn import PINNField


@pytest.mark.parametrize("shape", [(5, 6, 7), (6, 5, 8)])
def test_spectral_map_commutes_with_periodic_spatial_translation(shape):
    """A Fourier multiplier must remain a convolution with arbitrary weights.

    This also exercises the conjugate-pair projection at the real-FFT boundary
    planes; identity weights alone would not detect a wrong quadrant mapping.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(613)
        model = SpectralConv3D(2, 3, modes=3).double()
        values = torch.randn(2, 2, *shape, dtype=torch.float64)
    shifts = (1, -2, 3)
    actual = model(torch.roll(values, shifts, dims=(-3, -2, -1)))
    expected = torch.roll(model(values), shifts, dims=(-3, -2, -1))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_complex_spectral_weight_gradient_matches_finite_difference():
    """Check real and imaginary parameter derivatives in all four banks."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(614)
        model = SpectralConv3D(1, 1, modes=2).double()
        values = torch.randn(1, 1, 7, 8, 9, dtype=torch.float64)
        target = torch.randn(1, 1, 7, 8, 9, dtype=torch.float64)
    objective = lambda: (model(values) - target).square().mean()
    objective().backward()
    for bank in range(4):
        for part in (0, 1):
            index = (bank, 0, 0, 1, 1, 1, part)
            analytic = model.weights.grad[index].item()
            original = model.weights[index].item()
            energies = []
            for step in (1e-6, -1e-6):
                with torch.no_grad():
                    model.weights[index] = original + step
                energies.append(objective().item())
            with torch.no_grad():
                model.weights[index] = original
            assert analytic == pytest.approx((energies[0] - energies[1]) / 2e-6,
                                             rel=2e-6, abs=2e-9)


def test_pod_truncation_error_matches_independent_singular_spectrum():
    """POD rank K must attain the Eckart-Young squared reconstruction error."""
    generator = torch.Generator().manual_seed(615)
    fields = torch.randn(7, 3, 2, 3, 2, generator=generator, dtype=torch.float64)
    fields += torch.linspace(-3, 4, fields[0].numel(), dtype=fields.dtype).reshape_as(fields[0])
    centered = (fields - fields.mean(0)).flatten(1)
    singular = torch.linalg.svdvals(centered)
    state = fit_pod(fields, rank=3)
    coefficients = centered @ state["basis"].flatten(1).T
    reconstruction = PODField(state)(coefficients)
    torch.testing.assert_close((fields - reconstruction).square().sum(),
                               singular[3:].square().sum(), rtol=1e-12, atol=1e-12)


def test_physical_poiseuille_flow_has_zero_steady_ns_residual():
    """Nonzero pressure and viscosity must balance on an anisotropic domain."""
    bounds = torch.tensor([[-2., 4.], [1., 5.], [-3., 7.]], dtype=torch.float64)
    viscosity, amplitude = 0.13, 0.7
    field = PINNField(bounds, (3, 4, 5), hidden=4, layers=1,
                      viscosity=viscosity, collocation_points=17)

    class Poiseuille(nn.Module):
        def forward(self, normalized):
            physical = bounds[:, 0] + (normalized + 1) * (bounds[:, 1] - bounds[:, 0]) / 2
            x, y, z = physical.unbind(dim=1)
            u = amplitude * (y - bounds[1, 0]) * (bounds[1, 1] - y)
            pressure = -2 * viscosity * amplitude * x
            return torch.stack((u, z * 0, z * 0, pressure), dim=1)

    field.network = Poiseuille()
    losses = field.physics_loss()
    assert losses["divergence"].item() < 1e-24
    assert losses["momentum"].item() < 1e-24


def test_rectangular_geometry_accepts_walls_and_rejects_internal_solid():
    """Do not impose fluid physics or replay through an internal obstacle."""
    from flow_baselines.common import check_rectangular_geometry

    solid = torch.zeros(5, 6, 7, dtype=torch.bool)
    solid[[0, -1], :, :] = True
    solid[:, [0, -1], :] = True
    solid[:, :, [0, -1]] = True
    check_rectangular_geometry(solid)
    solid[2, 3, 3] = True
    with pytest.raises(ValueError):
        check_rectangular_geometry(solid)
