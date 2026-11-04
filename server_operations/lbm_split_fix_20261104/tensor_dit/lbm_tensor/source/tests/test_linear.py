"""Numerical and axis-convention checks for training-only field representations."""

import pytest
import torch

from flow_baselines.linear import PODField, RBFField, fit_pod


def test_pod_low_rank_reconstruction_and_training_variance():
    generator = torch.Generator().manual_seed(17)
    raw_modes = torch.randn(3 * 2 * 3 * 4, 2, dtype=torch.float64, generator=generator)
    modes = torch.linalg.qr(raw_modes).Q.T
    coefficients = torch.tensor([[-2., -1.], [-1., 1.], [1., 1.], [2., -1.]], dtype=torch.float64)
    fields = (3 + coefficients @ modes).reshape(4, 3, 2, 3, 4)
    state = fit_pod(fields, rank=10)
    assert state["requested_rank"] == 10
    assert state["effective_rank"] == state["numerical_rank"] == 2
    assert state["training_sample_count"] == 4
    basis = state["basis"].flatten(1)
    torch.testing.assert_close(basis @ basis.T, torch.eye(2, dtype=torch.float64))
    projected = (fields - state["mean"]).flatten(1) @ basis.T
    torch.testing.assert_close(PODField(state)(projected), fields)
    torch.testing.assert_close(projected.var(0), state["eigenvalues"])
    pivots = basis.abs().argmax(1)
    assert bool((basis[torch.arange(2), pivots] > 0).all())
    torch.testing.assert_close(PODField(state).prior_regularizer(projected), torch.tensor(0.75, dtype=torch.float64))


def test_pod_fit_does_not_retain_training_autograd_or_mutable_storage():
    fields = torch.randn(5, 3, 2, 2, 2, requires_grad=True)
    state = fit_pod(fields, rank=2)
    original_mean = state["mean"].clone()
    with torch.no_grad():
        fields.add_(100)
    torch.testing.assert_close(state["mean"], original_mean)
    assert not state["mean"].requires_grad
    assert not state["basis"].requires_grad
    coeff = torch.zeros(1, 2, requires_grad=True)
    model = PODField(state)
    model(coeff).square().sum().backward()
    assert coeff.grad is not None and bool(torch.isfinite(coeff.grad).all())
    assert list(model.parameters()) == []


@pytest.mark.parametrize("sample_count,rank", [(1, 32), (3, 32), (4, 32), (4, 0)])
def test_pod_constant_and_zero_rank_are_usable(sample_count, rank):
    fields = torch.full((sample_count, 3, 2, 3, 4), 0.1, dtype=torch.float64)
    state = fit_pod(fields, rank)
    assert state["effective_rank"] == 0
    model = PODField(state)
    coeff = torch.empty(1, 0, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(model(coeff), fields[:1])
    (model(coeff).sum() + model.prior_regularizer(coeff)).backward()
    assert coeff.grad is not None and coeff.grad.numel() == 0


def test_pod_requested_rank_zero_keeps_mean_on_varying_data():
    fields = torch.randn(4, 3, 2, 2, 2)
    state = fit_pod(fields, 0)
    assert state["effective_rank"] == 0
    assert state["numerical_rank"] == 3
    torch.testing.assert_close(PODField(state)(torch.empty(1, 0)), fields.mean(0, keepdim=True))


@pytest.mark.parametrize("rank", [-1, 1.5, True])
def test_pod_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="rank"):
        fit_pod(torch.zeros(2, 3, 2, 2, 2), rank)


def test_pod_rejects_nonfinite_and_invalid_state():
    fields = torch.zeros(2, 3, 2, 2, 2)
    fields[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_pod(fields)
    with pytest.raises(ValueError, match="positive"):
        PODField({"mean": torch.zeros(1, 3, 2, 2, 2), "basis": torch.ones(1, 3, 2, 2, 2), "eigenvalues": torch.zeros(1)})


def test_rbf_constant_field_has_zero_penalty_and_parameter_gradients():
    model = RBFField([[0, 2], [10, 20], [-1, 1]], (2, 3, 4))
    assert model.coefficients.shape == (3, 65)
    with torch.no_grad():
        model.coefficients[:, 0].copy_(torch.tensor([0.3, -0.8, 1.7]))
    expected = torch.tensor([0.3, -0.8, 1.7]).reshape(1, 3, 1, 1, 1).expand(1, 3, 2, 3, 4)
    torch.testing.assert_close(model(), expected)
    assert model.regularization() == 0
    model().square().mean().backward()
    assert model.coefficients.grad is not None
    assert bool(torch.isfinite(model.coefficients.grad).all())
    assert bool((model.coefficients.grad[:, 1:] != 0).all())


def test_rbf_center_and_grid_axis_order_on_anisotropic_domain():
    model = RBFField(torch.tensor([[2., 12.], [-3., 3.], [100., 102.]], dtype=torch.float64), (3, 4, 5), centers_per_axis=2, length_scale=0.2)
    # Center ordering is Z,Y,X nested; index 1 is (xmax,ymin,zmin).
    torch.testing.assert_close(model.centers[1], torch.tensor([12., -3., 100.], dtype=torch.float64))
    with torch.no_grad():
        model.coefficients[0, 2] = 1  # constant column then center 1
    field = model()[0, 0]
    assert field[0, 0, -1] == 1
    assert field[0, 0, 0] < 1e-5
    assert field[-1, 0, -1] < 1e-5
    assert field[0, -1, -1] < 1e-5
    torch.testing.assert_close(model.regularization(), torch.tensor(1e-4, dtype=torch.float64))


def test_rbf_single_center_is_centered_and_buffers_follow_dtype():
    model = RBFField([[0, 1], [0, 1], [0, 1]], (3, 3, 3), centers_per_axis=1).double()
    torch.testing.assert_close(model.normalized_centers, torch.full((1, 3), 0.5, dtype=torch.float64))
    assert model().dtype == torch.float64
    assert model.design[1].reshape(3, 3, 3)[1, 1, 1] == 1


@pytest.mark.parametrize("kwargs", [{"centers_per_axis": 0}, {"length_scale": 0}, {"ridge": -1}, {"spatial_shape": (0, 2, 2)}])
def test_rbf_rejects_invalid_configuration(kwargs):
    arguments = {"bounds": [[0, 1], [0, 1], [0, 1]], "spatial_shape": (2, 2, 2)}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        RBFField(**arguments)
