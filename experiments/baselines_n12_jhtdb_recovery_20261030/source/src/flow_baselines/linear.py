"""Training-only POD and Gaussian RBF field representations.

All velocity arrays use ``[batch, 3, D, H, W]`` (components XYZ, grid ZYX).
This module deliberately knows nothing about observations or evaluation cases.
The RBF model is a field parameterization for an *adapted projected trajectory
fitting* baseline, not a claim that classical RBF interpolation resolves unknown
particle depths without an inverse-problem solver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def fit_pod(fields: Tensor, rank: int = 32) -> dict[str, Any]:
    """Fit centered snapshot POD to the supplied training fields only.

    The snapshot Gram eigensystem is evaluated in float64. ``basis`` consists
    of orthonormal spatial modes; a coefficient's sample variance is the
    corresponding ``eigenvalues`` entry. Requested ranks above the centered
    numerical rank are truncated explicitly. One sample and constant training
    data yield an empty basis and a usable mean field.
    """
    rank = _nonnegative_integer(rank, "rank")
    if not isinstance(fields, Tensor) or fields.ndim != 5 or fields.shape[1] != 3:
        raise ValueError("fields must have shape [M, 3, D, H, W]")
    if fields.shape[0] == 0 or any(size == 0 for size in fields.shape[2:]):
        raise ValueError("fields must contain at least one nonempty training field")
    if not fields.is_floating_point() or not bool(torch.isfinite(fields).all()):
        raise ValueError("fields must be finite floating point values")
    output_dtype = fields.dtype if fields.dtype in (torch.float32, torch.float64) else torch.float32
    shape = tuple(fields.shape[1:])
    with torch.no_grad():
        flat = fields.detach().to(torch.float64).flatten(1)
        # Subtract one sample before summation: identical decimal-valued
        # float64 snapshots must not acquire a spurious mode from mean roundoff.
        reference = flat[:1]
        mean = reference + (flat - reference).mean(0, keepdim=True)
        centered = flat - mean
        sample_count, feature_count = centered.shape
        basis = centered.new_empty((0, feature_count))
        eigenvalues = centered.new_empty((0,))
        total_variance = centered.square().sum() / max(sample_count - 1, 1)
        numerical_rank = 0
        if sample_count > 1 and bool(total_variance > 0):
            # Eigh orders the spectrum ascending. The tolerance removes roundoff
            # eigenvalues created by the Gram product, including the centered
            # constant vector. No numerical zero is inverted below.
            gram = centered @ centered.T
            spectrum, vectors = torch.linalg.eigh(gram)
            order = torch.argsort(spectrum, descending=True)
            spectrum, vectors = spectrum[order], vectors[:, order]
            tolerance = torch.finfo(torch.float64).eps * max(sample_count, feature_count) * spectrum[0].clamp_min(0)
            positive = spectrum > tolerance
            numerical_rank = min(int(positive.sum()), sample_count - 1)
            effective_rank = min(rank, numerical_rank)
            if effective_rank:
                selected = spectrum[:effective_rank]
                basis = (vectors[:, :effective_rank].T @ centered) / selected.sqrt()[:, None]
                # Resolve arbitrary eigenvector signs for reproducible artifacts.
                pivots = basis.abs().argmax(1)
                signs = basis[torch.arange(effective_rank, device=basis.device), pivots].sign()
                basis = basis * signs[:, None]
                eigenvalues = selected / (sample_count - 1)
        return {
            "mean": mean.reshape(1, *shape).to(output_dtype),
            "basis": basis.reshape(-1, *shape).to(output_dtype),
            "eigenvalues": eigenvalues.to(output_dtype),
            "requested_rank": rank,
            "effective_rank": int(eigenvalues.numel()),
            "numerical_rank": numerical_rank,
            "training_sample_count": sample_count,
            "total_variance": float(total_variance),
            "coefficient_convention": "orthonormal_spatial_modes_sample_variance",
        }


class PODField(nn.Module):
    """A fixed training mean and basis, driven by external modal coefficients."""

    def __init__(self, state: Mapping[str, Any]):
        super().__init__()
        mean = torch.as_tensor(state["mean"]).detach().clone()
        basis = torch.as_tensor(state["basis"], dtype=mean.dtype, device=mean.device).detach().clone()
        variance = torch.as_tensor(state["eigenvalues"], dtype=mean.dtype, device=mean.device).detach().clone()
        if mean.ndim != 5 or mean.shape[:2] != (1, 3) or any(size < 1 for size in mean.shape[2:]):
            raise ValueError("POD mean must have shape [1, 3, D, H, W]")
        if basis.ndim != 5 or basis.shape[1:] != mean.shape[1:]:
            raise ValueError("POD basis must have shape [K, 3, D, H, W]")
        if variance.shape != (basis.shape[0],):
            raise ValueError("POD eigenvalues must have shape [K]")
        if not mean.is_floating_point() or not all(bool(torch.isfinite(x).all()) for x in (mean, basis, variance)):
            raise ValueError("POD state must contain finite floating point values")
        if bool((variance <= 0).any()):
            raise ValueError("retained POD eigenvalues must be positive")
        self.register_buffer("mean", mean)
        self.register_buffer("basis", basis)
        self.register_buffer("eigenvalues", variance)
        self.rank = basis.shape[0]

    def _validate_coeff(self, coeff: Tensor) -> None:
        if coeff.ndim != 2 or coeff.shape[1] != self.rank or coeff.shape[0] < 1:
            raise ValueError(f"coeff must have shape [B, {self.rank}] with B >= 1")

    def forward(self, coeff: Tensor) -> Tensor:
        self._validate_coeff(coeff)
        return self.mean + (coeff @ self.basis.flatten(1)).reshape(coeff.shape[0], *self.mean.shape[1:])

    def prior_regularizer(self, coeff: Tensor) -> Tensor:
        """Mean squared coefficients after whitening by training modal SDs."""
        self._validate_coeff(coeff)
        if self.rank == 0:
            return coeff.sum() * 0.0
        return (coeff.square() / self.eigenvalues).mean()


class RBFField(nn.Module):
    """A constant vector plus Gaussian RBFs on a normalized unit XYZ domain.

    ``centers_per_axis=4`` gives 64 centers. ``length_scale`` is the Gaussian
    standard deviation in normalized [0, 1] coordinates, independent of the
    physical domain's aspect ratio. ``ridge`` multiplies the sum of squared RBF
    coefficients; the constant vector is unpenalized. Coefficients initialize
    to zero and can be fitted to observations by an external solver.
    """

    def __init__(
        self,
        bounds: Tensor | Sequence[Sequence[float]],
        spatial_shape: Sequence[int],
        centers_per_axis: int = 4,
        length_scale: float = 0.35,
        ridge: float = 1e-4,
    ):
        super().__init__()
        centers_per_axis = _nonnegative_integer(centers_per_axis, "centers_per_axis")
        if centers_per_axis < 1:
            raise ValueError("centers_per_axis must be at least one")
        if len(spatial_shape) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in spatial_shape):
            raise ValueError("spatial_shape must be three positive integers in D, H, W order")
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("length_scale must be finite and positive")
        if not math.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        bounds = torch.as_tensor(bounds)
        if not bounds.is_floating_point():
            bounds = bounds.to(torch.float32)
        elif bounds.dtype not in (torch.float32, torch.float64):
            bounds = bounds.to(torch.float32)
        if bounds.shape != (3, 2) or not bool(torch.isfinite(bounds).all()) or not bool((bounds[:, 1] > bounds[:, 0]).all()):
            raise ValueError("bounds must be finite [3, 2] XYZ intervals with upper > lower")
        dtype, device = bounds.dtype, bounds.device
        self.spatial_shape = tuple(int(n) for n in spatial_shape)
        self.centers_per_axis = centers_per_axis
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        axis = torch.linspace(0, 1, centers_per_axis, dtype=dtype, device=device) if centers_per_axis > 1 else torch.full((1,), 0.5, dtype=dtype, device=device)
        cz, cy, cx = torch.meshgrid(axis, axis, axis, indexing="ij")
        normalized_centers = torch.stack((cx, cy, cz), -1).reshape(-1, 3)
        depth, height, width = self.spatial_shape
        gz, gy, gx = torch.meshgrid(
            torch.linspace(0, 1, depth, dtype=dtype, device=device),
            torch.linspace(0, 1, height, dtype=dtype, device=device),
            torch.linspace(0, 1, width, dtype=dtype, device=device),
            indexing="ij",
        )
        points = torch.stack((gx, gy, gz), -1).reshape(-1, 3)
        squared_distances = (points[None] - normalized_centers[:, None]).square().sum(-1)
        gaussian = torch.exp(-squared_distances / (2 * self.length_scale**2))
        design = torch.cat((torch.ones_like(gaussian[:1]), gaussian), 0)
        self.register_buffer("bounds", bounds.detach().clone())
        self.register_buffer("normalized_centers", normalized_centers)
        self.register_buffer("centers", bounds[:, 0] + normalized_centers * (bounds[:, 1] - bounds[:, 0]))
        self.register_buffer("design", design)
        self.coefficients = nn.Parameter(torch.zeros((3, design.shape[0]), dtype=dtype, device=device))

    def forward(self) -> Tensor:
        return (self.coefficients @ self.design).reshape(1, 3, *self.spatial_shape)

    def regularization(self) -> Tensor:
        return self.ridge * self.coefficients[:, 1:].square().sum()
