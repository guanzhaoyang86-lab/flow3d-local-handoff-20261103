"""Steady, physical velocity/pressure PINN for trajectory reconstruction.

No reference field, pressure or case metadata enters this module. Viscosity is
explicitly known kinematic viscosity; pressure denotes pressure divided by the
constant fluid density. Boundary constraints and residual weights are external.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _coordinate_gradient(value: Tensor, coordinates: Tensor) -> Tensor:
    """Pointwise derivative, with support for exactly constant/linear outputs."""
    if not value.requires_grad:
        return coordinates * 0.0
    gradient = torch.autograd.grad(
        value, coordinates, grad_outputs=torch.ones_like(value),
        create_graph=True, retain_graph=True, allow_unused=True,
    )[0]
    # The explicit zero dependency permits second derivatives of constants.
    return coordinates * 0.0 if gradient is None else gradient + coordinates * 0.0


class PINNField(nn.Module):
    """Steady XYZ velocity and pressure represented by a pointwise tanh MLP.

    ``bounds`` are physical XYZ [3,2]; ``spatial_shape`` is ZYX (D,H,W).
    ``layers`` counts hidden Linear+tanh layers, excluding the output layer.
    Inputs are normalized to [-1,1], while autograd operates on original physical
    coordinates, automatically retaining domain scales in first/second spatial
    derivatives. Pressure is gauged to zero at the domain center.

    The unforced, steady incompressible residual is
    (v dot grad)v + grad(p/rho) - viscosity*Laplacian(v), together with div(v).
    There is no implicit no-slip, moving lid, forcing or reference conditioning.
    """

    def __init__(self, bounds: Tensor | Sequence[Sequence[float]],
                 spatial_shape: Sequence[int], hidden: int = 128, layers: int = 5,
                 viscosity: float = 0.01, collocation_points: int = 1024,
                 seed: int = 31) -> None:
        super().__init__()
        physical_bounds = torch.as_tensor(bounds).detach().clone()
        if not physical_bounds.is_floating_point():
            physical_bounds = physical_bounds.to(torch.get_default_dtype())
        if physical_bounds.shape != (3, 2):
            raise ValueError("bounds must have shape [3,2] in physical XYZ order")
        if not bool(torch.isfinite(physical_bounds).all()):
            raise ValueError("bounds must be finite")
        if not bool((physical_bounds[:, 1] > physical_bounds[:, 0]).all()):
            raise ValueError("each upper bound must exceed its lower bound")
        if len(spatial_shape) != 3 or any(int(n) != n or n < 2 for n in spatial_shape):
            raise ValueError("spatial_shape must contain three integers >=2 in ZYX order")
        if int(hidden) != hidden or hidden < 1 or int(layers) != layers or layers < 1:
            raise ValueError("hidden and layers must be positive integers")
        if int(collocation_points) != collocation_points or collocation_points < 1:
            raise ValueError("collocation_points must be a positive integer")
        if not bool(torch.isfinite(torch.tensor(viscosity))) or viscosity < 0:
            raise ValueError("viscosity must be a finite nonnegative known value")
        self.spatial_shape = tuple(int(n) for n in spatial_shape)
        self.hidden, self.layers = int(hidden), int(layers)
        self.viscosity, self.seed = float(viscosity), int(seed)
        self.num_collocation_points = int(collocation_points)
        self.register_buffer("bounds", physical_bounds)

        # Local initialization does not change caller training/sampling RNGs.
        with torch.random.fork_rng(devices=[]):
            # Do not call torch.manual_seed here: it also resets CUDA generators
            # that are not covered by this CPU-only initialization context.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(self.seed).get_state())
            modules: list[nn.Module] = []
            incoming = 3
            for _ in range(self.layers):
                linear = nn.Linear(incoming, self.hidden)
                nn.init.xavier_normal_(linear.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(linear.bias)
                modules.extend((linear, nn.Tanh()))
                incoming = self.hidden
            output = nn.Linear(incoming, 4)
            nn.init.xavier_normal_(output.weight)
            nn.init.zeros_(output.bias)
            modules.append(output)
            self.network = nn.Sequential(*modules).to(device=physical_bounds.device,
                                                      dtype=physical_bounds.dtype)
        lo = physical_bounds[:, 0]
        span = physical_bounds[:, 1] - lo
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        unit_points = torch.rand((self.num_collocation_points, 3), generator=generator,
                                 dtype=physical_bounds.dtype).to(physical_bounds.device)
        self.register_buffer("collocation_coords", lo + unit_points * span)
        depth, height, width = self.spatial_shape
        axes = [torch.linspace(physical_bounds[i, 0], physical_bounds[i, 1], n,
                               device=physical_bounds.device, dtype=physical_bounds.dtype)
                for i, n in enumerate((width, height, depth))]
        zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        self.register_buffer("grid_coords", torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3))

    def query(self, coords: Tensor) -> Tensor:
        """Return physical [u,v,w,p/rho] at physical XYZ coordinates [M,3]."""
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape [M,3] in physical XYZ order")
        normalized = 2.0 * (coords - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]) - 1.0
        output = self.network(normalized)
        center = normalized.new_zeros((1, 3))
        center_pressure = self.network(center)[:, 3:4]
        return torch.cat((output[:, :3], output[:, 3:4] - center_pressure), dim=-1)

    def forward(self) -> Tensor:
        """Return physical velocity [1,3,D,H,W], XYZ components / ZYX storage."""
        velocity = self.query(self.grid_coords)[:, :3]
        return velocity.reshape(*self.spatial_shape, 3).permute(3, 0, 1, 2).unsqueeze(0)

    def physics_loss(self) -> dict[str, Tensor]:
        """Mean squared divergence and mean squared momentum-vector norm.

        The two dimensional losses remain separate; weights must be selected by
        the caller. Fixed interior collocation uses only domain bounds and seed.
        """
        with torch.enable_grad():
            coordinates = self.collocation_coords.detach().clone().requires_grad_(True)
            values = self.query(coordinates)
            velocity, pressure = values[:, :3], values[:, 3]
            gradients = torch.stack([
                _coordinate_gradient(velocity[:, component], coordinates)
                for component in range(3)], dim=1)
            divergence = torch.diagonal(gradients, dim1=1, dim2=2).sum(dim=1)
            pressure_gradient = _coordinate_gradient(pressure, coordinates)
            convection = (gradients * velocity[:, None, :]).sum(dim=2)
            laplacian = torch.stack([
                torch.stack([
                    _coordinate_gradient(gradients[:, component, axis], coordinates)[:, axis]
                    for axis in range(3)], dim=1).sum(dim=1)
                for component in range(3)], dim=1)
            momentum = convection + pressure_gradient - self.viscosity * laplacian
            return {"divergence": divergence.square().mean(),
                    "momentum": momentum.square().sum(dim=1).mean()}
