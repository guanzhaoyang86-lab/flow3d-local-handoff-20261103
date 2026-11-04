"""Gaussian Euler-transition likelihoods from the supplied Section 2.2.

The default projected residual is dy - P v(U(v, y0, depth), t) dt.
This is deliberately distinct from a Gaussian error on replayed positions.
Coordinates/components are XYZ; velocity volume axes are ZYX.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from ..advection import advect_particles
from ..interpolation import sample_velocity


def _times_sigma(times, sigma, like):
    times = torch.as_tensor(times, dtype=like.dtype, device=like.device)
    if times.ndim != 1 or len(times) < 2 or not bool(torch.isfinite(times).all()):
        raise ValueError('At least two finite observation times are required')
    if not bool((times.diff() > 0).all()):
        raise ValueError('Observation times must increase strictly')
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive, in displacement units')
    return times


def _mask_data(tracks, mask):
    if not isinstance(tracks, Tensor) or not tracks.is_floating_point():
        raise TypeError('Observed trajectories must be floating tensors')
    if any(s < 1 for s in tracks.shape):
        raise ValueError('Observed trajectories cannot have empty dimensions')
    mask = torch.ones(tracks.shape[:-1], dtype=torch.bool, device=tracks.device) if mask is None else torch.as_tensor(mask, device=tracks.device)
    if mask.dtype != torch.bool or mask.shape != tracks.shape[:-1]:
        raise ValueError('Boolean observation mask must match all non-coordinate axes')
    if not bool(torch.isfinite(tracks[mask]).all()):
        raise ValueError('Valid observations must be finite')
    if not bool((mask[..., 1:] & mask[..., :-1]).any()):
        raise ValueError('No consecutive observed positions are available')
    return torch.where(mask[..., None], tracks, torch.zeros_like(tracks)), mask


def _bounds(bounds, like):
    bounds = torch.as_tensor(bounds, dtype=like.dtype, device=like.device)
    if bounds.shape != (3, 2) or not bool(torch.isfinite(bounds).all()) or not bool((bounds[:, 1] > bounds[:, 0]).all()):
        raise ValueError('bounds must be finite increasing XYZ intervals [3,2]')
    return bounds


class ProjectedTrajectoryLikelihood:
    """One observed flow, synchronized tracks [V,N,T,2], with unknown depth.

    Same physical particle indices in every view. Rank-three views determine
    time-zero positions from observations. Rank-two views leave one bounded
    null-space coordinate per particle; no true depth or 3D track is read.
    """

    def __init__(self, tracks, projections, times, bounds, *, sigma,
                 mask=None, integrator='euler', num_substeps=1, likelihood='transition'):
        if tracks.ndim != 4 or tracks.shape[-1] != 2:
            raise ValueError('Projected tracks must be [V,N,T,2]')
        self.tracks, self.mask = _mask_data(tracks.detach(), mask)
        self.times = _times_sigma(times, sigma, tracks)
        if tracks.shape[-2] != len(self.times):
            raise ValueError('Track and time axes disagree')
        self.bounds = _bounds(bounds, tracks)
        self.projections = torch.as_tensor(projections, dtype=tracks.dtype, device=tracks.device).detach()
        if self.projections.shape != (tracks.shape[0], 2, 3) or not bool(torch.isfinite(self.projections).all()):
            raise ValueError('Projection matrices must be finite [V,2,3]')
        if any(int(torch.linalg.matrix_rank(p)) != 2 for p in self.projections):
            raise ValueError('Every camera must have rank two')
        if integrator not in ('euler', 'rk4') or not isinstance(num_substeps, int) or num_substeps < 1:
            raise ValueError('Use euler/rk4 with positive integer substeps')
        if likelihood not in ('transition', 'position'):
            raise ValueError('likelihood must be transition or position')
        self.sigma, self.integrator = float(sigma), integrator
        self.num_substeps, self.likelihood = num_substeps, likelihood
        self.particular, self.direction, self.lower, self.upper = self._initial_geometry()
        self.depth_active = self.upper > self.lower

    def _initial_geometry(self):
        particular, direction, lower, upper = [], [], [], []
        for particle in range(self.tracks.shape[1]):
            valid = self.mask[:, particle, 0]
            a = self.projections[valid].reshape(-1, 3)
            if not len(a):
                raise ValueError('Each particle needs a valid initial 2D observation')
            y = self.tracks[valid, particle, 0].reshape(-1)
            _, _, vh = torch.linalg.svd(a, full_matrices=True)
            rank = int(torch.linalg.matrix_rank(a))
            p = torch.linalg.pinv(a) @ y
            # The screenshot conditions on exact initial projections. Sigma
            # belongs to displacement residuals, not noisy time-zero geometry:
            # least-squares triangulation must not silently average conflicting
            # camera observations. Allow only scale-aware floating roundoff.
            scale = torch.maximum(y.abs(), (a.abs() * p.abs()[None]).sum(-1)).clamp_min(1)
            tolerance = 64 * torch.finfo(a.dtype).eps * scale
            if not bool(((a @ p - y).abs() <= tolerance).all()):
                raise ValueError('Initial camera observations are inconsistent for the same physical particle')
            q = torch.zeros_like(p) if rank == 3 else vh[-1]
            if rank not in (2, 3):
                raise ValueError('Initial view rank must be two or three')
            lo, hi = p.new_tensor(-torch.inf), p.new_tensor(torch.inf)
            if rank == 3:
                if not bool(((p >= self.bounds[:, 0] - 1e-5) & (p <= self.bounds[:, 1] + 1e-5)).all()):
                    raise ValueError('Triangulated initial position is outside the domain')
                p = p.clamp(self.bounds[:, 0], self.bounds[:, 1])
                lo = hi = p.new_tensor(0.)
            else:
                for axis in range(3):
                    if abs(float(q[axis])) < 1e-7:
                        if not bool((p[axis] >= self.bounds[axis, 0] - 1e-5) & (p[axis] <= self.bounds[axis, 1] + 1e-5)):
                            raise ValueError('Initial observation ray misses the domain')
                    else:
                        ends = (self.bounds[axis] - p[axis]) / q[axis]
                        lo = torch.maximum(lo, ends.min()); hi = torch.minimum(hi, ends.max())
                if not bool(torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)):
                    raise ValueError('Initial observation ray misses the domain')
            particular.append(p); direction.append(q); lower.append(lo); upper.append(hi)
        return tuple(torch.stack(v).detach() for v in (particular, direction, lower, upper))

    def initialize_depth(self, generator=None, *, random=True):
        if random and bool(self.depth_active.any()):
            fraction = torch.rand(self.lower.shape, dtype=self.lower.dtype, device=self.lower.device, generator=generator)
        else:
            fraction = torch.full_like(self.lower, .5)
        return self.lower + (self.upper - self.lower) * fraction

    def reflect_depth(self, depth):
        width = self.upper - self.lower
        phase = torch.remainder(depth - self.lower, 2 * width.clamp_min(1e-12))
        return torch.where(self.depth_active, self.lower + torch.minimum(phase, 2 * width - phase), self.lower)

    def initial_positions(self, depth):
        if depth is None or depth.shape != self.lower.shape:
            raise ValueError('A depth coordinate per observed particle is required')
        return self.particular + self.direction * depth[:, None]

    def replay(self, field, depth):
        if field.ndim != 5 or field.shape[:2] != (1, 3):
            raise ValueError('One steady field [1,3,D,H,W] is required')
        return advect_particles(field, self.initial_positions(depth)[None], self.times,
            integrator=self.integrator, num_substeps=self.num_substeps,
            domain_bounds=self.bounds, boundary_mode='clamp', return_validity=True)

    def residual(self, field, depth):
        positions, validity = self.replay(field, depth)
        if self.likelihood == 'position':
            predicted = torch.einsum('ntc,voc->vnto', positions[0], self.projections)
            # Initial 2D coordinates are conditioned upon, as in the screenshot.
            mask = self.mask.clone(); mask[..., 0] = False
            residual = self.tracks - predicted
        else:
            n, t = positions.shape[1:3]
            velocity = sample_velocity(field, positions[:, :, :-1].reshape(1, -1, 3),
                self.times[:-1].expand(n, t - 1).reshape(1, -1), self.bounds, 'clamp').reshape(n, t - 1, 3)
            projected_velocity = torch.einsum('ntc,voc->vnto', velocity, self.projections)
            residual = self.tracks.diff(dim=-2) - projected_velocity * self.times.diff()[None, None, :, None]
            mask = self.mask[..., :-1] & self.mask[..., 1:]
        # Candidate invalidity is reported, never used to hide a residual.
        return torch.where(mask[..., None], residual, torch.zeros_like(residual)), mask, validity

    def nll(self, field, depth):
        residual, _, _ = self.residual(field, depth)
        return residual.square().sum() / (2 * self.sigma ** 2)

    def diagnostics(self, field, depth):
        residual, mask, validity = self.residual(field, depth)
        positions, _ = self.replay(field, depth)
        projected = torch.einsum('ntc,voc->vnto', positions[0], self.projections)
        error = torch.where(self.mask[..., None], projected - self.tracks, torch.zeros_like(projected))
        return {'nll': float(residual.square().sum() / (2 * self.sigma ** 2)),
            'residual_rmse': float((residual.square().sum() / (2 * mask.sum())).sqrt()),
            'observed_track_rmse_cells': float((error.square().sum() / (2 * self.mask.sum())).sqrt()),
            'valid_replay_fraction': float(validity.float().mean()),
            'all_replay_particles_valid': bool(validity.all()),
            'unknown_initial_depth_count': int(self.depth_active.sum())}


class FullTrajectoryLikelihood:
    """Equation (2) specialized to one steady three-dimensional velocity field.

    The NLL uses observed positions, as in Equation (2), independently of the
    integrator selected for replay diagnostics. Replay clamps to the finite
    volume and reports attempts to leave it; this is a boundary surrogate,
    without an extra no-exit likelihood penalty.
    """

    def __init__(self, tracks, times, bounds, *, sigma, mask=None,
                 integrator='euler', num_substeps=1):
        if tracks.ndim != 3 or tracks.shape[-1] != 3:
            raise ValueError('Full observed tracks must be [N,T,3]')
        self.tracks, self.mask = _mask_data(tracks.detach(), mask)
        self.times = _times_sigma(times, sigma, tracks)
        if tracks.shape[1] != len(self.times):
            raise ValueError('Track and time axes disagree')
        self.bounds, self.sigma = _bounds(bounds, tracks), float(sigma)
        if integrator not in ('euler', 'rk4') or not isinstance(num_substeps, int) or num_substeps < 1:
            raise ValueError('Use euler/rk4 with positive integer substeps')
        self.integrator, self.num_substeps = integrator, num_substeps

    @staticmethod
    def _steady_field(field):
        if field.ndim != 5 or field.shape[:2] != (1, 3):
            raise ValueError('One steady field [1,3,D,H,W] is required')

    def residual(self, field, depth=None):
        del depth
        self._steady_field(field)
        n, t = self.tracks.shape[:2]
        velocity = sample_velocity(field, self.tracks[:, :-1].reshape(1, -1, 3),
            self.times[:-1].expand(n, t - 1).reshape(1, -1), self.bounds, 'clamp').reshape(n, t - 1, 3)
        residual = self.tracks.diff(dim=1) - velocity * self.times.diff()[None, :, None]
        mask = self.mask[:, :-1] & self.mask[:, 1:]
        return torch.where(mask[..., None], residual, torch.zeros_like(residual)), mask

    def nll(self, field, depth=None):
        residual, _ = self.residual(field, depth)
        return residual.square().sum() / (2 * self.sigma ** 2)

    def replay(self, field, depth=None):
        del depth
        self._steady_field(field)
        if not bool(self.mask[:, 0].all()):
            raise ValueError('Replay diagnostics require each initial 3D position to be observed')
        return advect_particles(field, self.tracks[:, 0][None], self.times,
            integrator=self.integrator, num_substeps=self.num_substeps,
            domain_bounds=self.bounds, boundary_mode='clamp', return_validity=True)

    def diagnostics(self, field, depth=None):
        residual, mask = self.residual(field, depth)
        positions, validity = self.replay(field, depth)
        error = torch.where(self.mask[..., None], positions[0] - self.tracks, torch.zeros_like(self.tracks))
        return {'nll': float(residual.square().sum() / (2 * self.sigma ** 2)),
            'residual_rmse': float((residual.square().sum() / (3 * mask.sum())).sqrt()),
            'observed_track_rmse_cells': float((error.square().sum() / (3 * self.mask.sum())).sqrt()),
            'valid_replay_fraction': float(validity.float().mean()),
            'all_replay_particles_valid': bool(validity.all()),
            'unknown_initial_depth_count': 0}
