"""Permutation-invariant conditioning for projected particle tracks.

The encoder treats particles as an unordered set while preserving the
time/view structure inside each particle.  It accepts padded batches: masked
observations and particles do not contribute to either the mean or max pools.
"""

from __future__ import annotations

import math
from collections.abc import Sized

import torch
from torch import Tensor, nn


def _numeric_mask(value: Tensor, *, name: str, device: torch.device) -> Tensor:
    mask = torch.as_tensor(value, device=device)
    if mask.dtype == torch.bool:
        return mask
    if not (mask.is_floating_point() or mask.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }):
        raise TypeError(f"{name} must be boolean or numeric")
    if not bool(torch.isfinite(mask).all().detach()) or not bool(
        ((mask == 0) | (mask == 1)).all().detach()
    ):
        raise ValueError(f"{name} must contain only zeros and ones")
    return mask.to(torch.bool)


def _masked_mean_and_max(
    values: Tensor, mask: Tensor, *, dimension: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool one axis and return mean, max, and whether any entry was valid."""

    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum(dim=dimension)
    mean = (values * weights).sum(dim=dimension) / count.clamp_min(1.0)
    minimum = torch.finfo(values.dtype).min
    maximum = values.masked_fill(~mask.unsqueeze(-1), minimum).amax(dim=dimension)
    valid = count.squeeze(-1) > 0
    maximum = torch.where(valid.unsqueeze(-1), maximum, torch.zeros_like(maximum))
    mean = torch.where(valid.unsqueeze(-1), mean, torch.zeros_like(mean))
    return mean, maximum, valid


class _ResidualProjection(nn.Module):
    """Preserve an explicit input path alongside a normalized nonlinear branch."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.skip = nn.Linear(input_dim, output_dim, bias=False)
        self.branch = nn.Sequential(
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.Linear(output_dim, output_dim, bias=False),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim, bias=False),
        )

    def forward(self, values: Tensor) -> Tensor:
        projected = self.skip(values)
        return projected + self.branch(projected)


class TrackSetEncoder(nn.Module):
    """Encode multi-view trajectories into one condition vector per example.

    Args:
        condition_dim: Size of the returned condition vector.
        token_dim: Hidden size used for individual time/view observations.
        particle_dim: Hidden size used after pooling a particle's observations.

    Inputs follow the repository convention: ``tracks`` has shape
    ``[B,V,N,T,2]``, ``mask`` has shape ``[B,V,N,T]``, cameras are
    ``[V,2,3]`` or ``[B,V,2,3]``, times are ``[T]`` or ``[B,T]``, and bounds
    are ``[3,2]`` or ``[B,3,2]``.  ``particle_mask`` is optional ``[B,N]``
    padding metadata.

    Each observation token contains normalized projected position, displacement
    from the first visible point, finite-difference motion, normalized time,
    the six camera-matrix entries, and a validity flag.  Pooling first across
    time/views and then across particles makes the result invariant to particle
    order.  The logarithm of the valid-particle count lets one shared encoder
    distinguish two-particle from denser observations.
    """

    token_feature_dim = 2 + 2 + 2 + 1 + 6 + 1

    def __init__(
        self,
        condition_dim: int = 128,
        *,
        token_dim: int = 64,
        particle_dim: int = 128,
        variant: str = "legacy",
        motion_scales: list[float] | tuple[float, ...] | None = None,
        condition_flow_grid: int = 4,
    ) -> None:
        super().__init__()
        if min(condition_dim, token_dim, particle_dim) < 1:
            raise ValueError("encoder dimensions must all be positive")
        self.condition_dim = int(condition_dim)
        self.token_dim = int(token_dim)
        self.particle_dim = int(particle_dim)
        if variant not in {"legacy", "motion-residual-v2"}:
            raise ValueError(f"unknown encoder variant: {variant}")
        self.variant = variant
        if variant == "motion-residual-v2":
            if motion_scales is None:
                raise ValueError("motion-residual-v2 requires train-fitted motion_scales")
            scales = torch.as_tensor(motion_scales, dtype=torch.float32)
            if scales.shape != (4,) or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
                raise ValueError("motion_scales must contain four finite positive values")
            if isinstance(condition_flow_grid, bool) or not isinstance(condition_flow_grid, int) or condition_flow_grid < 1:
                raise ValueError("condition_flow_grid must be a positive integer")
            self.condition_flow_grid = condition_flow_grid
            self.register_buffer("motion_scales", scales.clone())
            self.token_mlp = nn.Sequential(
                _ResidualProjection(self.token_feature_dim, self.token_dim),
                nn.LayerNorm(self.token_dim, elementwise_affine=False),
            )
            self.particle_mlp = nn.Sequential(
                _ResidualProjection(2 * self.token_dim, self.particle_dim),
                nn.LayerNorm(self.particle_dim, elementwise_affine=False),
            )
            self.output_mlp = nn.Sequential(
                _ResidualProjection(2 * self.particle_dim + 1, self.condition_dim),
                nn.LayerNorm(self.condition_dim, elementwise_affine=False),
            )
            self.condition_flow_head = nn.Sequential(
                nn.Linear(self.condition_dim, self.condition_dim),
                nn.SiLU(),
                nn.Linear(self.condition_dim, 3 * condition_flow_grid**3),
            )
            return

        # Do not change the modules, RNG consumption, or state keys of legacy
        # encoders: all existing checkpoints must reproduce their old outputs.
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_feature_dim, self.token_dim),
            nn.SiLU(),
            nn.Linear(self.token_dim, self.token_dim),
            nn.SiLU(),
        )
        self.particle_mlp = nn.Sequential(
            nn.Linear(2 * self.token_dim, self.particle_dim),
            nn.SiLU(),
            nn.Linear(self.particle_dim, self.particle_dim),
            nn.SiLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * self.particle_dim + 1, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.LayerNorm(self.condition_dim),
        )

    def forward(
        self,
        tracks: Tensor,
        mask: Tensor,
        projection_matrices: Tensor,
        observation_times: Tensor,
        domain_bounds: Tensor,
        particle_mask: Tensor | None = None,
    ) -> Tensor:
        if self.variant == "motion-residual-v2":
            if not isinstance(tracks, Tensor):
                raise TypeError("tracks must be a torch.Tensor")
            if not tracks.is_floating_point():
                raise TypeError("tracks must be floating point")
            # Small physical motions must not disappear in mixed precision.
            with torch.autocast(device_type=tracks.device.type, enabled=False):
                return self._encode(
                    tracks.float(), mask, projection_matrices, observation_times,
                    domain_bounds, particle_mask,
                )
        return self._encode(
            tracks, mask, projection_matrices, observation_times,
            domain_bounds, particle_mask,
        )

    def _encode(
        self, tracks: Tensor, mask: Tensor, projection_matrices: Tensor,
        observation_times: Tensor, domain_bounds: Tensor,
        particle_mask: Tensor | None = None,
    ) -> Tensor:
        token_features, valid = self.token_features(
            tracks, mask, projection_matrices, observation_times,
            domain_bounds, particle_mask,
        )
        if self.variant == "motion-residual-v2":
            token_features = torch.cat((
                token_features[..., :2],
                token_features[..., 2:6] / self.motion_scales,
                token_features[..., 6:],
            ), dim=-1)
        batch, views, particles, times_count, _ = tracks.shape
        token_embeddings = self.token_mlp(token_features)

        # Keep N as the set axis and flatten the within-particle V,T axes.
        within_particle = token_embeddings.permute(0, 2, 1, 3, 4).reshape(
            batch, particles, views * times_count, self.token_dim
        )
        within_mask = valid.permute(0, 2, 1, 3).reshape(
            batch, particles, views * times_count
        )
        token_mean, token_max, particle_valid = _masked_mean_and_max(
            within_particle, within_mask, dimension=2
        )
        particle_embeddings = self.particle_mlp(
            torch.cat((token_mean, token_max), dim=-1)
        )
        if not bool(particle_valid.any(dim=1).all().detach()):
            raise ValueError("each batch example must retain at least one valid particle")
        particle_mean, particle_max, _ = _masked_mean_and_max(
            particle_embeddings, particle_valid, dimension=1
        )
        particle_count = particle_valid.sum(dim=1, dtype=tracks.dtype)
        count_feature = torch.log1p(particle_count).unsqueeze(-1)
        return self.output_mlp(
            torch.cat((particle_mean, particle_max, count_feature), dim=-1)
        )

    def predict_condition_flow(self, condition: Tensor) -> Tensor:
        """Decode the final, undropped condition for training-only supervision."""
        if self.variant != "motion-residual-v2":
            raise ValueError("condition flow prediction requires motion-residual-v2")
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError("condition must have shape [B,condition_dim]")
        with torch.autocast(device_type=condition.device.type, enabled=False):
            prediction = self.condition_flow_head(condition.float())
        return prediction.reshape(condition.shape[0], 3, *([self.condition_flow_grid] * 3))

    @staticmethod
    def token_features(
        tracks: Tensor, mask: Tensor, projection_matrices: Tensor,
        observation_times: Tensor, domain_bounds: Tensor,
        particle_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build unscaled physical features and validity with legacy semantics.

        Shared by training-only RMS fitting and both encoder variants; this
        method performs no learned transformation or statistics fitting.
        """
        if not isinstance(tracks, Tensor):
            raise TypeError("tracks must be a torch.Tensor")
        if tracks.ndim != 5 or tracks.shape[-1] != 2:
            raise ValueError("tracks must have shape [B,V,N,T,2]")
        if not tracks.is_floating_point():
            raise TypeError("tracks must be floating point")
        batch, views, particles, times_count, _ = tracks.shape
        if min(batch, views, particles, times_count) < 1:
            raise ValueError("tracks dimensions B,V,N,T must all be non-empty")

        valid = _numeric_mask(mask, name="mask", device=tracks.device)
        if tuple(valid.shape) != (batch, views, particles, times_count):
            raise ValueError("mask must have shape [B,V,N,T] matching tracks")

        if particle_mask is None:
            valid_particles_requested = torch.ones(
                (batch, particles), dtype=torch.bool, device=tracks.device
            )
        else:
            valid_particles_requested = _numeric_mask(
                particle_mask, name="particle_mask", device=tracks.device
            )
            if valid_particles_requested.ndim == 1:
                if valid_particles_requested.shape[0] != particles:
                    raise ValueError("a 1D particle_mask must have shape [N]")
                valid_particles_requested = valid_particles_requested.unsqueeze(0)
            if valid_particles_requested.shape[0] == 1 and batch > 1:
                valid_particles_requested = valid_particles_requested.expand(batch, -1)
            if tuple(valid_particles_requested.shape) != (batch, particles):
                raise ValueError("particle_mask must have shape [B,N] or [N]")
        valid = valid & valid_particles_requested[:, None, :, None]

        cameras = TrackSetEncoder._cameras(
            projection_matrices,
            batch=batch,
            views=views,
            reference=tracks,
        )
        times = TrackSetEncoder._times(
            observation_times,
            batch=batch,
            count=times_count,
            reference=tracks,
        )
        bounds = TrackSetEncoder._bounds(domain_bounds, batch=batch, reference=tracks)

        coordinate_valid = valid.unsqueeze(-1).expand_as(tracks)
        if not bool(torch.isfinite(tracks[coordinate_valid]).all().detach()):
            raise ValueError("tracks must be finite wherever mask is true")
        safe_tracks = torch.where(valid.unsqueeze(-1), tracks, torch.zeros_like(tracks))

        center = bounds.mean(dim=-1)
        half_extent = 0.5 * (bounds[..., 1] - bounds[..., 0])
        projected_center = torch.einsum("bvoc,bc->bvo", cameras, center)
        projected_radius = torch.einsum(
            "bvoc,bc->bvo", cameras.abs(), half_extent
        ).clamp_min(torch.finfo(tracks.dtype).eps)
        position = (
            safe_tracks - projected_center[:, :, None, None]
        ) / projected_radius[:, :, None, None]

        first_index = valid.to(torch.int64).argmax(dim=-1)
        gather_index = first_index[..., None, None].expand(
            batch, views, particles, 1, 2
        )
        first_position = position.gather(3, gather_index).squeeze(3)
        displacement = position - first_position.unsqueeze(3)
        displacement = torch.where(
            valid.unsqueeze(-1), displacement, torch.zeros_like(displacement)
        )

        normalized_times = TrackSetEncoder._normalize_times(times)
        difference = torch.zeros_like(position)
        if times_count > 1:
            time_delta = normalized_times[:, 1:] - normalized_times[:, :-1]
            pair_valid = valid[..., 1:] & valid[..., :-1]
            motion = (position[..., 1:, :] - position[..., :-1, :]) / time_delta[
                :, None, None, :, None
            ].clamp_min(torch.finfo(tracks.dtype).eps)
            difference[..., 1:, :] = torch.where(
                pair_valid.unsqueeze(-1), motion, torch.zeros_like(motion)
            )

        camera_features = cameras.reshape(batch, views, 1, 1, 6).expand(
            batch, views, particles, times_count, 6
        )
        time_features = normalized_times[:, None, None, :, None].expand(
            batch, views, particles, times_count, 1
        )
        token_features = torch.cat(
            (
                position,
                displacement,
                difference,
                time_features,
                camera_features,
                valid.unsqueeze(-1).to(dtype=tracks.dtype),
            ),
            dim=-1,
        )
        return token_features, valid

    @staticmethod
    def _cameras(
        value: Tensor,
        *,
        batch: int,
        views: int,
        reference: Tensor,
    ) -> Tensor:
        cameras = torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        )
        if cameras.ndim == 3:
            if tuple(cameras.shape) != (views, 2, 3):
                raise ValueError("projection_matrices must have shape [V,2,3]")
            cameras = cameras.unsqueeze(0)
        elif cameras.ndim != 4 or tuple(cameras.shape[1:]) != (views, 2, 3):
            raise ValueError(
                "projection_matrices must have shape [V,2,3] or [B,V,2,3]"
            )
        if cameras.shape[0] == 1 and batch > 1:
            cameras = cameras.expand(batch, -1, -1, -1)
        if cameras.shape[0] != batch:
            raise ValueError("projection_matrices batch must be 1 or match tracks")
        if not bool(torch.isfinite(cameras).all().detach()):
            raise ValueError("projection_matrices must be finite")
        return cameras

    @staticmethod
    def _times(
        value: Tensor,
        *,
        batch: int,
        count: int,
        reference: Tensor,
    ) -> Tensor:
        times = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if times.ndim == 1:
            if times.shape[0] != count:
                raise ValueError("observation_times must have shape [T]")
            times = times.unsqueeze(0)
        elif times.ndim != 2 or times.shape[1] != count:
            raise ValueError("observation_times must have shape [T] or [B,T]")
        if times.shape[0] == 1 and batch > 1:
            times = times.expand(batch, -1)
        if times.shape[0] != batch:
            raise ValueError("observation_times batch must be 1 or match tracks")
        if not bool(torch.isfinite(times).all().detach()):
            raise ValueError("observation_times must be finite")
        if count > 1 and not bool((times[:, 1:] > times[:, :-1]).all().detach()):
            raise ValueError("observation_times must be strictly increasing")
        return times

    @staticmethod
    def _bounds(value: Tensor, *, batch: int, reference: Tensor) -> Tensor:
        bounds = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if bounds.ndim == 2:
            if tuple(bounds.shape) != (3, 2):
                raise ValueError("domain_bounds must have shape [3,2]")
            bounds = bounds.unsqueeze(0)
        elif bounds.ndim != 3 or tuple(bounds.shape[1:]) != (3, 2):
            raise ValueError("domain_bounds must have shape [3,2] or [B,3,2]")
        if bounds.shape[0] == 1 and batch > 1:
            bounds = bounds.expand(batch, -1, -1)
        if bounds.shape[0] != batch:
            raise ValueError("domain_bounds batch must be 1 or match tracks")
        if not bool(torch.isfinite(bounds).all().detach()) or not bool(
            (bounds[..., 1] > bounds[..., 0]).all().detach()
        ):
            raise ValueError("domain_bounds must be finite and strictly increasing")
        return bounds

    @staticmethod
    def _normalize_times(times: Tensor) -> Tensor:
        if times.shape[1] == 1:
            return torch.zeros_like(times)
        return (times - times[:, :1]) / (times[:, -1:] - times[:, :1])

    def extra_repr(self) -> str:
        return (
            f"condition_dim={self.condition_dim}, token_dim={self.token_dim}, "
            f"particle_dim={self.particle_dim}, variant={self.variant!r}"
        )


@torch.no_grad()
def fit_condition_motion_scales(
    dataset: Sized, device: torch.device | str, *, max_samples: int | None = None,
) -> list[float]:
    """Fit fixed displacement/motion RMS using only a supplied training split.

    Iterates examples in order without changing dataset epoch or random state.
    Missing observations are excluded; motion additionally requires a valid
    adjacent pair. A zero-only feature uses scale 1, otherwise scales are
    floored at 1e-6. No per-case normalization can erase speed differences.
    The caller must pass its training dataset, never validation or test data.
    """
    split = getattr(dataset, "split", None)
    if split is not None and split != "train":
        raise ValueError("motion scales may only be fitted on the train split")
    count = len(dataset)
    if max_samples is not None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        count = min(count, max_samples)
    if count < 1:
        raise ValueError("cannot fit motion scales from an empty training dataset")
    device = torch.device(device)
    squares = torch.zeros(4, dtype=torch.float64, device=device)
    counts = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.autocast(device_type=device.type, enabled=False):
        for index in range(count):
            example = dataset[index]
            tracks = torch.as_tensor(example["tracks"], dtype=torch.float32, device=device).unsqueeze(0)
            mask = torch.as_tensor(example["mask"], device=device).unsqueeze(0)
            particle_mask = example.get("particle_mask")
            if particle_mask is not None:
                particle_mask = torch.as_tensor(particle_mask, device=device)
            features, valid = TrackSetEncoder.token_features(
                tracks, mask, example["projection"], example["times"],
                example["bounds"], particle_mask,
            )
            pair_valid = torch.zeros_like(valid)
            pair_valid[..., 1:] = valid[..., 1:] & valid[..., :-1]
            feature_mask = torch.cat((
                valid.unsqueeze(-1).expand(*valid.shape, 2),
                pair_valid.unsqueeze(-1).expand(*valid.shape, 2),
            ), dim=-1)
            motion = features[..., 2:6].double()
            squares += torch.where(feature_mask, motion.square(), 0.0).reshape(-1, 4).sum(0)
            counts += feature_mask.reshape(-1, 4).sum(0)
    if not bool((counts[:2] > 0).all()):
        raise ValueError("training data contain no valid trajectory observations")
    rms = (squares / counts.clamp_min(1)).sqrt()
    scales = torch.where(rms > 0, rms.clamp_min(1e-6), torch.ones_like(rms))
    return scales.float().cpu().tolist()


__all__ = ["TrackSetEncoder", "fit_condition_motion_scales"]
