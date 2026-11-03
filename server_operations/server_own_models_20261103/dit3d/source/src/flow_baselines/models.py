"""Deterministic decoders of a shared trajectory condition into steady 3-D flow.

All public decoders consume only ``[B, condition_dim]`` observations encoded by
the caller and return normalized XYZ velocity in ZYX tensor storage. Coordinates
are normalized to [-1, 1]; neither a target field nor a boundary value is an
input. The FNO transforms three *spatial* axes, never an axis of time.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(values: Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("spatial_shape must contain (D,H,W)")
    return tuple(_positive_integer(v, "spatial dimension") for v in values)


def coordinate_grid(spatial_shape: Sequence[int]) -> Tensor:
    """Return [1,3,D,H,W] with channel order x, y, z and tensor order z,y,x."""
    d, h, w = _shape(spatial_shape)
    axes = [torch.linspace(-1.0, 1.0, n) if n > 1 else torch.zeros(1) for n in (d, h, w)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x, y, z), dim=0).unsqueeze(0)


def _groups(channels: int) -> int:
    # At least two channels per group also supports a 1x1x1 bottleneck at B=1.
    groups = min(8, max(1, channels // 2))
    while channels % groups:
        groups -= 1
    return groups


class _Decoder(nn.Module):
    def __init__(self, spatial_shape: Sequence[int], condition_dim: int) -> None:
        super().__init__()
        self.spatial_shape = _shape(spatial_shape)
        self.condition_dim = _positive_integer(condition_dim, "condition_dim")
        self.register_buffer("coordinates", coordinate_grid(self.spatial_shape), persistent=False)

    def _condition(self, condition: Tensor) -> Tensor:
        if not isinstance(condition, Tensor) or not condition.is_floating_point():
            raise TypeError("condition must be a floating-point tensor")
        if condition.ndim != 2 or condition.shape[0] < 1 or condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition must have shape [B,{self.condition_dim}] with B>=1")
        if not bool(torch.isfinite(condition).all().detach()):
            raise ValueError("condition must contain only finite values")
        if condition.device != self.coordinates.device:
            raise ValueError("condition and decoder must be on the same device")
        return condition

    def _coordinates(self, condition: Tensor) -> Tensor:
        return self.coordinates.to(dtype=condition.dtype).expand(condition.shape[0], -1, -1, -1, -1)


class _ConditionedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.modulation = nn.Linear(condition_dim, 2 * out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        return self.skip(values) + self.conv2(self.dropout(F.silu(hidden)))


class DeterministicUNet3D(_Decoder):
    """Coordinate-lifted, FiLM-conditioned 3-D U-Net for direct regression."""
    def __init__(self, spatial_shape: Sequence[int], condition_dim: int = 128, *,
                 base_channels: int = 32, channel_multipliers: Sequence[int] = (1, 2, 4),
                 dropout: float = 0.0) -> None:
        super().__init__(spatial_shape, condition_dim)
        base_channels = _positive_integer(base_channels, "base_channels")
        multipliers = tuple(_positive_integer(m, "channel multiplier") for m in channel_multipliers)
        if not multipliers:
            raise ValueError("channel_multipliers must not be empty")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        channels = tuple(base_channels * m for m in multipliers)
        if min(channels) < 2:
            raise ValueError("U-Net feature channels must be at least 2")
        self.condition_lift = nn.Linear(condition_dim, channels[0])
        self.coordinate_lift = nn.Conv3d(3, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = channels[0]
        for index, width in enumerate(channels):
            self.down_blocks.append(_ConditionedResidual(current, width, condition_dim, dropout))
            current = width
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv3d(current, current, 3, stride=2, padding=1))
        self.middle = _ConditionedResidual(current, current, condition_dim, dropout)
        self.up_blocks = nn.ModuleList()
        for width in reversed(channels):
            self.up_blocks.append(_ConditionedResidual(current + width, width, condition_dim, dropout))
            current = width
        self.output = nn.Conv3d(current, 3, 3, padding=1)

    def forward(self, condition: Tensor) -> Tensor:
        condition = self._condition(condition)
        hidden = self.coordinate_lift(self._coordinates(condition))
        hidden = hidden + self.condition_lift(condition)[:, :, None, None, None]
        skips = []
        for index, block in enumerate(self.down_blocks):
            hidden = block(hidden, condition)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.middle(hidden, condition)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            if hidden.shape[-3:] != skip.shape[-3:]:
                hidden = F.interpolate(hidden, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            hidden = block(torch.cat((hidden, skip), dim=1), condition)
        return self.output(F.silu(hidden))


class ResidualCNN3D(_Decoder):
    """Single-resolution coordinate-lifted residual CNN with local 3-D kernels."""
    def __init__(self, spatial_shape: Sequence[int], condition_dim: int = 128, *,
                 width: int = 32, depth: int = 4) -> None:
        super().__init__(spatial_shape, condition_dim)
        width = _positive_integer(width, "width")
        depth = _positive_integer(depth, "depth")
        if width < 2:
            raise ValueError("CNN width must be at least 2")
        self.coordinate_lift = nn.Conv3d(3, width, 3, padding=1)
        self.condition_lift = nn.Linear(condition_dim, width)
        self.blocks = nn.ModuleList(_ConditionedResidual(width, width, condition_dim) for _ in range(depth))
        self.output = nn.Conv3d(width, 3, 3, padding=1)

    def forward(self, condition: Tensor) -> Tensor:
        condition = self._condition(condition)
        hidden = self.coordinate_lift(self._coordinates(condition))
        hidden = hidden + self.condition_lift(condition)[:, :, None, None, None]
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(F.silu(hidden))


class SpectralConv3D(nn.Module):
    """Truncated real FFT convolution on spatial Z,Y,X axes.

    Four independent weight banks cover (+Z,+Y), (-Z,+Y), (+Z,-Y),
    (-Z,-Y); X uses the nonnegative half-spectrum of rfftn. Small grids use
    disjoint positive/negative slices, retaining every available frequency
    when modes are sufficiently large, including odd grid sizes. Weights are
    stored as real/imaginary pairs so parameter counts include both degrees
    of freedom and standard real-valued optimizers remain applicable.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int | Sequence[int] = 8) -> None:
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        if isinstance(modes, int) and not isinstance(modes, bool):
            modes = (modes, modes, modes)
        self.modes = _shape(modes)
        self.weights = nn.Parameter(torch.randn(4, in_channels, out_channels, *self.modes, 2)
                                    / math.sqrt(2 * in_channels))

    def forward(self, values: Tensor) -> Tensor:
        if not isinstance(values, Tensor) or values.ndim != 5 or values.shape[1] != self.in_channels or not values.is_floating_point():
            raise ValueError(f"spectral input must be floating [B,{self.in_channels},D,H,W]")
        d, h, w = values.shape[-3:]
        if min(d, h, w) < 1:
            raise ValueError("spectral spatial dimensions must be nonempty")
        original_dtype = values.dtype
        # CUDA half FFT requires powers of two and CPU FFT cannot consume bf16.
        # Explicitly disable autocast and use fp32 for all lower-precision inputs.
        fft_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=values.device.type, enabled=False):
            spectrum = torch.fft.rfftn(values.to(fft_dtype), dim=(-3, -2, -1), norm="ortho")
            result = spectrum.new_zeros((values.shape[0], self.out_channels, d, h, w // 2 + 1))
            mz, my, mx = self.modes
            zparts = ((slice(0, min(mz, (d + 1) // 2)), min(mz, (d + 1) // 2)),
                      (slice(d - min(mz, d // 2), d), min(mz, d // 2)))
            yparts = ((slice(0, min(my, (h + 1) // 2)), min(my, (h + 1) // 2)),
                      (slice(h - min(my, h // 2), h), min(my, h // 2)))
            nx = min(mx, w // 2 + 1)
            for iz, (zs, nz) in enumerate(zparts):
                for iy, (ys, ny) in enumerate(yparts):
                    if nz == 0 or ny == 0:
                        continue
                    bank = iz + 2 * iy
                    weights = torch.view_as_complex(self.weights[bank, :, :, :nz, :ny, :nx].contiguous())
                    weights = weights.to(dtype=spectrum.dtype)
                    result[:, :, zs, ys, :nx] = torch.einsum(
                        "bizyx,iozyx->bozyx", spectrum[:, :, zs, ys, :nx], weights)
            output = torch.fft.irfftn(result, s=(d, h, w), dim=(-3, -2, -1), norm="ortho")
        return output.to(original_dtype)


class FourierNeuralOperator3D(_Decoder):
    """Nonperiodic padded FNO with a global observation-condition lift."""
    def __init__(self, spatial_shape: Sequence[int], condition_dim: int = 128, *,
                 width: int = 32, modes: int | Sequence[int] = 8, depth: int = 4,
                 padding: int = 4) -> None:
        super().__init__(spatial_shape, condition_dim)
        width = _positive_integer(width, "width")
        depth = _positive_integer(depth, "depth")
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a nonnegative integer")
        self.padding = padding
        self.coordinate_lift = nn.Conv3d(3, width, 1)
        self.condition_lift = nn.Linear(condition_dim, width)
        self.spectral = nn.ModuleList(SpectralConv3D(width, width, modes) for _ in range(depth))
        self.local = nn.ModuleList(nn.Conv3d(width, width, 1) for _ in range(depth))
        self.output = nn.Sequential(nn.Conv3d(width, 2 * width, 1), nn.GELU(), nn.Conv3d(2 * width, 3, 1))

    def forward(self, condition: Tensor) -> Tensor:
        condition = self._condition(condition)
        hidden = self.coordinate_lift(self._coordinates(condition))
        hidden = hidden + self.condition_lift(condition)[:, :, None, None, None]
        if self.padding:
            p = self.padding
            hidden = F.pad(hidden, (p, p, p, p, p, p))
        for spectral, local in zip(self.spectral, self.local):
            hidden = F.gelu(spectral(hidden) + local(hidden))
        if self.padding:
            p = self.padding
            d, h, w = self.spatial_shape
            hidden = hidden[:, :, p:p+d, p:p+h, p:p+w]
        return self.output(hidden)


def _mlp(in_features: int, out_features: int, hidden: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules.extend((nn.Linear(in_features, hidden), nn.Tanh()))
        in_features = hidden
    modules.append(nn.Linear(in_features, out_features))
    return nn.Sequential(*modules)


class DeepONet3D(_Decoder):
    """Branch(condition) times trunk(x,y,z), with a separate basis per velocity.

    ``layers`` is the number of hidden layers in each network. The trajectory
    encoder is external; this is an inverse-operator adaptation of DeepONet,
    rather than its original fixed-sensor input representation.
    """
    def __init__(self, spatial_shape: Sequence[int], condition_dim: int = 128, *,
                 latent: int = 128, hidden: int = 128, layers: int = 3) -> None:
        super().__init__(spatial_shape, condition_dim)
        self.latent = _positive_integer(latent, "latent")
        hidden = _positive_integer(hidden, "hidden")
        layers = _positive_integer(layers, "layers")
        self.branch = _mlp(condition_dim, latent, hidden, layers)
        self.trunk = _mlp(3, 3 * latent, hidden, layers)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, condition: Tensor) -> Tensor:
        condition = self._condition(condition)
        coordinates = self.coordinates[0].reshape(3, -1).transpose(0, 1).to(dtype=condition.dtype)
        branch = self.branch(condition)
        trunk = self.trunk(coordinates).reshape(-1, 3, self.latent)
        field = torch.einsum("bl,pcl->bcp", branch, trunk) / math.sqrt(self.latent)
        field = field + self.bias[None, :, None]
        return field.reshape(condition.shape[0], 3, *self.spatial_shape)


_DECODERS = {"unet": DeterministicUNet3D, "cnn": ResidualCNN3D,
             "fno": FourierNeuralOperator3D, "deeponet": DeepONet3D}
_CONFIG_KEYS = {"unet": {"base_channels", "channel_multipliers", "dropout"},
                "cnn": {"width", "depth"}, "fno": {"width", "modes", "depth", "padding"},
                "deeponet": {"latent", "hidden", "layers"}}


def build_decoder(kind: str, spatial_shape: Sequence[int], condition_dim: int = 128, **config: object) -> nn.Module:
    """Build a deterministic baseline; reject misspelled/irrelevant parameters."""
    if kind not in _DECODERS:
        raise ValueError(f"unknown decoder {kind!r}; choose {', '.join(_DECODERS)}")
    unknown = set(config) - _CONFIG_KEYS[kind]
    if unknown:
        raise ValueError(f"unsupported {kind} configuration keys: {sorted(unknown)}")
    return _DECODERS[kind](spatial_shape, condition_dim=condition_dim, **config)
