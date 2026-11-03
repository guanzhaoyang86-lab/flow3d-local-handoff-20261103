"""Training targets that anchor trajectory representations to unknown fluid flow."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def condition_flow_target(flow: Tensor, solid_mask: Tensor, boundary_mask: Tensor,
                          grid_size: int) -> tuple[Tensor, Tensor]:
    """Average normalized velocity over unknown fluid cells in each coarse bin.

    Known boundary/solid values cannot dominate the auxiliary supervision. Empty
    bins are excluded. No inference-time field information enters the encoder.
    """
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow must have shape [B,3,D,H,W]")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    expected = (flow.shape[0], *flow.shape[2:])
    if tuple(solid_mask.shape) != expected or tuple(boundary_mask.shape) != expected:
        raise ValueError("fluid masks must have shape [B,D,H,W]")
    with torch.autocast(device_type=flow.device.type, enabled=False):
        valid = (~solid_mask.bool() & ~boundary_mask.bool()).unsqueeze(1)
        mass = F.adaptive_avg_pool3d(valid.float(), grid_size)
        target = F.adaptive_avg_pool3d(torch.where(valid, flow.float(), 0), grid_size) / mass.clamp_min(1e-12)
    return target.detach(), (mass > 0)


def condition_flow_loss(encoder, condition: Tensor, batch: dict) -> Tensor:
    prediction = encoder.predict_condition_flow(condition)
    target, valid = condition_flow_target(batch['flow'], batch['solid_mask'], batch['boundary_mask'], prediction.shape[-1])
    selected = valid.expand_as(prediction)
    if not bool(selected.any()):
        raise ValueError("condition supervision has no unknown fluid cells")
    return (prediction.float() - target).square()[selected].mean()


@torch.no_grad()
def fit_condition_flow_reference(dataset, grid_size: int, device) -> tuple[list, list]:
    """Fit a constant coarse-flow reference on training examples only."""
    if getattr(dataset, 'split', None) != 'train':
        raise ValueError('coarse flow reference must be fitted on the train split')
    sums = torch.zeros(1, 3, grid_size, grid_size, grid_size, dtype=torch.float64, device=device)
    counts = torch.zeros(1, 1, grid_size, grid_size, grid_size, dtype=torch.float64, device=device)
    for index in range(len(dataset)):
        item = dataset[index]
        flow = item['flow'].to(device).unsqueeze(0)
        solid = item['solid_mask'].to(device).unsqueeze(0)
        boundary = item['boundary_mask'].to(device).unsqueeze(0)
        target, valid = condition_flow_target(flow, solid, boundary, grid_size)
        sums += target.double() * valid
        counts += valid
    if not bool((counts > 0).any()):
        raise ValueError('training data have no unknown fluid cells')
    return (sums / counts.clamp_min(1))[0].float().cpu().tolist(), (counts > 0)[0].cpu().tolist()
