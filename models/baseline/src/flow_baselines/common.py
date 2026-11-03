"""Shared data protocol and model construction."""
from pathlib import Path
import torch
from torch import nn
from flow_observation.sparse_dataset import SparseFlowDataset
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.unet3d import ConditionalUNet3D


def check_rectangular_geometry(solid_mask):
    """Require a nonempty rectangular cavity without internal obstacles."""
    solid = torch.as_tensor(solid_mask)
    if solid.dtype != torch.bool or solid.ndim != 3 or solid.numel() == 0:
        raise ValueError('solid_mask must be a nonempty three-dimensional boolean array')
    if bool(solid[1:-1, 1:-1, 1:-1].any()):
        raise ValueError('Internal solid obstacles are unsupported; use a rectangular cavity')
    if not bool((~solid).any()):
        raise ValueError('Geometry must contain fluid voxels')


class TrainingDataset(SparseFlowDataset):
    def __init__(self, manifest, split, config, stats):
        self.config = config
        super().__init__(manifest, split, num_particles=config['num_particles'],
            seed=config['seed'] if split == 'train' else config['seed']+1,
            normalize=True, normalization_stats=stats)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        check_rectangular_geometry(row['solid_mask'])
        if self.config['views'] is not None:
            indices = torch.as_tensor(self.config['views'], dtype=torch.long)
            if indices.ndim != 1 or len(indices) < 1 or len(set(indices.tolist())) != len(indices) or bool((indices < 0).any()) or bool((indices >= row['tracks'].shape[0]).any()):
                raise ValueError('Invalid or repeated camera indices')
            for key in ('tracks', 'mask', 'projection'):
                row[key] = row[key][indices]
        if self.config['boundary_projection'] == 'none':
            row['boundary_mask'] = row['solid_mask'].clone()
            row['boundary_values'] = torch.zeros_like(row['boundary_values'])
        return row


def encode(encoder, row):
    return encoder(row['tracks'], row['mask'], row['projection'], row['times'], row['bounds'])


def to_device(row, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in row.items()}


def build_system(kind, shape, config, motion_scales):
    from .models import build_decoder
    encoder = TrackSetEncoder(config['condition_dim'], variant='motion-residual-v2',
        motion_scales=motion_scales, condition_flow_grid=config['condition_flow_grid'])
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(config['seed']+104729).get_state())
        if kind == 'unet_diffusion':
            decoder = ConditionalUNet3D(condition_dim=config['condition_dim'], **config[kind])
        else:
            decoder = build_decoder(kind, shape, condition_dim=config['condition_dim'], **config[kind])
    return nn.ModuleDict({'encoder': encoder, 'decoder': decoder})


def masked_mse(prediction, target, solid):
    valid = (~solid.bool())[:, None].expand_as(target)
    if not bool(valid.any()):
        raise ValueError('No fluid voxels for supervised loss')
    return (prediction.float()-target.float()).square()[valid].mean()


def no_output_collision(output):
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Output or completion record exists; choose a fresh path')
