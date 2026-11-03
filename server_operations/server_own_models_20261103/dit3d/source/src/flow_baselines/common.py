"""Shared data protocol and model construction."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from flow_observation.sparse_dataset import SparseFlowDataset, _stable_seed, _particle_count
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.unet3d import ConditionalUNet3D
from flow_observation.models.dit import ConditionalDiT3D


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
            num_probe_particles=config['num_probes'], seed=config['observation_seed'],
            normalize=True, normalization_stats=stats)

    def _load_case(self, case):
        loaded = super()._load_case(case)
        self._selection_arrays = loaded['arrays']
        return loaded

    def __getitem__(self, index):
        row = super().__getitem__(index)
        arrays = self._selection_arrays
        del self._selection_arrays
        # Epoch-zero order matches the observation-only inference loader exactly.
        # Reserve its probe tail permanently; only the remaining observation pool
        # changes across training epochs. Model initialization seeds never enter.
        generator = torch.Generator().manual_seed(_stable_seed(
            self.config['observation_seed'], 0, self.split, row['case_id'],
            int(row['observation_index'])))
        source = int(torch.randint(arrays['trajectories_2d'].shape[0], (1,), generator=generator))
        count = _particle_count(arrays, source)
        permutation = torch.randperm(count, generator=generator)
        probes = permutation[-self.num_probe_particles:] if self.num_probe_particles else permutation[:0]
        candidates = permutation[:-self.num_probe_particles] if self.num_probe_particles else permutation
        if self.split == 'train' and self.epoch:
            epoch_generator = torch.Generator().manual_seed(_stable_seed(
                self.config['observation_seed'], self.epoch, self.split,
                row['case_id'], int(row['observation_index']), 'observations_only'))
            candidates = candidates[torch.randperm(len(candidates), generator=epoch_generator)]
        selected = candidates[:self.num_particles]
        for prefix, indices in (('', selected), ('probe_', probes)):
            row[prefix+'tracks'] = torch.from_numpy(np.array(arrays['trajectories_2d'][source][:, indices.numpy()], copy=True)).to(self.dtype)
            row[prefix+'mask'] = torch.from_numpy(np.array(arrays['observation_mask'][source][:, indices.numpy()], copy=True)).bool()
        row['particle_indices'], row['probe_particle_indices'] = selected, probes
        row['source_sample_index'] = torch.tensor(source, dtype=torch.long)
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
        if kind == 'dit3d':
            decoder = ConditionalDiT3D(spatial_shape=shape, condition_dim=config['condition_dim'], **config[kind])
        elif kind == 'unet_diffusion':
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
