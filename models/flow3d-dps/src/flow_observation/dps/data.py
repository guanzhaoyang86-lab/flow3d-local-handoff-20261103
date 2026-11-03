"""Separate flow-only training and observation-only inference readers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from ..sparse_dataset import (FlowNormalizationStats, _boundary_tensors,
    _decode_metadata, _particle_count, _stable_seed, normalize_flow)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    temp.replace(path)


def read_prior_field(path):
    # Access only Eulerian fields / geometry: even archives with no trajectories
    # can train the unconditional prior.
    with np.load(path, allow_pickle=False) as arrays:
        field = torch.from_numpy(np.array(arrays['flow_field'], copy=True)).float()
        solid = torch.from_numpy(np.array(arrays['solid_mask'], copy=True)).bool()
    if field.ndim != 5 or field.shape[:2] != (1, 3) or min(field.shape[2:]) < 2 or not bool(torch.isfinite(field).all()):
        raise ValueError('Prior fields must be finite steady [1,3,D,H,W] volumes')
    if solid.shape != field.shape[2:] or bool(solid.all()):
        raise ValueError('Prior geometry must match field and contain fluid nodes')
    return field[0], solid


def fit_prior_stats(train_cases):
    total, squares, count = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
    for case in train_cases:
        field, solid = read_prior_field(case.path)
        values = field[:, ~solid].double()
        total += values.sum(1); squares += values.square().sum(1); count += values.shape[1]
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    return FlowNormalizationStats(mean.float(), std.float(), count, True)


class FlowPriorDataset(Dataset):
    def __init__(self, cases, stats):
        self.cases, self.stats = cases, stats

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        field, _ = read_prior_field(self.cases[index].path)
        return normalize_flow(field, self.stats)


def load_observation_case(case, split, num_particles, *, observation_seed=47,
                          num_probes=16, views=None, observation_type='projected',
                          known_boundaries=False):
    """Load explicitly selected observations, never a reference velocity field.

    Selection exactly matches SparseFlowDataset at epoch0 / observation0.
    Probe observations are returned separately and never passed to the NLL.
    Only explicit full3d mode reads trajectories_3d as observed measurements.
    Known velocity-boundary metadata is opt-in, rather than part of the image's
    trajectory-only conditioning by default.
    """
    if num_particles < 1 or num_probes < 0:
        raise ValueError('Particle count must be positive; probe count nonnegative')
    if observation_type not in ('projected', 'full3d'):
        raise ValueError('observation_type must be projected or full3d')
    if observation_type == 'full3d' and views is not None:
        raise ValueError('Camera selection does not apply to full3d observations')
    with np.load(case.path, allow_pickle=False) as arrays:
        key = 'trajectories_2d' if observation_type == 'projected' else 'trajectories_3d'
        tracks = arrays[key]
        if observation_type == 'projected':
            if tracks.ndim != 5 or tracks.shape[-1] != 2:
                raise ValueError('Stored projected observations must be [S,V,N,T,2]')
        elif tracks.ndim != 4 or tracks.shape[-1] != 3:
            raise ValueError('Stored full3d observations must be [S,N,T,3]')
        generator = torch.Generator().manual_seed(_stable_seed(observation_seed, 0, split, case.case_id, 0))
        source = int(torch.randint(tracks.shape[0], (1,), generator=generator))
        if observation_type == 'projected':
            count = _particle_count(arrays, source)
        else:
            count = int(tracks.shape[1])
            if 'particle_counts' in arrays:
                counts = np.asarray(arrays['particle_counts']).reshape(-1)
                if len(counts) not in (1, tracks.shape[0]):
                    raise ValueError('particle_counts must have one value or one per observation sample')
                count = int(counts[0 if len(counts) == 1 else source])
                if not 1 <= count <= tracks.shape[1]:
                    raise ValueError('Invalid stored full3d particle count')
        if num_particles + num_probes > count:
            raise ValueError('Not enough distinct particles for observed and probe pools')
        permutation = torch.randperm(count, generator=generator)
        indices = permutation[:num_particles].numpy()
        probes = permutation[-num_probes:].numpy() if num_probes else np.empty(0, dtype=np.int64)
        if observation_type == 'projected':
            view_indices = np.arange(tracks.shape[1]) if views is None else np.array(views, dtype=np.int64)
            if view_indices.ndim != 1 or len(view_indices) < 1 or len(set(view_indices.tolist())) != len(view_indices) or np.any((view_indices < 0) | (view_indices >= tracks.shape[1])):
                raise ValueError('View indices must be unique, valid and nonempty')
            selected = np.array(tracks[source][view_indices], copy=True)
            mask = np.array(arrays['observation_mask'][source][view_indices], copy=True)
            projection = torch.from_numpy(np.array(arrays['projection_matrix'][view_indices], copy=True)).float()
            selected_tracks, probe_tracks = selected[:, indices], selected[:, probes]
            selected_mask, probe_mask = mask[:, indices], mask[:, probes]
        else:
            view_indices = np.empty(0, dtype=np.int64)
            selected = np.array(tracks[source], copy=True)
            mask = (np.array(arrays['observation_mask_3d'][source], copy=True)
                    if 'observation_mask_3d' in arrays else np.ones(selected.shape[:-1], dtype=bool))
            selected_tracks, probe_tracks = selected[indices], selected[probes]
            selected_mask, probe_mask = mask[indices], mask[probes]
            projection = None
        if mask.shape != selected.shape[:-1]:
            raise ValueError('Stored observation mask must match trajectory axes')
        if mask.dtype != np.bool_:
            if not np.isin(mask, (0, 1)).all():
                raise ValueError('Observation mask must contain booleans or zeros/ones')
            selected_mask, probe_mask = selected_mask.astype(bool), probe_mask.astype(bool)
        solid = torch.from_numpy(np.array(arrays['solid_mask'], copy=True)).bool()
        if known_boundaries:
            metadata = _decode_metadata(arrays['metadata'] if 'metadata' in arrays else None)
            boundary, values = _boundary_tensors(solid, metadata, torch.float32)
        else:
            boundary, values = solid.clone(), torch.zeros((3, *solid.shape))
        result = {'tracks': torch.from_numpy(selected_tracks).float(),
            'mask': torch.from_numpy(selected_mask).bool(),
            'times': torch.from_numpy(np.array(arrays['observation_times'], copy=True)).float(),
            'bounds': torch.from_numpy(np.array(arrays['domain_bounds'], copy=True)).float(),
            'solid_mask': solid, 'boundary_mask': boundary, 'boundary_values': values,
            'probe_tracks': torch.from_numpy(probe_tracks).float(),
            'probe_mask': torch.from_numpy(probe_mask).bool(),
            'particle_indices': torch.from_numpy(indices), 'probe_particle_indices': torch.from_numpy(probes),
            'view_indices': view_indices.tolist(), 'source_sample_index': source,
            'case_id': case.case_id, 'flow_group_id': case.flow_group_id,
            'observation_type': observation_type, 'known_velocity_boundaries': known_boundaries}
        if projection is not None:
            result['projection'] = projection
    return result


def load_prior(path, device='cpu', *, raw_weights=False):
    from ..diffusion import GaussianDiffusion
    from ..models.factory import build_denoiser
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('method') != 'unconditional-flow-prior' or 'encoder' in value:
        raise ValueError('DPS requires a separately trained unconditional-prior checkpoint')
    if value.get('normalization_fit_split') != 'train' or value.get('conditioning') != 'none':
        raise ValueError('Prior must use train-only normalization and no trajectory conditioning')
    model = build_denoiser(value['model_config']).to(device)
    state = value['model'] if raw_weights else value['ema_model']
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError('Nonfinite prior weights')
    model.load_state_dict(state, strict=True)
    config = value['diffusion_config']
    diffusion = GaussianDiffusion(**config).to(device)
    stats = FlowNormalizationStats.from_dict(value['normalization'])
    return model, diffusion, stats, value
