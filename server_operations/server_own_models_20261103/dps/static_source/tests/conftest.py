from pathlib import Path
import json
import numpy as np
import pytest
import torch


@pytest.fixture
def small_manifest(tmp_path):
    cameras = np.array([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]], dtype=np.float32)
    times = np.array([0., .1, .2], dtype=np.float32)
    initial = np.array([[.23, .34, .45], [.43, .54, .65], [.62, .26, .39]], dtype=np.float32)
    splits = {'train': [], 'validation': [], 'test': []}
    for i, split in enumerate(('train', 'train', 'validation', 'test')):
        velocity = np.array([.08 + .02*i, .04 + .01*i, .02 + .005*i], dtype=np.float32)
        field = np.broadcast_to(velocity[None, :, None, None, None], (1, 3, 4, 4, 4)).copy()
        positions = initial[:, None] + times[None, :, None] * velocity
        tracks = np.einsum('ntc,voc->vnto', positions, cameras)[None]
        path = tmp_path / f'case_{i}.npz'
        np.savez(path, flow_field=field, solid_mask=np.zeros((4, 4, 4), dtype=bool),
                 trajectories_2d=tracks, observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
                 projection_matrix=cameras, observation_times=times,
                 domain_bounds=np.array([[0., 1.]]*3, dtype=np.float32),
                 # Poisoned truth-only fields catch accidental reads at inference.
                 initial_hidden_depth=np.array([object()], dtype=object),
                 trajectories_3d=np.array([object()], dtype=object))
        splits[split].append({'case_id': f'case_{i}', 'flow_group_id': f'flow_{i}', 'path': path.name})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'format_version': 1, 'mode': 'scientific', 'splits': splits}))
    return manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
