"""Scientific protocol regressions discovered in the screenshot conformance review."""
import json
import numpy as np
import pytest
import torch

from flow_observation.diffusion import GaussianDiffusion
from flow_observation.dps import inference
from flow_observation.dps.data import load_observation_case
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.dps.sampler import likelihood_gradient
from flow_observation.dps.training import train
from flow_observation.sparse_dataset import load_sparse_flow_manifest
from test_dps import training_args, LinearPrior, constant_case, CAMERAS, BOUNDS


def _sample_args(prior, manifest, output, extra=()):
    return inference.parser().parse_args(['--checkpoint', str(prior), '--manifest', str(manifest),
        '--output', str(output), '--device', 'cpu', '--sigma', '.03', '--num-particles', '2',
        '--num-probes', '1', '--num-samples', '1', '--sampler', 'ddim', '--sampling-steps', '4', *extra])


def _rewrite(path, **updates):
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files if k not in ('initial_hidden_depth', 'trajectories_3d')}
    values.update(updates)
    np.savez(path, **values)


def test_explicit_full3d_cli_runs_without_projected_tracks_or_hidden_truth(small_manifest, tmp_path):
    _, cases = load_sparse_flow_manifest(small_manifest)
    case = cases['validation'][0]
    with np.load(case.path, allow_pickle=False) as arrays:
        projected = arrays['trajectories_2d']
        xyz = np.stack([projected[:, 0, :, :, 0], projected[:, 0, :, :, 1], projected[:, 1, :, :, 1]], axis=-1)
        values = {k: arrays[k] for k in ('flow_field', 'solid_mask', 'domain_bounds', 'observation_times')}
    # No 2D observations or camera matrices are available at all in this test.
    np.savez(case.path, **values, trajectories_3d=xyz,
             initial_hidden_depth=np.array([object()], dtype=object))
    prior = train(training_args(small_manifest, tmp_path/'full3d_prior', 1))
    output = tmp_path/'full3d.npz'
    inference.sample(_sample_args(prior, small_manifest, output, ['--observation-type', 'full3d', '--evaluate']))
    with np.load(output, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays['metadata']))
        assert arrays['observed_tracks'].shape == (2, 3, 3)
        assert arrays['initial_depth_samples'].shape == (1, 0)
        assert 'projection_matrix' not in arrays
        assert np.isfinite(arrays['posterior_samples']).all()
    assert metadata['observation_type'] == 'full3d'
    assert metadata['hidden_depth_update'] == 'not_applicable'
    assert metadata['diagnostics'][0]['unknown_initial_depth_count'] == 0
    assert len(metadata['posthoc_evaluation']['probe_metrics']) == 1
    assert metadata['boundary_velocity_conditioning'] == 'none'


def test_default_does_not_use_true_lid_speed_and_known_boundary_is_opt_in(small_manifest, tmp_path):
    _, cases = load_sparse_flow_manifest(small_manifest)
    case = cases['validation'][0]
    prior = train(training_args(small_manifest, tmp_path/'boundary_prior', 1))
    samples = []
    for index, speed in enumerate((.4, 100.)):
        _rewrite(case.path, metadata=np.array(json.dumps({'lid_face': 'x=max', 'lid_velocity_xyz': [0., 0., speed]})))
        output = tmp_path/f'boundary_{index}.npz'
        inference.sample(_sample_args(prior, small_manifest, output))
        with np.load(output) as arrays:
            samples.append(arrays['posterior_samples'].copy())
            assert json.loads(str(arrays['metadata']))['boundary_velocity_conditioning'] == 'none'
    assert np.array_equal(*samples)
    explicit = tmp_path/'known_boundary.npz'
    inference.sample(_sample_args(prior, small_manifest, explicit, ['--boundary-projection', 'known']))
    with np.load(explicit) as arrays:
        values = arrays['posterior_samples']
        assert np.all(values[:, 2, :, :, -1] == 100.)
        assert np.all(values[:, :2, :, :, -1] == 0.)
        assert json.loads(str(arrays['metadata']))['boundary_velocity_conditioning'] == 'known'


def test_zero_guidance_control_is_not_labeled_as_likelihood_guided(small_manifest, tmp_path):
    prior = train(training_args(small_manifest, tmp_path/'zero_guidance_prior', 1))
    output = tmp_path/'prior_only.npz'
    inference.sample(_sample_args(prior, small_manifest, output, ['--guidance-step', '0', '--depth-step', '0']))
    with np.load(output) as arrays:
        metadata = json.loads(str(arrays['metadata']))
    assert metadata['conditioning'] == 'unconditional-prior-only'
    assert metadata['field_likelihood_guidance_enabled'] is False


def test_dps_rejects_a_detached_likelihood_gradient_path():
    field, _, tracks, times = constant_case()
    observation = ProjectedTrajectoryLikelihood(tracks, CAMERAS, times, BOUNDS, sigma=.1)
    diffusion = GaussianDiffusion(8, prediction_type='v').double()
    with pytest.raises(RuntimeError, match='disconnected'):
        likelihood_gradient(LinearPrior(), diffusion, field, torch.tensor([3]), observation,
                            lambda clean: clean.detach(), observation.initialize_depth())


def test_require_valid_replay_does_not_publish_invalid_success(small_manifest, tmp_path, monkeypatch):
    prior = train(training_args(small_manifest, tmp_path/'validity_prior', 1))
    output = tmp_path/'must_not_exist.npz'
    def invalid(*args, **kwargs):
        shape = args[2]
        return {'sample': torch.full(shape, 1e6), 'depth': kwargs['initial_depth'], 'trace': []}
    monkeypatch.setattr(inference, 'dps_sample', invalid)
    with pytest.raises(RuntimeError, match='leaves the domain'):
        inference.sample(_sample_args(prior, small_manifest, output, ['--require-valid-replay']))
    assert not output.exists()
    assert not output.with_suffix('.json').exists()
