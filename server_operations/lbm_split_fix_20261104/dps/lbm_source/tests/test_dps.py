import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from flow_observation.diffusion import GaussianDiffusion
from flow_observation.dps.observations import FullTrajectoryLikelihood, ProjectedTrajectoryLikelihood
from flow_observation.dps.sampler import dps_sample, likelihood_gradient
from flow_observation.dps.data import fit_prior_stats, load_observation_case, load_prior, read_prior_field
from flow_observation.dps.training import parser as train_parser, train
from flow_observation.dps.inference import parser as sample_parser, sample
from flow_observation.dps.audit import verify_prior, verify_posterior
from flow_observation.sparse_dataset import SparseFlowDataset, load_sparse_flow_manifest

DTYPE = torch.float64
BOUNDS = torch.tensor([[0., 1.]]*3, dtype=DTYPE)
CAMERAS = torch.tensor([[[1., 0, 0], [0, 1, 0]], [[1., 0, 0], [0, 0, 1]], [[0., 1, 0], [0, 0, 1]]], dtype=DTYPE)


def constant_case():
    velocity = torch.tensor([.08, .04, .02], dtype=DTYPE)
    times = torch.tensor([0., .1, .4], dtype=DTYPE)
    initial = torch.tensor([[.23, .34, .45], [.43, .54, .65]], dtype=DTYPE)
    trajectories = initial[:, None] + velocity * times[None, :, None]
    projected = torch.einsum('ntc,voc->vnto', trajectories, CAMERAS)
    field = velocity.reshape(1, 3, 1, 1, 1).expand(1, 3, 4, 4, 4).clone()
    return field, trajectories, projected, times


class LinearPrior(nn.Module):
    def __init__(self, weight=.2):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(weight, dtype=DTYPE))

    def forward(self, value, times, condition):
        assert condition is None
        return self.weight * value


def test_full_3d_likelihood_matches_equation_two_and_sigma_scaling():
    field, tracks, _, times = constant_case()
    likelihood = FullTrajectoryLikelihood(tracks, times, BOUNDS, sigma=.1)
    assert float(likelihood.nll(field)) < 1e-24
    wrong = field.clone(); wrong[:, 0] += .03
    expected = len(tracks) * (.03 * times.diff()).square().sum() / (2 * .1**2)
    assert torch.allclose(likelihood.nll(wrong), expected, atol=1e-14)
    smaller_sigma = FullTrajectoryLikelihood(tracks, times, BOUNDS, sigma=.05)
    assert torch.allclose(smaller_sigma.nll(wrong), 4 * expected, atol=1e-14)


def test_projected_euler_likelihood_and_triangulation_use_same_physical_particles():
    field, tracks, projected, times = constant_case()
    likelihood = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)
    depth = likelihood.initialize_depth()
    assert not bool(likelihood.depth_active.any())
    assert torch.allclose(likelihood.initial_positions(depth), tracks[:, 0], atol=1e-12)
    assert float(likelihood.nll(field, depth)) < 1e-24


def test_single_view_depth_gradient_and_recovery_without_true_depth():
    times = torch.tensor([0., 1., 2.], dtype=DTYPE)
    # Shear: observed x motion depends on the unobserved initial z coordinate.
    zz = torch.linspace(0, 1, 5, dtype=DTYPE)[:, None, None].expand(5, 5, 5)
    field = torch.stack([.1 * zz, torch.zeros_like(zz), torch.zeros_like(zz)])[None]
    observed = torch.tensor([[[[.2, .3], [.27, .3], [.34, .3]]]], dtype=DTYPE)
    likelihood = ProjectedTrajectoryLikelihood(observed, CAMERAS[:1], times, BOUNDS, sigma=.05)
    assert int(likelihood.depth_active.sum()) == 1
    depth = likelihood.initialize_depth(random=False).requires_grad_()
    before = likelihood.nll(field, depth)
    gradient = torch.autograd.grad(before, depth)[0]
    epsilon = 1e-5
    numerical = (likelihood.nll(field, depth + epsilon) - likelihood.nll(field, depth - epsilon)) / (2 * epsilon)
    assert torch.allclose(gradient[0], numerical, atol=1e-8)
    assert abs(float(gradient[0])) > .1
    for _ in range(30):
        depth = depth.detach().requires_grad_()
        gradient = torch.autograd.grad(likelihood.nll(field, depth), depth)[0]
        depth = likelihood.reflect_depth(depth - .05 * gradient).detach()
    assert float(likelihood.nll(field, depth)) < float(before.detach()) * .001
    assert abs(float(likelihood.initial_positions(depth)[0, 2]) - .7) < .01


def test_masked_nonfinite_observations_are_ignored_but_valid_nans_are_rejected():
    field, _, projected, times = constant_case()
    projected[0, 0, 1] = torch.nan
    mask = torch.ones(projected.shape[:-1], dtype=torch.bool); mask[0, 0, 1] = False
    likelihood = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1, mask=mask)
    assert torch.isfinite(likelihood.nll(field, likelihood.initialize_depth()))
    with pytest.raises(ValueError, match='finite'):
        ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)


def test_invalid_replayed_particles_are_reported_and_not_removed_from_likelihood():
    field, _, projected, times = constant_case()
    likelihood = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)
    field[:, 0] = 100.
    diagnostics = likelihood.diagnostics(field, likelihood.initialize_depth())
    assert diagnostics['all_replay_particles_valid'] is False
    assert diagnostics['nll'] > 100


def test_noisy_state_gradient_includes_denoiser_jacobian_and_matches_finite_difference():
    field, _, projected, times = constant_case()
    observation = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)
    depth = observation.initialize_depth(); model = LinearPrior(.3)
    diffusion = GaussianDiffusion(8, prediction_type='v').double()
    noisy = torch.full_like(field, .1); t = torch.tensor([3])
    _, gradient, _, _ = likelihood_gradient(model, diffusion, noisy, t, observation, lambda x:x, depth)
    location = (0, 0, 1, 1, 1); epsilon = 1e-5
    def energy(value):
        clean = diffusion.model_predictions(value, t, model(value, t, None))['predicted_x0']
        return observation.nll(clean, depth)
    plus = noisy.clone(); minus = noisy.clone(); plus[location] += epsilon; minus[location] -= epsilon
    numerical = (energy(plus) - energy(minus)) / (2 * epsilon)
    assert abs(float(gradient[location])) > 1e-6
    assert torch.allclose(gradient[location], numerical, atol=1e-8, rtol=1e-5)
    # A detached-clean approximation loses this chain factor.
    clean = diffusion.model_predictions(noisy, t, model(noisy, t, None))['predicted_x0'].detach().requires_grad_()
    clean_gradient = torch.autograd.grad(observation.nll(clean, depth), clean)[0]
    assert not torch.allclose(gradient, clean_gradient)


def test_zero_guidance_reproduces_unconditional_ddim_and_restores_model_state():
    field, _, projected, times = constant_case()
    observation = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)
    diffusion = GaussianDiffusion(8, prediction_type='v').double()
    model = LinearPrior(); model.train()
    initial = torch.randn(field.shape, dtype=DTYPE, generator=torch.Generator().manual_seed(11))
    expected = diffusion.ddim_sample(model, tuple(field.shape), None, initial_noise=initial,
        sampling_steps=4, eta=0., guidance_scale=1.)
    actual = dps_sample(model, diffusion, tuple(field.shape), observation, initial_noise=initial,
        method='ddim', sampling_steps=4, eta=0., guidance_step=0., depth_step=0.)
    assert torch.allclose(actual['sample'], expected, atol=1e-12)
    assert model.training and model.weight.requires_grad
    assert all(r['guidance_update_rms_raw'] == 0. for r in actual['trace'])


def test_ddpm_is_seed_reproducible_and_guidance_changes_the_sample():
    field, _, projected, times = constant_case()
    observation = ProjectedTrajectoryLikelihood(projected, CAMERAS, times, BOUNDS, sigma=.1)
    diffusion = GaussianDiffusion(6, prediction_type='v').double(); model = LinearPrior()
    def draw(step):
        return dps_sample(model, diffusion, tuple(field.shape), observation,
            generator=torch.Generator().manual_seed(5), dtype=DTYPE, guidance_step=step)['sample']
    a, b, prior = draw(.01), draw(.01), draw(0.)
    assert torch.equal(a, b) and torch.isfinite(a).all()
    assert not torch.equal(a, prior)
    with pytest.raises(ValueError, match='all timesteps'):
        dps_sample(model, diffusion, tuple(field.shape), observation, sampling_steps=3)


def training_args(manifest, out, epochs=2, resume=None):
    args = ['--manifest', str(manifest), '--output-dir', str(out), '--device', 'cpu',
        '--architecture', 'dit3d', '--epochs', str(epochs), '--dit-hidden-dim', '24',
        '--dit-depth', '1', '--dit-heads', '3', '--patch-size', '2', '--diffusion-steps', '8', '--save-every', '1']
    if resume:
        args += ['--resume', str(resume)]
    return train_parser().parse_args(args)


def test_flow_only_prior_reader_works_without_any_track_arrays(tmp_path):
    path = tmp_path / 'flow_only.npz'
    np.savez(path, flow_field=np.ones((1, 3, 4, 4, 4), dtype=np.float32), solid_mask=np.zeros((4,4,4), dtype=bool))
    field, solid = read_prior_field(path)
    assert field.shape == (3,4,4,4) and not solid.any()


def test_observation_reader_matches_existing_selection_and_never_reads_truth(small_manifest, tmp_path):
    _, cases = load_sparse_flow_manifest(small_manifest)
    case = cases['validation'][0]
    current = load_observation_case(case, 'validation', 2, num_probes=1)
    reference = SparseFlowDataset(small_manifest, 'validation', num_particles=2, num_probe_particles=1,
        observations_per_flow=1, seed=47, normalize=False)[0]
    for key in ('tracks', 'mask', 'projection', 'times', 'bounds', 'particle_indices', 'probe_particle_indices'):
        assert torch.equal(current[key], reference[key]), key
    with np.load(case.path, allow_pickle=False) as archive:
        # Rewrite the reference flow as a pickle-poisoned member: loading the
        # observations must still work because the flow member is never read.
        safe = {k: archive[k] for k in archive.files if k not in ('flow_field', 'trajectories_3d', 'initial_hidden_depth')}
    np.savez(case.path, **safe, flow_field=np.array([object()], dtype=object))
    checked = load_observation_case(case, 'validation', 2, num_probes=1)
    assert torch.equal(checked['tracks'], current['tracks'])


def test_prior_training_resume_matches_continuous_training(small_manifest, tmp_path):
    first = tmp_path / 'resumed'; continuous = tmp_path / 'continuous'
    train(training_args(small_manifest, first, 1))
    train(training_args(small_manifest, first, 2, first / 'latest.pt'))
    train(training_args(small_manifest, continuous, 2))
    a = torch.load(first / 'latest.pt', weights_only=True)
    b = torch.load(continuous / 'latest.pt', weights_only=True)
    assert a['epoch'] == b['epoch'] == 2
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert all(torch.equal(a['ema_model'][k], b['ema_model'][k]) for k in a['ema_model'])
    assert a['conditioning'] == 'none' and 'encoder' not in a


def test_train_and_posterior_sample_end_to_end_with_posthoc_metrics(small_manifest, tmp_path):
    prior = train(training_args(small_manifest, tmp_path / 'prior'))
    output = tmp_path / 'posterior.npz'
    args = sample_parser().parse_args(['--checkpoint', str(prior), '--manifest', str(small_manifest),
        '--output', str(output), '--device', 'cpu', '--sigma', '.03', '--num-particles', '2',
        '--num-probes', '1', '--num-samples', '2', '--sampler', 'ddim', '--sampling-steps', '4', '--evaluate'])
    sample(args)
    with np.load(output, allow_pickle=False) as result:
        assert result['posterior_samples'].shape == (2,3,4,4,4)
        assert np.isfinite(result['posterior_samples']).all()
        metadata = json.loads(str(result['metadata']))
    assert metadata['reference_used_during_inference'] is False
    assert metadata['conditioning'] == 'trajectory-likelihood-gradient-through-denoiser'
    assert len(metadata['posthoc_evaluation']['per_sample_field_relative_l2_unknown_interior']) == 2
    assert len(metadata['posthoc_evaluation']['probe_metrics']) == 2
    assert (output.with_suffix('.json')).is_file()
    assert verify_prior(prior.parent, 2)['status'] == 'verified'
    assert verify_posterior(output, 2, 2)['status'] == 'verified'


def test_single_view_dps_and_probe_evaluation_run_without_truth_depth(small_manifest, tmp_path):
    prior = train(training_args(small_manifest, tmp_path / 'single_view_prior', 1))
    output = tmp_path / 'single_view.npz'
    args = sample_parser().parse_args(['--checkpoint', str(prior), '--manifest', str(small_manifest),
        '--output', str(output), '--device', 'cpu', '--sigma', '.03', '--num-particles', '2',
        '--num-probes', '1', '--num-samples', '1', '--views', '0', '--sampler', 'ddim',
        '--sampling-steps', '4', '--evaluate'])
    sample(args)
    with np.load(output, allow_pickle=False) as result:
        assert result['observed_tracks'].shape == (1, 2, 3, 2)
        assert np.isfinite(result['initial_depth_samples']).all()
        metadata = json.loads(str(result['metadata']))
    assert metadata['diagnostics'][0]['unknown_initial_depth_count'] == 2
    assert metadata['hidden_depth_update'] == 'langevin'
    assert metadata['posthoc_evaluation']['probe_metrics'][0]['probe_track_rmse_cells'] is not None


def test_posterior_is_independent_of_reference_flow_and_probe_trajectory_values(small_manifest, tmp_path):
    prior = train(training_args(small_manifest, tmp_path / 'independence_prior', 1))
    outputs = []
    _, cases = load_sparse_flow_manifest(small_manifest)
    case = cases['validation'][0]
    selected = load_observation_case(case, 'validation', 2, num_probes=1)
    for run in range(2):
        output = tmp_path / f'independence_{run}.npz'
        args = sample_parser().parse_args(['--checkpoint', str(prior), '--manifest', str(small_manifest),
            '--output', str(output), '--device', 'cpu', '--sigma', '.03', '--num-particles', '2',
            '--num-probes', '1', '--num-samples', '1', '--sampler', 'ddim', '--sampling-steps', '4'])
        sample(args)
        with np.load(output) as result:
            outputs.append(result['posterior_samples'].copy())
        if run == 0:
            with np.load(case.path, allow_pickle=False) as archive:
                safe = {k: archive[k] for k in archive.files if k not in ('flow_field', 'trajectories_3d', 'initial_hidden_depth')}
            safe['trajectories_2d'][:, :, selected['probe_particle_indices'].numpy(), :, :] += 1000
            np.savez(case.path, **safe, flow_field=np.array([object()], dtype=object))
    assert np.array_equal(*outputs)


def test_conditional_checkpoint_is_not_accepted_as_an_unconditional_prior(tmp_path):
    path = tmp_path / 'conditional.pt'
    torch.save({'model_config': {}, 'encoder': {}}, path)
    with pytest.raises(ValueError, match='unconditional'):
        load_prior(path)


def test_empty_launcher_output_cannot_pass_artifact_checks(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_prior(tmp_path, 100)
    with pytest.raises(FileNotFoundError):
        verify_posterior(tmp_path/'missing.npz', 2, 2)


@pytest.mark.parametrize('script', ['train_flow_prior.py', 'sample_flow_dps.py'])
def test_cli_help(script):
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run([sys.executable, str(root/'scripts'/script), '--help'], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert '--manifest' in process.stdout
