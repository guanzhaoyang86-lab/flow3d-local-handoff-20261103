"""Adversarial checks against the supplied posterior-sampling equations."""

import pytest
import torch

from flow_observation.dps.observations import (
    FullTrajectoryLikelihood,
    ProjectedTrajectoryLikelihood,
)


def case(dtype=torch.float64):
    bounds = torch.tensor([[0., 1.]] * 3, dtype=dtype)
    times = torch.tensor([0., .1, .4], dtype=dtype)
    initial = torch.tensor([[.23, .34, .45], [.43, .54, .65]], dtype=dtype)
    velocity = torch.tensor([.08, .04, .02], dtype=dtype)
    tracks = initial[:, None] + times[None, :, None] * velocity
    field = (velocity + .01).reshape(1, 3, 1, 1, 1).expand(1, 3, 4, 4, 4).clone()
    cameras = torch.tensor([[[1., 0, 0], [0, 1, 0]],
                            [[1., 0, 0], [0, 0, 1]]], dtype=dtype)
    projected = torch.einsum('ntc,voc->vnto', tracks, cameras)
    return bounds, times, tracks, field, cameras, projected


@pytest.mark.parametrize('rank', [2, 3])
def test_contradictory_initial_camera_observations_are_rejected(rank):
    bounds, times, _, _, cameras, _ = case()
    if rank == 2:
        cameras = cameras[:1].repeat(2, 1, 1)
    # Both cameras observe x; no single physical particle has both x=.2 and .8.
    observed = torch.tensor([[[[.2, .3], [.2, .3], [.2, .3]]],
                             [[[.8, .4], [.8, .4], [.8, .4]]]], dtype=torch.float64)
    with pytest.raises(ValueError, match='inconsistent'):
        ProjectedTrajectoryLikelihood(observed, cameras, times, bounds, sigma=.1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('rank', [2, 3])
def test_consistent_scaled_oblique_cameras_survive_floating_roundoff(dtype, rank):
    bounds, times, tracks, _, _, _ = case(dtype)
    cameras = torch.tensor([[[1., .3, .2], [.1, 1., .2]],
                            [[1., .2, .4], [.2, .3, 1.]]], dtype=dtype) * 100
    if rank == 2:
        cameras = cameras[:1].repeat(2, 1, 1)
    # Non-unit camera scales and a 31-cell domain exercise scale-aware tolerance.
    tracks = tracks * 31
    projected = torch.einsum('ntc,voc->vnto', tracks, cameras)
    likelihood = ProjectedTrajectoryLikelihood(projected, cameras, times, bounds * 31, sigma=.1)
    depth = likelihood.initialize_depth(random=False)
    reconstructed = torch.einsum('nc,voc->vno', likelihood.initial_positions(depth), cameras)
    torch.testing.assert_close(reconstructed, projected[:, :, 0], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('projected', [False, True])
def test_independent_particle_likelihood_is_additive(projected):
    bounds, times, tracks, field, cameras, projections = case()
    if projected:
        joint = ProjectedTrajectoryLikelihood(projections, cameras, times, bounds, sigma=.1)
        value = joint.nll(field, joint.initialize_depth())
        individual = []
        for particle in range(2):
            likelihood = ProjectedTrajectoryLikelihood(projections[:, particle:particle + 1], cameras, times, bounds, sigma=.1)
            individual.append(likelihood.nll(field, likelihood.initialize_depth()))
    else:
        value = FullTrajectoryLikelihood(tracks, times, bounds, sigma=.1).nll(field)
        individual = [FullTrajectoryLikelihood(track[None], times, bounds, sigma=.1).nll(field)
                      for track in tracks]
    assert value > 0
    torch.testing.assert_close(value, sum(individual))


def test_full_trajectory_rejects_unsupported_time_varying_field():
    bounds, times, tracks, field, _, _ = case()
    likelihood = FullTrajectoryLikelihood(tracks, times + 10, bounds, sigma=.1)
    with pytest.raises(ValueError, match='steady'):
        likelihood.nll(field.repeat(2, 1, 1, 1, 1))
    with pytest.raises(ValueError, match='steady'):
        likelihood.replay(field.repeat(2, 1, 1, 1, 1))


def test_full_trajectory_diagnostics_replay_observed_initial_positions():
    bounds, times, tracks, field, _, _ = case()
    # Correct constant velocity gives zero Euler residual and zero replay error.
    field -= .01
    likelihood = FullTrajectoryLikelihood(tracks, times, bounds, sigma=.1)
    positions, validity = likelihood.replay(field)
    torch.testing.assert_close(positions[0], tracks)
    assert validity.all()
    diagnostics = likelihood.diagnostics(field)
    assert diagnostics['nll'] < 1e-25
    assert diagnostics['residual_rmse'] < 1e-14
    assert diagnostics['observed_track_rmse_cells'] < 1e-14
    assert diagnostics['all_replay_particles_valid']
    assert diagnostics['valid_replay_fraction'] == 1.
    assert diagnostics['unknown_initial_depth_count'] == 0


def test_full_trajectory_replay_does_not_invent_missing_initial_positions():
    bounds, times, tracks, field, _, _ = case()
    mask = torch.ones(tracks.shape[:-1], dtype=torch.bool)
    mask[:, 0] = False
    likelihood = FullTrajectoryLikelihood(tracks, times, bounds, sigma=.1, mask=mask)
    assert torch.isfinite(likelihood.nll(field))  # Later observed transitions remain usable.
    with pytest.raises(ValueError, match='initial 3D position'):
        likelihood.diagnostics(field)


def test_projected_clamp_surrogate_reports_hidden_domain_escape_without_new_penalty():
    bounds, _, _, field, cameras, _ = case()
    times = torch.tensor([0., 1.], dtype=torch.float64)
    observed = torch.tensor([[[[.5, .5], [.5, .5]]]], dtype=torch.float64)
    field.zero_(); field[:, 2] = 10.
    likelihood = ProjectedTrajectoryLikelihood(observed, cameras[:1], times, bounds, sigma=.1)
    # The screenshot's projected residual cannot see this hidden-direction exit.
    # We report it explicitly; adding a no-exit penalty would change the model.
    diagnostics = likelihood.diagnostics(field, likelihood.initialize_depth(random=False))
    assert diagnostics['nll'] == 0.
    assert diagnostics['observed_track_rmse_cells'] == 0.
    assert diagnostics['all_replay_particles_valid'] is False
    assert diagnostics['valid_replay_fraction'] == .5


def test_full_trajectory_clamp_replay_reports_domain_escape():
    bounds, _, _, field, _, _ = case()
    times = torch.tensor([0., 1.], dtype=torch.float64)
    observed = torch.tensor([[[.5, .5, .5], [.5, .5, .5]]], dtype=torch.float64)
    field.zero_(); field[:, 2] = 10.
    diagnostics = FullTrajectoryLikelihood(observed, times, bounds, sigma=.1).diagnostics(field)
    assert diagnostics['nll'] > 0
    assert diagnostics['all_replay_particles_valid'] is False
    assert diagnostics['valid_replay_fraction'] == .5
