import copy
import torch
from torch import nn
import runtime as r
from flow_observation.dps.sampler import likelihood_gradient, dps_sample

def fixture():
    times = torch.tensor([0., .07, .23], dtype=torch.float64)
    bounds = torch.tensor([[0., 1.], [0., 1.], [0., 1.]], dtype=torch.float64)
    projection = torch.tensor([[[1., 0., 0.], [0., 0., 1.]]], dtype=torch.float64)
    tracks = torch.tensor([[[[.2, .3], [.213, .3], [.26, .3]],
                            [[.35, .6], [.363, .6], [.41, .6]]]], dtype=torch.float64)
    observation = {'tracks': tracks, 'mask': torch.ones(1, 2, 3, dtype=torch.bool),
        'times': times, 'projection': projection, 'bounds': bounds, 'shape': (4, 4, 4)}
    y = torch.linspace(0, 1, 4, dtype=torch.float64).view(1, 1, 4, 1)
    field = torch.zeros(3, 3, 4, 4, 4, dtype=torch.float64)
    field[:, 0] = ((.08 + .15 * y) * torch.tensor([1., 1.8, 3.]).view(3, 1, 1, 1)).expand(-1, 4, -1, 4)
    return observation, field, {'integrator': 'rk4', 'trajectory_substeps': 4, 'likelihood': 'transition'}

class LinearPrior(nn.Module):
    def forward(self, field, timestep, condition):
        assert len(timestep) == len(field) and condition is None
        return field * .2

def test_native_time_likelihood_and_depth_gradients():
    observation, field, sampling = fixture()
    likelihood = r.make_likelihood(observation, .05, sampling)
    assert torch.equal(likelihood.times, observation['times'])
    depth = likelihood.initialize_depth(torch.Generator().manual_seed(47)).requires_grad_()
    field = field.requires_grad_(); nll = likelihood.nll(field, depth)
    fg, dg = torch.autograd.grad(nll, (field, depth))
    assert torch.isfinite(fg).all() and torch.isfinite(dg).all()
    assert (fg.flatten(1).abs().sum(1) > 0).sum() >= 2 and dg.abs().max() > 1e-8
    coordinate = int(dg.abs().argmax()); epsilon = 1e-5
    plus = depth.detach().clone(); minus = plus.clone()
    plus[coordinate] += epsilon; minus[coordinate] -= epsilon
    numerical = (likelihood.nll(field.detach(), plus) - likelihood.nll(field.detach(), minus)) / (2 * epsilon)
    assert torch.allclose(numerical, dg[coordinate], rtol=2e-4, atol=1e-6)
    uniform = copy.deepcopy(observation); uniform['times'] = torch.tensor([0., .115, .23], dtype=torch.float64)
    assert not torch.allclose(nll, r.make_likelihood(uniform, .05, sampling).nll(field, depth))
    positions, valid = likelihood.replay(field.detach(), depth.detach())
    assert positions.shape == (1, 2, 3, 3) and valid.all()

def test_full_time_batch_dps_and_noisy_state_gradient():
    observation, field, sampling = fixture()
    likelihood = r.make_likelihood(observation, .05, sampling)
    diffusion = r.GaussianDiffusion(num_steps=10, prediction_type='v', schedule='cosine', clip_x0=None).double()
    model = LinearPrior(); depth = likelihood.initialize_depth(torch.Generator().manual_seed(47))
    timestep = torch.full((3,), 4, dtype=torch.long)
    _, gradient, dg, nll = likelihood_gradient(model, diffusion, field, timestep, likelihood, lambda x: x, depth)
    coordinate = int(gradient.abs().flatten().argmax()); epsilon = 1e-5
    plus = field.clone(); minus = field.clone()
    plus.flatten()[coordinate] += epsilon; minus.flatten()[coordinate] -= epsilon
    def energy(value):
        return likelihood.nll(diffusion.model_predictions(value, timestep, model(value, timestep, None))['predicted_x0'], depth)
    numerical = (energy(plus) - energy(minus)) / (2 * epsilon)
    assert torch.allclose(numerical, gradient.flatten()[coordinate], rtol=2e-4, atol=1e-6)
    result = dps_sample(model, diffusion, tuple(field.shape), likelihood, method='ddim', sampling_steps=3,
        eta=0., initial_noise=field, initial_depth=depth, guidance_step=1e-5, depth_step=1e-5,
        generator=torch.Generator().manual_seed(47), dtype=torch.float64)
    assert result['sample'].shape == field.shape and len(result['trace']) == 3
    assert torch.isfinite(result['sample']).all() and torch.isfinite(result['depth']).all()
    assert ((result['depth'] >= likelihood.lower) & (result['depth'] <= likelihood.upper)).all()

def test_float32_posterior_native_times_without_reference(monkeypatch):
    observation, field, sampling = fixture()
    observation = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point()
        and k != 'times' else v) for k, v in observation.items()}
    def forbidden(*a, **kw):
        raise AssertionError('Reference read during posterior reconstruction')
    monkeypatch.setattr(r.physics, 'load_reference', forbidden)
    monkeypatch.setattr(r.np, 'load', forbidden)
    diffusion = r.GaussianDiffusion(num_steps=10, prediction_type='v', schedule='cosine', clip_x0=None)
    sampling.update({'num_samples': 2, 'seed': 47, 'sampler': 'ddim', 'sampling_steps': 3,
        'eta': 0., 'guidance_step': 1e-5, 'gradient_objective': 'nll', 'max_update_rms': .1,
        'depth_step': 1e-5, 'depth_mode': 'langevin'})
    fields, depths, traces, valid = r.reconstruct(LinearPrior(), diffusion,
        {'mean': [0., 0., 0.], 'std': [1., 1., 1.]}, observation, sampling, .05, 0)
    assert fields.shape == (2, 3, 3, 4, 4, 4) and fields.dtype == torch.float32
    assert observation['times'].dtype == torch.float64
    assert torch.isfinite(fields).all() and torch.isfinite(depths).all()
    assert len(traces) == 2 and all(len(t) == 3 for t in traces)
