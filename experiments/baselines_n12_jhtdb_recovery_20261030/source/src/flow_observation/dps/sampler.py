"""DPS gradients through the denoiser, with DDPM or DDIM prior transitions.

Unlike the legacy clean-space guidance hook, the likelihood gradient is with
respect to x_t and includes d x0_hat / d x_t. Hidden depths can be updated by
reflected unadjusted Langevin dynamics (approximate) or explicitly labeled MAP.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import torch


@contextmanager
def frozen_prior(model):
    training = model.training
    flags = [p.requires_grad for p in model.parameters()]
    model.eval().requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(training)


def likelihood_gradient(model, diffusion, noisy, timesteps, observation, decode, depth=None,
                        objective='nll'):
    """Expose the true noisy-state gradient for numerical verification."""
    with torch.enable_grad():
        x = noisy.detach().requires_grad_(True)
        d = depth.detach().requires_grad_(True) if depth is not None else None
        output = model(x, timesteps, None)
        clean = diffusion.model_predictions(x, timesteps, output)['predicted_x0']
        nll = observation.nll(decode(clean), d)
        if nll.ndim or not bool(torch.isfinite(nll)):
            raise RuntimeError('Trajectory NLL must be a finite scalar')
        if objective == 'nll':
            energy = nll
        elif objective == 'residual_norm':
            energy = (2 * nll + 1e-12).sqrt()
        else:
            raise ValueError('objective must be nll or residual_norm')
        if not energy.requires_grad:
            raise RuntimeError('Trajectory likelihood is disconnected from the noisy flow; do not detach the denoised field')
        inputs = (x, d) if d is not None else (x,)
        # Full-rank multiview positions can make the depth derivative zero.
        gradients = torch.autograd.grad(energy, inputs, allow_unused=True)
        if gradients[0] is None:
            raise RuntimeError('Trajectory likelihood is disconnected from the noisy flow; do not detach the denoised field')
        grad = gradients[0]
        depth_grad = None if d is None else (torch.zeros_like(d) if gradients[1] is None else gradients[1])
    return clean.detach(), grad.detach(), None if depth_grad is None else depth_grad.detach(), float(nll.detach())


def _proposal(diffusion, x, clean, timestep, previous, method, eta, generator):
    alpha = diffusion.alpha_bars[timestep].to(x)
    if previous < 0:
        return clean
    alpha_previous = diffusion.alpha_bars[previous].to(x)
    noise = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt().clamp_min(1e-12)
    if method == 'ddpm':
        beta = diffusion.betas[timestep].to(x)
        mean = (alpha_previous.sqrt() * beta * clean + (1 - beta).sqrt() * (1 - alpha_previous) * x) / (1 - alpha)
        variance = beta * (1 - alpha_previous) / (1 - alpha)
        return mean + variance.sqrt() * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    sigma = eta * (((1 - alpha_previous) / (1 - alpha)) * (1 - alpha / alpha_previous).clamp_min(0)).sqrt()
    draw = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if eta > 0 else torch.zeros_like(x)
    return alpha_previous.sqrt() * clean + (1 - alpha_previous - sigma.square()).clamp_min(0).sqrt() * noise + sigma * draw


def dps_sample(model, diffusion, shape, observation, *, decode=lambda value: value,
               method='ddpm', sampling_steps=None, eta=1., guidance_step=1e-4,
               objective='nll', max_update_rms=.1, depth_step=1e-4,
               depth_mode='langevin', initial_depth=None, initial_noise=None,
               generator=None, device='cpu', dtype=torch.float32):
    """Return normalized samples, latent depths and a bounded scalar trace.

    DDPM uses all training timesteps. DDIM is an explicitly recorded accelerated
    variant. Results are approximate posterior samples; clipping and discretized
    latent Langevin updates do not establish exact posterior calibration.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('DPS requires autograd; do not use torch.inference_mode')
    if len(shape) != 5 or shape[:2] != (1, 3) or min(shape) < 1:
        raise ValueError('DPS currently reconstructs one volume [1,3,D,H,W]')
    for name, value in [('guidance_step', guidance_step), ('depth_step', depth_step), ('max_update_rms', max_update_rms)]:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if method not in ('ddpm', 'ddim') or depth_mode not in ('langevin', 'map', 'fixed'):
        raise ValueError('Invalid sampler or depth mode')
    if objective not in ('nll', 'residual_norm') or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError('Invalid likelihood objective or DDIM eta')
    if method == 'ddpm' and eta != 1.:
        raise ValueError('DDPM uses its ancestral variance; use eta=1 or choose DDIM')
    steps = diffusion.num_steps if sampling_steps is None else int(sampling_steps)
    if not 1 <= steps <= diffusion.num_steps or (method == 'ddpm' and steps != diffusion.num_steps):
        raise ValueError('DDPM needs all timesteps; DDIM steps must be within the training schedule')
    if initial_noise is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        x = initial_noise.detach().clone()
        if tuple(x.shape) != shape or not bool(torch.isfinite(x).all()):
            raise ValueError('Initial noise must be finite and match shape')
    depth = initial_depth
    if hasattr(observation, 'initialize_depth'):
        depth = observation.initialize_depth(generator) if depth is None else depth.detach().clone()
        if depth.shape != observation.lower.shape or not bool(torch.isfinite(depth).all()):
            raise ValueError('Invalid initial depth')
        if bool(((depth < observation.lower) | (depth > observation.upper)).any()):
            raise ValueError('Initial depth is outside its feasible interval')
    trace = []
    schedule = torch.linspace(diffusion.num_steps - 1, 0, steps, device=x.device, dtype=torch.float64).round().long().tolist()
    with frozen_prior(model):
        for index, timestep in enumerate(schedule):
            previous = schedule[index + 1] if index + 1 < len(schedule) else -1
            time = torch.full((1,), timestep, dtype=torch.long, device=x.device)
            need_depth = depth is not None and depth_step > 0 and depth_mode != 'fixed' and bool(observation.depth_active.any())
            if guidance_step > 0 or need_depth:
                clean, grad, depth_grad, nll = likelihood_gradient(model, diffusion, x, time, observation, decode, depth, objective)
            else:
                with torch.no_grad():
                    clean = diffusion.model_predictions(x, time, model(x, time, None))['predicted_x0']
                    nll = float(observation.nll(decode(clean), depth))
                grad, depth_grad = torch.zeros_like(x), None
            if not bool(torch.isfinite(grad).all()) or not math.isfinite(nll):
                raise RuntimeError(f'Nonfinite posterior gradient at timestep {timestep}')
            update = guidance_step * grad
            rms = float(update.square().mean().sqrt())
            scale = min(1., max_update_rms / max(rms, 1e-30)) if max_update_rms > 0 else 1.
            with torch.no_grad():
                x = (_proposal(diffusion, x, clean, timestep, previous, method, eta, generator) - update * scale).detach()
                if need_depth:
                    if not bool(torch.isfinite(depth_grad).all()):
                        raise RuntimeError('Nonfinite depth gradient')
                    # Depth uses the Gaussian NLL even if the field step uses a
                    # residual-normalized objective as in the official DPS code.
                    if objective == 'residual_norm':
                        depth_grad = depth_grad * math.sqrt(2 * nll + 1e-12)
                    proposed = depth - depth_step * depth_grad
                    if depth_mode == 'langevin':
                        proposed = proposed + math.sqrt(2 * depth_step) * torch.randn(depth.shape, device=depth.device, dtype=depth.dtype, generator=generator)
                    depth = observation.reflect_depth(proposed).detach()
                if not bool(torch.isfinite(x).all()):
                    raise RuntimeError(f'Nonfinite posterior sample at timestep {timestep}')
            trace.append({'step': index, 'timestep': timestep, 'nll_at_clean_prediction': nll,
                          'guidance_update_rms_raw': rms, 'guidance_update_scale': scale})
    return {'sample': x, 'depth': depth, 'trace': trace}
