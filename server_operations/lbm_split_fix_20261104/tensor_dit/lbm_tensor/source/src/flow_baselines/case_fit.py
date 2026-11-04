"""Observation-only point estimates; these fits are not posterior samples."""
import math
import torch
from .common import check_rectangular_geometry


def fit_case(kind, inputs, config, observation, *, pod_state=None, decode=lambda x:x):
    check_rectangular_geometry(inputs['solid_mask'])
    from .linear import PODField, RBFField
    from .pinn import PINNField
    if observation.likelihood != 'transition':
        raise ValueError('Baseline case fitting uses the declared Euler-transition likelihood')
    device, dtype = inputs['tracks'].device, inputs['tracks'].dtype
    shape = tuple(inputs['solid_mask'].shape)
    seed = config['seed']
    if kind == 'pod':
        if pod_state is None:
            raise ValueError('POD requires a train-only fitted basis')
        model = PODField(pod_state).to(device=device, dtype=dtype)
        coefficients = torch.nn.Parameter(torch.zeros((1, model.rank), device=device, dtype=dtype))
        parameters = [coefficients] if model.rank else []
        raw_field = lambda: model(coefficients)
        regularizer = lambda: config['coefficient_weight']*model.prior_regularizer(coefficients)
    elif kind == 'rbf':
        model = RBFField(inputs['bounds'], shape, **config['rbf']).to(device=device, dtype=dtype)
        parameters = list(model.parameters())
        raw_field = model
        regularizer = model.regularization
    elif kind == 'pinn':
        options = {k: config['pinn'][k] for k in ('hidden', 'layers', 'viscosity', 'collocation_points')}
        model = PINNField(inputs['bounds'], shape, seed=seed, **options).to(device=device, dtype=dtype)
        parameters = list(model.parameters())
        raw_field = model
        regularizer = lambda: inputs['tracks'].new_zeros(())
    else:
        raise ValueError('Case fitting supports pod, rbf and pinn')
    generator = torch.Generator(device=device).manual_seed(seed+2000000011)
    depth = torch.nn.Parameter(observation.initialize_depth(generator))
    groups = []
    learning_rate = config['pinn']['learning_rate'] if kind == 'pinn' else config['fit_learning_rate']
    steps = config['pinn']['fit_steps'] if kind == 'pinn' else config['fit_steps']
    if not isinstance(steps, int) or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Invalid per-case optimization settings')
    if parameters:
        groups.append({'params': parameters, 'lr': learning_rate})
    if bool(observation.depth_active.any()):
        groups.append({'params': [depth], 'lr': config['depth_learning_rate']})
    optimizer = torch.optim.Adam(groups) if groups else None
    # Count all visible coordinate residuals; never mask by candidate validity.
    count = 2*(observation.mask[..., 1:] & observation.mask[..., :-1]).sum()
    best, best_field, best_depth, trace = math.inf, None, None, []
    for iteration in range(steps+1 if optimizer else 1):
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        reflected = observation.reflect_depth(depth)
        field = decode(raw_field())
        nll = observation.nll(field, reflected)
        loss = nll/count + regularizer()
        physics = {}
        if kind == 'pinn':
            physics = model.physics_loss()
            loss = loss + config['pinn']['divergence_weight']*physics['divergence'] + config['pinn']['momentum_weight']*physics['momentum']
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite per-case fitting loss')
        value = float(loss.detach())
        if value < best:
            best, best_field, best_depth = value, field.detach().clone(), reflected.detach().clone()
        if iteration % 100 == 0 or iteration == steps:
            trace.append({'iteration': iteration, 'objective': value, 'nll': float(nll.detach()),
                          **{k:float(v.detach()) for k,v in physics.items()}})
        if optimizer is None or iteration == steps:
            break
        loss.backward()
        trainable = [p for group in groups for p in group['params']]
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            depth.copy_(observation.reflect_depth(depth))
    return best_field, best_depth, {'mode': 'per_case_point_estimate', 'best_objective': best,
        'steps_executed': iteration, 'trace': trace,
        'fitted_parameters': sum(p.numel() for p in parameters),
        'latent_depth_count': int(observation.depth_active.sum())}


def fit_depth_for_fixed_field(field, observation, *, steps=100, learning_rate=.01, seed=47):
    """Post-prediction diagnostic depth fit; never updates the generated field."""
    if not isinstance(steps,int) or steps<0 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Invalid diagnostic depth optimization settings')
    generator = torch.Generator(device=field.device).manual_seed(seed)
    depth = torch.nn.Parameter(observation.initialize_depth(generator))
    if not bool(observation.depth_active.any()):
        return depth.detach()
    optimizer = torch.optim.Adam([depth], lr=learning_rate)
    best, selected = math.inf, depth.detach().clone()
    for index in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        value = observation.nll(field.detach(), observation.reflect_depth(depth))
        if not bool(torch.isfinite(value)):
            raise RuntimeError('Nonfinite diagnostic depth fitting')
        if float(value.detach()) < best:
            best, selected = float(value.detach()), observation.reflect_depth(depth).detach().clone()
        if index == steps:
            break
        value.backward(); optimizer.step()
        with torch.no_grad():
            depth.copy_(observation.reflect_depth(depth))
    return selected
