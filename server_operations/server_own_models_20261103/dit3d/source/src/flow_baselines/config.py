"""Explicit baseline defaults; hyperparameters are not yet validation-tuned."""
from copy import deepcopy
import json
import math
from pathlib import Path

MODELS = ('mean', 'pod', 'unet', 'fno', 'cnn', 'deeponet', 'unet_diffusion', 'dit3d', 'pinn', 'rbf')
SUPERVISED = ('unet', 'fno', 'cnn', 'deeponet', 'unet_diffusion', 'dit3d')
DEFAULTS = {
    'seed': 31, 'num_particles': 2, 'views': None, 'boundary_projection': 'none',
    'epochs': 100, 'batch_size': 1, 'learning_rate': 2e-4, 'weight_decay': 1e-4,
    'gradient_clip': 1., 'ema_decay': .999, 'amp': True,
    'condition_dim': 128, 'condition_flow_grid': 4, 'condition_flow_weight': .1,
    'condition_dropout': .1, 'diffusion_steps': 1000, 'prediction_type': 'v',
    'sampling_steps': 50, 'sampling_eta': 1., 'cfg_scale': 1., 'clip_x0': 8.,
    'observation_seed': 47, 'sampling_seed': 47, 'num_probes': 16, 'num_samples': 2,
    'fit_steps': 1000, 'fit_learning_rate': .01, 'sigma': .05,
    'coefficient_weight': 1e-4, 'depth_learning_rate': .01,
    'unet': {'base_channels': 32, 'channel_multipliers': [1, 2, 4], 'dropout': 0.},
    'cnn': {'width': 32, 'depth': 4},
    'fno': {'width': 32, 'depth': 4, 'modes': 8, 'padding': 4},
    'deeponet': {'latent': 128, 'hidden': 128, 'layers': 3},
    'unet_diffusion': {'base_channels': 16, 'channel_multipliers': [1, 2, 4],
                       'time_embedding_dim': 64, 'dropout': 0.},
    'dit3d': {'hidden_dim': 192, 'depth': 4, 'num_heads': 6, 'patch_size': 4, 'dropout': 0.},
    'pod': {'rank': 32},
    'rbf': {'centers_per_axis': 4, 'length_scale': .35, 'ridge': 1e-4},
    'pinn': {'hidden': 128, 'layers': 5, 'viscosity': None, 'collocation_points': 1024,
             'fit_steps': 2000, 'learning_rate': .001,
             'divergence_weight': 1., 'momentum_weight': 1.},
}


def resolve_config(path=None, overrides=None):
    config = deepcopy(DEFAULTS)
    changes = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(changes,dict) or (overrides is not None and not isinstance(overrides,dict)):
        raise ValueError('Configuration must be a JSON object')
    changes.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key, value in changes.items():
        if key not in config:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or set(value) - set(config[key]):
                raise ValueError(f'Unknown or invalid {key} configuration')
            config[key].update(value)
        else:
            config[key] = value
    if not isinstance(config['amp'],bool):
        raise ValueError('amp must be a JSON boolean')
    for key in ('seed','sampling_seed','observation_seed'):
        if isinstance(config[key],bool) or not isinstance(config[key],int) or not 0<=config[key]<2**63:
            raise ValueError(f'{key} must be an integer in [0,2**63)')
    if config['views'] is not None:
        views=config['views']
        if not isinstance(views,(list,tuple)) or not views or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in views) or len(set(views))!=len(views):
            raise ValueError('views must contain unique nonnegative integer indices')
        config['views']=list(views)
    if config['clip_x0'] is not None and (isinstance(config['clip_x0'],bool) or not isinstance(config['clip_x0'],(int,float)) or not math.isfinite(config['clip_x0']) or config['clip_x0']<=0):
        raise ValueError('clip_x0 must be finite and positive or null')
    for key in ('epochs','batch_size','condition_dim','condition_flow_grid','diffusion_steps',
                'sampling_steps','num_samples','num_particles','fit_steps'):
        if isinstance(config[key],bool) or not isinstance(config[key],int) or config[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate','gradient_clip','fit_learning_rate','sigma','depth_learning_rate'):
        if not math.isfinite(config[key]) or config[key]<=0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('weight_decay','condition_flow_weight','coefficient_weight','cfg_scale'):
        if not math.isfinite(config[key]) or config[key]<0:
            raise ValueError(f'{key} must be finite and nonnegative')
    for key in ('ema_decay','condition_dropout'):
        if not math.isfinite(config[key]) or not 0<=config[key]<1:
            raise ValueError(f'Invalid {key}')
    if config['boundary_projection'] not in ('none','known'):
        raise ValueError('boundary_projection must be none or known')
    if config['prediction_type'] not in ('v','epsilon'):
        raise ValueError('prediction_type must be v or epsilon')
    if not isinstance(config['num_probes'],int) or config['num_probes']<0:
        raise ValueError('num_probes must be nonnegative')
    if not math.isfinite(config['sampling_eta']) or not 0<=config['sampling_eta']<=1:
        raise ValueError('sampling_eta must be in [0,1]')
    if config['sampling_steps']>config['diffusion_steps']:
        raise ValueError('sampling_steps exceeds diffusion_steps')
    if config['pinn']['viscosity'] is not None and (not math.isfinite(config['pinn']['viscosity']) or config['pinn']['viscosity']<=0):
        raise ValueError('Known kinematic viscosity must be positive')
    for key in ('divergence_weight','momentum_weight'):
        if not math.isfinite(config['pinn'][key]) or config['pinn'][key]<0:
            raise ValueError('PINN weights must be finite and nonnegative')
    return config
