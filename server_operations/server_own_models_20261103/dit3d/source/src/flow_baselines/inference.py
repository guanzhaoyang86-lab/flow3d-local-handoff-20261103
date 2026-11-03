"""Observation-only inference; references are opened only after fields are fixed."""
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from flow_observation.dps.data import load_observation_case, sha256, write_json
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.dps.inference import _posthoc_evaluate
from flow_observation.sparse_dataset import FlowNormalizationStats, denormalize_flow, load_sparse_flow_manifest
from flow_observation.sparse_physics import apply_hard_boundary_conditions
from flow_observation.diffusion import GaussianDiffusion
from .common import build_system, check_rectangular_geometry, encode, no_output_collision, to_device
from .config import MODELS, SUPERVISED, resolve_config
from .case_fit import fit_case, fit_depth_for_fixed_field


def predict(manifest, output, *, kind=None, checkpoint=None, config=None,
            split='validation', case_index=0, device='cuda', evaluate=False, require_valid=False):
    output, manifest = Path(output), Path(manifest)
    no_output_collision(output)
    manifest_data, splits = load_sparse_flow_manifest(manifest)
    if split not in ('validation', 'test') or not 0 <= case_index < len(splits[split]):
        raise ValueError('Invalid evaluation split or case index')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    if saved:
        if saved.get('format') != 'flow3d-baseline-v1' or saved['manifest_sha256'] != sha256(manifest):
            raise ValueError('Checkpoint format or manifest mismatch')
        if kind is not None and saved['model'] != kind:
            raise ValueError('Checkpoint model differs from requested model')
        kind = saved['model']
        if saved['normalization_fit_split'] != 'train':
            raise ValueError('Normalization must be train-only')
        case = splits[split][case_index]
        if case.case_id in saved['train_case_ids'] or case.flow_group_id in saved['train_flow_group_ids']:
            raise ValueError('Evaluation case overlaps fitted training cases')
        cfg = resolve_config(overrides=saved['config'])
        if config:
            allowed = {'num_particles','views','boundary_projection','num_samples','num_probes',
                'sampling_steps','sampling_eta','cfg_scale','observation_seed','sampling_seed',
                'fit_steps','fit_learning_rate','sigma','coefficient_weight','depth_learning_rate','seed'}
            for key, value in config.items():
                if value != cfg.get(key) and key not in allowed:
                    raise ValueError(f'Cannot change fitted model configuration: {key}')
                cfg[key] = value
            cfg = resolve_config(overrides=cfg)
    else:
        cfg = resolve_config(overrides=config)
        if kind not in ('rbf','pinn'):
            raise ValueError('This model requires a fitted checkpoint')
    if kind not in MODELS:
        raise ValueError('Unknown model')
    if kind=='pinn' and cfg['pinn']['viscosity'] is None:
        raise ValueError('PINN requires explicit known kinematic viscosity; set pinn.viscosity or --viscosity')
    if kind=='pinn' and cfg['boundary_projection']=='known':
        raise ValueError('PINN does not support hard postprojection: its continuous NS residual would describe a different field')
    case = splits[split][case_index]
    device = torch.device(device)
    inputs = load_observation_case(case, split, cfg['num_particles'], observation_seed=cfg['observation_seed'],
        num_probes=cfg['num_probes'], views=cfg['views'], known_boundaries=cfg['boundary_projection']=='known')
    check_rectangular_geometry(inputs['solid_mask'])
    data = to_device(inputs, device)
    observation = ProjectedTrajectoryLikelihood(data['tracks'], data['projection'], data['times'],
        data['bounds'], sigma=cfg['sigma'], mask=data['mask'])
    shape = tuple(data['solid_mask'].shape)
    if saved and list(shape) != saved['spatial_shape']:
        raise ValueError('Observation grid differs from fitted model')
    def project(field):
        if cfg['boundary_projection'] == 'known':
            return apply_hard_boundary_conditions(field, data['boundary_mask'], data['boundary_values'])
        return field
    began = time.monotonic()
    fields, depths, fitting, diagnostics = [], [], [], []
    if kind in SUPERVISED:
        torch.manual_seed(saved['config']['seed'])
        system = build_system(kind, shape, saved['config'], saved['motion_scales']).to(device)
        system.load_state_dict(saved['ema_system'], strict=True)
        system.eval().requires_grad_(False)
        stats = FlowNormalizationStats.from_dict(saved['normalization'])
        batch = {k:v[None] for k,v in data.items() if k in ('tracks','mask','projection','times','bounds')}
        with torch.no_grad():
            condition = encode(system['encoder'], batch)
        diffusion = GaussianDiffusion(cfg['diffusion_steps'], schedule='cosine', clip_x0=cfg['clip_x0'],
            prediction_type=cfg['prediction_type']).to(device) if kind in ('unet_diffusion', 'dit3d') else None
        for index in range(cfg['num_samples'] if diffusion is not None else 1):
            with torch.no_grad():
                if diffusion is not None:
                    seed = cfg['sampling_seed']+index+case_index*100003
                    initial = torch.randn((1,3,*shape), device=device, generator=torch.Generator(device=device).manual_seed(seed))
                    generator = torch.Generator(device=device).manual_seed(seed+1000000007)
                    def diffusion_projection(value, timestep):
                        # Projection uses physical units; conversion returns normalized field.
                        from flow_observation.sparse_dataset import normalize_flow
                        return normalize_flow(project(denormalize_flow(value, stats)), stats)
                    normalized = diffusion.ddim_sample(system['decoder'], (1,3,*shape), condition,
                        sampling_steps=cfg['sampling_steps'], eta=cfg['sampling_eta'], guidance_scale=cfg['cfg_scale'],
                        generator=generator, initial_noise=initial,
                        projection=diffusion_projection if cfg['boundary_projection']=='known' else None)
                else:
                    normalized = system['decoder'](condition)
                field = project(denormalize_flow(normalized, stats))
            fields.append(field.detach())
    elif kind=='mean':
        fields.append(project(saved['mean'].to(device)).detach())
    else:
        field, depth, fit = fit_case(kind, data, cfg, observation,
            pod_state=saved.get('pod_state') if saved else None, decode=project)
        fields.append(field.detach()); depths.append(depth); fitting.append(fit)
    # Fitting depth here is a diagnostic only: generated fields stay fixed.
    for index, field in enumerate(fields):
        if not bool(torch.isfinite(field).all()):
            raise RuntimeError('Nonfinite reconstructed field')
        if len(depths) <= index:
            depths.append(fit_depth_for_fixed_field(field, observation, steps=100,
                learning_rate=cfg['depth_learning_rate'], seed=cfg['sampling_seed']))
        with torch.no_grad():
            row = observation.diagnostics(field, depths[index])
        if require_valid and not row['all_replay_particles_valid']:
            raise RuntimeError('Invalid final replay; refusing completion')
        diagnostics.append(row)
    samples = torch.cat(fields).cpu()
    checkpoint_selection = None
    if saved:
        if kind in ('mean', 'pod'):
            checkpoint_selection = 'statistical_fit'
        elif Path(checkpoint).name == 'best.pt':
            checkpoint_selection = 'best_validation'
        elif saved.get('epoch') == saved['config']['epochs']:
            checkpoint_selection = 'final_epoch'
        else:
            checkpoint_selection = 'explicit_epoch'
    metadata = {'status':'completed', 'format':'flow3d-baseline-prediction-v1', 'model':kind,
        'scope':'steady_3d_projected_trajectory_reconstruction', 'split':split,
        'case_id':case.case_id, 'flow_group_id':case.flow_group_id,
        'manifest_sha256':sha256(manifest), 'source_case_sha256':sha256(case.path),
        'checkpoint_sha256':sha256(checkpoint) if checkpoint else None,
        'checkpoint_epoch':saved.get('epoch') if saved else None,
        'checkpoint_selection':checkpoint_selection,
        'config':cfg, 'reference_used_during_inference':False,
        'training_seed':saved['config']['seed'] if kind in SUPERVISED else None,
        'training_num_particles':saved['config']['num_particles'] if kind in SUPERVISED else None,
        'training_views':saved['config']['views'] if kind in SUPERVISED else None,
        'training_boundary_projection':saved['config']['boundary_projection'] if kind in SUPERVISED else None,
        'fit_seed':cfg['seed'] if kind in ('pod','rbf','pinn') else None,
        'views':inputs['view_indices'],
        'known_velocity_boundary_conditioning':cfg['boundary_projection'],
        'observation_protocol':'single_view' if len(inputs['view_indices'])==1 else 'multiview',
        'particle_indices':inputs['particle_indices'].tolist(),
        'probe_particle_indices':inputs['probe_particle_indices'].tolist(),
        'num_observed_particles':cfg['num_particles'], 'actual_predictions':len(fields),
        'prediction_type':'conditional_diffusion_samples' if kind in ('unet_diffusion', 'dit3d') else 'point_estimate',
        'uncertainty_available':kind in ('unet_diffusion', 'dit3d'),
        'deterministic_variance_is_uncertainty':False,
        'depth_diagnostics':'joint_fit' if fitting else 'post_prediction_fit_observations_only',
        'fixed_field_depth_fit':{'max_steps':100, 'learning_rate':cfg['depth_learning_rate'],
            'seed':cfg['sampling_seed']} if not fitting else None,
        'forward_domain_policy':'clamp_surrogate_with_reported_invalidity',
        'valid_prediction_count':sum(row['all_replay_particles_valid'] for row in diagnostics),
        'diagnostics':diagnostics, 'fitting':fitting,
        'generation_and_diagnostics_seconds':time.monotonic()-began,
        'hyperparameters_validation_tuned':False,
        'physical_information':{'viscosity':cfg['pinn']['viscosity'], 'known_incompressible_steady_NS':True}
            if kind=='pinn' else {},
        'scientific_dataset':manifest_data.get('mode','scientific')=='scientific'}
    payload = {'predictions':samples.numpy(), 'prediction_mean':samples.mean(0).numpy(),
        'initial_depth_estimates':torch.stack(depths).detach().cpu().numpy(),
        'observed_tracks':inputs['tracks'].numpy(), 'observation_mask':inputs['mask'].numpy(),
        'projection_matrix':inputs['projection'].numpy(), 'observation_times':inputs['times'].numpy(),
        'domain_bounds':inputs['bounds'].numpy()}
    if kind in ('unet_diffusion', 'dit3d'):
        payload['sample_variance'] = samples.var(0,unbiased=False).numpy()
    if evaluate:
        args = SimpleNamespace(split=split,num_particles=cfg['num_particles'],observation_seed=cfg['observation_seed'],
            num_probes=cfg['num_probes'],observation_type='projected',trajectory_substeps=1)
        results = _posthoc_evaluate(samples.to(device), case, args)
        payload['reference_field'] = results.pop('reference')
        metadata['evaluation'] = results
        metadata['evaluation_protocol'] = {
            'field_region':'unknown_fluid_interior',
            'probe_error_views':'all_stored_cameras',
            'probe_initialization':'triangulated_from_all_stored_cameras_posthoc_only',
            'extra_probe_information_used_for_generation':False}
    payload['metadata'] = np.array(json.dumps(metadata,allow_nan=False))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**payload)
    temporary.replace(output)
    write_json(output.with_suffix('.json'), {**metadata,'output_sha256':sha256(output)})
    return output
