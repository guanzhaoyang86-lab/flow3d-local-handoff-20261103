"""Observation-only posterior reconstruction and optional post-hoc evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..advection import advect_particles
from ..multiview import triangulate_initial_positions
from ..recovery import project_trajectory_views, masked_trajectory_mse
from ..sparse_dataset import denormalize_flow, load_sparse_flow_manifest
from ..sparse_physics import apply_hard_boundary_conditions
from .data import load_observation_case, load_prior, read_prior_field, sha256, write_json
from .observations import FullTrajectoryLikelihood, ProjectedTrajectoryLikelihood
from .sampler import dps_sample
from .training import positive


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--case-index', type=int, default=0)
    p.add_argument('--num-particles', type=positive, default=2)
    p.add_argument('--num-probes', type=int, default=16)
    p.add_argument('--observation-type', choices=['projected', 'full3d'], default='projected',
                   help='Full3d explicitly treats stored XYZ trajectories as observed data')
    p.add_argument('--views', help='Comma-separated view indices; omit for all views')
    p.add_argument('--boundary-projection', choices=['none', 'known'], default='none',
                   help='Known velocity boundaries are extra information, enabled only explicitly')
    p.add_argument('--observation-seed', type=int, default=47)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--num-samples', type=positive, default=2)
    p.add_argument('--sigma', type=float, required=True, help='Gaussian displacement-residual scale in physical coordinate units')
    p.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    p.add_argument('--sampling-steps', type=positive)
    p.add_argument('--eta', type=float, default=1.)
    p.add_argument('--guidance-step', type=float, default=1e-4)
    p.add_argument('--gradient-objective', choices=['nll', 'residual_norm'], default='nll')
    p.add_argument('--max-update-rms', type=float, default=.1, help='0 disables guidance-step clipping')
    p.add_argument('--depth-step', type=float, default=1e-4)
    p.add_argument('--depth-mode', choices=['langevin', 'map', 'fixed'], default='langevin')
    p.add_argument('--likelihood', choices=['transition', 'position'], default='transition')
    p.add_argument('--integrator', choices=['euler', 'rk4'], default='euler')
    p.add_argument('--trajectory-substeps', type=positive, default=1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--raw-weights', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Read reference flow only AFTER all posterior samples are fixed')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--require-valid-replay', action='store_true',
                   help='Reject final invalid trajectories rather than save a completed invalid reconstruction')
    return p


def _posthoc_evaluate(samples, case, args):
    # Called after the final posterior sample is fixed, outside any inference graph.
    reference, _ = read_prior_field(case.path)
    reference = reference.to(samples)
    # Boundary metadata is permitted here only AFTER generation is complete.
    evaluation = load_observation_case(case, args.split, args.num_particles,
        observation_seed=args.observation_seed, num_probes=args.num_probes,
        observation_type=args.observation_type, known_boundaries=True)
    interior = (~evaluation['solid_mask'] & ~evaluation['boundary_mask']).to(samples.device)
    if not bool(interior.any()):
        raise ValueError('No unknown fluid interior is available for evaluation')
    weights = interior[None].expand_as(reference)
    denominator = torch.linalg.vector_norm(reference[weights])
    errors = [float(torch.linalg.vector_norm((sample - reference)[weights]) / denominator)
              if float(denominator) > 0 else None for sample in samples]
    mean_error = float(torch.linalg.vector_norm((samples.mean(0) - reference)[weights]) / denominator) if float(denominator) > 0 else None
    # Probe time-zero positions use the original camera observations only in this
    # post-hoc evaluator. They are never visible to the posterior likelihood.
    probe_rows = []
    if args.num_probes:
        tracks = evaluation['probe_tracks'].to(samples)[None]
        mask = evaluation['probe_mask'].to(samples.device)[None]
        try:
            if args.observation_type == 'projected':
                camera = evaluation['projection'].to(samples)
                initial = triangulate_initial_positions(tracks, camera, mask)
            else:
                if not bool(mask[:, :, 0].all()):
                    raise ValueError('Probe initial 3D observations are missing')
                initial = tracks[:, :, 0]
        except ValueError:
            probe_rows = [{'probe_track_rmse_cells': None,
                           'reason': 'Probe initial depth is not identifiable from evaluation views'} for _ in samples]
        else:
            for sample in samples:
                trajectory, valid = advect_particles(sample[None], initial, evaluation['times'].to(samples),
                    integrator='rk4', num_substeps=args.trajectory_substeps,
                    domain_bounds=evaluation['bounds'].to(samples), boundary_mode='clamp', return_validity=True)
                if args.observation_type == 'projected':
                    replay = project_trajectory_views(trajectory, camera)
                    rmse = float(masked_trajectory_mse(replay, tracks, mask).sqrt())
                else:
                    difference = torch.where(mask[..., None], trajectory - tracks, torch.zeros_like(trajectory))
                    rmse = float((difference.square().sum() / (3 * mask.sum())).sqrt())
                probe_rows.append({'probe_track_rmse_cells': rmse,
                    'all_probe_particles_valid': bool(valid.all()), 'probe_valid_fraction': float(valid.float().mean())})
    return {'per_sample_field_relative_l2_unknown_interior': errors,
            'posterior_mean_field_relative_l2_unknown_interior': mean_error,
            'probe_metrics': probe_rows, 'reference': reference.detach().cpu().numpy()}


def sample(args):
    if args.output.exists() and not args.overwrite:
        raise ValueError('Output exists; choose another path or explicitly use --overwrite')
    manifest, cases = load_sparse_flow_manifest(args.manifest)
    if not 0 <= args.case_index < len(cases[args.split]):
        raise ValueError('Case index is outside the selected split')
    device = torch.device(args.device)
    model, diffusion, stats, prior = load_prior(args.checkpoint, device, raw_weights=args.raw_weights)
    if sha256(args.manifest) != prior['manifest_sha256']:
        raise ValueError('Use the same split manifest as the prior checkpoint')
    case = cases[args.split][args.case_index]
    if case.case_id in prior['train_case_ids'] or case.flow_group_id in prior['train_flow_group_ids']:
        raise ValueError('Evaluation case overlaps prior-training data')
    views = None if args.views is None else [int(v) for v in args.views.split(',')]
    if args.observation_type == 'full3d' and (views is not None or args.likelihood != 'transition'):
        raise ValueError('Full3d uses the image Euler-transition likelihood and has no camera selection')
    inputs = load_observation_case(case, args.split, args.num_particles,
        observation_seed=args.observation_seed, num_probes=args.num_probes, views=views,
        observation_type=args.observation_type, known_boundaries=args.boundary_projection == 'known')
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if args.observation_type == 'projected':
        observation = ProjectedTrajectoryLikelihood(data['tracks'], data['projection'], data['times'], data['bounds'],
            sigma=args.sigma, mask=data['mask'], integrator=args.integrator,
            num_substeps=args.trajectory_substeps, likelihood=args.likelihood)
    else:
        if not bool(data['mask'][:, 0].all()):
            raise ValueError('Full3d sampling requires an observed initial XYZ position per particle')
        observation = FullTrajectoryLikelihood(data['tracks'], data['times'], data['bounds'],
            sigma=args.sigma, mask=data['mask'], integrator=args.integrator, num_substeps=args.trajectory_substeps)
    shape = (1, 3, *data['solid_mask'].shape)
    if list(shape[2:]) != prior['model_config']['spatial_shape']:
        raise ValueError('Observation geometry differs from the trained prior grid')

    def decode(normalized):
        physical = denormalize_flow(normalized, stats)
        if args.boundary_projection == 'known':
            physical = apply_hard_boundary_conditions(physical, data['boundary_mask'], data['boundary_values'])
        return physical

    steps = args.sampling_steps
    if steps is None:
        steps = diffusion.num_steps if args.sampler == 'ddpm' else min(50, diffusion.num_steps)
    fields, depths, traces, diagnostics = [], [], [], []
    model.eval()
    for index in range(args.num_samples):
        seed = args.seed + index
        # Match the completed conditional evaluation's initial/update streams.
        initial_seed = seed + args.case_index * 100003
        update_seed = initial_seed + 1000000007
        initial = torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(initial_seed))
        generator = torch.Generator(device=device).manual_seed(update_seed)
        depth = (observation.initialize_depth(torch.Generator(device=device).manual_seed(initial_seed + 2000000011))
                 if args.observation_type == 'projected' else None)
        outcome = dps_sample(model, diffusion, shape, observation, decode=decode,
            method=args.sampler, sampling_steps=steps, eta=args.eta, guidance_step=args.guidance_step,
            objective=args.gradient_objective, max_update_rms=args.max_update_rms,
            depth_step=args.depth_step, depth_mode=args.depth_mode, generator=generator, device=device,
            initial_noise=initial, initial_depth=depth)
        with torch.no_grad():
            field = decode(outcome['sample'])
            row = observation.diagnostics(field, outcome['depth'])
            if not bool(torch.isfinite(field).all()):
                raise RuntimeError('Nonfinite final posterior field')
        if args.require_valid_replay and not row['all_replay_particles_valid']:
            raise RuntimeError('Final replay leaves the domain; no valid-reconstruction completion will be written')
        fields.append(field[0].detach().cpu())
        depths.append(outcome['depth'].detach().cpu() if outcome['depth'] is not None else torch.empty(0))
        diagnostics.append({'sampling_seed': seed, 'initial_seed': initial_seed,
            'update_seed': update_seed, 'initial_noise_sha256': __import__('hashlib').sha256(initial.cpu().numpy().tobytes()).hexdigest(), **row})
        traces.append(outcome['trace'])
        print(json.dumps({'sample': index + 1, **diagnostics[-1]}, allow_nan=False), flush=True)
    samples = torch.stack(fields)
    results = None
    if args.evaluate:
        results = _posthoc_evaluate(samples, case, args)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {'method': 'unconditional-prior-trajectory-dps', 'sampler': args.sampler,
        'sampling_steps': steps,
        'conditioning': 'trajectory-likelihood-gradient-through-denoiser' if args.guidance_step > 0 else 'unconditional-prior-only',
        'field_likelihood_guidance_enabled': args.guidance_step > 0,
        'likelihood': args.likelihood,
        'sigma_units': 'physical displacement coordinates' if args.likelihood == 'transition' else 'physical position coordinates',
        'observation_type': args.observation_type,
        'observation_protocol': ('full_3d' if args.observation_type == 'full3d' else
                                 'projected_single_view' if len(inputs['view_indices']) == 1 else 'projected_multiview_extension'),
        'velocity_field_scope': 'steady_3d',
        'boundary_velocity_conditioning': args.boundary_projection,
        'camera_noise_assumption': 'independent Gaussian residuals per view' if args.observation_type == 'projected' else 'not_applicable',
        'forward_domain_policy': 'clamp_surrogate_with_reported_invalidity',
        'valid_sample_count': sum(row['all_replay_particles_valid'] for row in diagnostics),
        'invalid_sample_count': sum(not row['all_replay_particles_valid'] for row in diagnostics),
        'hidden_depth_update': args.depth_mode if args.observation_type == 'projected' else 'not_applicable',
        'posterior_is_approximate': True,
        'prior_checkpoint_sha256': sha256(args.checkpoint), 'prior_epoch': prior['epoch'],
        'source_case_sha256': sha256(case.path),
        'manifest_sha256': prior['manifest_sha256'], 'split': args.split, 'case_id': case.case_id,
        'flow_group_id': case.flow_group_id, 'num_observed_particles': args.num_particles,
        'views': inputs['view_indices'], 'observation_seed': args.observation_seed,
        'particle_indices': inputs['particle_indices'].tolist(), 'probe_particle_indices': inputs['probe_particle_indices'].tolist(),
        'source_sample_index': inputs['source_sample_index'],
        'reference_used_during_inference': False, 'normalization_fit_split': 'train',
        'scientific_result': bool(prior['scientific_result']) and manifest.get('mode', 'scientific') == 'scientific',
        'hyperparameter_validation_status': 'not_established_by_code_validation',
        'diagnostics': diagnostics, 'config': config}
    payload = {'posterior_samples': samples.numpy(), 'posterior_mean': samples.mean(0).numpy(),
        'posterior_variance': samples.var(0, unbiased=False).numpy(), 'initial_depth_samples': torch.stack(depths).numpy(),
        'observed_tracks': inputs['tracks'].numpy(), 'observation_mask': inputs['mask'].numpy(),
        'observation_times': inputs['times'].numpy(),
        'domain_bounds': inputs['bounds'].numpy(), 'solid_mask': inputs['solid_mask'].numpy(),
        'particle_indices': inputs['particle_indices'].numpy(), 'metadata': np.array(json.dumps(metadata, allow_nan=False))}
    if args.observation_type == 'projected':
        payload['projection_matrix'] = inputs['projection'].numpy()
    if results is not None:
        payload['reference_field'] = results.pop('reference')
        metadata['posthoc_evaluation'] = results
        payload['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temp.replace(args.output)
    write_json(args.output.with_suffix('.json'), {'status': 'completed', **metadata,
        'output_sha256': sha256(args.output), 'trace': traces})
    return args.output
