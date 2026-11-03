"""Train-only statistical fitting and supervised training with exact resume."""
import copy
import json
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from flow_observation.dps.data import fit_prior_stats, read_prior_field, save_checkpoint, sha256, write_json
from flow_observation.dps.training import _optimizer_step
from flow_observation.models.trajectory_encoder import fit_condition_motion_scales
from flow_observation.sparse_dataset import FlowNormalizationStats, load_sparse_flow_manifest
from flow_observation.diffusion import GaussianDiffusion
from flow_observation.conditioning import condition_flow_loss
from .common import TrainingDataset, build_system, check_rectangular_geometry, encode, masked_mse, to_device
from .config import SUPERVISED, resolve_config


def _loss(system, batch, kind, config, diffusion=None, generator=None, training=False):
    condition = encode(system['encoder'], batch)
    if kind == 'unet_diffusion':
        used = condition
        if training and config['condition_dropout']:
            dropped = torch.rand((len(condition), 1), device=condition.device) < config['condition_dropout']
            used = torch.where(dropped, torch.zeros_like(condition), condition)
        times = torch.randint(diffusion.num_steps, (len(condition),), device=condition.device, generator=generator)
        noise = torch.randn(batch['flow'].shape, device=condition.device, generator=generator)
        primary = diffusion.training_loss(system['decoder'], batch['flow'], used, timesteps=times, noise=noise)['loss']
    else:
        primary = masked_mse(system['decoder'](condition), batch['flow'], batch['solid_mask'])
    # A zero weight explicitly disables auxiliary supervision. Evaluating an
    # otherwise unused target can reject valid primary-only batches (for
    # example all fluid cells are known boundary cells) or propagate 0 * NaN.
    if config['condition_flow_weight'] > 0:
        primary = primary + config['condition_flow_weight']*condition_flow_loss(system['encoder'], condition, batch)
    return primary


def _restore_history(path, checkpoint_epoch):
    """Discard records that were not committed in the resumed checkpoint.

    The JSONL append precedes the atomic checkpoint write, so preemption can
    leave an extra full or partial line. A checkpoint, rather than a history
    append, is the authoritative epoch boundary for exact resume.
    """
    if not path.exists():
        return
    lines = path.read_text().splitlines(keepends=True)
    committed = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith('\n'):
                break
            raise ValueError('Training history contains an invalid complete record')
        epoch = record.get('epoch')
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError('Training history contains an invalid epoch')
        if epoch <= checkpoint_epoch:
            committed[epoch] = record
    temp = path.with_name(path.name + '.resume.tmp')
    temp.write_text(''.join(json.dumps(committed[epoch], allow_nan=False)+'\n'
                            for epoch in sorted(committed)))
    temp.replace(path)


def train(kind, manifest, output_dir, config=None, *, device='cuda', resume=None):
    if kind not in (*SUPERVISED, 'mean', 'pod'):
        raise ValueError('PINN/RBF fit per case through predict')
    config = resolve_config(overrides=config)
    manifest, output_dir = Path(manifest), Path(output_dir)
    manifest_data, splits = load_sparse_flow_manifest(manifest)
    digest = sha256(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output_dir/name).exists() for name in ('latest.pt', 'run.json', 'history.jsonl')):
        raise ValueError('Training artifacts exist; use resume or a new directory')
    if kind in ('mean', 'pod') and resume:
        raise ValueError('Statistical fits do not support resume')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    old = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if old:
        invariant = {k: v for k, v in config.items() if k != 'epochs'}
        saved = {k: v for k, v in old['config'].items() if k != 'epochs'}
        if old.get('format') != 'flow3d-baseline-v1' or old.get('model') != kind or old.get('manifest_sha256') != digest or invariant != saved:
            raise ValueError('Resume model, manifest or configuration differs')
        stats = FlowNormalizationStats.from_dict(old['normalization'])
    else:
        stats = fit_prior_stats(splits['train'])
    first_field, first_solid = read_prior_field(splits['train'][0].path)
    check_rectangular_geometry(first_solid)
    shape = list(first_field.shape[1:])
    state = {'format': 'flow3d-baseline-v1', 'model': kind, 'config': config,
        'spatial_shape': shape, 'manifest_sha256': digest,
        'train_case_ids': [c.case_id for c in splits['train']],
        'train_flow_group_ids': [c.flow_group_id for c in splits['train']],
        'normalization': stats.to_dict(), 'normalization_fit_split': 'train',
        'scientific_dataset': manifest_data.get('mode', 'scientific') == 'scientific'}
    if kind in ('mean', 'pod'):
        write_json(output_dir/'run.json', {'status': 'running', 'model': kind, 'config': config})
        fields = [] if kind == 'pod' else None
        total = torch.zeros((3, *shape), dtype=torch.float64)
        for case in splits['train']:
            field, solid = read_prior_field(case.path)
            check_rectangular_geometry(solid)
            if list(field.shape[1:]) != shape:
                raise ValueError('Statistical baselines require a common grid')
            total += field.double()
            if fields is not None:
                fields.append(field)
        state['mean'] = (total/len(splits['train'])).float()[None]
        if kind == 'pod':
            from .linear import fit_pod
            state['pod_state'] = fit_pod(torch.stack(fields), config['pod']['rank'])
        state['epoch'] = 0
        save_checkpoint(output_dir/'latest.pt', state)
    else:
        train_set = TrainingDataset(manifest, 'train', config, stats)
        val_set = TrainingDataset(manifest, 'validation', config, stats)
        motion = old['motion_scales'] if old else fit_condition_motion_scales(train_set, 'cpu')
        system = build_system(kind, shape, config, motion).to(device)
        ema = copy.deepcopy(system).eval().requires_grad_(False)
        diffusion = GaussianDiffusion(config['diffusion_steps'], schedule='cosine',
            prediction_type=config['prediction_type'], clip_x0=config['clip_x0']).to(device) if kind == 'unet_diffusion' else None
        optimizer = torch.optim.AdamW(system.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        amp = bool(config['amp']) and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        shuffle = torch.Generator().manual_seed(config['seed'])
        start, best, updates, skips = 1, math.inf, 0, 0
        if old:
            system.load_state_dict(old['system']); ema.load_state_dict(old['ema_system'])
            optimizer.load_state_dict(old['optimizer']); scaler.load_state_dict(old['scaler'])
            torch.set_rng_state(old['rng_state']); shuffle.set_state(old['shuffle_rng_state'])
            if device.type == 'cuda' and old['cuda_rng_state'] is not None:
                torch.cuda.set_rng_state_all(old['cuda_rng_state'])
            start, best, updates, skips = old['epoch']+1, old['best_validation_loss'], old['optimizer_steps'], old['amp_skips']
            if start > config['epochs']:
                raise ValueError('Resume checkpoint reached requested final epoch')
        else:
            torch.manual_seed(config['seed']+27183)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(config['seed']+27183)
        if old:
            _restore_history(output_dir/'history.jsonl', old['epoch'])
        write_json(output_dir/'run.json', {'status': 'running', 'model': kind, 'config': config})
        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, generator=shuffle)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'])
        for epoch in range(start, config['epochs']+1):
            began = time.monotonic(); system.train(); train_set.set_epoch(epoch-1)
            train_sum, train_count = 0., 0
            for row in train_loader:
                row = to_device(row, device); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = _loss(system, row, kind, config, diffusion, training=True)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                updated = _optimizer_step(loss, system, optimizer, scaler, config['gradient_clip'])
                if updated:
                    updates += 1
                    with torch.no_grad():
                        for target, source in zip(ema.parameters(), system.parameters()):
                            target.lerp_(source, 1-config['ema_decay'])
                        for target, source in zip(ema.buffers(), system.buffers()):
                            target.copy_(source)
                else:
                    skips += 1
                train_sum += float(loss.detach())*len(row['flow']); train_count += len(row['flow'])
            ema.eval(); val_sum, val_count = 0., 0
            generator = torch.Generator(device=device).manual_seed(config['seed']+100000)
            devices = [device.index or 0] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for row in val_loader:
                    row = to_device(row, device)
                    loss = _loss(ema, row, kind, config, diffusion, generator=generator)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite validation loss')
                    val_sum += float(loss)*len(row['flow']); val_count += len(row['flow'])
            val_loss = val_sum/val_count
            improved = val_loss < best; best = min(best, val_loss)
            record = {'epoch': epoch, 'train_loss': train_sum/train_count,
                'validation_loss': val_loss, 'optimizer_steps': updates, 'amp_skips': skips,
                'elapsed_seconds': time.monotonic()-began}
            with (output_dir/'history.jsonl').open('a') as out:
                out.write(json.dumps(record, allow_nan=False)+'\n')
            state.update(epoch=epoch, motion_scales=motion, system=system.state_dict(), ema_system=ema.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), best_validation_loss=best,
                optimizer_steps=updates, amp_skips=skips, rng_state=torch.get_rng_state(),
                shuffle_rng_state=shuffle.get_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            save_checkpoint(output_dir/'latest.pt', state)
            if improved:
                save_checkpoint(output_dir/'best.pt', state)
            if epoch % 10 == 0 or epoch == config['epochs']:
                save_checkpoint(output_dir/f'epoch_{epoch:04d}.pt', state)
            print(json.dumps(record, allow_nan=False), flush=True)
        if updates < 1:
            raise RuntimeError('No optimizer updates; refusing completion')
    write_json(output_dir/'run.json', {'status': 'completed', 'model': kind, 'config': config,
        'epoch': state['epoch'], 'normalization_fit_split': 'train',
        'checkpoint_sha256': sha256(output_dir/'latest.pt')})
    return output_dir/'latest.pt'
