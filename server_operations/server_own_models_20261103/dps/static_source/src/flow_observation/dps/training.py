"""Train an N-independent Eulerian flow prior; no trajectory encoder exists."""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..diffusion import GaussianDiffusion
from ..models.factory import build_denoiser
from ..sparse_dataset import FlowNormalizationStats, load_sparse_flow_manifest
from .data import (FlowPriorDataset, fit_prior_stats, read_prior_field, save_checkpoint,
                   sha256, write_json)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--architecture', choices=['dit3d', 'unet3d'], default='dit3d')
    p.add_argument('--epochs', type=positive, default=100)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--batch-size', type=positive, default=1)
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--save-every', type=positive, default=10)
    p.add_argument('--diffusion-steps', type=positive, default=1000)
    p.add_argument('--prediction-type', choices=['v', 'epsilon'], default='v')
    p.add_argument('--dit-hidden-dim', type=positive, default=192)
    p.add_argument('--dit-depth', type=positive, default=4)
    p.add_argument('--dit-heads', type=positive, default=6)
    p.add_argument('--patch-size', type=positive, default=4)
    p.add_argument('--base-channels', type=positive, default=32)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', type=Path)
    return p


def _validation(model, diffusion, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0., 0
    with torch.no_grad():
        for field in loader:
            field = field.to(device)
            times = torch.randint(diffusion.num_steps, (len(field),), device=device, generator=generator)
            noise = torch.randn(field.shape, device=device, generator=generator)
            loss = diffusion.training_loss(model, field, None, timesteps=times, noise=noise)['loss']
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite validation loss')
            total += float(loss) * len(field); count += len(field)
    return total / count


def _optimizer_step(loss, model, optimizer, scaler, gradient_clip):
    """Recover from scaled AMP overflow without updating the optimizer or EMA.

    ``unscale_`` records overflow for GradScaler. Let ``step`` skip such an
    update and ``update`` reduce its scale before trying the next minibatch.
    Checking the gradient norm first with error_if_nonfinite=True would abort
    this normal loss-scale adjustment. Full-precision nonfinite gradients are
    still fatal.
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if scaler.is_enabled() and any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()):
        scaler.step(optimizer)
        scaler.update()
        return False
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return True


def train(args):
    for name in ('learning_rate', 'weight_decay', 'gradient_clip'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != 'weight_decay' and value == 0):
            raise ValueError(f'Invalid {name}')
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        raise ValueError('EMA decay must be in [0,1)')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    manifest, cases = load_sparse_flow_manifest(args.manifest)
    digest = sha256(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'latest.pt').exists() and args.resume is None:
        raise ValueError('Output already has a checkpoint; use --resume or another directory')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    old = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if old and (old.get('method') != 'unconditional-flow-prior' or old.get('manifest_sha256') != digest):
        raise ValueError('Resume requires the same unconditional prior and manifest')
    stats = FlowNormalizationStats.from_dict(old['normalization']) if old else fit_prior_stats(cases['train'])
    shape = list(read_prior_field(cases['train'][0].path)[0].shape[1:])
    model_config = {'architecture': args.architecture, 'condition_dim': 1, 'dropout': 0.,
        'spatial_shape': shape, 'dit_hidden_dim': args.dit_hidden_dim, 'dit_depth': args.dit_depth,
        'dit_heads': args.dit_heads, 'patch_size': args.patch_size, 'base_channels': args.base_channels,
        'time_embedding_dim': 128, 'channel_multipliers': [1, 2, 4]}
    diffusion_config = {'num_steps': args.diffusion_steps, 'prediction_type': args.prediction_type,
                        'schedule': 'cosine', 'clip_x0': None}
    invariant_keys = ('seed', 'batch_size', 'learning_rate', 'weight_decay', 'ema_decay', 'gradient_clip', 'amp')
    if old and (old['model_config'] != model_config or old['diffusion_config'] != diffusion_config or
                any(old['train_config'].get(k) != getattr(args, k) for k in invariant_keys)):
        raise ValueError('Resume cannot change model, objective or optimizer configuration')
    model = build_denoiser(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(**diffusion_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    shuffle = torch.Generator().manual_seed(args.seed)
    start, best = 1, math.inf
    successful_optimizer_steps, skipped_amp_steps = 0, 0
    if old:
        model.load_state_dict(old['model']); ema.load_state_dict(old['ema_model'])
        optimizer.load_state_dict(old['optimizer']); scaler.load_state_dict(old['scaler'])
        torch.set_rng_state(old['rng_state']); shuffle.set_state(old['shuffle_rng_state'])
        if device.type == 'cuda' and old.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(old['cuda_rng_state'])
        start, best = old['epoch'] + 1, old['best_validation_loss']
        # Earlier checkpoints aborted on overflow, so every completed step in
        # those checkpoints necessarily updated the optimizer.
        successful_optimizer_steps = old.get('successful_optimizer_steps',
            old['epoch'] * math.ceil(len(cases['train']) / args.batch_size))
        skipped_amp_steps = old.get('skipped_amp_steps', 0)
        if start > args.epochs:
            raise ValueError('Resume checkpoint already reached the requested epoch')
    train_loader = DataLoader(FlowPriorDataset(cases['train'], stats), batch_size=args.batch_size,
                              shuffle=True, generator=shuffle)
    val_loader = DataLoader(FlowPriorDataset(cases['validation'], stats), batch_size=args.batch_size)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output_dir / 'run.json', {'status': 'running', 'method': 'unconditional-flow-prior',
        'conditioning': 'none', 'normalization_fit_split': 'train', 'train_count': len(cases['train']),
        'validation_count': len(cases['validation']), 'scientific_result': manifest.get('mode', 'scientific') == 'scientific',
        'config': config})
    for epoch in range(start, args.epochs + 1):
        began = time.monotonic(); model.train(); total, count = 0., 0
        epoch_optimizer_steps, epoch_amp_skips = 0, 0
        for field in train_loader:
            field = field.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = diffusion.training_loss(model, field, None)['loss']
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite prior training loss')
            updated = _optimizer_step(loss, model, optimizer, scaler, args.gradient_clip)
            if updated:
                successful_optimizer_steps += 1; epoch_optimizer_steps += 1
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 1 - args.ema_decay)
                    for target, source in zip(ema.buffers(), model.buffers()):
                        target.copy_(source)
            else:
                skipped_amp_steps += 1; epoch_amp_skips += 1
            total += float(loss.detach()) * len(field); count += len(field)
        validation = _validation(ema, diffusion, val_loader, device, args.seed + 100000)
        improved = validation < best; best = min(best, validation)
        row = {'epoch': epoch, 'train_loss': total / count, 'validation_ema_loss': validation,
               'optimizer_steps_this_epoch': epoch_optimizer_steps, 'amp_skipped_steps_this_epoch': epoch_amp_skips,
               'successful_optimizer_steps': successful_optimizer_steps, 'skipped_amp_steps': skipped_amp_steps,
               'elapsed_seconds': time.monotonic() - began}
        with (args.output_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(__import__('json').dumps(row, allow_nan=False) + '\n')
        print(__import__('json').dumps(row, allow_nan=False), flush=True)
        state = {'format_version': 1, 'method': 'unconditional-flow-prior', 'conditioning': 'none',
            'epoch': epoch, 'model_config': model_config, 'diffusion_config': diffusion_config,
            'model': model.state_dict(), 'ema_model': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(), 'normalization': stats.to_dict(), 'normalization_fit_split': 'train',
            'manifest_sha256': digest, 'train_case_ids': [c.case_id for c in cases['train']],
            'train_flow_group_ids': [c.flow_group_id for c in cases['train']], 'train_config': config,
            'scientific_result': manifest.get('mode', 'scientific') == 'scientific',
            'best_validation_loss': best, 'rng_state': torch.get_rng_state(),
            'successful_optimizer_steps': successful_optimizer_steps, 'skipped_amp_steps': skipped_amp_steps,
            'shuffle_rng_state': shuffle.get_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        save_checkpoint(args.output_dir / 'latest.pt', state)
        if improved:
            save_checkpoint(args.output_dir / 'best.pt', state)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output_dir / f'epoch_{epoch:04d}.pt', state)
    final = torch.load(args.output_dir / 'latest.pt', map_location='cpu', weights_only=True)
    if final['successful_optimizer_steps'] < 1:
        raise RuntimeError('Prior training performed no successful optimizer updates; refusing completion')
    if final['epoch'] != args.epochs or not all(bool(torch.isfinite(v).all()) for v in final['ema_model'].values()):
        raise RuntimeError('Final checkpoint verification failed')
    write_json(args.output_dir / 'run.json', {'status': 'completed', 'audit': 'verified',
        'method': 'unconditional-flow-prior', 'conditioning': 'none', 'epoch': args.epochs,
        'normalization_fit_split': 'train', 'train_count': len(cases['train']),
        'validation_count': len(cases['validation']), 'scientific_result': final['scientific_result'],
        'successful_optimizer_steps': successful_optimizer_steps, 'skipped_amp_steps': skipped_amp_steps,
        'checkpoint_sha256': sha256(args.output_dir / 'latest.pt'), 'config': config})
    return args.output_dir / 'latest.pt'
