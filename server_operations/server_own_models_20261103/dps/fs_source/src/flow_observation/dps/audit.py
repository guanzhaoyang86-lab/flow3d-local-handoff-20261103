"""Artifact checks run after srun, so an empty launcher cannot pass a job."""
from pathlib import Path
import json
import numpy as np
import torch

from .data import sha256


def verify_prior(directory, epochs):
    directory = Path(directory)
    record = json.loads((directory / 'run.json').read_text())
    value = torch.load(directory / 'latest.pt', map_location='cpu', weights_only=True)
    if record.get('status') != 'completed' or record.get('audit') != 'verified':
        raise ValueError('Prior training has no verified completion record')
    if value.get('method') != 'unconditional-flow-prior' or value.get('conditioning') != 'none':
        raise ValueError('Not an unconditional flow prior')
    if value.get('epoch') != epochs or record.get('epoch') != epochs:
        raise ValueError('Prior did not finish all requested epochs')
    if record.get('checkpoint_sha256') != sha256(directory / 'latest.pt'):
        raise ValueError('Prior checkpoint checksum mismatch')
    for name in ('model', 'ema_model'):
        if not value.get(name) or not all(bool(torch.isfinite(v).all()) for v in value[name].values()):
            raise ValueError('Missing or nonfinite prior weights')
    if value.get('normalization_fit_split') != 'train' or not value.get('train_case_ids'):
        raise ValueError('Missing train-only data provenance')
    if 'successful_optimizer_steps' in value:
        if value['successful_optimizer_steps'] < 1 or record.get('successful_optimizer_steps') != value['successful_optimizer_steps']:
            raise ValueError('No verified successful optimizer update')
    return {'status': 'verified', 'epoch': epochs, 'conditioning': 'none'}


def verify_posterior(path, num_particles, num_samples):
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    if record.get('status') != 'completed' or record.get('method') != 'unconditional-prior-trajectory-dps':
        raise ValueError('DPS reconstruction did not complete')
    if record.get('output_sha256') != sha256(path) or record.get('num_observed_particles') != num_particles:
        raise ValueError('Posterior checksum or observed-particle count mismatch')
    with np.load(path, allow_pickle=False) as arrays:
        samples = arrays['posterior_samples']
        if samples.ndim != 5 or samples.shape[:2] != (num_samples, 3) or not np.isfinite(samples).all():
            raise ValueError('Missing or nonfinite posterior samples')
        metadata = json.loads(str(arrays['metadata']))
        if any(record.get(key) != value for key, value in metadata.items()):
            raise ValueError('Posterior metadata and completion record disagree')
    if record.get('reference_used_during_inference') is not False:
        raise ValueError('Reference-leakage declaration missing')
    if len(record.get('trace', [])) != num_samples or any(len(trace) != record['sampling_steps'] for trace in record['trace']):
        raise ValueError('DPS did not execute the requested sampling steps')
    return {'status': 'verified', 'num_samples': num_samples, 'num_particles': num_particles}
