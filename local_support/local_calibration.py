#!/usr/bin/env python3
"""Portable POD/RBF validation selection; preserves the frozen 3 x 8 protocol."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import local_baselines as base

SIGMAS = (0.02, 0.05, 0.1)


def commands(args, output, data, checkpoint):
    root = base.experiment(args.dataset)
    rows = []
    for sigma in SIGMAS:
        config = base.read(root / 'configs' / f'{args.model}_s31.json')
        config['sigma'] = sigma
        if config.get('fit_steps') != 1000 or config.get('observation_seed') != 47:
            raise ValueError('Frozen calibration protocol differs from 1000 steps / observation seed 47')
        cfg_path = output / f'config_sigma_{sigma}.json'
        for index in range(min(8, data['split_counts']['validation'])):
            case = output / f'sigma_{sigma}' / f'case_{index:04d}'
            if args.dataset == 'jhtdb':
                cmd = [sys.executable, '-B', str(root/'source/scripts/run_baseline.py'), 'predict',
                       '--case-index', str(index), '--output', str(case.with_suffix('.npz')), '--evaluate']
            else:
                cmd = [sys.executable, '-B', str(root/'launch/fs_baseline.py'), 'evaluate',
                       '--case-indices', str(index), '--output-dir', str(case), '--resume']
            cmd += ['--manifest', data['manifest'], '--model', args.model, '--config', str(cfg_path),
                    '--num-particles', '12', '--seed', '31', '--split', 'validation', '--device', args.device]
            if checkpoint:
                cmd += ['--checkpoint', str(checkpoint)]
            rows.append(dict(sigma=sigma, case_index=index, output=str(case), config=config,
                             config_path=str(cfg_path), command=cmd))
    return rows


def resume_command(args, row, data, checkpoint):
    if args.dataset != 'jhtdb':
        return row['command']
    output = Path(row['output']).with_suffix('.npz')
    meta = output.with_suffix('.json')
    if not output.exists() and not meta.exists():
        return row['command']
    if not args.resume or not output.is_file() or not meta.is_file():
        raise ValueError('Existing/partial prediction requires verified explicit resume; no overwrite: ' + str(output))
    value = base.read(meta)
    case_id = base.read(data['manifest'])['splits']['validation'][row['case_index']]['case_id']
    required = dict(model=args.model, config=row['config'], manifest_sha256=data['manifest_sha256'],
                    split='validation', case_id=case_id, output_sha256=base.sha(output))
    if checkpoint:
        required['checkpoint_sha256'] = base.sha(checkpoint)
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError('Existing prediction provenance differs from the calibration request')
    return [sys.executable, '-B', str(base.experiment(args.dataset)/'source/scripts/run_baseline.py'), 'verify', str(output)]


def score(dataset, row):
    path = Path(row['output'])
    if dataset == 'jhtdb':
        value = base.read(path.with_suffix('.json'))['evaluation']['posterior_mean_field_relative_l2_unknown_interior']
    else:
        value = base.read(path/'summary.json')['summary']['physical_velocity_relative_l2']
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite validation error')
    return value


def select(rows, count):
    scores = {}
    for sigma in SIGMAS:
        selected = [r for r in rows if r['sigma'] == sigma]
        if sorted(r['case_index'] for r in selected) != list(range(count)) or count < 1:
            raise ValueError('Incomplete calibration grid')
        values = [r['validation_relative_l2'] for r in selected]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite validation error')
        scores[sigma] = sum(values) / len(values)
    return min(scores, key=lambda s: (scores[s], s)), scores


def run(args):
    base.assert_training_unassigned(args.dataset, args.model, args.server_snapshot, stage='calibration')
    checks = base.preflight(args)
    data = checks['data']
    checkpoint = base.checkpoint_path(args)
    checkpoint_info = base.validate_checkpoint(checkpoint, args, data) if checkpoint else None
    name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or (args.resume and not args.run_name):
        raise ValueError('Use a safe --run-name; --resume requires the original run name')
    output = base.ROOT/'local_runs'/args.dataset/args.model/('calibration_'+name)
    rows = commands(args, output, data, checkpoint)
    request = dict(dataset=args.dataset, model=args.model, manifest_sha256=data['manifest_sha256'],
                   frozen_manifest_sha256=checks['source']['frozen_manifest_sha256'], checkpoint=checkpoint_info,
                   split='validation', test_used=False, training_seed=31, observation_seed=47, N=12,
                   cases=[r['case_id'] for r in base.read(data['manifest'])['splits']['validation'][:8]],
                   configs=[r['config'] for r in rows if r['case_index'] == 0])
    if output.exists():
        if not args.resume or base.read(output/'calibration_request.json') != request:
            raise ValueError('Existing calibration directory requires matching request and --resume')
    print(json.dumps(dict(output_dir=str(output), execution_requested=args.execute, fits=len(rows),
                          commands=[r['command'] for r in rows], request=request), indent=2))
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    base.write(output/'calibration_request.json', request)
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    results = []
    for row in rows:
        base.write(row['config_path'], row['config'])
        subprocess.run(resume_command(args, row, data, checkpoint), cwd=base.ROOT, env=env, check=True)
        results.append(dict(sigma=row['sigma'], case_index=row['case_index'], output=row['output'],
                            validation_relative_l2=score(args.dataset, row)))
        base.write(output/'calibration_progress.json', dict(completed_fits=len(results), planned_fits=len(rows)))
    sigma, means = select(results, min(8, data['split_counts']['validation']))
    config = base.read(base.experiment(args.dataset)/'configs'/f'{args.model}_s31.json')
    config['sigma'] = sigma
    base.write(output/'selected_config.json', config)
    base.write(output/'CALIBRATION_COMPLETE.json', dict(status='completed', model=args.model, dataset=args.dataset,
               split='validation', seed=31, N=12, sigma_candidates=SIGMAS, outputs=results,
               mean_validation_errors=means, selected_sigma=sigma,
               selection_status='frozen_from_predeclared_validation_subset', test_used=False, request=request))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('jhtdb', 'fs_sph'), required=True)
    p.add_argument('--model', choices=('pod', 'rbf'), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--server-snapshot', type=Path)
    p.add_argument('--run-name')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
