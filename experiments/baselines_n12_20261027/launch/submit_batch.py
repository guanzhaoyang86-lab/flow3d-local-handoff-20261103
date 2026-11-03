#!/usr/bin/env python3
"""Prepare, test, submit, and inspect the explicitly authorized N12 baseline batch."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/sw/user/python/miniforge3-pytorch-2.10.0/bin/python'
LEARNED = ['cnn', 'deeponet', 'unet', 'fno', 'unet_diffusion']
SEEDS = [31, 32, 33]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def build_plan():
    output = '/work/hdd/biup/zguan2/results/baselines_n12_20261027'
    datasets = {
        'jhtdb': {'raw_root': '/work/hdd/biup/zguan2/datasets/jhtdb_iso1024_native32_3125_t1_20261024',
                  'derived_root': '/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32'},
        'fs_sph': {'raw_root': '/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_rods_full_v2',
                   'raw_run': '/work/nvme/biup/zguan2/fs_sph_gpu/runs/3429811',
                   'derived_root': '/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027'},
    }
    jobs = []
    for dataset in datasets:
        prep, pilot = f'{dataset}_prepare', f'{dataset}_pilot'
        jobs.append({'key': prep, 'dataset': dataset, 'stage': 'prepare', 'time': '04:00:00' if dataset == 'jhtdb' else '12:00:00',
                     'cpus': 8 if dataset == 'jhtdb' else 16, 'memory': '64G', 'dependencies': []})
        jobs.append({'key': pilot, 'dataset': dataset, 'stage': 'pilot', 'time': '01:00:00',
                     'cpus': 8, 'memory': '64G', 'dependencies': [['afterok', prep]]})
        previous = pilot
        for model in ('mean', 'pod', *LEARNED):
            statistical = model in ('mean', 'pod')
            seeds = [31] if statistical else SEEDS
            if statistical:
                hours = 1 if model == 'mean' else (4 if dataset == 'jhtdb' else 2)
            else:
                hours = (8 if model in ('cnn', 'deeponet') else 12) if dataset == 'jhtdb' else 4
            key = f'{dataset}_{model}'
            tasks = [{'dataset': dataset, 'stage': 'fit_statistics' if statistical else 'train',
                      'model': model, 'N': 12, 'seed': seed,
                      'epochs': 0 if statistical else 100,
                      'output_dir': str(Path(output) / dataset / (model if statistical else f'{model}_s{seed}'))}
                     for seed in seeds]
            deps = [['afterok', pilot]]
            if previous != pilot:
                deps.append(['afterany', previous])
            jobs.append({'key': key, 'dataset': dataset, 'stage': 'fit_statistics' if statistical else 'train',
                         'time': f'{hours:02d}:00:00', 'cpus': 8, 'memory': '64G',
                         'dependencies': deps, 'tasks': tasks, 'array_concurrency': 1})
            previous = key
        for model in ('pod', 'rbf'):
            key = f'{dataset}_{model}_calibration'
            deps = [['afterok', pilot], ['afterany', previous]]
            if model == 'pod':
                deps.append(['afterok', f'{dataset}_pod'])
            else:
                # A failed POD fit may cancel its calibration before training
                # ends; do not let that cancellation release RBF early.
                deps.append(['afterany', f'{dataset}_unet_diffusion'])
            jobs.append({'key': key, 'dataset': dataset, 'stage': 'calibration', 'time': '08:00:00',
                         'cpus': 8, 'memory': '64G', 'dependencies': deps,
                         'tasks': [{'dataset': dataset, 'stage': 'calibration', 'model': model, 'N': 12,
                                    'seed': 31, 'output_dir': str(Path(output) / dataset / 'calibration' / model)}]})
            previous = key
    def hours(job):
        hh, mm, ss = map(int, job['time'].split(':'))
        return (hh + mm / 60 + ss / 3600) * len(job.get('tasks', [None]))
    return {'schema': 'baselines_n12_submission_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'N': 12, 'input_views': ['xz'], 'training_seeds': SEEDS, 'epochs': 100,
            'datasets': datasets, 'result_root': output, 'learned_models': LEARNED,
            'excluded_models': ['pinn', 'dit3d', 'tensor_dit'], 'account': 'biup-dtai-gh',
            'partition': 'ghx4', 'gpus_per_task': 1, 'python': PYTHON,
            'max_concurrent_allocations': 2, 'concurrency_policy': 'One serial dependency chain per dataset; each seed array concurrency1.',
            'sigma_candidates': [.02, .05, .1], 'calibration_cases': 8,
            'evaluation_phase': 'Validation calibration only; test is held out until validation choices are frozen.',
            'jobs': jobs, 'sbatch_submission_count': len(jobs),
            'logical_allocations': sum(len(j.get('tasks', [None])) for j in jobs),
            'requested_gpu_hour_upper_bound': sum(hours(j) for j in jobs),
            'resource_note': 'Caps, not measured durations. Actual dataset GPU pilot logs timing before training starts. Latest completed epochs are resumable; no automatic requeue.',
            'initial_account_balance_gpu_hours': 805}


def command(job, dependency_ids=None, *, test_only=False):
    args = ['sbatch', '--test-only' if test_only else '--parsable', '--job-name=b12_' + job['key'],
            '--account=biup-dtai-gh', '--partition=ghx4', '--nodes=1', '--ntasks=1',
            '--gpus-per-node=1', f'--cpus-per-task={job["cpus"]}', '--mem=' + job['memory'],
            '--time=' + job['time'], '--no-requeue', '--kill-on-invalid-dep=yes',
            '--chdir=' + str(ROOT), '--export=ALL', '--open-mode=append',
            '--output=' + str(ROOT / 'logs' / (job['key'] + '_%A_%a.out')),
            '--error=' + str(ROOT / 'logs' / (job['key'] + '_%A_%a.err'))]
    count = len(job.get('tasks', [None]))
    if count > 1:
        args.append(f'--array=0-{count-1}%1')
    if dependency_ids is not None and job['dependencies']:
        parts = []
        for kind, key in job['dependencies']:
            if key not in dependency_ids:
                raise ValueError('Dependency has not been recorded: ' + key)
            parts.append(kind + ':' + dependency_ids[key])
        args.append('--dependency=' + ','.join(parts))
    args.append(str(ROOT / 'launch/allocation.slurm'))
    return args


def prepare():
    if (ROOT / 'submission/submission.json').exists():
        raise RuntimeError('Submission already recorded; inspect rather than rebuild')
    plan_path = ROOT / 'plan.json'
    if plan_path.exists():
        raise RuntimeError('Plan already prepared; do not silently replace frozen settings')
    plan = build_plan()
    for model in ('mean', 'pod', 'rbf', *LEARNED):
        template = read(ROOT / 'configs' / f'{model}.json')
        for seed in SEEDS if model in LEARNED else [31]:
            cfg = dict(template, seed=seed, num_particles=12, views=[0], num_probes=32, epochs=100)
            write(ROOT / 'configs' / f'{model}_s{seed}.json', cfg)
    for name in ('logs', 'submission', 'reports'):
        (ROOT / name).mkdir(exist_ok=True)
    write(plan_path, plan)
    files = []
    for subtree in ('launch', 'source', 'configs', 'protocol'):
        files.extend(p for p in (ROOT / subtree).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    files.append(plan_path)
    lock = {'schema': 'frozen_baseline_inputs_v1', 'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}
    write(ROOT / 'frozen_manifest.json', lock)
    # Files remain reviewable; on-node fingerprint checks detect any later edit.
    print(json.dumps({'status': 'prepared', 'plan': str(plan_path),
                      'frozen_sha256': sha(ROOT / 'frozen_manifest.json'),
                      'logical_allocations': plan['logical_allocations'],
                      'requested_gpu_hour_upper_bound': plan['requested_gpu_hour_upper_bound']}, indent=2))


def verify_lock():
    for name, digest in read(ROOT / 'frozen_manifest.json')['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen input changed: ' + name)


def test_only():
    verify_lock()
    plan = read(ROOT / 'plan.json')
    def check(job):
        result = subprocess.run(command(job, test_only=True), capture_output=True, text=True, timeout=60)
        return {'key': job['key'], 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check, plan['jobs']))
    report = {'passed': all(c['returncode'] == 0 for c in checks), 'checks': checks,
              'plan_sha256': sha(ROOT / 'plan.json'), 'frozen_sha256': sha(ROOT / 'frozen_manifest.json')}
    write(ROOT / 'reports/slurm_test_only.json', report)
    print(json.dumps({'passed': report['passed'], 'requests': len(checks)}, indent=2))
    if not report['passed']:
        raise RuntimeError('Slurm test-only rejected a request; inspect report before submission')


def submit():
    import fcntl
    guard = (ROOT / 'submission/.submit.lock').open('a')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    record_path = ROOT / 'submission/submission.json'
    if record_path.exists():
        raise RuntimeError('Submission was already attempted; never duplicate or silently retry')
    verify_lock()
    tested = read(ROOT / 'reports/slurm_test_only.json')
    if not tested['passed'] or tested['plan_sha256'] != sha(ROOT / 'plan.json') or tested['frozen_sha256'] != sha(ROOT / 'frozen_manifest.json'):
        raise RuntimeError('Current frozen plan has not passed scheduler preflight')
    cpu_tests = read(ROOT / 'reports/pre_submit_validation.json')
    if not cpu_tests.get('passed') or cpu_tests.get('frozen_sha256') != sha(ROOT / 'frozen_manifest.json'):
        raise RuntimeError('Frozen code has no passing integration test record')
    plan = read(ROOT / 'plan.json')
    if plan['requested_gpu_hour_upper_bound'] > plan['initial_account_balance_gpu_hours']:
        raise RuntimeError('Request caps exceed last verified allocation balance')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONPATH=str(ROOT / 'source/src'), BASELINE_FROZEN_SHA256=sha(ROOT / 'frozen_manifest.json'))
    ids = {}
    record = {'status': 'submitting', 'plan_sha256': sha(ROOT / 'plan.json'),
              'frozen_sha256': env['BASELINE_FROZEN_SHA256'], 'jobs': {},
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'requests': []}
    write(record_path, record)
    for job in plan['jobs']:
        env['BASELINE_JOB_KEY'] = job['key']
        args = command(job, ids)
        request = {'key': job['key'], 'command': args, 'state': 'attempting'}
        record['requests'].append(request)
        write(record_path, record)
        try:
            result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)
        except BaseException:
            record['status'] = 'ambiguous_submission_stopped_no_retry'
            write(record_path, record)
            raise
        request.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s]+)?', result.stdout.strip()):
            record['status'] = 'partial_or_failed_submission_stopped_no_retry'
            request['state'] = 'unconfirmed'
            write(record_path, record)
            raise RuntimeError('Submission not confirmed; preserve record and inspect queue, never blindly retry')
        job_id = result.stdout.strip().split(';')[0]
        ids[job['key']] = job_id
        request.update(state='accepted', job_id=job_id)
        record['jobs'][job['key']] = job_id
        write(record_path, record)
        print(job['key'], job_id, flush=True)
    record.update(status='submitted', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  logical_allocations=plan['logical_allocations'])
    write(record_path, record)
    print(json.dumps({'status': record['status'], 'jobs': ids, 'logical_allocations': plan['logical_allocations']}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'test-only', 'submit', 'status'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'test-only':
        test_only()
    elif args.action == 'submit':
        submit()
    else:
        print(json.dumps(read(ROOT / 'submission/submission.json'), indent=2))


if __name__ == '__main__':
    main()
