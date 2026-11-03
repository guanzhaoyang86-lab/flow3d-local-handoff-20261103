#!/usr/bin/env python3
"""Freeze, preflight, and submit the authorized two-dataset Tensor-DiT batch."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / 'baselines_n12_20261027'
PYTHON = '/sw/user/python/miniforge3-pytorch-2.10.0/bin/python'
SEEDS = [31, 32, 33]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def build_plan():
    baseline_plan = read(BASELINE / 'plan.json')
    submitted = read(BASELINE / 'submission/submission.json')
    if submitted.get('status') != 'submitted' or submitted['plan_sha256'] != sha(BASELINE / 'plan.json'):
        raise RuntimeError('Existing baseline data pipeline is not a confirmed, unchanged submission')
    if submitted['frozen_sha256'] != sha(BASELINE / 'frozen_manifest.json'):
        raise RuntimeError('Existing baseline frozen fingerprint changed')
    external = {}
    jobs = []
    output = Path('/work/hdd/biup/zguan2/results/tensor_dit_n12_20261027')
    for dataset in ('jhtdb', 'fs_sph'):
        for suffix in ('prepare', 'unet_diffusion', 'rbf_calibration'):
            key = 'baseline_' + dataset + '_' + suffix
            external[key] = submitted['jobs'][dataset + '_' + suffix]
        prepare, pilot, train = [dataset + '_' + stage for stage in ('tensor_prepare', 'pilot', 'tensor_dit')]
        # Wait directly for the final training array as well as calibration:
        # cancellation of calibration must not open an early overlapping chain.
        deps = [['afterok', 'baseline_' + dataset + '_prepare'],
                ['afterany', 'baseline_' + dataset + '_unet_diffusion'],
                ['afterany', 'baseline_' + dataset + '_rbf_calibration']]
        jobs.append({'key': prepare, 'dataset': dataset, 'stage': 'tensor_prepare',
                     'time': '08:00:00', 'cpus': 8, 'memory': '64G', 'dependencies': deps})
        jobs.append({'key': pilot, 'dataset': dataset, 'stage': 'pilot',
                     'time': '01:00:00', 'cpus': 8, 'memory': '64G',
                     'dependencies': [['afterok', prepare]]})
        jobs.append({'key': train, 'dataset': dataset, 'stage': 'train',
                     'time': '12:00:00' if dataset == 'jhtdb' else '04:00:00',
                     'cpus': 8, 'memory': '64G', 'dependencies': [['afterok', prepare], ['afterok', pilot]],
                     'tasks': [{'model': 'tensor_dit', 'dataset': dataset, 'N': 12, 'epochs': 100,
                                'seed': seed, 'output_dir': str(output / dataset / f'tensor_dit_s{seed}')}
                               for seed in SEEDS], 'array_concurrency': 1})
    hours = sum(int(job['time'].split(':')[0]) * len(job.get('tasks', [None])) for job in jobs)
    return {'schema': 'tensor_dit_n12_submission_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'N': 12, 'architecture': 'tensor_dit', 'epochs': 100, 'training_seeds': SEEDS,
            'observation_seed': 47, 'input_views': ['xz'], 'num_probes': 32,
            'datasets': baseline_plan['datasets'], 'result_root': str(output),
            'account': baseline_plan['account'], 'partition': baseline_plan['partition'],
            'gpus_per_task': 1, 'python': PYTHON, 'jobs': jobs, 'external_dependencies': external,
            'baseline_root': str(BASELINE), 'baseline_frozen_sha256': submitted['frozen_sha256'],
            'rank_candidates': [4, 8, 12, 16], 'hooi_iterations': 2,
            'rank_validation_mean_max': 0.05, 'rank_validation_worst_max': 0.20,
            'rank_failure_policy': 'Stop without training if every candidate fails validation; never inspect test.',
            'max_concurrent_allocations_including_baseline': 2,
            'concurrency_policy': 'Append one serial Tensor chain per dataset after its existing baseline chain; seed arrays throttle1.',
            'requested_gpu_hour_upper_bound': hours,
            'baseline_gpu_hour_upper_bound': baseline_plan['requested_gpu_hour_upper_bound'],
            'combined_gpu_hour_upper_bound': hours + baseline_plan['requested_gpu_hour_upper_bound'],
            'verified_account_balance_gpu_hours': 805,
            'sbatch_submission_count': len(jobs), 'logical_allocations': sum(len(j.get('tasks', [None])) for j in jobs),
            'evaluation_phase': 'Training and validation only. No test evaluation is scheduled.',
            'resource_note': 'Walltime caps, not measured durations. Real CUDA pilot runs before all formal training.'}


def command(job, dependency_ids=None, *, test_only=False):
    args = ['sbatch', '--test-only' if test_only else '--parsable', '--job-name=t12_' + job['key'],
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
        pieces = []
        for kind, key in job['dependencies']:
            if key not in dependency_ids:
                raise ValueError('Unrecorded dependency: ' + key)
            jid = str(dependency_ids[key])
            if not re.fullmatch(r'[0-9]+', jid):
                raise ValueError('Invalid dependency job ID')
            pieces.append(kind + ':' + jid)
        args.append('--dependency=' + ','.join(pieces))
    args.append(str(ROOT / 'launch/allocation.slurm'))
    return args


def prepare():
    if (ROOT / 'submission/submission.json').exists() or (ROOT / 'plan.json').exists():
        raise RuntimeError('Plan/submission already exists; do not replace a frozen batch')
    from tensor_model import default_config
    plan = build_plan()
    for dataset in plan['datasets']:
        for seed in SEEDS:
            write(ROOT / 'configs' / f'{dataset}_s{seed}.json', default_config(dataset, seed))
    write(ROOT / 'plan.json', plan)
    files = []
    for subtree in ('launch', 'source', 'configs', 'protocol', 'tests'):
        files.extend(p for p in (ROOT / subtree).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    files.extend([ROOT / 'plan.json', ROOT / 'README.txt'])
    write(ROOT / 'frozen_manifest.json', {'schema': 'frozen_tensor_inputs_v1',
          'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}})
    print(json.dumps({'status': 'prepared', 'frozen_sha256': sha(ROOT / 'frozen_manifest.json'),
                      'logical_allocations': plan['logical_allocations'], 'gpu_hour_cap': plan['requested_gpu_hour_upper_bound']}, indent=2))


def verify_lock():
    for name, digest in read(ROOT / 'frozen_manifest.json')['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Frozen file changed: ' + name)


def test_only():
    verify_lock()
    plan = read(ROOT / 'plan.json')
    def check(job):
        result = subprocess.run(command(job, test_only=True), capture_output=True, text=True, timeout=60)
        return {'key': job['key'], 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check, plan['jobs']))
    report = {'passed': all(row['returncode'] == 0 for row in checks), 'checks': checks,
              'plan_sha256': sha(ROOT / 'plan.json'), 'frozen_sha256': sha(ROOT / 'frozen_manifest.json')}
    write(ROOT / 'reports/slurm_test_only.json', report)
    print(json.dumps({'passed': report['passed'], 'requests': len(checks)}, indent=2))
    if not report['passed']:
        raise RuntimeError('Slurm preflight rejected request')


def submit():
    import fcntl
    guard = (ROOT / 'submission/.submit.lock').open('a')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    record_path = ROOT / 'submission/submission.json'
    if record_path.exists():
        raise RuntimeError('Submission already attempted; inspect queue, never blindly resubmit')
    verify_lock()
    plan = read(ROOT / 'plan.json')
    tested = read(ROOT / 'reports/slurm_test_only.json')
    validation = read(ROOT / 'reports/pre_submit_validation.json')
    for report in (tested, validation):
        if not report.get('passed') or report.get('frozen_sha256') != sha(ROOT / 'frozen_manifest.json'):
            raise RuntimeError('Current frozen inputs lack passing validation/preflight')
    if tested['plan_sha256'] != sha(ROOT / 'plan.json'):
        raise RuntimeError('Preflight belongs to a different plan')
    if validation.get('cpu_tests_xml_sha256') != sha(ROOT / 'reports/cpu_tests.xml'):
        raise RuntimeError('CPU test evidence changed')
    if plan['combined_gpu_hour_upper_bound'] > plan['verified_account_balance_gpu_hours']:
        raise RuntimeError('Combined batch time caps exceed verified account balance')
    if sha(Path(plan['baseline_root']) / 'frozen_manifest.json') != plan['baseline_frozen_sha256']:
        raise RuntimeError('Existing baseline fingerprint changed')
    # Ensure every external dependency still refers to the recorded user job.
    for key, jid in plan['external_dependencies'].items():
        state = subprocess.run(['scontrol', '-o', 'show', 'job', jid], text=True, capture_output=True, check=True).stdout
        if not re.search(r'(?:^|\s)UserId=zguan2\(', state):
            raise RuntimeError('External dependency is not the authorized user job: ' + key)
        if re.search(r'JobState=(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)', state) and key.endswith('_prepare'):
            raise RuntimeError('Required dataset preparation already failed: ' + key)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONPATH=str(ROOT / 'source/src'), TENSOR_FROZEN_SHA256=sha(ROOT / 'frozen_manifest.json'))
    ids = dict(plan['external_dependencies'])
    record = {'status': 'submitting', 'plan_sha256': sha(ROOT / 'plan.json'),
              'frozen_sha256': env['TENSOR_FROZEN_SHA256'], 'external_dependencies': plan['external_dependencies'],
              'jobs': {}, 'requests': [], 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    write(record_path, record)
    for job in plan['jobs']:
        env['TENSOR_JOB_KEY'] = job['key']
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
            request['state'] = 'unconfirmed'
            record['status'] = 'partial_or_failed_submission_stopped_no_retry'
            write(record_path, record)
            raise RuntimeError('Unconfirmed submission; preserve record and inspect queue, do not retry')
        jid = result.stdout.strip().split(';')[0]
        ids[job['key']] = jid
        record['jobs'][job['key']] = jid
        request.update(state='accepted', job_id=jid)
        write(record_path, record)
        print(job['key'], jid, flush=True)
    record.update(status='submitted', completed_at_utc=datetime.now(timezone.utc).isoformat(), logical_allocations=plan['logical_allocations'])
    write(record_path, record)
    print(json.dumps({'status': record['status'], 'jobs': record['jobs'], 'logical_allocations': plan['logical_allocations']}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'test-only', 'submit', 'status'))
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'test-only':
        test_only()
    elif action == 'submit':
        submit()
    else:
        print(json.dumps(read(ROOT / 'submission/submission.json'), indent=2))


if __name__ == '__main__':
    main()
