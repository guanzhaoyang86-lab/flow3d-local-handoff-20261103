#!/usr/bin/env python3
"""One allocated GPU: frozen Tucker preparation, pilot, then seed-31 training."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verified_plan(expected=None):
    operation = ROOT / 'operation_manifest.json'
    if expected is not None:
        require(sha(operation) == expected, 'Operation fingerprint changed after submission')
    for name, digest in read(operation)['files'].items():
        require(sha(ROOT / name) == digest, 'Operation file changed: ' + name)
    plan = read(ROOT / 'plan.json')
    require(plan['training_seeds'] == [31] and plan['epochs'] == 100, 'Only seed 31 and 100 epochs authorized')
    require(plan['rank_candidates'] == [4, 8, 12, 16] and plan['hooi_iterations'] == 2, 'Frozen Tucker protocol changed')
    frozen = Path(plan['frozen_root'])
    require(sha(frozen / 'frozen_manifest.json') == plan['frozen_sha256'], 'Frozen Tensor fingerprint changed')
    for name, digest in read(frozen / 'frozen_manifest.json')['files'].items():
        require(sha(frozen / name) == digest, 'Frozen Tensor source changed: ' + name)
    historical_plan = read(frozen / 'plan.json')
    require(sha(Path(historical_plan['baseline_root']) / 'frozen_manifest.json') == historical_plan['baseline_frozen_sha256'], 'Frozen baseline data pipeline changed')
    for dataset, spec in plan['datasets'].items():
        config = read(ROOT / spec['config'])
        require(config == read(frozen / 'configs' / f'{dataset}_s31.json'), 'Config differs from frozen seed-31 recipe')
    return plan


def data_ready(dataset, spec):
    manifest = Path(spec['manifest'])
    marker_path = manifest.parent / 'COMPLETE.json'
    require(sha(manifest) == spec['manifest_sha256'], 'Dataset manifest changed')
    require(sha(marker_path) == spec['complete_sha256'], 'Dataset completion evidence changed')
    marker = read(marker_path)
    require(marker.get('status') in ('completed', 'validated'), 'Dataset is incomplete')
    require(marker.get('manifest_sha256') == sha(manifest), 'Completion marker does not bind manifest')
    document = read(manifest)
    if dataset == 'jhtdb':
        require(document.get('mode') == 'scientific', 'JHTDB requires its scientific spatial manifest')
    else:
        require(document.get('schema') == 'fs_transient_baseline_manifest_v1' and document.get('mode') == 'within_case_temporal', 'FS requires its temporal-window manifest')
    seen_ids, seen_paths, groups = set(), set(), {}
    for split in ('train', 'validation', 'test'):
        rows = document.get('splits', {}).get(split, [])
        require(len(rows) == spec['split_counts'][split] and rows, 'Unexpected split count: ' + split)
        groups[split] = set()
        for row in rows:
            path = (manifest.parent / row['path']).resolve()
            require(path.is_relative_to(manifest.parent.resolve()) and path.is_file(), 'Missing or escaping archive: ' + str(path))
            require(row['case_id'] not in seen_ids and str(path) not in seen_paths, 'Duplicate archive/case across splits')
            seen_ids.add(row['case_id']); seen_paths.add(str(path)); groups[split].add(row['flow_group_id'])
    if dataset == 'jhtdb':
        require(not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test']), 'Spatial group leakage')
    else:
        require(len(set.union(*groups.values())) == 1, 'FS must retain its one physical case')
        edges = [(min(r['window_id'] for r in document['splits'][s]), max(r['window_id'] for r in document['splits'][s])) for s in ('train', 'validation', 'test')]
        require(edges[0][1] < edges[1][0] and edges[1][1] < edges[2][0], 'FS temporal windows overlap')
    return manifest


def load_runtime(plan):
    frozen = Path(plan['frozen_root'])
    sys.path.insert(0, str(frozen / 'launch'))
    sys.path.insert(0, str(frozen / 'source/src'))
    module_spec = importlib.util.spec_from_file_location('frozen_tensor_runtime', frozen / 'launch/runtime.py')
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    os.environ['TENSOR_FROZEN_SHA256'] = plan['frozen_sha256']
    runtime.verify_inputs()
    return runtime


def completed_formal(output, dataset, manifest, artifact, config, runtime):
    """Only skip a fully verified completed formal checkpoint."""
    if not (output / 'latest.pt').is_file() or not (output / 'run.json').is_file():
        return False
    marker = read(output / 'run.json')
    if marker.get('status') != 'completed' or marker.get('epoch') != 100:
        return False
    state = runtime.completed_checkpoint(output, dataset, 100, manifest, artifact, config)
    runtime.formal_scope(state, manifest)
    return True


def run_stages(plan, dataset, runtime, update):
    spec = plan['datasets'][dataset]
    manifest = data_ready(dataset, spec)
    output = Path(spec['output_dir'])
    cache, pilot, formal = output / 'tensor_space', output / 'pilot', output / 'tensor_dit_s31'
    config = ROOT / spec['config']
    frozen = Path(plan['frozen_root'])
    update('tensor_prepare', 'running')
    runtime.run([plan['python'], '-B', frozen / 'launch/tensor_data.py', 'prepare',
                 '--dataset', dataset, '--manifest', manifest, '--output-dir', cache,
                 '--device', 'cuda', '--ranks', '4,8,12,16', '--hooi-iterations', 2, '--resume'])
    artifact = runtime.artifact_ready(cache, manifest, dataset)
    update('tensor_prepare', 'completed')
    update('gpu_pilot', 'running')
    runtime.run_pilot(plan, dataset, manifest, artifact, pilot)
    runtime.pilot_ready(pilot, manifest, artifact)
    update('gpu_pilot', 'completed')
    update('train_seed31', 'running')
    if not completed_formal(formal, dataset, manifest, artifact, config, runtime):
        runtime.run(runtime.train_command(dataset, manifest, artifact, config, formal, 31, 100))
    state = runtime.completed_checkpoint(formal, dataset, 100, manifest, artifact, config)
    runtime.formal_scope(state, manifest)
    update('train_seed31', 'completed')
    return {'checkpoint': str(formal / 'latest.pt'), 'checkpoint_sha256': sha(formal / 'latest.pt'), 'epoch': 100, 'seed': 31, 'test_evaluated': False}


def execute(dataset, expected):
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Training requires a real Slurm allocation')
    require(not os.environ.get('SLURM_ARRAY_TASK_ID'), 'This workflow does not use a seed array')
    plan = verified_plan(expected)
    require(dataset in plan['datasets'], 'Dataset not present in submitted plan')
    require(Path(sys.executable).resolve() == Path(plan['python']).resolve(), 'Use the pinned GH200 Python interpreter')
    data_ready(dataset, plan['datasets'][dataset])
    output = Path(plan['datasets'][dataset]['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.continuous.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'status': 'running', 'dataset': dataset, 'model': 'tensor_dit', 'seed': 31,
                  'job_id': os.environ['SLURM_JOB_ID'], 'operation_sha256': expected,
                  'started_unix': time.time(), 'stages': {}, 'original_source_unchanged': True}
        status_path = output / 'continuous_status.json'
        def update(stage, state):
            status['stage'] = stage
            status['stages'][stage] = {'status': state, 'updated_unix': time.time()}
            write(status_path, status)
            print(json.dumps({'stage': stage, 'status': state, 'dataset': dataset}), flush=True)
        try:
            runtime = load_runtime(plan)
            status['gpu'] = runtime.gpu_check()
            write(status_path, status)
            result = run_stages(plan, dataset, runtime, update)
            status.update(status='completed', completed_unix=time.time(), result=result)
            write(status_path, status)
        except BaseException as error:
            status.update(status='failed', failed_unix=time.time(), error=str(error), traceback=traceback.format_exc())
            write(status_path, status)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'run'))
    parser.add_argument('--dataset', choices=('jhtdb', 'fs_sph'))
    parser.add_argument('--operation-fingerprint')
    args = parser.parse_args()
    if args.action == 'run':
        require(args.dataset and args.operation_fingerprint, 'Run requires dataset and submission-time fingerprint')
        execute(args.dataset, args.operation_fingerprint)
    else:
        plan = verified_plan(args.operation_fingerprint)
        selected = [args.dataset] if args.dataset else list(plan['datasets'])
        report = {dataset: {'manifest': str(data_ready(dataset, plan['datasets'][dataset])),
                            'split_counts': plan['datasets'][dataset]['split_counts']} for dataset in selected}
        print(json.dumps({'passed': True, 'datasets': report, 'operation_sha256': sha(ROOT / 'operation_manifest.json'),
                          'scope': 'Frozen bytes, completion/manifest hashes, split identity and archive presence only; no CUDA/Tucker/training run.'}, indent=2))


if __name__ == '__main__':
    main()
