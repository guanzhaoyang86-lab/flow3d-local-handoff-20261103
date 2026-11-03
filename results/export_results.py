#!/usr/bin/env python3
"""Read server result records; export small metadata and inventory weights only.

No scheduler, GPU, training, Git, or network operations are performed. All writes
are confined to the directory containing this script. Requires Python 3.10+.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil


OUTPUT = Path(__file__).resolve().parent
PROJECT = Path('/u/zguan2/projects/flow3d_runs')
BASELINES = Path('/work/hdd/biup/zguan2/results/baselines_n12_20261027')
LEGACY_PLAN = PROJECT / 'deliverables/flow3d_full_training_20261020/prepared_plan.json'
MODELS = ('cnn', 'deeponet', 'unet', 'fno', 'unet_diffusion')
SMALL_SUFFIXES = {'.json', '.jsonl', '.csv', '.txt'}
WEIGHT_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path: Path, value):
    path.resolve().relative_to(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def history_metrics(path: Path):
    if not path.exists():
        return {'history_present': False}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    last = rows[-1] if rows else {}
    losses = [number(row.get('validation_loss')) for row in rows]
    losses = [loss for loss in losses if loss is not None]
    seconds = [number(row.get('elapsed_seconds')) for row in rows]
    return {
        'history_present': True, 'recorded_epochs': len(rows),
        'last_epoch': last.get('epoch'), 'last_train_loss': number(last.get('train_loss')),
        'last_validation_loss': number(last.get('validation_loss')),
        'minimum_recorded_validation_loss': min(losses) if losses else None,
        'sum_recorded_epoch_seconds': sum(value for value in seconds if value is not None),
        'timing_scope': 'Server epoch training plus validation; excludes queue and initialization; not laptop timing.',
        'metric_scope': 'Training/validation objective values, not final held-out test reconstruction metrics.',
    }


class Export:
    def __init__(self, limit):
        self.limit = limit
        self.files = []
        self.weights = []
        self.notes = []

    def copy(self, source, relative, scope):
        source, target = Path(source), OUTPUT / relative
        target.resolve().relative_to(OUTPUT)
        if not source.is_file():
            return None
        size = source.stat().st_size
        if source.suffix not in SMALL_SUFFIXES or size > self.limit:
            self.notes.append({'path': str(source), 'reason': 'Not an allowed small metadata file', 'bytes': size})
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        row = {'source_path': str(source), 'export_path': str(target.relative_to(OUTPUT)),
               'bytes': size, 'sha256': sha(target), 'scope': scope}
        self.files.append(row)
        return row['export_path']

    def small_directory(self, source, relative, scope, recursive=False):
        source = Path(source)
        if not source.is_dir():
            return
        paths = source.rglob('*') if recursive else source.iterdir()
        for path in sorted(paths):
            if path.is_file() and path.suffix in SMALL_SUFFIXES:
                self.copy(path, Path(relative) / path.relative_to(source), scope)

    def checkpoints(self, directory, *, group, dataset, model, seed=None, recorded=None):
        directory = Path(directory)
        recorded = recorded or {}
        if not directory.is_dir():
            return []
        rows = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix not in WEIGHT_SUFFIXES:
                continue
            row = {'source_path': str(path), 'bytes': path.stat().st_size,
                   'group': group, 'dataset': dataset, 'model': model, 'seed': seed,
                   'filename': path.name, 'copied': False,
                   'recorded_sha256': recorded.get(path.name),
                   'checksum_scope': 'Existing completion record only; large checkpoint bytes were not rehashed.'}
            self.weights.append(row)
            rows.append(row)
        return rows


def current_baselines(export):
    runs, calibrations, pilots = [], [], []
    for dataset in ('fs_sph', 'jhtdb'):
        root = BASELINES / dataset
        if not root.is_dir():
            export.notes.append({'path': str(root), 'reason': 'Result directory absent'})
            continue
        for directory in sorted(root.iterdir()):
            match = re.fullmatch(r'(mean|pod|cnn|deeponet|unet|fno|unet_diffusion)(?:_s(\d+))?', directory.name)
            if not directory.is_dir() or not match:
                continue
            model, seed = match.group(1), int(match.group(2)) if match.group(2) else None
            record = directory / 'run.json'
            data = read(record) if record.exists() else {}
            row = {'dataset': dataset, 'model': model, 'seed': seed, 'run_directory': str(directory),
                   'status': data.get('status', 'no_completion_record'), 'epoch': data.get('epoch'),
                   'current_single_seed_policy': seed in (None, 31),
                   'metrics_scope': 'Training/validation only; no final held-out test score is asserted.'}
            if data.get('status') == 'completed':
                relative = Path('current_baselines') / dataset / directory.name
                export.small_directory(directory, relative, 'current_new_dataset_completed_training')
                row['exported_run_record'] = str(relative / 'run.json')
                row['history_metrics'] = history_metrics(directory / 'history.jsonl')
                row['checkpoints'] = export.checkpoints(
                    directory, group='current_baseline', dataset=dataset, model=model, seed=seed,
                    recorded={'latest.pt': data.get('checkpoint_sha256')})
            runs.append(row)
        for model in ('pod', 'rbf'):
            directory = root / 'calibration' / model
            marker = directory / 'CALIBRATION_COMPLETE.json'
            data = read(marker) if marker.exists() else {}
            row = {'dataset': dataset, 'model': model, 'record': str(marker),
                   'status': data.get('status', 'not_completed'), 'split': 'validation',
                   'scope': 'Hyperparameter calibration, not final test evaluation.'}
            if data.get('status') == 'completed':
                export.small_directory(directory, Path('current_calibration') / dataset / model,
                                       'validation_calibration_only', recursive=True)
                row['selected_sigma'] = data.get('selected_sigma')
            calibrations.append(row)
        pilot = root / 'pilot'
        marker = pilot / 'PILOT_COMPLETE.json'
        if marker.exists():
            data = read(marker)
            export.copy(marker, Path('pilot_smoke_only') / dataset / marker.name, 'smoke_test_not_formal_result')
            for subdirectory in ('smoke_predictions', 'pod_fit_smoke', 'rbf_fit_smoke'):
                export.small_directory(pilot / subdirectory,
                                       Path('pilot_smoke_only') / dataset / subdirectory,
                                       'smoke_test_not_formal_result', recursive=True)
            pilots.append({'dataset': dataset, 'status': data.get('status'), 'test_used': data.get('test_used'),
                           'scope': 'Reduced one-epoch smoke test; not a formal 100-epoch model or test score.'})
    return runs, calibrations, pilots


def legacy_training(export):
    if not LEGACY_PLAN.exists():
        export.notes.append({'path': str(LEGACY_PLAN), 'reason': 'Legacy plan absent'})
        return [], {}
    plan = read(LEGACY_PLAN)
    export.copy(LEGACY_PLAN, 'legacy_conditional/provenance/prepared_plan.json', 'legacy_old_lbm_training_plan')
    rows = []
    for task in plan.get('tasks', []):
        directory = Path(task['run_dir'])
        training = Path(task['training_dir'])
        path = directory / 'results.json'
        result = read(path) if path.exists() else {}
        audit = result.get('audit', {})
        completed = result.get('status') == 'completed' and audit.get('status') == 'verified'
        row = {'key': task['key'], 'model': task['architecture'], 'N': task['num_particles'],
               'seed': task['seed'], 'status': 'completed_verified' if completed else result.get('status', 'no_completion_record'),
               'run_directory': str(directory), 'scope': 'Old LBM dataset; distinct from current FS-SPH/JHTDB/LBM500k.',
               'metrics_scope': 'Training/validation losses, not a current-data test result.'}
        if completed:
            relative = Path('legacy_conditional/completed_training') / task['key']
            export.small_directory(directory, relative, 'legacy_old_lbm_completed_training')
            export.small_directory(training, relative / 'training', 'legacy_old_lbm_training_objectives')
            summary_path = training / 'training_summary.json'
            summary = read(summary_path) if summary_path.exists() else {}
            row.update(epoch=audit.get('epoch'), elapsed_seconds=result.get('elapsed_seconds'),
                       best_validation_total_loss=summary.get('best_validation_total_loss'))
            known = {name: value.get('sha256') for name, value in audit.get('checkpoints', {}).items()}
            row['checkpoints'] = export.checkpoints(training, group='legacy_conditional', dataset='old_lbm_20261001',
                                                    model=task['architecture'], seed=task['seed'], recorded=known)
        rows.append(row)
    for task in plan.get('reused_N2', []):
        training = Path(task['training_dir'])
        summary_path = training / 'training_summary.json'
        data = read(summary_path) if summary_path.exists() else {}
        complete = data.get('status') == 'completed' and data.get('epochs') == 100
        row = {'key': task['key'], 'model': task['architecture'], 'N': 2, 'seed': task['seed'],
               'status': 'completed_reused_verified_summary' if complete else 'reuse_not_verified',
               'training_directory': str(training), 'scope': 'Old LBM dataset, reused N2 training.',
               'metrics_scope': 'Training/validation loss only.'}
        if complete:
            relative = Path('legacy_conditional/reused_N2_training') / task['key']
            export.small_directory(training, relative, 'legacy_old_lbm_reused_training')
            row.update(epoch=100, elapsed_seconds=data.get('elapsed_seconds'),
                       best_validation_total_loss=data.get('best_validation_total_loss'))
            row['checkpoints'] = export.checkpoints(
                training, group='legacy_conditional_reused_N2', dataset='old_lbm_20261001',
                model=task['architecture'], seed=task['seed'],
                recorded={Path(task['checkpoint']).name: task.get('checkpoint_sha256')})
        rows.append(row)
    provenance = {'manifest': plan.get('manifest'), 'train_case_count': plan.get('train_case_count'),
                  'validation_case_count': plan.get('validation_case_count'),
                  'source_commit': plan.get('source_commit'), 'source_plan': str(LEGACY_PLAN),
                  'warning': 'This is an older dataset and protocol; never merge with new FS-SPH/JHTDB metrics.'}
    return rows, provenance


def legacy_comparisons(export):
    result = []
    for name in ('flow3d_final_comparison_20261020', 'flow3d_unet_comparison_20261020',
                 'flow3d_unet_mean_std_20261020'):
        directory = PROJECT / 'deliverables' / name
        export.small_directory(directory, Path('legacy_comparisons') / name,
                               'legacy_comparison_with_its_own_dataset_and_split')
        result.append({'source': str(directory), 'exported_directory': str(Path('legacy_comparisons') / name),
                       'scope': 'Historical comparison only; consult copied scope/provenance for its dataset and split.'})
    final = PROJECT / 'deliverables/flow3d_final_comparison_20261020/final_comparison.json'
    if final.exists():
        result[0]['protocol'] = read(final).get('scope')
        result[0]['split_warning'] = 'N2 comparison on 44 validation recipients; not current-data held-out test performance.'
    return result


def dataset_provenance(export):
    roots = {
        'jhtdb': Path('/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32'),
        'fs_sph': Path('/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027'),
    }
    result = {}
    for name, root in roots.items():
        marker, manifest = root / 'COMPLETE.json', root / 'manifest.json'
        data = read(manifest) if manifest.exists() else {}
        result[name] = {'server_directory': str(root), 'split_counts': {key: len(value) for key, value in data.get('splits', {}).items()},
                        'adapter_completion': read(marker) if marker.exists() else None,
                        'data_payload_included': False}
        for path in (marker, manifest):
            export.copy(path, Path('dataset_provenance') / name / path.name, 'prepared_dataset_provenance_only')
    return result


def csv_rows(path, rows, columns):
    path.resolve().relative_to(OUTPUT)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-limit-mib', type=int, default=16)
    parser.add_argument('--queue-snapshot', type=Path, default=OUTPUT / 'server_queue.json')
    args = parser.parse_args()
    if args.metadata_limit_mib < 1:
        parser.error('--metadata-limit-mib must be positive')
    export = Export(args.metadata_limit_mib * 1024 ** 2)
    runs, calibrations, pilots = current_baselines(export)
    legacy, legacy_source = legacy_training(export)
    comparisons = legacy_comparisons(export)
    datasets = dataset_provenance(export)
    queue = read(args.queue_snapshot) if args.queue_snapshot.exists() else None
    completed = [row for row in runs if row['status'] == 'completed']
    primary = [row for row in completed if row['current_single_seed_policy']]
    missing = {dataset: [model for model in MODELS if not any(
        row['dataset'] == dataset and row['model'] == model and row['seed'] == 31 for row in primary)]
        for dataset in ('fs_sph', 'jhtdb')}
    now = datetime.now(timezone.utc).isoformat()
    total_weights = sum(row['bytes'] for row in export.weights)
    overview = {
        'schema': 'flow3d_completed_results_summary_v1', 'exported_at_utc': now,
        'current_policy': 'Training seed31; previously completed seed32/33 results retained as historical extras.',
        'completed_primary_runs': [{key: row.get(key) for key in ('dataset', 'model', 'seed', 'epoch', 'status')}
                                   for row in primary],
        'completed_extra_seed_runs': [{key: row.get(key) for key in ('dataset', 'model', 'seed', 'epoch')}
                                      for row in completed if not row['current_single_seed_policy']],
        'not_completed_primary_models': missing,
        'calibrations': calibrations,
        'final_test_metrics_for_current_datasets_available': False,
        'metric_warning': 'Copied train_loss/validation_loss are objectives, not final test velocity/replay metrics; losses across model objectives are not directly comparable.',
        'legacy_training_status_counts': dict(Counter(row['status'] for row in legacy)),
        'legacy_warning': legacy_source.get('warning'),
        'checkpoint_policy': 'Inventory only: no checkpoint, optimizer state, NPZ dataset or prediction array was copied.',
        'checkpoint_file_count': len(export.weights), 'checkpoint_total_bytes_all_versions': total_weights,
        'checkpoint_larger_than_100_mib': sum(row['bytes'] > 100 * 1024 ** 2 for row in export.weights),
        'copied_metadata_files': len(export.files), 'copied_metadata_bytes': sum(row['bytes'] for row in export.files),
        'queue_snapshot_file': args.queue_snapshot.name if queue is not None else None,
        'server_jobs_policy': 'Export reads existing artifacts only. It never cancels, submits or duplicates server jobs.',
        'methods_requiring_future_work': [
            {'method': 'Tensor-DiT', 'scope': 'New JHTDB/FS-SPH dataset batch',
             'status': 'Historical frozen plan exists; no current pending Tensor task is asserted by this exporter.',
             'next_step': 'Use current live queue snapshot and explicit single-seed policy before any future submission.'},
            {'method': 'DiT3D', 'scope': 'New JHTDB/FS-SPH dataset comparisons',
             'status': 'Code and older-data results exist; no completed new-data run found in this baseline result root.'},
            {'method': 'Flow3D-DPS', 'scope': 'Steady JHTDB/LBM data',
             'status': 'Separate unconditional-prior route; no matched new-data completed result found here.',
             'limitation': 'Current DPS implementation rejects time-dependent FS-SPH sequences.'},
        ],
    }
    inventory = {
        'schema': 'flow3d_results_inventory_v1', 'exported_at_utc': now,
        'scope': 'Current completed new-data baselines plus separately labelled legacy conditional training/comparisons.',
        'summary': overview, 'datasets': datasets, 'current_baseline_runs': runs,
        'validation_calibration_records': calibrations, 'pilot_smoke_records': pilots,
        'legacy_conditional_provenance': legacy_source, 'legacy_conditional_runs': legacy,
        'legacy_comparisons': comparisons, 'checkpoint_files_inventory_only': export.weights,
        'copied_metadata_files': export.files, 'skipped_or_missing_notes': export.notes,
        'server_queue_snapshot': queue,
    }
    write(OUTPUT / 'summary.json', overview)
    write(OUTPUT / 'results_inventory.json', inventory)
    csv_rows(OUTPUT / 'checkpoint_files.csv', export.weights,
             ('group', 'dataset', 'model', 'seed', 'filename', 'bytes', 'source_path', 'recorded_sha256', 'copied'))
    metrics = []
    for row in completed:
        history = row.get('history_metrics', {})
        metrics.append({key: row.get(key) for key in ('dataset', 'model', 'seed', 'epoch', 'current_single_seed_policy')} | history)
    csv_rows(OUTPUT / 'current_training_objectives.csv', metrics,
             ('dataset', 'model', 'seed', 'epoch', 'current_single_seed_policy', 'last_train_loss',
              'last_validation_loss', 'minimum_recorded_validation_loss', 'sum_recorded_epoch_seconds', 'metric_scope'))
    print(json.dumps({key: overview[key] for key in ('copied_metadata_files', 'copied_metadata_bytes',
                     'checkpoint_file_count', 'checkpoint_total_bytes_all_versions',
                     'legacy_training_status_counts', 'not_completed_primary_models')}, indent=2))


if __name__ == '__main__':
    main()
