"""Measured parameter inventories and protocol-preserving result summaries."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULTS, MODELS, SUPERVISED


def _read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def describe(output_dir):
    """Count constructor parameters only; never fit data or run a forward pass."""
    from .common import build_system
    from .linear import RBFField
    from .pinn import PINNField
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(DEFAULTS)
    rows = []
    bounds = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
    for kind in MODELS:
        row = {'model': kind, 'encoder_parameters': 0, 'decoder_parameters': 0,
               'total_trainable_parameters': 0, 'dataset_gradient_training': kind in SUPERVISED,
               'case_fit_parameters': 'none', 'viscosity_required': kind == 'pinn',
               'hyperparameters_validation_tuned': False,
               'hyperparameters': deepcopy(config.get(kind, {})), 'notes': ''}
        if kind in SUPERVISED:
            with torch.random.fork_rng(devices=[]):
                system = build_system(kind, (32, 32, 32), config, [1., 1., 1., 1.])
            row.update(encoder_parameters=_parameters(system['encoder']),
                       decoder_parameters=_parameters(system['decoder']),
                       total_trainable_parameters=_parameters(system))
            row['notes'] = 'Counts include the shared motion-residual-v2 encoder and auxiliary flow head.'
            del system
        elif kind == 'mean':
            row['notes'] = 'Training-only empirical mean; zero learned gradient parameters.'
        elif kind == 'pod':
            row['requested_rank'] = config['pod']['rank']
            row['effective_rank'] = None
            row['case_fit_parameters'] = 'K + N*(3-rank(P_initial)); K=min(requested_rank,training_numerical_rank)'
            row['notes'] = 'Training-only POD fit; effective rank cannot be known before fitting training snapshots.'
        else:
            if kind == 'rbf':
                module = RBFField(bounds, (32, 32, 32), **config['rbf'])
            else:
                settings = {key: config['pinn'][key] for key in ('hidden','layers','collocation_points')}
                module = PINNField(bounds, (32, 32, 32), viscosity=1., **settings)
            row['decoder_parameters'] = row['total_trainable_parameters'] = _parameters(module)
            row['case_fit_parameters'] = f"{row['total_trainable_parameters']} + N*(3-rank(P_initial))"
            row['notes'] = ('Per-case optimization, not a network pretrained across the dataset. '
                            'Depth unknowns are zero for full-rank multiview initial geometry.')
            if kind == 'pinn':
                row['notes'] += ' Viscosity=1 was used only to count parameters; configured viscosity remains None and must be supplied.'
            del module
        rows.append(row)
    report = {'format': 'flow3d-baseline-parameters-v1', 'spatial_shape': [32,32,32],
              'hyperparameters_validation_tuned': False, 'counts_are_real_scalar_parameters': True,
              'parameter_count_motion_scales_are_placeholders': True,
              'defaults': config, 'models': rows}
    _write_json(output_dir/'parameters.json', report)
    columns = ['model','encoder_parameters','decoder_parameters','total_trainable_parameters',
               'dataset_gradient_training','case_fit_parameters','viscosity_required',
               'hyperparameters_validation_tuned','hyperparameters','notes']
    with (output_dir/'parameters.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'hyperparameters': json.dumps(row['hyperparameters'],sort_keys=True)})
    lines = ['Flow3D baseline defaults: 32 x 32 x 32; hyperparameters NOT validation-tuned.',
             'Counts are real scalar trainable parameters; statistical fits and per-case optimization are distinguished.', '']
    for row in rows:
        lines += [f"{row['model']}: encoder={row['encoder_parameters']:,}; decoder={row['decoder_parameters']:,}; total={row['total_trainable_parameters']:,}",
                  f"  settings: {json.dumps(row['hyperparameters'],sort_keys=True)}",
                  f"  per-case: {row['case_fit_parameters']}", f"  {row['notes']}"]
    (output_dir/'parameters.txt').write_text('\n'.join(lines)+'\n')
    return report


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    if isinstance(value,float):
        return math.isfinite(value)
    return True


def _prediction_record(record):
    if record.get('format') != 'flow3d-baseline-prediction-v1' or record.get('status') != 'completed':
        raise ValueError('Expected a completed baseline prediction record')
    if record.get('model') not in MODELS:
        raise ValueError('Unknown baseline prediction model')
    if record.get('reference_used_during_inference') is not False:
        raise ValueError('Prediction must explicitly exclude reference use during inference')


def verify(path):
    """Verify completed checkpoint directories or matching NPZ/JSON pairs."""
    path = Path(path)
    if path.is_dir():
        record, checkpoint = _read_json(path/'run.json'), path/'latest.pt'
        if record.get('status') != 'completed' or record.get('model') not in MODELS:
            raise ValueError('Training run is not a completed baseline')
        if record.get('checkpoint_sha256') != _sha256(checkpoint):
            raise ValueError('Checkpoint checksum mismatch')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        kind = record['model']
        if state.get('format') != 'flow3d-baseline-v1' or state.get('model') != kind:
            raise ValueError('Checkpoint format/model differs from completion record')
        if state.get('config') != record.get('config'):
            raise ValueError('Checkpoint configuration differs from completion record')
        if 'epoch' in record and record['epoch'] != state.get('epoch'):
            raise ValueError('Checkpoint epoch differs from completion record')
        if state.get('normalization_fit_split') != 'train':
            raise ValueError('Checkpoint normalization was not fitted on training data')
        if kind in SUPERVISED:
            if 'epochs' in state.get('config',{}) and state.get('epoch',-1) < state['config']['epochs']:
                raise ValueError('Training did not reach requested final epoch')
            updates = state.get('optimizer_steps')
            if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
                raise ValueError('No successful optimizer updates')
            for key in ('system','ema_system'):
                weights = state.get(key)
                if not isinstance(weights, dict) or not weights or not all(isinstance(v,torch.Tensor) for v in weights.values()) or not _finite_tensors(weights):
                    raise ValueError(f'Missing or nonfinite {key} weights')
        elif kind == 'mean':
            mean = state.get('mean')
            if not isinstance(mean, torch.Tensor) or mean.ndim != 5 or mean.shape[:2] != (1,3) or min(mean.shape[2:])<1 or not mean.is_floating_point() or not _finite_tensors(mean):
                raise ValueError('Missing or invalid mean field')
        elif kind == 'pod':
            pod = state.get('pod_state')
            if not isinstance(pod, dict) or not {'mean','basis','eigenvalues'} <= pod.keys() or not _finite_tensors(pod):
                raise ValueError('Missing or nonfinite POD state')
            from .linear import PODField
            PODField(pod)
        else:
            raise ValueError('Per-case models do not have dataset training checkpoints')
        return {'status':'verified','verified':True,'artifact_type':'training_checkpoint',
                'model':kind,'path':str(path),'checkpoint_sha256':record['checkpoint_sha256']}
    if path.suffix not in ('.npz','.json'):
        raise ValueError('Expected a checkpoint directory, prediction NPZ, or prediction JSON')
    archive = path.with_suffix('.npz')
    sidecar = archive.with_suffix('.json')
    record = _read_json(sidecar)
    _prediction_record(record)
    if record.get('output_sha256') != _sha256(archive):
        raise ValueError('Prediction checksum mismatch')
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        if {k:v for k,v in record.items() if k != 'output_sha256'} != metadata:
            raise ValueError('NPZ metadata and JSON completion record differ')
        predictions = data['predictions']
        actual = record.get('actual_predictions')
        if isinstance(actual,bool) or not isinstance(actual,int) or actual<1:
            raise ValueError('Invalid actual_predictions count')
        if predictions.ndim != 5 or predictions.shape[:2] != (actual,3) or min(predictions.shape[2:]) < 1:
            raise ValueError('Prediction dimensions differ from actual_predictions')
        if not np.issubdtype(predictions.dtype,np.floating) or not np.isfinite(predictions).all():
            raise ValueError('Predictions must be finite floating point fields')
        if 'prediction_mean' in data:
            mean = data['prediction_mean']
            if mean.shape != predictions.shape[1:] or not np.isfinite(mean).all() or not np.allclose(mean,predictions.mean(0),rtol=1e-5,atol=1e-6):
                raise ValueError('Invalid prediction_mean')
    return {'status':'verified','verified':True,'artifact_type':'prediction',
            'model':record['model'],'path':str(archive),'actual_predictions':actual,
            'output_sha256':record['output_sha256']}


def _mean(values):
    values = list(values)
    available = []
    for value in values:
        if value is None:
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Metrics must be finite numbers or explicit null')
        available.append(float(value))
    if available and len(available) != len(values):
        raise ValueError('Incomplete metric coverage: refusing to silently omit null samples, cases, or seeds')
    return float(np.mean(available)) if available else None


def _case_metrics(record):
    evaluation = record.get('evaluation')
    if not isinstance(evaluation,dict):
        raise ValueError('Completed prediction has no posthoc evaluation')
    errors = evaluation.get('per_sample_field_relative_l2_unknown_interior')
    field = _mean(errors) if isinstance(errors,list) else _mean([evaluation.get('field_l2')])
    probes = evaluation.get('probe_metrics',[])
    valid_count, actual_count = record.get('valid_prediction_count'), record.get('actual_predictions')
    prediction_valid_fraction = None
    if valid_count is not None:
        if (isinstance(valid_count,bool) or not isinstance(valid_count,int)
                or isinstance(actual_count,bool) or not isinstance(actual_count,int)
                or actual_count<1 or not 0<=valid_count<=actual_count):
            raise ValueError('Invalid valid_prediction_count / actual_predictions')
        prediction_valid_fraction = valid_count/actual_count
    return {'field_relative_l2':field,
            'mean_field_relative_l2':_mean([evaluation.get('posterior_mean_field_relative_l2_unknown_interior')]),
            'probe_rmse':_mean([p.get('probe_track_rmse_cells') for p in probes]) if probes else _mean([evaluation.get('probe_rmse')]),
            'observed_track_rmse':_mean([p.get('observed_track_rmse_cells') for p in record.get('diagnostics',[])]),
            'replay_valid_fraction':_mean([p.get('valid_replay_fraction') for p in record.get('diagnostics',[])]),
            'prediction_valid_fraction':prediction_valid_fraction,
            'generation_fitting_and_diagnostics_seconds':_mean([record.get('generation_and_diagnostics_seconds')])}


def _protocol(record):
    config = deepcopy(record.get('config',{}))
    # Different training / fitting initializations are replicates; observations
    # and every other model/inference setting remain explicit grouping keys.
    config.pop('seed',None)
    selection = record.get('checkpoint_selection')
    return {'model':record['model'], 'num_particles':record.get('num_observed_particles',config.get('num_particles')),
            'training_num_particles':record.get('training_num_particles'),
            'training_views':record.get('training_views'),
            'training_boundary_projection':record.get('training_boundary_projection'),
            'checkpoint_selection':selection,
            # Validation-selected epochs may legitimately vary across training
            # seeds. Explicit/final epoch comparisons retain their actual epoch.
            'checkpoint_epoch':None if selection == 'best_validation' else record.get('checkpoint_epoch'),
            'views':record.get('views',config.get('views')),
            'boundary_projection':record.get('known_velocity_boundary_conditioning',config.get('boundary_projection')),
            'split':record.get('split'), 'manifest_sha256':record.get('manifest_sha256'),
            'scope':record.get('scope'), 'observation_protocol':record.get('observation_protocol'),
            'evaluation_protocol':record.get('evaluation_protocol',record.get('eval_protocol')),
            'prediction_type':record.get('prediction_type'), 'actual_predictions':record.get('actual_predictions'),
            'scientific_dataset':record.get('scientific_dataset'),
            'physical_information':record.get('physical_information',{}), 'config':config}


def summarize(inputs, output):
    """Average samples/cases within a seed, then compute SD across seed means.

    Supervised models use independent training seeds. POD/RBF/PINN use per-case
    fit seeds, explicitly labelled. Repeating a mean field under multiple seed
    labels is a duplicate, never an independent training replicate. All seeds
    within a group must cover the same case set to prevent composition effects.
    """
    paths = []
    ignored = 0
    for supplied in inputs:
        supplied = Path(supplied)
        candidates = sorted(supplied.rglob('*.json')) if supplied.is_dir() else [supplied]
        for path in candidates:
            if path.suffix == '.npz':
                path = path.with_suffix('.json')
            if path.suffix != '.json':
                raise ValueError('Summary inputs must be prediction JSON/NPZ files or directories')
            record = _read_json(path)
            if not isinstance(record,dict):
                raise ValueError('Summary JSON input must contain an object')
            if record.get('format') == 'flow3d-baseline-prediction-v1':
                paths.append(path)
            else:
                ignored += 1
    if not paths:
        raise ValueError('No completed evaluated prediction records supplied; no results were fabricated')
    groups = {}
    for path in paths:
        record = _read_json(path)
        _prediction_record(record)
        # Sidecar metrics alone cannot establish that a prediction completed:
        # require its finite field archive, checksum, and matching metadata.
        verify(path)
        protocol = _protocol(record)
        key = json.dumps(protocol,sort_keys=True,allow_nan=False)
        kind = record['model']
        replicate = 'training_seed' if kind in SUPERVISED else 'fit_seed' if kind in ('pod','rbf','pinn') else 'deterministic'
        seed = (record.get(replicate,record.get('seed',record.get('config',{}).get('seed')))
                if replicate != 'deterministic' else None)
        if replicate != 'deterministic' and (isinstance(seed,bool) or not isinstance(seed,int)):
            raise ValueError(f'Missing explicit {replicate}')
        case = record.get('case_id')
        if not isinstance(case,str) or not case:
            raise ValueError('Missing case_id')
        group = groups.setdefault(key,{'protocol':protocol,'replicate_kind':replicate,'seeds':{},
                                       'checkpoint_sources':{},'case_sources':{},'provenance_complete':True})
        checkpoint_hash = record.get('checkpoint_sha256')
        if seed in group['checkpoint_sources'] and group['checkpoint_sources'][seed] != checkpoint_hash:
            raise ValueError('Different checkpoint weights within one protocol/seed')
        group['checkpoint_sources'][seed] = checkpoint_hash
        source_hash = record.get('source_case_sha256')
        if case in group['case_sources'] and group['case_sources'][case] != source_hash:
            raise ValueError('Different source case contents across seeds within one protocol')
        group['case_sources'][case] = source_hash
        checkpoint_required = kind in SUPERVISED or kind in ('mean','pod')
        group['provenance_complete'] &= bool(source_hash) and (not checkpoint_required or bool(checkpoint_hash))
        cases = group['seeds'].setdefault(seed,{})
        if case in cases:
            raise ValueError(f'Duplicate case/seed: {case}/{seed} within one protocol')
        cases[case] = _case_metrics(record)
    results = []
    for key in sorted(groups):
        group = groups[key]
        case_sets = [set(cases) for cases in group['seeds'].values()]
        if any(cases != case_sets[0] for cases in case_sets[1:]):
            raise ValueError('Seeds cover different case sets; summarize the same cases for each seed')
        seed_means = []
        metric_names = ('field_relative_l2','mean_field_relative_l2','probe_rmse','observed_track_rmse',
                        'replay_valid_fraction','prediction_valid_fraction',
                        'generation_fitting_and_diagnostics_seconds')
        for seed,cases in sorted(group['seeds'].items(),key=lambda item:str(item[0])):
            values = {name:_mean([row[name] for row in cases.values()]) for name in metric_names}
            counts = {name:sum(row[name] is not None for row in cases.values()) for name in metric_names}
            seed_means.append({'seed':seed,'case_count':len(cases),'metrics':values,'valid_case_counts':counts,
                               'checkpoint_sha256':group['checkpoint_sources'][seed]})
        metrics = {}
        for name in metric_names:
            all_values = [row['metrics'][name] for row in seed_means]
            mean = _mean(all_values)
            values = [value for value in all_values if value is not None]
            metrics[name] = {'mean':mean,
                             'std':float(np.std(values,ddof=1)) if len(values)>1 else None,
                             'seed_count':len(values)}
        results.append({'protocol':group['protocol'],'replicate_kind':group['replicate_kind'],
                        'replicate_count':len(seed_means),'cases_per_replicate':len(case_sets[0]),
                        'case_ids':sorted(case_sets[0]),'seed_case_means':seed_means,'metrics':metrics,
                        'provenance_complete':bool(group['provenance_complete']),
                        'case_source_sha256':group['case_sources']})
    report = {'format':'flow3d-baseline-summary-v1','ddof':1,
              'aggregation':'sample average per case; case average within each seed; mean and sample SD across seed means',
              'training_seed_and_fit_seed_variability_are_distinct':True,
              'null_metric_policy':'all unavailable is null; partially unavailable is rejected',
              'prediction_artifacts_verified':True,
              'input_record_count':len(paths),'ignored_nonprediction_json_count':ignored,'groups':results}
    _write_json(output,report)
    return report
