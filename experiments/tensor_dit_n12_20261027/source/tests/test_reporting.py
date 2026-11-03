import copy
import hashlib
import json
import math
import numpy as np
import pytest
import torch
from flow_baselines.reporting import describe, summarize, verify

torch.set_num_threads(1)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prediction(tmp_path, case='a', seed=31, error=1., **changes):
    record = {'format':'flow3d-baseline-prediction-v1','status':'completed','model':'unet',
        'reference_used_during_inference':False,'actual_predictions':1,'case_id':case,
        'training_seed':seed,'config':{'seed':seed,'num_particles':2,'views':[0],
        'boundary_projection':'none','sampling_seed':47},'views':[0],
        'num_observed_particles':2,'split':'test','manifest_sha256':'same',
        'evaluation':{'per_sample_field_relative_l2_unknown_interior':[error],
        'probe_metrics':[{'probe_track_rmse_cells':error/2}]}}
    record.update(changes)
    path = tmp_path/f'{case}_{seed}_{len(list(tmp_path.glob("*.npz")))}.npz'
    values = np.zeros((record['actual_predictions'],3,2,2,2),dtype=np.float32)
    np.savez(path,predictions=values,prediction_mean=values.mean(0),metadata=np.array(json.dumps(record)))
    path.with_suffix('.json').write_text(json.dumps({**record,'output_sha256':digest(path)}))
    return path

def test_describe_counts_without_forward_or_fitting(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Parameter inventory must not execute forward')
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    report = describe(tmp_path)
    models = {row['model']:row for row in report['models']}
    assert len(models) == 9
    assert models['mean']['total_trainable_parameters'] == 0
    assert models['rbf']['total_trainable_parameters'] == 195
    assert models['pinn']['total_trainable_parameters'] == 67076
    assert models['pinn']['hyperparameters']['viscosity'] is None
    assert models['pinn']['viscosity_required'] is True
    assert models['pod']['requested_rank'] == 32
    assert models['pod']['effective_rank'] is None
    for kind in ('unet','cnn','fno','deeponet','unet_diffusion'):
        row = models[kind]
        assert row['total_trainable_parameters'] == row['encoder_parameters'] + row['decoder_parameters']
    assert (tmp_path/'parameters.csv').exists() and (tmp_path/'parameters.txt').exists()

def test_verify_prediction_checks_integrity_and_metadata(tmp_path):
    path = prediction(tmp_path)
    assert verify(path)['status'] == 'verified'
    record = json.loads(path.with_suffix('.json').read_text())
    record['training_seed'] = 999
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='metadata'):
        verify(path)

def test_verify_prediction_rejects_reference_leakage(tmp_path):
    path = prediction(tmp_path,reference_used_during_inference=True)
    with pytest.raises(ValueError,match='reference'):
        verify(path)

@pytest.mark.parametrize('fault',('count','nonfinite','components'))
def test_prediction_array_shape_and_finiteness_are_checked(tmp_path,fault):
    path = prediction(tmp_path)
    with np.load(path) as archive:
        metadata = archive['metadata'].copy()
    values = np.zeros((2 if fault=='count' else 1,2 if fault=='components' else 3,2,2,2),np.float32)
    if fault == 'nonfinite':
        values.flat[0] = np.nan
    np.savez(path,predictions=values,metadata=metadata)
    record = json.loads(path.with_suffix('.json').read_text())
    record['output_sha256'] = digest(path)
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='finite|dimensions'):
        verify(path)

def checkpoint(tmp_path, updates=1, finite=True):
    weights = {'weight':torch.tensor([1. if finite else float('nan')])}
    state = {'format':'flow3d-baseline-v1','model':'unet','config':{},
             'normalization_fit_split':'train','optimizer_steps':updates,
             'system':weights,'ema_system':copy.deepcopy(weights)}
    torch.save(state,tmp_path/'latest.pt')
    (tmp_path/'run.json').write_text(json.dumps({'status':'completed','model':'unet',
        'config':{},'checkpoint_sha256':digest(tmp_path/'latest.pt')}))

def test_verify_training_requires_finite_weights_and_real_updates(tmp_path):
    checkpoint(tmp_path)
    assert verify(tmp_path)['verified']
    checkpoint(tmp_path,updates=0)
    with pytest.raises(ValueError,match='optimizer'):
        verify(tmp_path)
    checkpoint(tmp_path,finite=False)
    with pytest.raises(ValueError,match='nonfinite'):
        verify(tmp_path)
    checkpoint(tmp_path)
    with (tmp_path/'latest.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path)

def test_summary_averages_cases_before_seed_sample_sd(tmp_path):
    inputs = [prediction(tmp_path,case,seed,error).with_suffix('.json')
              for seed,values in ((31,(1.,3.)),(32,(5.,7.)))
              for case,error in zip(('a','b'),values)]
    group = summarize(inputs,tmp_path/'summary.json')['groups'][0]
    field = group['metrics']['field_relative_l2']
    assert group['replicate_kind'] == 'training_seed'
    assert field['mean'] == 4.
    assert field['std'] == pytest.approx(math.sqrt(8))
    assert field['seed_count'] == 2
    assert [row['metrics']['field_relative_l2'] for row in group['seed_case_means']] == [2.,6.]

def test_summary_single_seed_duplicate_and_protocol_separation(tmp_path):
    first = prediction(tmp_path).with_suffix('.json')
    result = summarize([first],tmp_path/'summary.json')
    assert result['groups'][0]['metrics']['field_relative_l2']['std'] is None
    with pytest.raises(ValueError,match='Duplicate'):
        summarize([first,first],tmp_path/'summary.json')
    other = prediction(tmp_path,views=[1]).with_suffix('.json')
    assert len(summarize([first,other],tmp_path/'summary.json')['groups']) == 2

def test_mean_seed_labels_are_not_independent_replicates(tmp_path):
    one = prediction(tmp_path,model='mean',training_seed=None).with_suffix('.json')
    two = prediction(tmp_path,seed=32,model='mean',training_seed=None).with_suffix('.json')
    with pytest.raises(ValueError,match='Duplicate'):
        summarize([one,two],tmp_path/'summary.json')

def test_summary_rejects_unbalanced_cases_and_empty_input(tmp_path):
    one = prediction(tmp_path,case='a',seed=31).with_suffix('.json')
    two = prediction(tmp_path,case='b',seed=32).with_suffix('.json')
    with pytest.raises(ValueError,match='different case sets'):
        summarize([one,two],tmp_path/'summary.json')
    with pytest.raises(ValueError,match='No completed'):
        summarize([],tmp_path/'summary.json')

def test_summary_keeps_training_particle_counts_and_views_separate(tmp_path):
    paths = [prediction(tmp_path,training_num_particles=n,training_views=views).with_suffix('.json')
             for n,views in ((2,[0]),(4,[0]),(2,[0,1]))]
    groups = summarize(paths,tmp_path/'summary.json')['groups']
    assert len(groups) == 3
    assert {group['protocol']['training_num_particles'] for group in groups} == {2,4}

def test_summary_reports_replay_validity_and_end_to_end_inference_time(tmp_path):
    path = prediction(tmp_path,actual_predictions=2,valid_prediction_count=1,
        diagnostics=[{'valid_replay_fraction':1.},{'valid_replay_fraction':.5}],
        generation_and_diagnostics_seconds=12.).with_suffix('.json')
    metrics = summarize([path],tmp_path/'summary.json')['groups'][0]['metrics']
    assert metrics['replay_valid_fraction']['mean'] == .75
    assert metrics['prediction_valid_fraction']['mean'] == .5
    assert metrics['generation_fitting_and_diagnostics_seconds']['mean'] == 12.

def test_summary_rejects_invalid_prediction_validity_count(tmp_path):
    path = prediction(tmp_path,valid_prediction_count=2).with_suffix('.json')
    with pytest.raises(ValueError,match='valid_prediction_count'):
        summarize([path],tmp_path/'summary.json')
