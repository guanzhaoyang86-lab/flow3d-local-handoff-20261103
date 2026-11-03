"""Independent CPU checks of submission dependencies and scientific stage gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'launch'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_plan_exact_formal_work_and_resource_caps():
    module = load('tensor_submission_plan', 'submit_tensor.py')
    plan = module.build_plan()
    assert plan['sbatch_submission_count'] == 6
    assert plan['logical_allocations'] == 10
    assert plan['requested_gpu_hour_upper_bound'] == 66
    assert plan['combined_gpu_hour_upper_bound'] == 340
    assert plan['combined_gpu_hour_upper_bound'] < plan['verified_account_balance_gpu_hours']
    formal = [task for job in plan['jobs'] for task in job.get('tasks', [])]
    assert len(formal) == 6
    assert {t['model'] for t in formal} == {'tensor_dit'}
    assert {t['N'] for t in formal} == {12}
    assert {t['epochs'] for t in formal} == {100}
    for dataset in ('jhtdb','fs_sph'):
        assert {t['seed'] for t in formal if t['dataset']==dataset} == {31,32,33}


def test_external_failure_cannot_release_tensor_before_final_training():
    module = load('tensor_submission_dependencies', 'submit_tensor.py')
    plan = module.build_plan()
    for dataset in plan['datasets']:
        prep, pilot, train = [job for job in plan['jobs'] if job['dataset']==dataset]
        assert ['afterok','baseline_'+dataset+'_prepare'] in prep['dependencies']
        # The direct last-training dependency remains necessary even when a
        # calibration has already been cancelled by its own failed dependency.
        assert ['afterany','baseline_'+dataset+'_unet_diffusion'] in prep['dependencies']
        assert ['afterany','baseline_'+dataset+'_rbf_calibration'] in prep['dependencies']
        assert pilot['dependencies'] == [['afterok',prep['key']]]
        assert ['afterok',prep['key']] in train['dependencies']
        assert ['afterok',pilot['key']] in train['dependencies']
    ids = dict(plan['external_dependencies'])
    for index, job in enumerate(plan['jobs']):
        ids[job['key']] = str(8000000+index)
    for job in plan['jobs']:
        command = module.command(job,ids)
        assert '--gpus-per-node=1' in command
        assert '--kill-on-invalid-dep=yes' in command
        assert any(str(arg).startswith('--dependency=') for arg in command)
        if job['stage']=='train':
            assert '--array=0-2%1' in command
        else:
            assert not any(str(arg).startswith('--array=') for arg in command)
    ids['baseline_jhtdb_prepare']='invalid_job_id'
    with pytest.raises(ValueError,match='job ID'):
        module.command(plan['jobs'][0],ids)


def test_existing_submission_stops_before_external_calls(tmp_path,monkeypatch):
    module = load('tensor_submission_duplicate', 'submit_tensor.py')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    write(tmp_path/'submission/submission.json',{'status':'submitting'})
    def forbidden(*args,**kwargs):
        raise AssertionError('A duplicate attempt contacted scheduler')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='already attempted'):
        module.submit()


def test_dataset_marker_binds_actual_manifest(tmp_path):
    module = load('tensor_runtime_data', 'runtime.py')
    manifest=tmp_path/'manifest.json'
    write(manifest,{'splits':{split:[{'case_id':split}] for split in ('train','validation','test')}})
    write(tmp_path/'COMPLETE.json',{'status':'completed','manifest_sha256':module.sha(manifest)})
    assert module.data_ready({'derived_root':str(tmp_path)}) == manifest
    manifest.write_text(manifest.read_text()+' ')
    with pytest.raises(RuntimeError,match='bound'):
        module.data_ready({'derived_root':str(tmp_path)})


def test_tensor_artifact_marker_checked_before_loader(tmp_path,monkeypatch):
    module = load('tensor_runtime_artifact', 'runtime.py')
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    artifact=tmp_path/'cache/selected.pt'; artifact.parent.mkdir(); artifact.write_bytes(b'artifact')
    calls=[]
    monkeypatch.setitem(sys.modules,'tensor_data',SimpleNamespace(load_artifact=lambda *args:calls.append(args)))
    marker={'status':'completed','manifest_sha256':module.sha(manifest),'artifact_sha256':module.sha(artifact)}
    write(artifact.parent/'COMPLETE.json',marker)
    assert module.artifact_ready(artifact.parent,manifest,'jhtdb') == artifact
    assert len(calls)==1
    artifact.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='artifact differs'):
        module.artifact_ready(artifact.parent,manifest,'jhtdb')
    assert len(calls)==1


def test_completed_checkpoint_rejects_wrong_epoch_or_artifact(tmp_path):
    module=load('tensor_runtime_checkpoint','runtime.py')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    artifact=tmp_path/'selected.pt';artifact.write_bytes(b'artifact')
    config=tmp_path/'config.json';write(config,{'seed':31,'epochs':100})
    output=tmp_path/'run';output.mkdir()
    write(output/'run.json',{'status':'completed'})
    state={'model':'tensor_dit','dataset':'jhtdb','epoch':100,'optimizer_steps':3,
           'manifest_sha256':module.sha(manifest),'artifact_sha256':module.sha(artifact),
           'config':json.loads(config.read_text())}
    torch.save(state,output/'latest.pt')
    assert module.completed_checkpoint(output,'jhtdb',100,manifest,artifact,config)['epoch']==100
    state['epoch']=99;torch.save(state,output/'latest.pt')
    with pytest.raises(RuntimeError,match='epoch'):
        module.completed_checkpoint(output,'jhtdb',100,manifest,artifact,config)
    state['epoch']=100;state['artifact_sha256']='different';torch.save(state,output/'latest.pt')
    with pytest.raises(RuntimeError,match='artifact'):
        module.completed_checkpoint(output,'jhtdb',100,manifest,artifact,config)


def test_formal_command_has_no_smoke_limits_and_pilot_has_explicit_limits(tmp_path):
    module=load('tensor_runtime_limits','runtime.py')
    arguments=('jhtdb',tmp_path/'manifest',tmp_path/'artifact',tmp_path/'config',tmp_path/'out',31,100)
    formal=module.train_command(*arguments)
    assert '--max-train-cases' not in formal and '--max-validation-cases' not in formal
    pilot=module.train_command(*arguments,pilot=True)
    assert pilot[pilot.index('--max-train-cases')+1]==8
    assert pilot[pilot.index('--max-validation-cases')+1]==2


def test_pilot_gate_requires_real_optimizer_updates(tmp_path):
    module=load('tensor_runtime_pilot','runtime.py')
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    artifact=tmp_path/'selected.pt';artifact.write_bytes(b'artifact')
    marker={'status':'completed','model':'tensor_dit','manifest_sha256':module.sha(manifest),
            'artifact_sha256':module.sha(artifact),'optimizer_steps':0}
    write(tmp_path/'PILOT_COMPLETE.json',marker)
    with pytest.raises(RuntimeError,match='actual optimization'):
        module.pilot_ready(tmp_path,manifest,artifact)


@pytest.mark.parametrize('change',[
    {'pilot_subset':True}, {'max_train_cases':8}, {'max_validation_cases':2},
    {'limits':{'train':8}}, {'training_case_ids':['a']},
    {'validation_case_ids':['wrong_validation']},
])
def test_formal_scope_requires_unlimited_complete_split_membership(tmp_path,change):
    module=load('tensor_runtime_formal_membership','runtime.py')
    manifest=tmp_path/'manifest.json'
    write(manifest,{'splits':{'train':[{'case_id':'a'},{'case_id':'b'}],
                             'validation':[{'case_id':'v'}],'test':[{'case_id':'t'}]}})
    valid={'pilot_subset':False,'max_train_cases':None,'max_validation_cases':None,
           'training_case_ids':['a','b'],'validation_case_ids':['v']}
    module.formal_scope(valid,manifest)
    with pytest.raises(RuntimeError,match='limited|full'):
        module.formal_scope({**valid,**change},manifest)
