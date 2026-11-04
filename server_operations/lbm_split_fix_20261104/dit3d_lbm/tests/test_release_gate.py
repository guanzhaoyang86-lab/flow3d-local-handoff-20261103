import json
import pytest
from test_lbm_binding import m
@pytest.fixture
def release(tmp_path,monkeypatch):
    d=tmp_path/'data'; r=tmp_path/'raw'; d.mkdir(); r.mkdir(); (d/'fields').mkdir()
    monkeypatch.setattr(m,'DATA',d); monkeypatch.setattr(m,'RAW',r)
    counts=dict(train=1,validation=1,test=1); monkeypatch.setattr(m,'COUNTS',counts)
    (r/'SHA256SUMS').write_text('fixture'); v=m.sha(r/'SHA256SUMS')
    raw=dict(status='completed',num_cases=3,num_independent_flow_groups=3,physical_trajectories=500000,
        single_view_trajectories=500000,dataset_version=v,split_counts=counts)
    (r/'COMPLETE.json').write_text(json.dumps(raw))
    (r/'manifest.json').write_text('{}')
    binding=dict(source_sha256='a'*64,raw_dataset_version=v,
                 raw_manifest_sha256=m.sha(r/'manifest.json'))
    monkeypatch.setattr(m,'adapter_binding',lambda:binding)
    splits={}
    for i,s in enumerate(counts):
        n=f'case_{i:05d}'; (d/'fields'/f'{n}.npz').touch()
        splits[s]=[dict(path=f'fields/{n}.npz',case_id=n,source_case_id=n,flow_group_id=n,sha256='a'*64)]
    manifest=dict(schema='flow3d_lbm_frozen_manifest_v1',dataset_id=m.DATASET_ID,source_dataset_version=v,
        mode='scientific',split_unit='flow_group_id',allow_overlap_for_smoke_test=False,
        velocity_units='lattice cells per LBM step',time_units='LBM steps',unit_scale=[1.,1.,1.],
        observation_protocol=dict(N=12,probe_count=32,observation_seed=47,view='xz',training_seed=31),
        scientific_scope='fixture',splits=splits)
    marker=dict(status='completed',dataset_id=m.DATASET_ID,arrays_modified=False,source_data_mutated=False,
        training_or_rank_selection_performed=False,verified_fields=3,source_dataset_version=v,
        source_complete_sha256=m.sha(r/'COMPLETE.json'),split_counts=counts)
    marker['adapter_sha256']=binding['source_sha256']
    def publish():
        (d/'manifest.json').write_text(json.dumps(manifest))
        marker['manifest_sha256']=m.sha(d/'manifest.json')
        (d/'COMPLETE.json').write_text(json.dumps(marker))
    publish()
    return d,r,manifest,marker,publish
def test_lbm_identity(release):
    assert m.data_check()['dataset']=='lbm'
@pytest.mark.parametrize('kind',['incomplete','hash','protocol','overlap','units','missing','old_adapter','changed_raw'])
def test_bad_publish(release,kind):
    d,r,v,c,publish=release
    if kind=='incomplete': c['status']='running'
    if kind=='protocol': v['observation_protocol']['observation_seed']=17
    if kind=='overlap': v['splits']['test'][0]['flow_group_id']=v['splits']['train'][0]['flow_group_id']
    if kind=='units': v['velocity_units']='m/s'
    if kind=='missing': (d/'fields/case_00000.npz').unlink()
    if kind=='old_adapter': c['adapter_sha256']='0'*64
    if kind=='changed_raw': (r/'manifest.json').write_text('{"changed":true}')
    publish()
    if kind=='hash': (d/'manifest.json').write_text('{}')
    with pytest.raises(ValueError): m.data_check()
