"""Bounded CPU checks of advection physics, termination and observation isolation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

BATCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BATCH/'source/src'))
spec = importlib.util.spec_from_file_location('prepare_jhtdb',BATCH/'launch/prepare_jhtdb.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_constant_field_matches_analytic_motion_and_projection(tmp_path):
    spacing = 2*np.pi/1024
    velocity = np.asarray([8.,-3.,4.],dtype=np.float32)
    field = np.broadcast_to(velocity[None,:,None,None,None]*spacing,(1,3,32,32,32)).copy()
    payload = prep.build_case(field,spacing,'analytic')
    xyz = payload['trajectories_3d'][0]
    expected = xyz[:,0,None,:]+payload['observation_times'][None,:,None]*velocity
    np.testing.assert_allclose(xyz,expected,atol=8e-5,rtol=0)
    assert payload['observation_mask'].all()
    np.testing.assert_allclose(payload['flow_field'][0,:,0,0,0],velocity,rtol=1e-6)
    np.testing.assert_array_equal(payload['trajectories_2d'][0,0],xyz[:,:,[0,2]])
    payload['metadata']=np.asarray(json.dumps({'case_id':'analytic'}))
    archive=tmp_path/'case.npz'
    prep.save_case(archive,payload)
    assert len(prep.validate_case(archive))==64


def test_spatially_linear_velocity_matches_exponential_solution():
    spacing = 2*np.pi/1024
    field = np.zeros((1,3,32,32,32),dtype=np.float32)
    field[0,0]=np.arange(32,dtype=np.float32)[None,None,:]*20*spacing
    payload = prep.build_case(field,spacing,'linear')
    xyz = payload['trajectories_3d'][0]
    np.testing.assert_allclose(xyz[:,-1,0],xyz[:,0,0]*np.exp(20*prep.HORIZON),atol=1e-4,rtol=0)
    np.testing.assert_array_equal(xyz[:,-1,1:],xyz[:,0,1:])


def test_exit_terminates_without_periodic_wrap_or_valid_clamping():
    field=torch.zeros((1,3,4,4,4))
    field[:,0]=10
    initial=torch.tensor([[2.8,1.,1.],[0.1,1.,1.]])
    xyz,valid=prep.integrate(field,initial,torch.tensor([0.,0.1,0.2,1.]),4)
    assert valid[:,0].all() and not valid[:,-1].any()
    assert not torch.diff(valid.to(torch.int8),dim=1).gt(0).any()
    assert torch.all(xyz[valid]>=0) and torch.all(xyz[valid]<=3)
    assert torch.all(xyz[0,1:]==initial[0])


@pytest.fixture
def fixture_manifest(tmp_path):
    entries={}
    for split in ('train','validation','test'):
        case_id='fixture_'+split
        payload=prep.build_case(np.zeros((1,3,32,32,32),dtype=np.float32),1.,case_id)
        payload['metadata']=np.asarray(json.dumps({'case_id':case_id}))
        prep.save_case(tmp_path/(case_id+'.npz'),payload)
        entries[split]=[{'case_id':case_id,'path':case_id+'.npz','flow_group_id':'block_'+split}]
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'mode':'scientific','splits':entries}))
    return path


def test_shared_observations_ignore_model_seed_and_hold_probes_fixed(fixture_manifest):
    from flow_baselines.common import TrainingDataset
    from flow_baselines.config import resolve_config
    from flow_observation.sparse_dataset import FlowNormalizationStats
    stats=FlowNormalizationStats(torch.zeros(3),torch.ones(3),1)
    def dataset(seed,count=12):
        config=resolve_config(overrides={'seed':seed,'observation_seed':47,'num_probes':32,'num_particles':count,'views':[0]})
        return TrainingDataset(fixture_manifest,'train',config,stats)
    first,other,nested=dataset(31),dataset(33),dataset(31,24)
    fixed_probes=first[0]['probe_particle_indices']
    for epoch in (0,1,7):
        for item in (first,other,nested):
            item.set_epoch(epoch)
        left,right,larger=first[0],other[0],nested[0]
        assert torch.equal(left['particle_indices'],right['particle_indices'])
        assert torch.equal(left['particle_indices'],larger['particle_indices'][:12])
        assert torch.equal(left['probe_particle_indices'],fixed_probes)
        assert not set(left['particle_indices'].tolist()) & set(fixed_probes.tolist())
        assert left['tracks'].shape==(1,12,20,2)


@pytest.mark.parametrize('split',['validation','test'])
def test_frozen_selection_matches_observation_only_inference(fixture_manifest,split):
    from flow_baselines.common import TrainingDataset
    from flow_baselines.config import resolve_config
    from flow_observation.dps.data import load_observation_case
    from flow_observation.sparse_dataset import FlowNormalizationStats
    config=resolve_config(overrides={'seed':33,'num_particles':12,'num_probes':32,'observation_seed':47,'views':[0]})
    dataset=TrainingDataset(fixture_manifest,split,config,FlowNormalizationStats(torch.zeros(3),torch.ones(3),1))
    expected=load_observation_case(dataset.cases[0],split,12,observation_seed=47,num_probes=32,views=[0])
    dataset.set_epoch(9)  # Validation must never change when an epoch is supplied.
    actual=dataset[0]
    for key in ('particle_indices','probe_particle_indices','tracks','mask'):
        assert torch.equal(actual[key],expected[key])
    assert not actual['boundary_mask'].any()


def test_full_cpu_preparation_refused_before_io(tmp_path):
    args=SimpleNamespace(source_root=tmp_path/'missing',output_root=tmp_path/'must_not_exist',
                         device='cpu',smoke_case=None,resume=False)
    with pytest.raises(ValueError,match='requires Slurm'):
        prep.run(args)
    assert not args.output_root.exists()
