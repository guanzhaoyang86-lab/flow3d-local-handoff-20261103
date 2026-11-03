"""Synthetic CPU tests of protocol-preserving full-volume DiT3D adaptation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/src'))
from flow_baselines.config import resolve_config
from flow_baselines.training import train
from flow_baselines.common import TrainingDataset,build_system
from flow_observation.dps.data import fit_prior_stats,load_observation_case
from flow_observation.sparse_dataset import load_sparse_flow_manifest
from flow_observation.models.dit import ConditionalDiT3D
from test_fs_protocol import fs,dataset,train_args

torch.set_num_threads(1)

@pytest.fixture
def static_data(tmp_path):
    camera=np.array([[[1,0,0],[0,0,1]]],np.float32)
    times=np.array([0,.1,.2],np.float32)
    initial=np.random.default_rng(47).uniform(.2,.7,(48,3)).astype(np.float32)
    splits={'train':[],'validation':[],'test':[]}
    for index,split in enumerate(('train','train','validation','test')):
        velocity=np.array([.03+.01*index,.02,.04+.005*index],np.float32)
        field=np.broadcast_to(velocity[None,:,None,None,None],(1,3,8,8,8)).copy()
        positions=initial[:,None]+times[None,:,None]*velocity
        tracks=np.einsum('ntc,voc->vnto',positions,camera)[None]
        path=tmp_path/f'case_{index}.npz'
        np.savez(path,flow_field=field,solid_mask=np.zeros((8,8,8),bool),trajectories_2d=tracks,
                 observation_mask=np.ones(tracks.shape[:-1],bool),projection_matrix=camera,observation_times=times,
                 domain_bounds=np.array([[0.,1.]]*3,np.float32),particle_counts=np.array([48]),
                 initial_hidden_depth=np.array([object()],dtype=object),trajectories_3d=np.array([object()],dtype=object))
        splits[split].append({'case_id':f'case_{index}','flow_group_id':f'flow_{index}','path':path.name})
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'format_version':1,'mode':'scientific','splits':splits}))
    return path


def tiny(epochs=1):
    return resolve_config(overrides={'seed':31,'observation_seed':47,'num_particles':12,'num_probes':32,'views':[0],
           'condition_dim':8,'condition_flow_grid':2,'condition_flow_weight':.1,'amp':False,'epochs':epochs,
           'diffusion_steps':8,'sampling_steps':2,'num_samples':1,
           'dit3d':{'hidden_dim':24,'depth':1,'num_heads':3,'patch_size':2,'dropout':0.0}})


def test_static_observation_policy_matches_inference_and_probes_stay_reserved(static_data):
    _,splits=load_sparse_flow_manifest(static_data);stats=fit_prior_stats(splits['train'])
    data=TrainingDataset(static_data,'train',tiny(),stats)
    first=data[0]
    observation=load_observation_case(splits['train'][0],'train',12,observation_seed=47,num_probes=32,views=[0])
    assert torch.equal(first['particle_indices'],observation['particle_indices'])
    assert torch.equal(first['tracks'],observation['tracks'])
    assert torch.equal(first['projection'],torch.tensor([[[1.,0,0],[0,0,1.]]]))
    probes=first['probe_particle_indices'].clone();data.set_epoch(1);later=data[0]
    assert torch.equal(probes,later['probe_particle_indices'])
    assert not set(probes.tolist()) & set(later['particle_indices'].tolist())
    assert not first['boundary_mask'].any()


def test_static_dit3d_real_cpu_training_resume_exact(static_data,tmp_path):
    whole=train('dit3d',static_data,tmp_path/'whole',tiny(2),device='cpu')
    part=train('dit3d',static_data,tmp_path/'part',tiny(1),device='cpu')
    resumed=train('dit3d',static_data,tmp_path/'part',tiny(2),device='cpu',resume=part)
    a=torch.load(whole,weights_only=True);b=torch.load(resumed,weights_only=True)
    assert a['model']==b['model']=='dit3d' and a['epoch']==b['epoch']==2
    assert a['optimizer_steps']==b['optimizer_steps']==4
    for group in ('system','ema_system'):
        for key in a[group]:assert torch.equal(a[group][key],b[group][key]),(group,key)
    assert a['best_validation_loss']==b['best_validation_loss']


def test_fs_dit3d_real_cpu_training_resume_exact_and_no_truth_input(dataset,tmp_path):
    manifest,cfg_path,rows=dataset
    cfg=json.loads(cfg_path.read_text());cfg.update(tiny(2));cfg_path.write_text(json.dumps(cfg))
    a=fs.train(train_args(dataset,tmp_path/'whole',2,model='dit3d'))
    fs.train(train_args(dataset,tmp_path/'part',1,model='dit3d'))
    b=fs.train(train_args(dataset,tmp_path/'part',2,str(tmp_path/'part/latest.pt'),model='dit3d'))
    assert a['model']==b['model']=='dit3d' and a['optimizer_steps']==b['optimizer_steps']==4
    for group in ('system','ema_system'):
        for key in a[group]:assert torch.equal(a[group][key],b[group][key]),(group,key)
    assert a['diffusion_missing_target_policy'].startswith('independent_N01')
    obs=fs.load_observation(rows[-1]);pred,_=fs.reconstruct(obs,'dit3d',a['config'],a)
    with np.load(rows[-1]['path']) as archive:payload={key:archive[key] for key in archive.files}
    payload['flow_field']*=123;payload['support_mask']=~payload['support_mask'];payload['probe_xyz']*=9
    np.savez(rows[-1]['path'],**payload)
    changed,_=fs.reconstruct(fs.load_observation(rows[-1]),'dit3d',a['config'],a)
    assert torch.equal(pred,changed)


def test_fs_real_dit_padding_truth_is_ignored(dataset):
    _,splits=fs.read_manifest(dataset[0]);stats=fs.fit_statistics(splits['train']);cfg=tiny()
    system=fs.TemporalSystem('dit3d',cfg,stats)
    assert isinstance(system.decoder,ConditionalDiT3D)
    obs=fs.load_observation(splits['train'][0]);field,mask,times=fs.load_reference(splits['train'][0])
    condition=system.condition(obs,torch.tensor([0]))
    assert not torch.equal(condition,system.condition(obs,torch.tensor([1])))
    target=fs.normalize(field[:1],stats);changed=target.clone();changed[~mask[:1,None].expand_as(changed)]=torch.nan
    diffusion=fs.make_diffusion(cfg,'cpu')
    first=fs.partial_diffusion_loss(system,diffusion,target,mask[:1],condition,generator=torch.Generator().manual_seed(8))
    second=fs.partial_diffusion_loss(system,diffusion,changed,mask[:1],condition,generator=torch.Generator().manual_seed(8))
    assert torch.equal(first,second) and torch.isfinite(second)
    second.backward();assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
