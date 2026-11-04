import json
import numpy as np
import pytest
import torch
from flow_baselines.config import resolve_config
from flow_baselines.training import train
from flow_baselines.inference import predict
from flow_observation.sparse_dataset import load_sparse_flow_manifest


def tiny(**changes):
    cfg=resolve_config(overrides={'epochs':1,'amp':False,'condition_dim':8,'condition_flow_grid':2,
        'diffusion_steps':8,'sampling_steps':4,'num_samples':2,'num_probes':1,
        'fit_steps':3,'pod':{'rank':2},
        'unet':{'base_channels':4,'channel_multipliers':[1,2]},
        'unet_diffusion':{'base_channels':4,'channel_multipliers':[1,2],'time_embedding_dim':8},
        'fno':{'width':4,'depth':1,'modes':2,'padding':1},
        'cnn':{'width':4,'depth':1},'deeponet':{'hidden':8,'latent':8,'layers':1},
        'rbf':{'centers_per_axis':2},
        'pinn':{'hidden':8,'layers':2,'collocation_points':8,'viscosity':.01,'fit_steps':3}})
    cfg.update(changes)
    return cfg


@pytest.mark.parametrize('kind',['mean','pod','unet','fno','cnn','deeponet','unet_diffusion','rbf','pinn'])
def test_nine_models_end_to_end(kind,small_manifest,tmp_path):
    cfg=tiny()
    checkpoint=None if kind in ('rbf','pinn') else train(kind,small_manifest,tmp_path/kind,cfg,device='cpu')
    path=predict(small_manifest,tmp_path/f'{kind}.npz',kind=kind,checkpoint=checkpoint,
        config=cfg,device='cpu',evaluate=True)
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        assert np.isfinite(data['predictions']).all()
        assert data['predictions'].shape==(2 if kind=='unet_diffusion' else 1,3,8,8,8)
        assert ('sample_variance' in data)==(kind=='unet_diffusion')
        assert metadata['reference_used_during_inference'] is False
        assert metadata['evaluation']['per_sample_field_relative_l2_unknown_interior']
        assert metadata['evaluation_protocol']['extra_probe_information_used_for_generation'] is False
        assert metadata['checkpoint_epoch'] == (None if checkpoint is None else 0 if kind in ('mean','pod') else cfg['epochs'])
        assert metadata['checkpoint_selection'] == (None if checkpoint is None else 'statistical_fit' if kind in ('mean','pod') else 'final_epoch')
        assert metadata['training_boundary_projection'] == (cfg['boundary_projection'] if kind in ('unet','fno','cnn','deeponet','unet_diffusion') else None)
        assert metadata['fixed_field_depth_fit'] == (None if kind in ('pod','rbf','pinn') else {
            'max_steps':100,'learning_rate':cfg['depth_learning_rate'],'seed':cfg['sampling_seed']})
    from flow_baselines.reporting import verify
    assert verify(path)['status']=='verified'
    if checkpoint:
        assert verify(checkpoint.parent)['status']=='verified'


@pytest.mark.parametrize('kind',['mean','pod','cnn','rbf','pinn'])
def test_changing_reference_and_hidden_truth_cannot_change_generation(kind,small_manifest,tmp_path):
    cfg=tiny(views=[0])
    checkpoint=None if kind in ('rbf','pinn') else train(kind,small_manifest,tmp_path/kind,cfg,device='cpu')
    predictions=[]
    _,splits=load_sparse_flow_manifest(small_manifest)
    case=splits['validation'][0]
    for index in range(2):
        path=predict(small_manifest,tmp_path/f'{kind}_{index}.npz',kind=kind,checkpoint=checkpoint,
            config=cfg,device='cpu')
        with np.load(path) as data:
            predictions.append(data['predictions'].copy())
        if index==0:
            with np.load(case.path,allow_pickle=False) as data:
                values={k:data[k] for k in data.files if k not in ('initial_hidden_depth','trajectories_3d')}
            values['flow_field']=np.full_like(values['flow_field'],9999.)
            values['initial_hidden_depth']=np.array([object(),object()],dtype=object)
            values['trajectories_3d']=np.array([object(),object()],dtype=object)
            np.savez(case.path,**values)
    assert np.array_equal(*predictions)


@pytest.mark.parametrize('kind',['cnn','unet_diffusion'])
def test_resume_is_exact(kind,small_manifest,tmp_path):
    cfg=tiny(epochs=2)
    complete=train(kind,small_manifest,tmp_path/'complete',cfg,device='cpu')
    partial=train(kind,small_manifest,tmp_path/'partial',tiny(),device='cpu')
    resumed=train(kind,small_manifest,tmp_path/'partial',cfg,device='cpu',resume=partial)
    a=torch.load(complete,weights_only=True)
    b=torch.load(resumed,weights_only=True)
    for key in ('system','ema_system'):
        for name in a[key]:
            assert torch.equal(a[key][name],b[key][name]),(key,name)
    assert a['best_validation_loss']==b['best_validation_loss']
    with pytest.raises(ValueError,match='configuration'):
        train(kind,small_manifest,tmp_path/'partial',tiny(epochs=3,learning_rate=.1),device='cpu',resume=resumed)


def test_mean_uses_only_train(small_manifest,tmp_path):
    ckpt=train('mean',small_manifest,tmp_path/'mean',tiny(),device='cpu')
    saved=torch.load(ckpt,weights_only=True)
    _,splits=load_sparse_flow_manifest(small_manifest)
    expected=np.mean([np.load(c.path)['flow_field'] for c in splits['train']],axis=0)
    assert np.allclose(saved['mean'].numpy(),expected)
    assert saved['train_case_ids']==[c.case_id for c in splits['train']]


def test_pinn_requires_known_viscosity_and_consistent_field(small_manifest,tmp_path):
    cfg=tiny(); cfg['pinn']['viscosity']=None
    with pytest.raises(ValueError,match='viscosity'):
        predict(small_manifest,tmp_path/'missing.npz',kind='pinn',config=cfg,device='cpu')
    cfg=tiny(boundary_projection='known')
    with pytest.raises(ValueError,match='NS residual'):
        predict(small_manifest,tmp_path/'boundary.npz',kind='pinn',config=cfg,device='cpu')


def test_configuration_rejects_invalid_settings():
    with pytest.raises(ValueError,match='Unknown'):
        resolve_config(overrides={'learning_rtae':.001})
    with pytest.raises(ValueError,match='positive'):
        resolve_config(overrides={'fit_steps':0})
    with pytest.raises(ValueError,match='Unknown'):
        resolve_config(overrides={'fno':{'modez':4}})


def test_cli_train_predict_verify(small_manifest,tmp_path):
    from flow_baselines.cli import main
    config=tmp_path/'tiny.json'; config.write_text(json.dumps(tiny()))
    directory=tmp_path/'cli_mean'
    checkpoint=main(['train','--model','mean','--manifest',str(small_manifest),
        '--output-dir',str(directory),'--config',str(config),'--device','cpu'])
    main(['verify',str(directory)])
    output=tmp_path/'cli_mean.npz'
    main(['predict','--checkpoint',str(checkpoint),'--manifest',str(small_manifest),
        '--output',str(output),'--device','cpu','--evaluate'])
    main(['verify',str(output)])
    with pytest.raises(ValueError,match='exists'):
        main(['predict','--checkpoint',str(checkpoint),'--manifest',str(small_manifest),
            '--output',str(output),'--device','cpu'])


def test_cli_pinn_explicit_viscosity_and_steps(small_manifest,tmp_path):
    from flow_baselines.cli import main
    config=tmp_path/'tiny_pinn.json'; config.write_text(json.dumps(tiny()))
    output=tmp_path/'cli_pinn.npz'
    main(['predict','--model','pinn','--manifest',str(small_manifest),
        '--output',str(output),'--device','cpu','--config',str(config),
        '--viscosity','.03','--fit-steps','1'])
    row=json.loads(output.with_suffix('.json').read_text())
    assert row['physical_information']['viscosity']==.03
    assert row['fitting'][0]['steps_executed']==1
