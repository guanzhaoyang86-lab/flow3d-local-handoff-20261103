import importlib.util
from pathlib import Path
import subprocess
import pytest
R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('lbm',R/'run_lbm.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_config():
    a=R/'configs/lbm_s31.json'; b=m.MODEL_ROOT/'configs/jhtdb_s31.json'
    assert a.read_bytes()==b.read_bytes()
    assert m.read(a)['boundary_projection']=='none'
    c=m.command('lbm',Path('/manifest'),Path('/out'),100,True)
    assert str(m.MODEL_ROOT/'source/scripts/run_baseline.py') in c
    assert c[-2:]==['--resume','/out/latest.pt']
@pytest.mark.parametrize('start,fail,want',[(None,False,[1,100]),(7,False,[100]),(100,False,[]),(None,True,[1])])
def test_resume(tmp_path,monkeypatch,start,fail,want):
    a=m.allocation_controller(); a.RESULT_ROOT=tmp_path
    out=tmp_path/'lbm_s31'; out.mkdir()
    if start is not None: (out/'latest.pt').touch()
    monkeypatch.setenv('SLURM_JOB_ID','mock')
    monkeypatch.setattr(a,'source_check',lambda **k:{})
    monkeypatch.setattr(a,'data_check',lambda d:dict(manifest='fixture',manifest_sha256='fixture'))
    monkeypatch.setattr(a,'gpu_check',lambda:{})
    state=[start]; calls=[]
    def run(c,check):
        e=int(c[c.index('--epochs')+1]); calls.append(e)
        if e==100: assert '--resume' in c
        if fail: raise subprocess.CalledProcessError(1,c)
        state[0]=e
    monkeypatch.setattr(a.subprocess,'run',run)
    monkeypatch.setattr(a,'checkpoint_check',lambda *p,**k:dict(epoch=state[0]))
    if fail:
        with pytest.raises(subprocess.CalledProcessError): a.run('lbm')
        assert m.read(out/'allocation_status.json')['status']=='failed'
        assert not (out/'PILOT_COMPLETE.json').exists()
    else:
        a.run('lbm')
        assert m.read(out/'allocation_status.json')['status']=='completed'
    assert calls==want
