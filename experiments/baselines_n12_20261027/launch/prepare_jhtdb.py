#!/usr/bin/env python3
"""Create frozen-snapshot JHTDB baseline cases; full preparation requires Slurm/CUDA.

Raw DNS archives remain immutable. All observations use canonical cell coordinates;
physical time stays in DNS units. Other stored views/XYZ serve post-hoc evaluation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

SOURCE_ROOT = Path('/work/hdd/biup/zguan2/datasets/jhtdb_iso1024_native32_3125_t1_20261024')
OUTPUT_ROOT = Path('/work/hdd/biup/zguan2/datasets/experiment_adapters_20261027/jhtdb_frozen32')
SOURCE_MANIFEST_SHA256 = '2a453d05adace4e22dd7c6301f83bab9691a004e3f387a24b7532b9f10da8704'
HORIZON = 0.005216254525311773
PROJECTION = np.asarray([[[1,0,0],[0,0,1]], [[1,0,0],[0,1,0]], [[0,1,0],[0,0,1]]], dtype=np.float32)
COUNTS = {'train': 2500, 'validation': 312, 'test': 313}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('\x1f'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2**63-1)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def sample_velocity(field, positions):
    """Trilinear interpolation on XYZ cell coordinates without periodic wrapping."""
    side = torch.as_tensor(field.shape[-3:][::-1], dtype=positions.dtype, device=positions.device)-1
    grid = (positions / side * 2-1).reshape(1, -1, 1, 1, 3)
    sampled = F.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled[0, :, :, 0, 0].T


def integrate(field, initial, times, substeps=4):
    """Vectorized RK4 with persistent termination masks, including stage exits.

    Invalid paths retain the last finite valid position as storage padding.
    They never re-enter the valid set and no clamped position is valid motion.
    """
    require(field.ndim == 5 and field.shape[:2] == (1,3), 'field must be [1,3,Z,Y,X]')
    require(substeps >= 1 and len(times) >= 2, 'invalid integration schedule')
    side = torch.as_tensor(field.shape[-3:][::-1], dtype=initial.dtype, device=initial.device)-1
    def inside(points):
        return torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= side).all(-1)
    position = initial.clone()
    alive = inside(position)
    history, masks = [position.clone()], [alive.clone()]
    for index in range(1, len(times)):
        dt = (times[index]-times[index-1])/substeps
        for _ in range(substeps):
            k1 = sample_velocity(field, position)
            p2 = position+dt*k1/2
            k2 = sample_velocity(field, p2)
            p3 = position+dt*k2/2
            k3 = sample_velocity(field, p3)
            p4 = position+dt*k3
            k4 = sample_velocity(field, p4)
            proposal = position+dt*(k1+2*k2+2*k3+k4)/6
            alive = alive & inside(p2) & inside(p3) & inside(p4) & inside(proposal)
            position = torch.where(alive[:,None], proposal, position)
        history.append(position.clone())
        masks.append(alive.clone())
    return torch.stack(history, dim=1), torch.stack(masks, dim=1)


def build_case(flow_physical, spacing, case_id, *, device='cpu', substeps=4):
    """Pure case adapter usable by bounded CPU tests and scheduled CUDA prep."""
    raw = np.asarray(flow_physical)
    require(raw.shape == (1,3,32,32,32) and np.isfinite(raw).all(), 'invalid native JHTDB velocity shape/values')
    require(math.isfinite(spacing) and spacing > 0, 'spacing must be positive')
    canonical = np.asarray(raw/spacing, dtype=np.float32)
    generator = np.random.default_rng(stable_seed(47, 'jhtdb_uniform_initial', case_id))
    initial = generator.uniform(4.0, 27.0, size=(160,3)).astype(np.float32)
    field = torch.as_tensor(canonical, device=device)
    times = np.linspace(0, HORIZON, 20, dtype=np.float64)
    with torch.no_grad():
        tracks, valid = integrate(field, torch.as_tensor(initial, device=device), torch.as_tensor(times, device=device), substeps)
    tracks, valid = tracks.cpu().numpy(), valid.cpu().numpy()
    projected = np.einsum('ntc,voc->vnto', tracks, PROJECTION, optimize=True)[None]
    return {
        'flow_field': canonical, 'trajectories_3d': tracks[None],
        'trajectories_2d': projected.astype(np.float32),
        'observation_mask': np.broadcast_to(valid, (1,3,160,20)).copy(),
        'observation_mask_3d': valid[None], 'observation_times': times,
        'projection_matrix': PROJECTION.copy(), 'particle_counts': np.asarray([160],dtype=np.int64),
        'particle_ids': np.arange(160,dtype=np.int64),
        'domain_bounds': np.asarray([[0,31]]*3,dtype=np.float64),
        'solid_mask': np.zeros((32,32,32),dtype=bool),
    }


def validate_case(path):
    with np.load(path, allow_pickle=False) as archive:
        for key, shape in {'flow_field':(1,3,32,32,32), 'trajectories_3d':(1,160,20,3),
                           'trajectories_2d':(1,3,160,20,2), 'observation_mask':(1,3,160,20),
                           'observation_mask_3d':(1,160,20), 'projection_matrix':(3,2,3)}.items():
            require(archive[key].shape == shape, f'{path}: invalid {key} shape')
        xyz, valid = archive['trajectories_3d'], archive['observation_mask_3d']
        require(valid.dtype == np.bool_ and not np.any(np.diff(valid.astype(np.int8),axis=-1)>0), 'invalid termination mask')
        require(np.isfinite(xyz).all() and np.isfinite(archive['flow_field']).all(), 'nonfinite archive')
        require(np.all((xyz[valid] >= 0) & (xyz[valid] <= 31)), 'valid trajectory leaves crop')
        require(np.allclose(archive['trajectories_2d'][0], np.einsum('ntc,voc->vnto',xyz[0],PROJECTION),atol=1e-6), 'projection mismatch')
        require(not archive['solid_mask'].any(), 'JHTDB has no prescribed walls')
        require(np.allclose(archive['observation_times'], np.linspace(0,HORIZON,20),rtol=0,atol=1e-14), 'time mismatch')
        require(np.array_equal(archive['observation_mask'],np.broadcast_to(valid[:,None],(1,3,160,20))), 'view masks differ')
        meta = json.loads(str(archive['metadata'].item()))
        require('lid_face' not in meta and 'lid_velocity_xyz' not in meta, 'fabricated lid conditions')
    return digest(path)


def preparation_config(source):
    manifest_path = source/'manifest.json'
    require(digest(manifest_path) == SOURCE_MANIFEST_SHA256, 'source manifest differs from pinned release')
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((source/'COMPLETE.json').read_text())
    require(complete['status']=='completed' and complete['verified_fields']==3125, 'source release incomplete')
    speed = max(case['statistics']['speed_max'] for case in manifest['splits']['train'])
    spacing = 2*math.pi/1024
    require(math.isclose(HORIZON,0.75*4*spacing/speed,rel_tol=1e-12), 'train-only horizon no longer matches')
    config = {'schema':'jhtdb_frozen32_adapter_v1','source_root':str(source.resolve()),
              'source_manifest_sha256':SOURCE_MANIFEST_SHA256, 'adapter_sha256':digest(__file__),
              'spacing':spacing,'horizon':HORIZON,'horizon_fit_split':'train','training_speed_bound':speed,
              'seed':47,'particle_count':160,'frames':20,'integrator':'rk4','substeps':4,
              'margin_cells':4,'local_boundary':'terminate','input_view':'xz',
              'other_stored_views':'posthoc evaluation only','coordinates':'cell XYZ [0,31], velocity native/dx, native time'}
    return config, manifest


def save_case(path, payload):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,**payload)
    os.replace(temporary,path)


def run(args):
    source, output = args.source_root.resolve(), args.output_root.resolve()
    require(not output.is_relative_to(source), 'derived output may not be placed inside immutable raw release')
    if not args.smoke_case:
        require(args.device=='cuda' and bool(os.environ.get('SLURM_JOB_ID')), 'full preparation requires Slurm and --device cuda')
        require(torch.cuda.is_available(), 'CUDA unavailable inside allocation')
    config, source_manifest = preparation_config(source)
    flat = [(name,case) for old,name in [('train','train'),('val','validation'),('test','test')]
            for case in source_manifest['splits'][old]]
    if args.smoke_case:
        flat = [(split,case) for split,case in flat if case['case_id']==args.smoke_case]
        require(len(flat)==1,'unknown smoke case')
    else:
        require({name:sum(split==name for split,_ in flat) for name in COUNTS} == COUNTS,'raw split counts differ')
    config['smoke_case'] = args.smoke_case
    config_digest = hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/('.'+output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():
            existing = json.loads((output/'COMPLETE.json').read_text()) if (output/'COMPLETE.json').exists() else {}
            require(args.resume and existing.get('configuration_sha256')==config_digest,'output exists or differs; choose new output')
            require(existing.get('status')=='completed' and existing.get('split_counts')==COUNTS, 'completed publication counts differ')
            require(digest(output/'manifest.json')==existing['manifest_sha256'] and digest(output/'SHA256SUMS')==existing['checksum_list_sha256'], 'completed publication index changed')
            published=json.loads((output/'manifest.json').read_text())
            for split in COUNTS:
                require(len(published['splits'][split])==COUNTS[split], 'completed manifest counts differ')
                for case in published['splits'][split]:
                    require(validate_case(output/case['path'])==case['sha256'], 'completed archive changed')
            print(json.dumps({'status':'already_completed','output_root':str(output)}),flush=True)
            return
        stage = output.parent/('.'+output.name+'.staging')
        if stage.exists():
            require(args.resume and json.loads((stage/'PREPARATION.json').read_text())==config,'staging exists; explicit matching --resume required')
        else:
            stage.mkdir()
            (stage/'fields').mkdir()
            write_json(stage/'PREPARATION.json',config)
        splits = {name:[] for name in COUNTS}
        checksums = []
        preflight = None
        for index,(split,case) in enumerate(flat):
            path = stage/'fields'/(case['case_id']+'.npz')
            if not path.exists():
                rawpath = source/case['file']
                require(digest(rawpath)==case['sha256'],f"raw field changed: {case['case_id']}")
                with np.load(rawpath,allow_pickle=False) as archive:
                    raw = archive['flow_field'].copy()
                    bounds = archive['domain_bounds'].copy()
                payload = build_case(raw,config['spacing'],case['case_id'],device=args.device)
                if preflight is None and split=='train':
                    refined = build_case(raw,config['spacing'],case['case_id'],device=args.device,substeps=8)
                    valid = payload['observation_mask_3d'] & refined['observation_mask_3d']
                    error = float(np.max(np.abs(payload['trajectories_3d'][valid]-refined['trajectories_3d'][valid])))
                    require(error <= 0.002,'RK4 convergence preflight failed')
                    preflight = {'case_id':case['case_id'],'rk4_4_vs_8_max_abs_cells':error,
                                 'valid_fraction':float(payload['observation_mask_3d'].mean()),'test_used':False}
                    write_json(stage/'preflight.json',preflight)
                metadata = {'case_id':case['case_id'],'flow_group_id':case['macroblock_id'],
                            'source_flow_group_id':case['flow_group_id'],'macroblock_id':case['macroblock_id'],
                            'source_simulation':'isotropic1024coarse','source_dns_time':1.0,
                            'scope':'spatial blocked holdout within one frozen DNS snapshot, not independent DNS runs',
                            'synthetic_trajectories':True,'frozen_snapshot':True,'source_sha256':case['sha256'],
                            'source_bounds_xyz':bounds.tolist(),'physical_to_cell_scale_xyz':[1/config['spacing']]*3,
                            'velocity_to_physical_scale':config['spacing'],'configuration_sha256':config_digest,
                            'termination_padding':'last valid position, masked false after exit',
                            'input_views':['xz'],'posthoc_only_views':['xy','yz']}
                payload['metadata']=np.asarray(json.dumps(metadata,allow_nan=False))
                save_case(path,payload)
            casehash = validate_case(path)
            with np.load(path,allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata'].item()))
            require(metadata['configuration_sha256']==config_digest and metadata['source_sha256']==case['sha256'],'resumed case provenance differs')
            relative = str(path.relative_to(stage))
            splits[split].append({'case_id':case['case_id'],'path':relative,'flow_group_id':case['macroblock_id'],
                                 'source_flow_group_id':case['flow_group_id'],'source_case_id':case['case_id'],
                                 'macroblock_id':case['macroblock_id'],'sha256':casehash})
            checksums.append(f'{casehash}  {relative}\n')
            if index%25==0:
                print(json.dumps({'status':'preparing','completed':index+1,'total':len(flat)}),flush=True)
        if args.smoke_case:
            write_json(stage/'SMOKE_ONLY.json',{'scientific_result':False,'case_id':args.smoke_case,'configuration_sha256':config_digest})
            os.replace(stage,output)
            return
        manifest = {'format_version':1,'mode':'scientific','dataset_id':'jhtdb_frozen32_20261027',
                    'scope':'spatial blocked holdout from one DNS snapshot, not temporal/cross-simulation generalization',
                    'grouping':'original macroblock IDs; preserve source tile IDs separately',
                    'configuration_sha256':config_digest,'source_manifest_sha256':SOURCE_MANIFEST_SHA256,'splits':splits}
        write_json(stage/'manifest.json',manifest)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/src'))
        from flow_observation.sparse_dataset import load_sparse_flow_manifest
        load_sparse_flow_manifest(stage/'manifest.json')
        (stage/'SHA256SUMS').write_text(''.join(checksums))
        write_json(stage/'COMPLETE.json',{'status':'completed','verified_fields':len(flat),'split_counts':COUNTS,
                   'configuration_sha256':config_digest,'manifest_sha256':digest(stage/'manifest.json'),
                   'checksum_list_sha256':digest(stage/'SHA256SUMS'),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                   'device':args.device,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
                   'source_data_mutated':False})
        os.replace(stage,output)
        print(json.dumps({'status':'completed','output_root':str(output),'cases':len(flat)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    parser.add_argument('--output-root',type=Path,default=OUTPUT_ROOT)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-case',help='One raw crop only; output explicitly non-scientific and never COMPLETE')
    run(parser.parse_args())


if __name__=='__main__':
    main()
