#!/usr/bin/env python3
"""Read-only real-data compatibility and vendored-source checks; no model run."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from flow_observation.dps.data import load_observation_case,sha256,write_json
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.sparse_dataset import load_sparse_flow_manifest
from flow_baselines.common import check_rectangular_geometry


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/'PROVENANCE.json').read_text())
    for name,digest in provenance['copied_files'].items():
        if sha256(root/name)!=digest:
            raise ValueError('Vendored source changed: '+name)
    _,splits=load_sparse_flow_manifest(args.manifest)
    rows=[]
    for n in (2,4,6,12,24,48,96):
        for views in (None,[0]):
            inputs=load_observation_case(splits['validation'][13], 'validation', n,num_probes=16,views=views)
            check_rectangular_geometry(inputs['solid_mask'])
            likelihood=ProjectedTrajectoryLikelihood(inputs['tracks'],inputs['projection'],inputs['times'],
                inputs['bounds'],sigma=.05,mask=inputs['mask'])
            rows.append({'N':n,'views':inputs['view_indices'],'shape':list(inputs['tracks'].shape),
                'spatial_shape':list(inputs['solid_mask'].shape), 'rectangular_geometry_checked':True,
                'unknown_depth_count':int(likelihood.depth_active.sum())})
    write_json(args.output,{'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':sha256(args.manifest),'split_counts':{k:len(v) for k,v in splits.items()},
        'vendored_files_unchanged':len(provenance['copied_files']),
        'model_inference_invoked':False,'training_invoked':False,'observation_checks':rows})
    print('Passed: 14 real-data observation geometries; no model inference or training.')
