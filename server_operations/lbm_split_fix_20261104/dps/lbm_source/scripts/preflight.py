#!/usr/bin/env python3
"""Read-only real-data / provenance checks; no model or training is run."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flow_observation.dps.data import load_observation_case, read_prior_field, sha256, write_json
from flow_observation.dps.observations import FullTrajectoryLikelihood, ProjectedTrajectoryLikelihood
from flow_observation.sparse_dataset import load_sparse_flow_manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, default=13)
    parser.add_argument('--matched-capture', type=Path)
    parser.add_argument('--check-full3d', action='store_true', help='Also explicitly read stored XYZ trajectories as observations')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root/'BASELINE_PROVENANCE.json').read_text())
    for relative, expected in provenance['baseline_files'].items():
        if sha256(root/relative) != expected:
            raise ValueError('Copied baseline source changed: ' + relative)
    manifest, cases = load_sparse_flow_manifest(args.manifest)
    field, solid = read_prior_field(cases['train'][0].path)
    if not 0 <= args.case_index < len(cases['validation']):
        raise ValueError('Validation case index is out of range')
    case = cases['validation'][args.case_index]
    geometry = []
    for n in (2,4,6,12,24,48,96):
        for views in (None, [0]):
            inputs = load_observation_case(case, 'validation', n, num_probes=16, views=views)
            observation = ProjectedTrajectoryLikelihood(inputs['tracks'], inputs['projection'], inputs['times'], inputs['bounds'],
                sigma=.05, mask=inputs['mask'])
            geometry.append({'N': n, 'views': inputs['view_indices'],
                'track_shape': list(inputs['tracks'].shape), 'unknown_depth_count': int(observation.depth_active.sum())})
    if args.check_full3d:
        for n in (2, 96):
            inputs = load_observation_case(case, 'validation', n, num_probes=16, observation_type='full3d')
            FullTrajectoryLikelihood(inputs['tracks'], inputs['times'], inputs['bounds'], sigma=.05, mask=inputs['mask'])
            geometry.append({'N': n, 'observation_type': 'full3d', 'track_shape': list(inputs['tracks'].shape), 'unknown_depth_count': 0})
    matched = 'not_requested'
    if args.matched_capture:
        inputs = load_observation_case(case, 'validation', 2, num_probes=16)
        with np.load(args.matched_capture, allow_pickle=False) as archive:
            for key in ('tracks','mask','bounds','times'):
                if not np.array_equal(inputs[key].numpy(), archive[key]):
                    raise ValueError('Input differs from completed conditional evaluation: ' + key)
        matched = 'exact_tracks_mask_bounds_times'
    report = {'status':'passed', 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_invoked':False, 'model_inference_invoked':False,
        'manifest_sha256':sha256(args.manifest), 'split_counts':{k:len(v) for k,v in cases.items()},
        'prior_field_shape':list(field.shape), 'case_id':case.case_id,
        'baseline_source_hashes_verified':len(provenance['baseline_files']),
        'full3d_observations_checked':args.check_full3d,
        'existing_conditional_inputs_match':matched, 'observation_geometry':geometry}
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k != 'observation_geometry'},indent=2))
