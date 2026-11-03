"""One-time extraction of established FS-SPH physical equations."""
import ast
from pathlib import Path

here = Path(__file__).resolve().parent
source = Path('/u/zguan2/projects/flow3d_runs/experiments/baselines_n12_20261027/launch/fs_baseline.py')
text = source.read_text()
names = {'read_manifest', 'metadata', 'frame_weights', 'load_observation',
         'load_reference', 'normalize', 'denormalize', 'masked_mse',
         'nuisance_score_target', 'partial_score_loss', 'TransientLikelihood', 'evaluate_field'}
header = '''"""Frozen FS physical adapter; no baseline model or trajectory encoder."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.advection import advect_particles
from flow_observation.interpolation import sample_velocity

'''
parts = [ast.get_source_segment(text, node) for node in ast.parse(text).body
         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
assert len(parts) == len(names)
(here / 'physics.py').write_text(header + '\n\n'.join(parts) + '\n')
sampler = here.parent / 'fs_source/src/flow_observation/dps/sampler.py'
text = sampler.read_text().replace('shape[:2] != (1, 3)', 'shape[1] != 3')
text = text.replace('DPS currently reconstructs one volume [1,3,D,H,W]',
                    'Transient DPS reconstructs native-time volumes [T,3,D,H,W]')
text = text.replace('time = torch.full((1,), timestep', 'time = torch.full((shape[0],), timestep')
sampler.write_text(text)
print('Prepared frozen FS physical adapter and independent-frame sampler')
