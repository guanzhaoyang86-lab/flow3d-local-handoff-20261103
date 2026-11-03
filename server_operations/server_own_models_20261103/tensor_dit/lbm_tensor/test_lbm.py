"""CPU interface/gate regressions; no training, Tucker fitting, or GPU work."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'launch'))
import tensor_data as data
import tensor_model as model
import runtime

torch.set_num_threads(1)
RAW = Path('/work/hdd/biup/zguan2/datasets/lbm32_xz_500k_deltaai_arm_20261023')
ROW = {'path': str(RAW / 'case_00000.npz'), 'case_id': 'case_00000',
       'flow_group_id': 'flow_a2c2c89b9374', '_split': 'validation'}


class ObservationOnlyArchive:
    def __init__(self, archive):
        self.archive = archive

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()

    def __getitem__(self, key):
        if key in {'flow_field', 'trajectories_3d', 'initial_hidden_depth', 'density_field'}:
            raise AssertionError('Forbidden inference reference read: ' + key)
        if key == 'metadata':
            info = data.decode_metadata(self.archive)
            info['lid_velocity_xyz'] = [99999, -99999, 99999]
            return np.array(json.dumps(info))
        return self.archive[key]

    def get(self, key, default=None):
        return self[key] if key in self.archive else default


class LBMInterfaceTests(unittest.TestCase):
    def test_single_seed_static_configuration(self):
        cfg = model.resolve_config('lbm', ROOT / 'configs/lbm_s31.json')
        self.assertEqual((cfg['seed'], cfg['epochs'], cfg['num_particles']), (31, 100, 12))
        self.assertFalse(cfg['physical_time_adapter'])
        self.assertEqual(cfg['decoded_field_weight'], .1)

    def test_observation_has_no_reference_or_lid_condition(self):
        expected = data.load_observation(ROW, 'lbm')
        original_load = np.load
        with mock.patch.object(data.np, 'load', side_effect=lambda *a, **k: ObservationOnlyArchive(original_load(*a, **k))):
            actual = data.load_observation(ROW, 'lbm')
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            else:
                self.assertEqual(actual[key], expected[key], key)
        self.assertEqual(actual['tracks'].shape, (1, 12, 20, 2))
        self.assertTrue(torch.equal(actual['unit_scale'], torch.ones(3)))
        self.assertEqual(len(actual['probe_particle_indices']), 32)
        self.assertFalse(set(actual['particle_indices'].tolist()) & set(actual['probe_particle_indices'].tolist()))

    def test_real_support_and_masked_field_metrics(self):
        obs = data.load_observation(ROW, 'lbm')
        field, support, times = data.load_reference(ROW, 'lbm')
        with np.load(ROW['path'], allow_pickle=False) as archive:
            solid = torch.from_numpy(archive['solid_mask'].copy()).bool()
        self.assertEqual(field.shape, (1, 3, 32, 32, 32))
        self.assertTrue(torch.equal(support[0], ~solid))
        self.assertEqual(int(solid.sum()), 4868)
        altered = torch.where(support[:, None], field, torch.full_like(field, 999.))
        without_probes = {k: v for k, v in obs.items() if k != 'probe_particle_indices'}
        metrics = model.physical_metrics(altered, ROW, without_probes, 'lbm')
        self.assertEqual(metrics['physical_velocity_relative_l2'], 0.)
        self.assertEqual(metrics['velocity_units'], 'lattice cells per LBM step')
        self.assertFalse(metrics['per_case_lid_velocity_condition'])
        self.assertEqual(metrics['boundary_projection'], 'none')
        self.assertEqual(data.reconstruction_score(altered, field, support, times, obs['unit_scale']), 0.)

    def test_posthoc_probe_mask_without_observation_mask_3d(self):
        obs = data.load_observation(ROW, 'lbm')
        field, _, _ = data.load_reference(ROW, 'lbm')
        metrics = model.physical_metrics(field, ROW, obs, 'lbm')
        self.assertEqual(metrics['physical_velocity_relative_l2'], 0.)
        self.assertTrue(np.isfinite(metrics['heldout_projected_track_vector_rmse_native']))
        self.assertTrue(metrics['probe_initial_xyz_is_posthoc_only'])

    def test_jhtdb_all_fluid_guard_remains(self):
        with self.assertRaisesRegex(ValueError, 'solid/boundary'):
            data.load_observation(ROW, 'jhtdb')


class CompletionGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.derived = self.root / 'derived'; self.derived.mkdir()
        self.raw = self.root / 'raw'; self.raw.mkdir()
        self.spec = copy.deepcopy(json.loads((ROOT / 'plan.json').read_text())['datasets']['lbm'])
        self.spec.update(derived_root=str(self.derived), raw_root=str(self.raw))
        (self.raw / 'SHA256SUMS').write_text('fixture checksum list\n')
        runtime.write(self.raw / 'COMPLETE.json', {'status': 'completed', 'dataset_version': runtime.sha(self.raw / 'SHA256SUMS')})
        self.counts = {'train': 2500, 'validation': 312, 'test': 313}
        runtime.write(self.derived / 'manifest.json', {
            'dataset_id': self.spec['dataset_id'], 'schema': 'flow3d_lbm_frozen_manifest_v1',
            'mode': 'scientific', 'splits': {k: [{}] * n for k, n in self.counts.items()}})
        self.marker = {'status': 'completed', 'dataset_id': self.spec['dataset_id'],
            'manifest_sha256': runtime.sha(self.derived / 'manifest.json'),
            'adapter_sha256': self.spec['adapter_sha256'], 'verified_fields': 3125,
            'physical_trajectories': 500000, 'single_view_trajectories': 500000,
            'split_counts': self.counts, 'source_dataset_version': runtime.sha(self.raw / 'SHA256SUMS'),
            'source_complete_sha256': runtime.sha(self.raw / 'COMPLETE.json')}
        runtime.write(self.derived / 'COMPLETE.json', self.marker)

    def test_bound_full_completion_accepted(self):
        self.assertEqual(runtime.data_ready(self.spec), self.derived / 'manifest.json')

    def test_partial_counts_wrong_adapter_and_incomplete_status_rejected(self):
        for key, value in [('verified_fields', 1000), ('physical_trajectories', 160000),
                           ('adapter_sha256', 'wrong'), ('status', 'validated')]:
            with self.subTest(key=key):
                marker = dict(self.marker); marker[key] = value
                runtime.write(self.derived / 'COMPLETE.json', marker)
                with self.assertRaises(RuntimeError):
                    runtime.data_ready(self.spec)

    def test_manifest_and_raw_provenance_tampering_rejected(self):
        (self.derived / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'manifest'):
            runtime.data_ready(self.spec)
        self.setUp()
        (self.raw / 'SHA256SUMS').write_text('changed\n')
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            runtime.data_ready(self.spec)


if __name__ == '__main__':
    unittest.main(verbosity=2)
