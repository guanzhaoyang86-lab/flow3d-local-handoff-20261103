"""Read-only release regression tests; no archive copying or generation."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('revised_lbm_prepare', ROOT / 'prepare.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class CompletedReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marker, cls.manifest, cls.checksums = adapter.source_ready()

    def test_real_release_matches_counts_identity_and_metadata_hashes(self):
        self.assertEqual(self.marker['status'], 'completed')
        self.assertEqual(self.marker['split_counts'], {'train': 2499, 'validation': 313, 'test': 313})
        self.assertEqual({k: len(v) for k, v in self.manifest['splits'].items()}, adapter.COUNTS)
        self.assertEqual(self.marker['dataset_version'], adapter.SOURCE_DATASET_VERSION)
        self.assertEqual(self.checksums['manifest.json'], adapter.SOURCE_MANIFEST_SHA256)
        self.assertEqual(sum(k.endswith('.npz') for k in self.checksums), 3125)

    def test_original_split_algorithm_reproduces_completed_counts(self):
        path = Path('/u/zguan2/projects/flow3d_runs/lbm_generation_500k/source/scripts/build_sparse_flow_manifest.py')
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_split_counts')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(namespace['_split_counts'](3125, .8, .1), (2499, 313, 313))

    def test_previous_wrong_counts_and_partial_completion_rejected(self):
        original_read = adapter.read
        variants = [('split_counts', {'train': 2500, 'validation': 312, 'test': 313}),
                    ('status', 'running'), ('physical_trajectories', 160000), ('num_cases', 1000)]
        for key, value in variants:
            with self.subTest(key=key):
                marker = copy.deepcopy(self.marker); marker[key] = value
                def read(path):
                    return marker if Path(path) == adapter.SOURCE / 'COMPLETE.json' else original_read(path)
                with mock.patch.object(adapter, 'read', side_effect=read):
                    with self.assertRaises(ValueError):
                        adapter.source_ready()

    def test_other_completed_release_rejected(self):
        with mock.patch.object(adapter, 'SOURCE_DATASET_VERSION', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'reviewed completed dataset'):
                adapter.source_ready()

    def test_checksum_paths_and_duplicate_entries_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'SHA256SUMS'
            for text in [('a' * 64) + '  ../outside.npz\n',
                         (('a' * 64) + '  case_00000.npz\n') * 2]:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    adapter.parse_checksums(path)

    def test_one_existing_archive_matches_unchanged_cavity_contract(self):
        row = next(row for rows in self.manifest['splits'].values() for row in rows
                   if row['case_id'] == 'case_00000')
        adapter.check_archive(adapter.SOURCE / row['path'], row)


class CompletedOutputIsNotTrustedTests(unittest.TestCase):
    def test_existing_complete_does_not_skip_archive_reverification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'; source.mkdir()
            output = root / 'output'; (output / 'fields').mkdir(parents=True)
            raw = source / 'case_00000.npz'; raw.write_bytes(b'fixture original bytes')
            (output / 'fields' / raw.name).write_bytes(b'corrupted existing output')
            marker = {'dataset_version': 'fixture-version'}
            adapter.write(output / 'COMPLETE.json', {'source_dataset_version': marker['dataset_version']})
            row = {'case_id': 'case_00000', 'path': raw.name, 'flow_group_id': 'fixture-group'}
            manifest = {'splits': {'train': [row], 'validation': [], 'test': []}}
            with mock.patch.dict(adapter.os.environ, {'SLURM_JOB_ID': 'cpu-regression'}), \
                    mock.patch.object(adapter, 'source_ready', return_value=(marker, manifest, {raw.name: adapter.sha(raw)})), \
                    mock.patch.object(adapter, 'check_archive') as archive_check:
                with self.assertRaisesRegex(ValueError, 'Existing adapted archive differs'):
                    adapter.prepare(source, output)
                archive_check.assert_called_once_with(raw, row)
            self.assertEqual(raw.read_bytes(), b'fixture original bytes')
            self.assertEqual((output / 'fields' / raw.name).read_bytes(), b'corrupted existing output')


if __name__ == '__main__':
    unittest.main(verbosity=2)
