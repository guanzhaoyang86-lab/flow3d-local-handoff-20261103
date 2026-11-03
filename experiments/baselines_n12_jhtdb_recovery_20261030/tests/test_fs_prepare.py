"""Analytic, small-data tests for the FS transient reference preparation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_fs_test_module", ROOT / "launch/prepare_fs.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def analytic_native():
    # The first27 particles populate an exact3³ grid; IDs27/28 are held out,
    # and ID29 is substrate with a deliberately poisonous velocity.
    grid = np.array([[x, y, z] for z in (0, .5, 1) for y in (0, .5, 1) for x in (0, .5, 1)], dtype=np.float64)
    points = np.vstack((grid, [[.4, .4, .4], [.6, .6, .6], [.3, .3, .3]]))
    matrix = np.array([[2, -.2, .3], [.1, 1, -.4], [.5, .2, 1.5]])
    transport = points @ matrix.T + [.7, -.3, .2]
    transport[[27, 29]] = 1e7
    valid = np.ones(30, dtype=np.uint8)
    # An invalid transport velocity retains material geometry but cannot donate.
    valid[1] = 0
    transport[1] = -1e7
    fixed = np.zeros(30, dtype=np.uint8)
    fixed[26] = 4
    jointed = np.zeros(30, dtype=np.uint8)
    jointed[[10, 29]] = 1  # ID10 is an adhered initial rod and must remain a donor.
    return dict(points=points, fields=dict(unique_idx=np.arange(30), transport_velocity=transport,
                transport_velocity_valid=valid, velocity=np.zeros_like(transport), Fixed=fixed, Jointed=jointed)), matrix


class FSPrepareTests(unittest.TestCase):
    def frame(self):
        native, matrix = analytic_native()
        frame = prep.build_frame(native, cohort_ids=np.arange(29), excluded_ids=np.array([27, 28]),
                                 probe_ids=np.array([28]), physical_bounds=np.array([[0., 1.]] * 3), spacing=.5, grid_size=3)
        return native, matrix, frame

    def test_temporal_blocks_preserve_one_group_and_exclude_embargo(self):
        splits = prep.window_rows()
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 126, "validation": 18, "test": 18})
        rows = [row for group in splits.values() for row in group]
        ids = {row["window_id"] for row in rows}
        self.assertEqual(len(ids), 162)
        self.assertFalse(ids & (set(range(126, 135)) | set(range(153, 162))))
        self.assertEqual({row["flow_group_id"] for row in rows}, {prep.GROUP})
        self.assertEqual(splits["validation"][0]["frame_start"], 2700)
        self.assertEqual(splits["test"][-1]["frame_stop_exclusive"], 3600)

    def test_affine_gridding_excludes_heldout_substrate_and_invalid_velocities(self):
        native, matrix, frame = self.frame()
        expected = (native["points"][:27] @ matrix.T + [.7, -.3, .2]) * 2
        actual = frame["flow_field"].transpose(1, 2, 3, 0).reshape(-1, 3)
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        self.assertEqual(frame["diagnostics"]["donors"], 26)
        self.assertTrue(frame["material_mask"][0, 0, 1])
        self.assertTrue(frame["prescribed_mask"][2, 2, 2])
        self.assertFalse(frame["support_mask"][2, 2, 2])
        self.assertTrue(frame["support_mask"][0, 0, 1])
        self.assertFalse((frame["ghost_extension_mask"] & frame["material_mask"]).any())
        self.assertLess(frame["diagnostics"]["native_probe_fit_squared_error"], 1e-12)

    def test_missing_transport_never_falls_back_to_momentum(self):
        native, _ = analytic_native()
        del native["fields"]["transport_velocity"]
        with self.assertRaisesRegex(ValueError, "fallback is forbidden"):
            prep.build_frame(native, cohort_ids=np.arange(29), excluded_ids=[27, 28], probe_ids=[28],
                             physical_bounds=np.array([[0., 1.]] * 3), spacing=.5, grid_size=3)

    def test_canonical_transform_preserves_physical_time_and_anisotropic_units(self):
        bounds = np.array([[1., 3.], [-2., 2.], [10., 11.]])
        scale = prep.canonical_scale(bounds)
        np.testing.assert_allclose(scale, [15.5, 7.75, 31.])
        physical_position = np.array([2., 0., 10.5])
        physical_velocity = np.array([1., -3., .2])
        np.testing.assert_allclose((physical_position - bounds[:, 0]) * scale, [15.5] * 3)
        np.testing.assert_allclose((physical_velocity * scale) / scale, physical_velocity)
        np.testing.assert_allclose(prep.physical_time_weights([0., .1, 1.]), [.05, .5, .45])

    def test_window_schema_single_xz_and_eval_truth_separation(self):
        _, _, frame = self.frame()
        frames = [copy.deepcopy(frame), copy.deepcopy(frame)]
        xyz = np.array([[[.1, 1000., .2]], [[.3, -1000., .4]]])
        metadata = dict(config_sha256="config", flow_group_id=prep.GROUP, case_id="test", window_id=0,
                        raw_sources=[], unit_scale=[2.] * 3, grid_shape_zyx=[3] * 3)
        arrays = prep.assemble_window(frames, xyz, [10., 10.0001], [1, 2], [27], [28], [[0., 1.]] * 3, metadata)
        np.testing.assert_allclose(arrays["trajectories_2d"][0, 0, 0], [[.2, .4], [.6, .8]])
        self.assertEqual(arrays["trajectories_2d"].shape, (1, 1, 1, 2, 2))
        self.assertEqual(arrays["probe_xyz"].shape, (1, 2, 3))
        np.testing.assert_array_equal(arrays["observation_times"], [10., 10.0001])
        self.assertNotIn("trajectories_3d", arrays)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            row = dict(case_id="test", window_id=0, path="sample.npz", frame_start=0, frame_stop_exclusive=2)
            prep.atomic_npz(path, arrays)
            record = prep.validate_window(path, row, "config", grid_size=3, n_observed=1, n_probes=1)
            self.assertEqual(record["sha256"], prep.sha256(path))
            with self.assertRaisesRegex(ValueError, "source/config"):
                prep.validate_window(path, row, "changed", grid_size=3, n_observed=1, n_probes=1)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                prep.atomic_npz(path, arrays)

    def test_source_frame_mutation_rejected_during_resume(self):
        _, _, frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "frame.bin"
            raw.write_bytes(b"raw")
            record = dict(path=str(raw), sha256=prep.sha256(raw), **prep.file_stat(raw))
            metadata = dict(config_sha256="config", flow_group_id=prep.GROUP, case_id="test", window_id=0,
                            raw_sources=[record], unit_scale=[2.] * 3, grid_shape_zyx=[3] * 3)
            arrays = prep.assemble_window([copy.deepcopy(frame)] * 2, np.zeros((2, 1, 3)), [1., 2.], [1, 2],
                                         [27], [28], [[0., 1.]] * 3, metadata)
            path = Path(directory) / "sample.npz"
            prep.atomic_npz(path, arrays)
            raw.write_bytes(b"changed")
            row = dict(case_id="test", window_id=0, path="sample.npz", frame_start=0, frame_stop_exclusive=2)
            with self.assertRaisesRegex(ValueError, "Raw source changed"):
                prep.validate_window(path, row, "config", grid_size=3, n_observed=1, n_probes=1)

    def test_missing_required_complete_cohort_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            prep.id_rows(np.array([1, 2]), np.array([1, 3]))

    def test_raw_checksum_detects_same_size_same_timestamp_mutation(self):
        _, _, frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "frame.bin"
            raw.write_bytes(b"raw")
            original_stat = raw.stat()
            record = dict(path=str(raw), sha256=prep.sha256(raw), **prep.file_stat(raw))
            metadata = dict(config_sha256="config", flow_group_id=prep.GROUP, case_id="test", window_id=0,
                            raw_sources=[record], unit_scale=[2.] * 3, grid_shape_zyx=[3] * 3)
            arrays = prep.assemble_window([copy.deepcopy(frame)] * 2, np.zeros((2, 1, 3)), [1., 2.], [1, 2],
                                         [27], [28], [[0., 1.]] * 3, metadata)
            path = Path(directory) / "sample.npz"
            prep.atomic_npz(path, arrays)
            raw.write_bytes(b"bad")
            os.utime(raw, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            row = dict(case_id="test", window_id=0, path="sample.npz", frame_start=0, frame_stop_exclusive=2)
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                prep.validate_window(path, row, "config", grid_size=3, n_observed=1, n_probes=1)


if __name__ == "__main__":
    unittest.main()
