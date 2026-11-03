"""Small CPU synthetic-archive checks; never imports Taichi or runs LBM/CUDA."""
from __future__ import annotations

import copy
import builtins
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_generation as generation

_PACKAGED_SOURCE = Path(__file__).resolve().parent / "source"
_DEFAULT_SOURCE = _PACKAGED_SOURCE if _PACKAGED_SOURCE.is_dir() else Path(__file__).resolve().parents[1] / "flow3d-current"
SOURCE = Path(os.environ.get("FLOW3D_TEST_SOURCE_ROOT", str(_DEFAULT_SOURCE))).resolve()


def make_fixture(output: Path):
    collection = generation.load_collection(SOURCE)
    arguments = generation.collection_arguments(SOURCE, output / "unused-upstream", output)
    args = collection._build_parser().parse_args(arguments)
    args.num_cases = 3
    args.num_particles = 2
    args.grid_size = 8
    args.num_observation_times = 3
    cases = collection.plan_cases(args)
    records = []
    for case in cases:
        metadata = {
            "field": "taichi_lbm3d_lid_driven_cavity",
            "case_id": case.case_id, "flow_group_id": case.flow_group_id,
            "grid_size_xyz": [8, 8, 8], "warmup_steps": case.warmup_steps,
            "snapshot_iteration": case.warmup_steps, "seed": case.seed,
            "max_num_particles": 2, "num_observation_times": 3,
            "projection_names": ["xz"], "integration_substeps": args.integration_substeps,
            "candidate_multiplier": args.candidate_multiplier, "particle_selection_mode": "random",
            "upstream_niu_parameter": case.upstream_niu, "duration": args.duration,
            "particle_margin": args.particle_margin, "minimum_path_length": args.minimum_path_length,
            "lid_velocity_xyz": [0., 0., case.lid_speed], "lid_face": "x=max",
            "backend": "cuda", "upstream_commit": generation.UPSTREAM_COMMIT,
            "frozen_snapshot_used_for_advection": True,
        }
        tracks = np.array([[[[2., 3., 2.], [2.1, 3., 2.2], [2.2, 3., 2.4]],
                            [[4., 3., 4.], [4.1, 3., 4.2], [4.2, 3., 4.4]]]], dtype=np.float32)
        path = output / f"{case.case_id}.npz"
        np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)),
                            flow_field=np.ones((1, 3, 8, 8, 8), dtype=np.float32),
                            trajectories_3d=tracks, trajectories_2d=tracks[:, None, ..., [0, 2]],
                            projection_matrix=np.asarray([[[1., 0., 0.], [0., 0., 1.]]], dtype=np.float32),
                            observation_mask=np.ones((1, 1, 2, 3), dtype=bool),
                            particle_counts=np.asarray([2], dtype=np.int64),
                            observation_times=np.asarray([0., 60., 120.], dtype=np.float32),
                            domain_bounds=np.asarray([[0., 7.]] * 3, dtype=np.float32),
                            solid_mask=np.zeros((8, 8, 8), dtype=bool))
        records.append({**vars(case), "output": str(path), "command": [], "status": "validated"})
    return collection, args, cases, records


class GenerationTests(unittest.TestCase):
    def test_check_plan_never_imports_torch_or_taichi_or_writes_dataset(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "taichi"}:
                raise AssertionError(f"Planning attempted prohibited import: {name}")
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-be-created"
            argv = ["run_generation.py", "--source-root", str(SOURCE),
                    "--upstream-repo", "/unused/upstream", "--output-dir", str(output), "--check-plan"]
            stream = io.StringIO()
            with patch("sys.argv", argv), patch("builtins.__import__", guarded_import), redirect_stdout(stream):
                generation.main()
            self.assertEqual(json.loads(stream.getvalue())["single_view_trajectories"], 500000)
            self.assertFalse(output.exists())

    def test_arm_profile_requires_matching_validated_environment_and_preserves_venv(self):
        versions = {"torch": "2.10.0", "numpy": "2.4.2", "taichi": "1.8.0",
                    "sympy": "1.14.0", "pyevtk": "1.6.0"}
        executable = "/tmp/specific-arm-venv/bin/python"
        fake_torch = SimpleNamespace(__version__="2.10.0+cu129", version=SimpleNamespace(cuda="12.9"),
                                     cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                          get_device_name=lambda index: "NVIDIA GH200 120GB"))
        fake_taichi = SimpleNamespace(__version__=(1, 8, 0), _lib=SimpleNamespace(core=SimpleNamespace(
            get_commit_hash=lambda: generation.ARM_TAICHI_COMMIT,
            get_llvm_target_support=lambda: "20.1.8", get_version_string=lambda: "1.8.0")))
        manifest = {
            "schema_version": 1, "profile": generation.ARM_RUNTIME_PROFILE,
            "system": "Linux", "machine": "aarch64", "python_executable": executable,
            "python_version": "3.12.9", "taichi_version": "1.8.0",
            "taichi_commit": generation.ARM_TAICHI_COMMIT,
            "taichi_source_commit": generation.ARM_TAICHI_COMMIT,
            "torch_version": "2.10.0+cu129", "cuda_runtime": "12.9",
            "gpu_name": "NVIDIA GH200 120GB", "llvm_version": "20.1.8",
            "compiler_patch_sha256": "a" * 64,
            "cuda_kernel_validated": True, "lbm32_pilot_validated": True,
            "validated_at_utc": "2026-10-23T23:00:00+00:00",
            "extra_validation_evidence": {"example": "All contents must be preserved"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "validated_env.json"
            generation.atomic_json(marker, manifest)
            with patch.dict("os.environ", {"SLURM_JOB_ID": "mock-cpu-test"}), \
                    patch("run_generation.platform.system", return_value="Linux"), \
                    patch("run_generation.platform.machine", return_value="aarch64"), \
                    patch("run_generation.platform.python_version", return_value="3.12.9"), \
                    patch("run_generation.sys.executable", executable), \
                    patch("run_generation.version", side_effect=versions.__getitem__), \
                    patch.dict("sys.modules", {"torch": fake_torch, "taichi": fake_taichi}):
                with self.assertRaisesRegex(ValueError, "requires Linux x86_64"):
                    generation.runtime_provenance()
                with self.assertRaisesRegex(ValueError, "requires --environment-manifest"):
                    generation.runtime_provenance(generation.ARM_RUNTIME_PROFILE)
                actual = generation.runtime_provenance(generation.ARM_RUNTIME_PROFILE, marker)
                self.assertEqual(actual["python_executable"], executable)
                self.assertEqual(actual["environment_manifest"]["contents"], manifest)
                self.assertEqual(actual["environment_manifest"]["sha256"], generation.sha256(marker))
                versions["taichi"] = "1.7.4"
                with self.assertRaisesRegex(ValueError, "metadata and native core"):
                    generation.runtime_provenance(generation.ARM_RUNTIME_PROFILE, marker)
                versions["taichi"] = "1.8.0"
                for key, wrong in (("python_executable", "/usr/bin/python"), ("python_version", "3.11.0"),
                                   ("taichi_version", "wrong"), ("taichi_commit", "b" * 40),
                                   ("taichi_source_commit", "b" * 40), ("torch_version", "wrong"),
                                   ("cuda_runtime", "wrong"), ("gpu_name", "wrong"),
                                   ("llvm_version", "20.1.6"), ("profile", "unvalidated-arm"),
                                   ("cuda_kernel_validated", False), ("lbm32_pilot_validated", 1),
                                   ("compiler_patch_sha256", "missing"), ("validated_at_utc", "2026-10-23")):
                    with self.subTest(key=key):
                        changed = {**manifest, key: wrong}
                        generation.atomic_json(marker, changed)
                        with self.assertRaises(ValueError):
                            generation.runtime_provenance(generation.ARM_RUNTIME_PROFILE, marker)

    def test_arm_manifest_cannot_authorize_a_different_native_build(self):
        actual = {"python_executable": "/venv/bin/python", "python": "3.12.9", "gpu": "GH200",
                  "packages": {"taichi": "1.8.0"}, "taichi_commit": "b" * 40,
                  "torch_imported_version": "2.10.0", "cuda_runtime": "12.9", "llvm_version": "20.1.8"}
        manifest = {"schema_version": 1, "profile": generation.ARM_RUNTIME_PROFILE,
                    "system": "Linux", "machine": "aarch64", "python_executable": actual["python_executable"],
                    "python_version": actual["python"], "taichi_version": "1.8.0",
                    "taichi_commit": actual["taichi_commit"], "taichi_source_commit": generation.ARM_TAICHI_COMMIT,
                    "torch_version": "2.10.0", "cuda_runtime": "12.9", "gpu_name": "GH200", "llvm_version": "20.1.8"}
        with self.assertRaisesRegex(ValueError, "not the pinned ARM build"):
            generation.validate_arm_environment_manifest(manifest, actual)

    def test_exact_production_plan_and_chunks(self):
        collection = generation.load_collection(SOURCE)
        args = collection._build_parser().parse_args(generation.collection_arguments(SOURCE, Path("/unused"), Path("/tmp/unused")))
        cases = collection.plan_cases(args)
        self.assertEqual(len(cases), 3125)
        self.assertEqual(len({case.case_id for case in cases}), 3125)
        self.assertEqual(len({case.flow_group_id for case in cases}), 3125)
        self.assertEqual(len(cases) * args.num_particles, 500000)
        self.assertEqual(args.projections, "xz")
        self.assertEqual(args.grid_size, 32)
        chunks = [generation.chunk_bounds(i, 25) for i in range(25)]
        self.assertEqual([item for first, last in chunks for item in range(first, last)], list(range(3125)))
        self.assertTrue(all(last-first == 125 for first, last in chunks))
        self.assertEqual(cases, collection.plan_cases(args))

    def test_strict_resume_rejects_changed_configuration_and_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            config = {"source_sha256": {"source.py": "digest"}, "gpu": "GPU-model"}
            records = [{"case_id": "case_00000"}]
            generation.initialize(output, config, records, False)
            generation.initialize(output, config, records, True)
            changed = copy.deepcopy(config)
            changed["source_sha256"]["source.py"] = "changed"
            with self.assertRaisesRegex(ValueError, "identical source"):
                generation.initialize(output, changed, records, True)
            with self.assertRaisesRegex(ValueError, "Stored plan"):
                generation.initialize(output, config, [{"case_id": "different"}], True)

    def test_interrupted_initialization_is_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = [{"case_id": "case_00000"}]
            generation.atomic_json(output / "plan.json", records)
            (output / "generation.json.tmp").write_text("partial")
            generation.initialize(output, {"source": "same"}, records, True)
            self.assertEqual(json.loads((output / "generation.json").read_text()), {"source": "same"})

    def test_real_manifest_from_small_cpu_synthetic_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            collection, args, cases, records = make_fixture(output)
            generation.atomic_json(output / "generation.json", {"test": "CPU synthetic, not CFD"})
            generation.atomic_json(output / "plan.json", records)
            counts = generation.validate_archive(output / "case_00000.npz", collection, args, cases[0])
            self.assertEqual(counts["single_view_trajectories"], 2)
            with self.assertRaisesRegex(ValueError, "exactly 7"):
                generation.finalize(output, SOURCE, collection, args, cases, records, expected_tracks=7)
            result = generation.finalize(output, SOURCE, collection, args, cases, records, expected_tracks=6)
            self.assertEqual(result["single_view_trajectories"], 6)
            self.assertEqual(result["physical_trajectories"], 6)
            self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
            self.assertTrue((output / "COMPLETE.json").is_file())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["mode"], "scientific")

    def test_corrupt_projection_and_unexpected_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            collection, args, cases, records = make_fixture(output)
            path = output / "case_00000.npz"
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            payload["trajectories_2d"] = payload["trajectories_2d"] + 1.
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "projection mismatch"):
                generation.validate_archive(path, collection, args, cases[0])
            (output / "unplanned.npz").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected=1"):
                generation.finalize(output, SOURCE, collection, args, cases, records, expected_tracks=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
