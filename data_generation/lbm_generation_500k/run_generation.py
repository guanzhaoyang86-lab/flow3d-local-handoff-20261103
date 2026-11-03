#!/usr/bin/env python3
"""Generate exactly 500,000 single-view tracks with the frozen existing solver.

This wrapper only plans, launches and checks the existing collection/single-case
generators. It makes no changes to the LBM algorithm or observation integrator.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

UPSTREAM_COMMIT = "fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c"
NUM_CASES = 3125
NUM_PARTICLES = 160
EXPECTED_TRACKS = 500_000
PLANNER_SEED = 20260918
DEFAULT_RUNTIME_PROFILE = "delta-x86"
ARM_RUNTIME_PROFILE = "deltaai-arm-llvm20"
ARM_TAICHI_COMMIT = "93c190f603b676ec1f8871e4f46cf5d5be89548e"
ARM_LLVM_VERSION = "20.1.8"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_fingerprint(source: Path) -> dict[str, str]:
    paths = [source / "scripts" / name for name in (
        "generate_taichi_lbm3d_flow_collection.py",
        "generate_taichi_lbm3d_cavity_dataset.py",
        "build_sparse_flow_manifest.py",
    )]
    paths.extend(sorted((source / "src/flow_observation").rglob("*.py")))
    if not paths[3:]:
        raise ValueError("Source root is missing src/flow_observation Python sources")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing required source file: {path}")
    return {str(path.relative_to(source)): sha256(path) for path in paths}


def load_collection(source: Path):
    path = source / "scripts/generate_taichi_lbm3d_flow_collection.py"
    specification = importlib.util.spec_from_file_location("lbm500k_original_collection", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def collection_arguments(source: Path, upstream: Path, output: Path) -> list[str]:
    return [
        "--output-dir", str(output), "--sampling", "random",
        "--num-cases", str(NUM_CASES), "--planner-seed", str(PLANNER_SEED),
        "--python", sys.executable,
        "--generator", str(source / "scripts/generate_taichi_lbm3d_cavity_dataset.py"),
        "--upstream-repo", str(upstream), "--backend", "cuda", "--grid-size", "32",
        "--lid-speed-range", "0.03", "0.06", "--upstream-niu-range", "0.12", "0.21",
        "--warmup-step-range", "600", "1200", "--seed-range", "0", "2147483647",
        "--num-particles", str(NUM_PARTICLES), "--num-observation-times", "20",
        "--duration", "120", "--projections", "xz", "--integration-substeps", "4",
        "--particle-margin", "2", "--candidate-multiplier", "32",
        "--minimum-path-length", "0", "--particle-selection", "random",
        "--progress-interval", "200",
    ]


def chunk_bounds(index: int, count: int, total: int = NUM_CASES) -> tuple[int, int]:
    if count < 1 or count > total or index < 0 or index >= count:
        raise ValueError("Require 1 <= num_chunks <= num_cases and 0 <= chunk_index < num_chunks")
    return total * index // count, total * (index + 1) // count


@contextmanager
def file_lock(path: Path, blocking: bool = False):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise ValueError(f"Another process holds lock: {path}")
        yield


def check_upstream(upstream: Path) -> None:
    solver = upstream / "Single_phase/LBM_3D_SinglePhase_Solver.py"
    if not solver.is_file():
        raise ValueError(f"Missing upstream solver: {solver}")
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if actual != UPSTREAM_COMMIT or dirty:
        raise ValueError("Upstream must be clean and at the exact pinned commit")


def validate_arm_environment_manifest(manifest: dict, actual: dict) -> None:
    """Accept only the explicitly built and GPU/pilot-validated ARM environment."""
    if not isinstance(manifest, dict):
        raise ValueError("ARM environment manifest must be a JSON object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("ARM environment manifest requires schema_version=1")
    exact = {
        "profile": ARM_RUNTIME_PROFILE, "system": "Linux", "machine": "aarch64",
        "python_executable": actual["python_executable"], "python_version": actual["python"],
        "taichi_version": actual["packages"]["taichi"], "taichi_commit": actual["taichi_commit"],
        "taichi_source_commit": ARM_TAICHI_COMMIT,
        "torch_version": actual["torch_imported_version"], "cuda_runtime": actual["cuda_runtime"],
        "gpu_name": actual["gpu"], "llvm_version": ARM_LLVM_VERSION,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ValueError(f"ARM environment manifest {key}={manifest.get(key)!r}, actual/required {expected!r}")
    commit = actual["taichi_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit) or not ARM_TAICHI_COMMIT.startswith(commit):
        raise ValueError("Installed native Taichi commit is not the pinned ARM build")
    if actual["llvm_version"] != ARM_LLVM_VERSION:
        raise ValueError(f"Installed native Taichi must use LLVM {ARM_LLVM_VERSION}")
    for key in ("cuda_kernel_validated", "lbm32_pilot_validated"):
        if manifest.get(key) is not True:
            raise ValueError(f"ARM environment requires {key}=true after real GPU validation")
    if not isinstance(manifest.get("compiler_patch_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["compiler_patch_sha256"]):
        raise ValueError("ARM environment requires compiler_patch_sha256 as a lowercase 64-digit SHA256")
    timestamp = manifest.get("validated_at_utc")
    try:
        validated_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if validated_at.utcoffset() != timezone.utc.utcoffset(validated_at):
            raise ValueError("validation timestamp must be UTC")
    except (AttributeError, TypeError, ValueError):
        raise ValueError("ARM environment requires validated_at_utc as an ISO8601 UTC timestamp")


def runtime_provenance(runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
                       environment_manifest: Path | None = None) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Generate only in a Slurm allocation; --dry-run is allowed locally")
    if runtime_profile not in (DEFAULT_RUNTIME_PROFILE, ARM_RUNTIME_PROFILE):
        raise ValueError(f"Unsupported runtime profile: {runtime_profile}")
    expected_machine = "x86_64" if runtime_profile == DEFAULT_RUNTIME_PROFILE else "aarch64"
    if platform.system() != "Linux" or platform.machine() != expected_machine:
        raise ValueError(f"Runtime profile {runtime_profile} requires Linux {expected_machine}")
    if runtime_profile == ARM_RUNTIME_PROFILE and environment_manifest is None:
        raise ValueError("ARM runtime profile requires --environment-manifest from real GPU/pilot validation")
    if runtime_profile == DEFAULT_RUNTIME_PROFILE and environment_manifest is not None:
        raise ValueError("--environment-manifest is only supported by the explicit ARM runtime profile")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Generation requires exactly one visible working CUDA GPU")
    actual = {
        "profile": runtime_profile,
        "python": platform.python_version(), "python_executable": sys.executable,
        "machine": platform.machine(), "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda, "torch_imported_version": torch.__version__,
        "packages": {name: version(name) for name in ("torch", "numpy", "taichi", "sympy", "pyevtk")},
    }
    if runtime_profile == ARM_RUNTIME_PROFILE:
        import taichi
        try:
            core = taichi._lib.core
            actual["taichi_commit"] = core.get_commit_hash()
            actual["llvm_version"] = core.get_llvm_target_support()
            actual["taichi_runtime_version"] = core.get_version_string()
        except AttributeError as error:
            raise ValueError(f"Native Taichi build has no required introspection API: {error}") from error
        module_version = ".".join(map(str, taichi.__version__)) if isinstance(taichi.__version__, tuple) else str(taichi.__version__)
        if module_version != actual["taichi_runtime_version"]:
            raise ValueError("Taichi Python module and native core versions differ")
        distribution_version = re.fullmatch(
            r"(\d+\.\d+\.\d+)(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?(?:\+[A-Za-z0-9.]+)?",
            actual["packages"]["taichi"],
        )
        if distribution_version is None or distribution_version.group(1) != actual["taichi_runtime_version"]:
            raise ValueError("Taichi distribution metadata and native core release versions differ")
        manifest_path = environment_manifest.expanduser().absolute()
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
        validate_arm_environment_manifest(manifest, actual)
        actual["environment_manifest"] = {
            "path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest(),
            "contents": manifest,
        }
    return actual


def configuration(source: Path, upstream: Path, output: Path, num_chunks: int,
                  runtime_profile: str = DEFAULT_RUNTIME_PROFILE,
                  environment_manifest: Path | None = None) -> dict:
    return {
        "format_version": 1, "wrapper_sha256": sha256(Path(__file__).resolve()),
        "source_root": str(source), "source_sha256": source_fingerprint(source),
        "upstream_root": str(upstream), "upstream_commit": UPSTREAM_COMMIT,
        "collection_arguments": collection_arguments(source, upstream, output),
        "num_cases": NUM_CASES, "particles_per_case": NUM_PARTICLES,
        "expected_physical_trajectories": EXPECTED_TRACKS,
        "expected_single_view_trajectories": EXPECTED_TRACKS,
        "num_chunks": num_chunks, "projection_names": ["xz"],
        "scientific_scope": "Frozen snapshots of three-component lid-driven cavity flow; not validated turbulence",
        "runtime_profile": runtime_profile,
        "runtime": runtime_provenance(runtime_profile, environment_manifest),
    }


def initialize(output: Path, config: dict, records: list[dict], resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Simultaneously starting different array tasks may briefly serialize here.
    with file_lock(output / ".initialization.lock", blocking=True):
        marker = output / "generation.json"
        if marker.exists():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError("Resume requires identical source, wrapper, solver, packages, GPU, paths and parameters")
            if not resume:
                raise ValueError("Dataset already initialized; pass --resume to continue it")
            if (output / "COMPLETE.json").exists():
                raise ValueError("Dataset is already complete; use its manifest")
            if not (output / "plan.json").exists():
                raise ValueError("Stored generation metadata has no plan.json; initialization is incomplete")
            recorded_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            if recorded_plan != records:
                raise ValueError("Stored plan differs from the exact deterministic generation plan")
        else:
            plan = output / "plan.json"
            if resume and not plan.exists():
                raise ValueError("--resume requires an existing generation.json")
            # Publishing the plan before the generation marker makes interrupted
            # initialization recoverable without accepting any untracked data.
            allowed = {".initialization.lock", "plan.json", "plan.json.tmp", "generation.json.tmp"}
            other = [p for p in output.iterdir() if p.name not in allowed]
            if other:
                raise ValueError("A new dataset requires an empty output directory")
            if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != records:
                raise ValueError("Partial initialization plan differs from the deterministic plan")
            atomic_json(output / "plan.json", records)
            atomic_json(output / "generation.json", config)
        (output / "chunks").mkdir(exist_ok=True)
        (output / "attempts").mkdir(exist_ok=True)


def validate_archive(path: Path, collection, args, case) -> dict:
    """Check the original metadata contract plus actual arrays and trajectory count."""
    import numpy as np
    collection._validate_existing_archive(path, args, case)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("backend") != "cuda" or metadata.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError(f"Wrong backend or solver provenance: {path}")
        expected = {
            "flow_field": (1, 3, args.grid_size, args.grid_size, args.grid_size),
            "trajectories_3d": (1, args.num_particles, args.num_observation_times, 3),
            "trajectories_2d": (1, 1, args.num_particles, args.num_observation_times, 2),
            "observation_mask": (1, 1, args.num_particles, args.num_observation_times),
            "projection_matrix": (1, 2, 3), "particle_counts": (1,),
        }
        for name, shape in expected.items():
            if name not in archive.files or archive[name].shape != shape:
                raise ValueError(f"Wrong or missing array {name}, expected {shape}: {path}")
        for name in archive.files:
            value = archive[name]
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite values in {name}: {path}")
        if not bool(archive["observation_mask"].all()):
            raise ValueError(f"Incomplete observation trajectories: {path}")
        if archive["particle_counts"].tolist() != [args.num_particles]:
            raise ValueError(f"Wrong particle_counts: {path}")
        camera = np.asarray([[[1., 0., 0.], [0., 0., 1.]]], dtype=np.float32)
        if not np.array_equal(archive["projection_matrix"], camera):
            raise ValueError(f"Expected the xz single-view projection: {path}")
        if not np.allclose(archive["trajectories_2d"][0, 0], archive["trajectories_3d"][0][..., [0, 2]], rtol=0., atol=1e-6):
            raise ValueError(f"2D/3D trajectory projection mismatch: {path}")
        return {"physical_trajectories": int(archive["trajectories_3d"].shape[1]),
                "single_view_trajectories": int(archive["trajectories_2d"].shape[2]),
                "flow_group_id": metadata["flow_group_id"]}


def generate_chunk(index: int, num_chunks: int, output: Path, collection, args, cases, records) -> None:
    first, last = chunk_bounds(index, num_chunks, len(cases))
    state_path = output / "chunks" / f"chunk_{index:03d}.json"
    with file_lock(output / "chunks" / f"chunk_{index:03d}.lock"):
        started = time.monotonic()
        state = {"status": "running", "chunk_index": index, "case_start": first, "case_stop": last,
                 "num_cases": last - first, "generated": 0, "reused": 0,
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        atomic_json(state_path, state)
        try:
            for position in range(first, last):
                case = cases[position]
                path = output / f"{case.case_id}.npz"
                if path.exists():
                    validate_archive(path, collection, args, case)
                    state["reused"] += 1
                else:
                    log_path = output / "attempts" / f"{case.case_id}_{os.environ.get('SLURM_JOB_ID', 'local')}_{time.time_ns()}.log"
                    print(f"Generating {case.case_id} ({position + 1}/{len(cases)}); log={log_path}", flush=True)
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(records[position]["command"], cwd=args.generator.parent.parent,
                                       check=True, stdout=log, stderr=subprocess.STDOUT)
                    validate_archive(path, collection, args, case)
                    state["generated"] += 1
                state.update(last_case_id=case.case_id, completed_cases=state["generated"] + state["reused"],
                             elapsed_seconds=time.monotonic() - started)
                atomic_json(state_path, state)
                print(f"Chunk {index}: {state['completed_cases']}/{last-first} cases; "
                      f"{state['completed_cases'] * args.num_particles} single-view trajectories validated", flush=True)
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            atomic_json(state_path, state)


def finalize(output: Path, source: Path, collection, args, cases, records, expected_tracks: int = EXPECTED_TRACKS) -> dict:
    with file_lock(output / ".finalization.lock"):
        actual = set(output.glob("*.npz"))
        expected = {output / f"{case.case_id}.npz" for case in cases}
        if actual != expected:
            raise ValueError(f"Archive set differs from plan: missing={len(expected-actual)}, unexpected={len(actual-expected)}")
        totals = {"physical_trajectories": 0, "single_view_trajectories": 0}
        groups = set()
        checksums = {}
        for case in cases:
            path = output / f"{case.case_id}.npz"
            counts = validate_archive(path, collection, args, case)
            for key in totals:
                totals[key] += counts[key]
            groups.add(counts["flow_group_id"])
            checksums[path.name] = sha256(path)
        if any(value != expected_tracks for value in totals.values()):
            raise ValueError(f"Expected exactly {expected_tracks} trajectories; actual {totals}")
        if len(groups) != len(cases):
            raise ValueError("Planned independent physical flow groups are not unique")
        atomic_json(output / "collection.json", {
            "format_version": 1, "complete": True, "dry_run": False, "sampling": "random",
            "planner_seed": PLANNER_SEED, "num_cases": len(cases),
            "cases": [{**record, "status": "validated"} for record in records],
            "fixed_solver_facts": {"geometry": "lid-driven cavity", "lid_face": "x=max",
                                   "lid_direction": "+z", "initial_state": "fixed zero velocity",
                                   "frozen_snapshot": True},
            "split_policy": "flow_group_id is indivisible across train/validation/test",
            "scientific_scope": "Frozen cavity snapshots, not validated turbulence",
        })
        subprocess.run([sys.executable, "-B", str(source / "scripts/build_sparse_flow_manifest.py"),
                        "--input-dir", str(output), "--output", str(output / "manifest.json"),
                        "--seed", str(PLANNER_SEED)], check=True, cwd=source)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for name in ("manifest.json", "collection.json", "generation.json", "plan.json"):
            checksums[name] = sha256(output / name)
        temporary = output / "SHA256SUMS.tmp"
        temporary.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
        temporary.replace(output / "SHA256SUMS")
        result = {"status": "completed", "num_cases": len(cases), **totals,
                  "num_independent_flow_groups": len(groups), "projection_names": ["xz"],
                  "split_counts": {name: len(rows) for name, rows in manifest["splits"].items()},
                  "manifest": str(output / "manifest.json"),
                  "dataset_version": sha256(output / "SHA256SUMS"),
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "scientific_scope": "Data-contract validation only; frozen cavity snapshots are not validated turbulence"}
        atomic_json(output / "COMPLETE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-chunks", type=int, default=25)
    parser.add_argument("--runtime-profile", choices=(DEFAULT_RUNTIME_PROFILE, ARM_RUNTIME_PROFILE),
                        default=DEFAULT_RUNTIME_PROFILE)
    parser.add_argument("--environment-manifest", type=Path,
                        help="Exact GPU/pilot validation manifest, required only for deltaai-arm-llvm20")
    parser.add_argument("--chunk-index", type=int, help="Zero-based chunk; omit to run all chunks sequentially and finalize")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", "--check-plan", dest="dry_run", action="store_true",
                        help="Print the plan summary; no torch, solver, CUDA or files are initialized")
    parser.add_argument("--init-only", action="store_true", help="Initialize a new strict-provenance dataset within a Slurm GPU job")
    parser.add_argument("--finalize", action="store_true", help="Validate all archives, build manifest/checksums and publish COMPLETE.json")
    options = parser.parse_args()
    try:
        source = options.source_root.expanduser().resolve()
        upstream = options.upstream_repo.expanduser().resolve()
        output = options.output_dir.expanduser().resolve()
        if output == source or source in output.parents or output == upstream or upstream in output.parents:
            raise ValueError("Output directory must be outside both source and upstream repositories")
        chunk_bounds(options.chunk_index or 0, options.num_chunks)
        if sum((options.dry_run, options.init_only, options.finalize)) > 1:
            raise ValueError("--dry-run, --init-only and --finalize are mutually exclusive")
        if options.chunk_index is not None and (options.init_only or options.finalize):
            raise ValueError("--chunk-index cannot be combined with --init-only or --finalize")
        fingerprint = source_fingerprint(source)
        collection = load_collection(source)
        args = collection._build_parser().parse_args(collection_arguments(source, upstream, output))
        cases = collection.plan_cases(args)
        records = [{**vars(case), "output": str(output / f"{case.case_id}.npz"),
                    "command": collection.build_generator_command(args, case, output / f"{case.case_id}.npz"),
                    "status": "planned"} for case in cases]
        if len(cases) * NUM_PARTICLES != EXPECTED_TRACKS or len({case.flow_group_id for case in cases}) != NUM_CASES:
            raise ValueError("Plan does not contain exactly 3125 independent cases and 500000 tracks")
        if options.dry_run:
            print(json.dumps({"num_cases": len(cases), "particles_per_case": NUM_PARTICLES,
                              "physical_trajectories": EXPECTED_TRACKS, "single_view_trajectories": EXPECTED_TRACKS,
                              "projection_names": ["xz"], "num_chunks": options.num_chunks,
                              "chunk_ranges": [chunk_bounds(i, options.num_chunks) for i in range(options.num_chunks)],
                              "first_command": records[0]["command"], "source_file_count": len(fingerprint),
                              "runtime_profile": options.runtime_profile,
                              "environment_manifest": str(options.environment_manifest) if options.environment_manifest else None,
                              "scientific_scope": "Frozen cavity snapshots, not validated turbulence"}, indent=2))
            return
        check_upstream(upstream)
        config = configuration(source, upstream, output, options.num_chunks,
                               options.runtime_profile, options.environment_manifest)
        initialize(output, config, records, options.resume)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        # The same cache root is reused by sequential cases in an allocation.
        os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = str(output / "attempts" / f"taichi-cache-{os.environ['SLURM_JOB_ID']}")
        if options.init_only:
            print(f"Initialized {output}; all generation jobs must now pass --resume", flush=True)
            return
        if not options.finalize:
            selected = range(options.num_chunks) if options.chunk_index is None else [options.chunk_index]
            for index in selected:
                generate_chunk(index, options.num_chunks, output, collection, args, cases, records)
        if options.finalize or options.chunk_index is None:
            finalize(output, source, collection, args, cases, records)
    except (OSError, ValueError, subprocess.SubprocessError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
