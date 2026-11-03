#!/usr/bin/env python3
"""Prepare resumable, masked transient FS transport fields for the N12 baselines.

Production preprocessing requires a Slurm allocation.  --eligibility-only reads
metadata, small protocol arrays and one initial raw frame; --smoke-only runs
analytic tests and never scans or prepares the real simulation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np


SCHEMA = "fs_transient_baseline_manifest_v1"
GROUP = "fs_sph_3429811"
FRAME_COUNT = 20
DEFAULT_DATASET = Path("/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_rods_full_v2")
DEFAULT_OUTPUT = Path("/work/nvme/biup/zguan2/fs_sph_gpu/datasets/fs_sph_3429811_flow32_windows_20261027")
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2] / "jhtdb_fs_tensor_dit_20261026/protocol"
NATIVE_ROOT = Path(__file__).resolve().with_name("fs_native")
PROJECTION_XZ = np.array([[[1, 0, 0], [0, 0, 1]]], dtype=np.float32)
_WORKER_CONTEXT = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing to overwrite window {path}")
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def split_windows():
    """One physical case, chronological holdout; embargoes are never examples."""
    return {"train": list(range(126)), "validation": list(range(135, 153)), "test": list(range(162, 180))}


def window_rows():
    return {split: [dict(case_id=f"fs_window_{index:04d}", path=f"windows/window_{index:04d}.npz",
                         flow_group_id=GROUP, window_id=index, frame_start=20 * index,
                         frame_stop_exclusive=20 * (index + 1)) for index in indices]
            for split, indices in split_windows().items()}


def native_modules():
    modules = []
    for name in ("fs_particle_io", "fs_grid"):
        key = "baseline_fs_native_" + name
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, NATIVE_ROOT / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[key] = module
        modules.append(sys.modules[key])
    return modules


def file_stat(path):
    info = Path(path).stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def id_rows(ids, requested):
    ids = np.asarray(ids)
    order = np.argsort(ids)
    locations = np.searchsorted(ids[order], requested)
    require(bool((locations < len(ids)).all()), "Required complete-cohort particle missing from native frame")
    rows = order[locations]
    require(np.array_equal(ids[rows], requested), "Required complete-cohort particle missing from native frame")
    return rows


def canonical_scale(bounds, grid_size=32):
    bounds = np.asarray(bounds, dtype=np.float64)
    require(bounds.shape == (3, 2) and np.isfinite(bounds).all()
            and bool((bounds[:, 1] > bounds[:, 0]).all()), "Invalid native physical domain")
    return (grid_size - 1) / (bounds[:, 1] - bounds[:, 0])


def physical_time_weights(times):
    times = np.asarray(times, dtype=np.float64)
    require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all()
            and bool((np.diff(times) > 0).all()), "Invalid physical time vector")
    delta = np.diff(times)
    weights = np.concatenate(([delta[0] / 2], (delta[:-1] + delta[1:]) / 2, [delta[-1] / 2]))
    return weights / weights.sum()


def build_frame(native, *, cohort_ids, excluded_ids, probe_ids, physical_bounds, spacing, grid_size=32):
    """Grid only the initial complete rod cohort, independent of later adhesion.

    Excluded observed/probe IDs never enter velocity fits OR geometry donors.
    Transport-invalid points can locate material but cannot donate velocities.
    Substrate particles never donate, even if their coordinates overlap rods.
    """
    from scipy.spatial import cKDTree

    _, grid_module = native_modules()
    fields = native["fields"]
    require("transport_velocity" in fields and "transport_velocity_valid" in fields,
            "Native transport velocity and validity are mandatory; momentum fallback is forbidden")
    all_ids = np.asarray(fields["unique_idx"])
    cohort_rows = id_rows(all_ids, np.asarray(cohort_ids))
    points = np.asarray(native["points"])[cohort_rows].astype(np.float64)
    ids = all_ids[cohort_rows]
    velocity = np.asarray(fields["transport_velocity"])[cohort_rows].astype(np.float64)
    native_valid = np.asarray(fields["transport_velocity_valid"])[cohort_rows].astype(bool)
    native_valid &= np.isfinite(velocity).all(axis=1)
    require(np.isfinite(points).all(), "Nonfinite complete-cohort coordinates")
    held_out = np.isin(ids, excluded_ids)
    geometry_rows = np.flatnonzero(~held_out)
    donor_rows = np.flatnonzero(~held_out & native_valid)
    require(len(geometry_rows) >= 4 and len(donor_rows) >= 4, "Insufficient valid independent rod donors")
    safe_velocity = np.where(native_valid[:, None], velocity, 0.0)
    invalid_ids = ids[~native_valid]
    grid = grid_module.grid_particle_velocity(
        points, safe_velocity, physical_bounds, grid_size=grid_size, spacing=spacing,
        velocity_scale_factor=1.0, k=32, workers=1, support_radius_factor=7.0,
        material_radius_factor=0.9, particle_ids=ids,
        excluded_ids=np.union1d(excluded_ids, invalid_ids),
    )
    # Geometry uses all retained initial-cohort particles, without silently
    # declaring transport-invalid material to be empty space.
    query = grid["grid_points"].reshape(-1, 3)
    distance, nearest = cKDTree(points[geometry_rows]).query(query)
    nearest_rows = geometry_rows[nearest]
    shape = (grid_size,) * 3
    material = (distance <= 0.9 * spacing).reshape(shape)
    interpolation_support = grid["support_mask"].astype(bool)
    fixed = np.asarray(fields["Fixed"])[cohort_rows][nearest_rows].reshape(shape)
    prescribed = material & interpolation_support & (fixed >= 2)
    support = material & interpolation_support & ~prescribed
    physical_flow = np.where(interpolation_support[None], grid["flow_field"], 0.0)
    scale = canonical_scale(physical_bounds, grid_size)
    flow = (physical_flow.astype(np.float64) * scale[:, None, None, None]).astype(np.float32)
    require(np.isfinite(flow).all(), "Nonfinite canonical grid velocities")
    probe_rows = id_rows(ids, np.asarray(probe_ids))
    probe_valid = native_valid[probe_rows]
    probe_velocity = np.where(probe_valid[:, None], velocity[probe_rows], 0.0)
    probe_fit = grid_module.interpolate_particle_velocity(
        points, safe_velocity, points[probe_rows], spacing,
        velocity_scale_factor=1.0, k=32, workers=1, support_radius_factor=7.0,
        material_radius_factor=0.9, particle_ids=ids,
        excluded_ids=np.union1d(excluded_ids, invalid_ids),
    )
    fit_valid = probe_valid & probe_fit["support_mask"]
    error = probe_fit["velocity"][fit_valid] - velocity[probe_rows][fit_valid]
    return dict(
        flow_field=flow, support_mask=support, material_mask=material,
        interpolation_support_mask=interpolation_support, prescribed_mask=prescribed,
        ghost_extension_mask=interpolation_support & ~material,
        probe_xyz=((points[probe_rows] - np.asarray(physical_bounds)[:, 0]) * scale).astype(np.float32),
        probe_velocity=(probe_velocity * scale).astype(np.float32), probe_valid=probe_valid,
        probe_reference_support=probe_fit["support_mask"].astype(bool),
        diagnostics=dict(donors=len(donor_rows), geometry_donors=len(geometry_rows),
                         transport_invalid_count=int((~native_valid).sum()),
                         native_probe_fit_count=int(fit_valid.sum()),
                         native_probe_fit_squared_error=float(np.square(error).sum()),
                         native_probe_velocity_squared_energy=float(np.square(velocity[probe_rows][fit_valid]).sum()),
                         valid_grid_nodes=int(support.sum())),
    )


def source_context(dataset_root, protocol_root, run_root=None):
    """Bounded eligibility checks: metadata/small arrays and one raw frame."""
    dataset_root, protocol_root = Path(dataset_root).resolve(), Path(protocol_root).resolve()
    source_manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(source_manifest_path.read_text())
    require(manifest.get("schema") == "fs_sph_full_tracks_v2" and manifest.get("status") == "validated",
            "Use the validated complete FS v2 release")
    require(manifest["metadata"].get("flow_group_id") == GROUP
            and manifest["counts"].get("independent_cases") == 1,
            "Unexpected physical group or independent case count")
    require(manifest["counts"].get("particle_count") == 25990
            and manifest["counts"].get("frame_count") == 3613,
            "Expected the complete 25990 x 3613 release, not the old 160-track sample")
    arrays = {}
    fingerprints = {str(source_manifest_path): {"sha256": sha256(source_manifest_path), **file_stat(source_manifest_path)}}
    for key in ("positions", "observation_mask_3d", "particle_ids", "observation_times", "simulation_steps", "domain_bounds"):
        record = manifest["files"][key]
        path = (dataset_root / record["path"]).resolve()
        require(path.is_relative_to(dataset_root), "Release array path escapes root")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(list(array.shape) == record["shape"] and np.dtype(record["dtype"]) == array.dtype,
                f"Release layout changed: {key}")
        arrays[key] = array
        fingerprints[str(path)] = file_stat(path)
        if key not in {"positions", "observation_mask_3d"}:
            fingerprints[str(path)]["sha256"] = sha256(path)
    require(arrays["positions"].shape == (3613, 25990, 3)
            and arrays["observation_mask_3d"].shape == (3613, 25990), "Unexpected full-track array layout")
    require(np.isfinite(arrays["observation_times"]).all()
            and bool((np.diff(arrays["observation_times"]) > 0).all()), "Native times must increase")
    require(bool((np.diff(arrays["simulation_steps"]) > 0).all()), "Native steps must increase")
    cohort_ids = np.asarray(arrays["particle_ids"])
    require(len(np.unique(cohort_ids)) == 25990, "Invalid complete cohort IDs")
    id_files = dict(observed_ids=dataset_root / "protocol/particle_ids_N12_seed47.npy",
                    maximal_observed_ids=dataset_root / "protocol/particle_ids_N96_seed47.npy",
                    probe_ids=protocol_root / "fs_probe_ids_32_seed47.npy")
    for key, path in id_files.items():
        arrays[key] = np.load(path, allow_pickle=False).astype(np.int64)
        fingerprints[str(path)] = {"sha256": sha256(path), **file_stat(path)}
    require(len(arrays["observed_ids"]) == 12 and len(arrays["maximal_observed_ids"]) == 96
            and len(arrays["probe_ids"]) == 32, "Fixed IDs must have sizes N12/N96/probes32")
    permutation = np.random.default_rng(47).permutation(np.sort(cohort_ids))
    require(np.array_equal(arrays["observed_ids"], np.sort(permutation[:12]))
            and np.array_equal(arrays["maximal_observed_ids"], np.sort(permutation[:96])),
            "Published observation IDs differ from the declared nested seed47 protocol")
    require(np.isin(arrays["observed_ids"], arrays["maximal_observed_ids"]).all(), "N12 must nest in fixed N96")
    require(not np.intersect1d(arrays["maximal_observed_ids"], arrays["probe_ids"]).size, "Observed and probe IDs overlap")
    excluded_ids = np.union1d(arrays["maximal_observed_ids"], arrays["probe_ids"])
    require(len(excluded_ids) == 128 and np.isin(excluded_ids, cohort_ids).all(), "Observation/probe IDs must be complete rod members")
    arrays["excluded_ids"] = excluded_ids
    run_root = Path(run_root or manifest["source"]["run_dir"]).resolve()
    summary_path = run_root / "work/run_summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary.get("stop_reason") == "model_complete"
            and summary["completed_steps"] >= summary["full_model_steps"], "Native simulation is not complete")
    fingerprints[str(summary_path)] = {"sha256": sha256(summary_path), **file_stat(summary_path)}
    spacing = float(summary.get("particle_spacing_m", 0.00075))
    require(np.isfinite(spacing) and spacing > 0, "Invalid native particle spacing")
    raw_path = run_root / "work/particle_snapshots/particles_000000000000.bin"
    native_reader, _ = native_modules()
    initial = native_reader.read_compact(raw_path)
    require(initial["step"] == 0 and initial["physical_time_s"] == 0.0, "Unexpected initial native frame")
    rows = id_rows(initial["fields"]["unique_idx"], cohort_ids)
    require(bool((initial["fields"]["Fixed"][rows] == 0).all())
            and bool((initial["fields"]["Jointed"][rows] == 0).all()), "Full release does not match initial deformable rods")
    require(np.array_equal(initial["points"][rows], np.asarray(arrays["positions"][0])), "Initial raw/release positions differ")
    fingerprints[str(raw_path)] = {"sha256": sha256(raw_path), **file_stat(raw_path)}
    require(np.allclose(arrays["observation_times"], arrays["simulation_steps"] * summary["simulation_dt"]
                        * summary["velocity_scale_factor"], rtol=1e-10, atol=1e-14), "Native time mapping differs from validated release")
    config = dict(schema=SCHEMA, group=GROUP, input_views=["xz"], num_particles=12, probes=32,
                  grid_size=32, frames_per_window=20, splits=split_windows(),
                  cohort_policy="initial complete deformable rods, identity retained after adhesion",
                  velocity_kind="native_transport_no_fallback", native_velocity_scale=1.0,
                  coordinate_transform="x_cell=(x_m-lower_m)*unit_scale; v_cell=v_m_s*unit_scale",
                  domain_side_information="Published union of all material points over process; common declared known domain",
                  physical_bounds=np.asarray(arrays["domain_bounds"]).tolist(),
                  unit_scale=canonical_scale(arrays["domain_bounds"]).tolist(),
                  spacing_m=spacing, spacing_source="run_summary" if "particle_spacing_m" in summary else "known solver default 0.00075m",
                  interpolation=dict(k=32, material_radius_factor=0.9, support_radius_factor=7.0),
                  dataset_root=str(dataset_root), run_root=str(run_root), protocol_root=str(protocol_root),
                  source_fingerprints=fingerprints,
                  source_code_sha256={str(path): sha256(path) for path in
                                      (Path(__file__).resolve(), NATIVE_ROOT / "fs_particle_io.py", NATIVE_ROOT / "fs_grid.py")})
    return dict(arrays=arrays, cohort_ids=cohort_ids, run_root=run_root, config=config,
                config_sha256=stable_hash(config), spacing=spacing)


def assemble_window(frames, observations_xyz, times, steps, observed_ids, probe_ids, physical_bounds, metadata):
    scale = canonical_scale(physical_bounds, frames[0]["flow_field"].shape[-1])
    observed_cell = ((np.asarray(observations_xyz, dtype=np.float64) - np.asarray(physical_bounds)[:, 0]) * scale).astype(np.float32)
    # Source observations_xyz axes are T,N,XYZ.  Only XZ enters reconstruction.
    tracks = np.transpose(observed_cell[:, :, [0, 2]], (1, 0, 2))[None, None]
    result = {key: np.stack([frame[key] for frame in frames]) for key in
              ("flow_field", "support_mask", "material_mask", "interpolation_support_mask", "prescribed_mask", "ghost_extension_mask")}
    for key in ("probe_xyz", "probe_velocity", "probe_valid", "probe_reference_support"):
        result[key] = np.stack([frame[key] for frame in frames], axis=1)
    result.update(trajectories_2d=tracks, observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
                  projection_matrix=PROJECTION_XZ.copy(), observation_times=np.asarray(times, dtype=np.float64),
                  velocity_times=np.asarray(times, dtype=np.float64), simulation_steps=np.asarray(steps, dtype=np.int64),
                  domain_bounds=np.array([[0, frames[0]["flow_field"].shape[-1] - 1]] * 3, dtype=np.float64),
                  particle_ids=np.asarray(observed_ids, dtype=np.int64), probe_ids=np.asarray(probe_ids, dtype=np.int64),
                  physical_domain_bounds=np.asarray(physical_bounds, dtype=np.float64),
                  metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    return result


def init_worker(dataset_root, protocol_root, run_root, config_sha256, output_root):
    global _WORKER_CONTEXT
    # mmap the release in each worker; only 44 selected positions per frame are copied.
    context = source_context(dataset_root, protocol_root, run_root)
    require(context["config_sha256"] == config_sha256, "Source/config changed before worker started")
    context["output_root"] = Path(output_root)
    _WORKER_CONTEXT = context


def build_window(row):
    context = _WORKER_CONTEXT
    arrays = context["arrays"]
    times = arrays["observation_times"][row["frame_start"]:row["frame_stop_exclusive"]]
    steps = arrays["simulation_steps"][row["frame_start"]:row["frame_stop_exclusive"]]
    observed_rows = id_rows(context["cohort_ids"], arrays["observed_ids"])
    probe_rows = id_rows(context["cohort_ids"], arrays["probe_ids"])
    frames, raw_sources = [], []
    native_reader, _ = native_modules()
    started = time.monotonic()
    for local_index, (step, native_time) in enumerate(zip(steps, times)):
        frame_index = row["frame_start"] + local_index
        path = context["run_root"] / "work/particle_snapshots" / f"particles_{int(step):012d}.bin"
        before = file_stat(path)
        native = native_reader.read_compact(path)
        require(native["step"] == int(step) and np.isclose(native["physical_time_s"], native_time, rtol=1e-12, atol=1e-14),
                f"Native header differs from release at frame {frame_index}")
        selected_ids = np.concatenate((arrays["observed_ids"], arrays["probe_ids"]))
        native_rows = id_rows(native["fields"]["unique_idx"], selected_ids)
        reference_rows = np.concatenate((observed_rows, probe_rows))
        require(bool(arrays["observation_mask_3d"][frame_index, reference_rows].all()), "Selected full tracks are not complete")
        require(np.array_equal(native["points"][native_rows], np.asarray(arrays["positions"][frame_index, reference_rows])),
                f"Native/release XYZ mismatch at frame {frame_index}")
        frame = build_frame(native, cohort_ids=context["cohort_ids"], excluded_ids=arrays["excluded_ids"],
                            probe_ids=arrays["probe_ids"], physical_bounds=arrays["domain_bounds"], spacing=context["spacing"])
        require(frame["support_mask"].any(), f"No valid unknown material nodes at frame {frame_index}")
        raw_sources.append(dict(path=str(path), sha256=sha256(path), **before))
        require(file_stat(path) == before, f"Raw frame changed while read: {path}")
        frames.append(frame)
    diagnostics = [frame.pop("diagnostics") for frame in frames]
    weights = physical_time_weights(times)
    probe_count = float(np.dot(weights, [frame["native_probe_fit_count"] for frame in diagnostics]))
    probe_error = float(np.dot(weights, [frame["native_probe_fit_squared_error"] for frame in diagnostics]))
    probe_energy = float(np.dot(weights, [frame["native_probe_velocity_squared_energy"] for frame in diagnostics]))
    metadata = dict(schema="fs_transient_baseline_window_v1", case_id=row["case_id"], flow_group_id=GROUP,
                    original_physical_group_id=GROUP, window_id=row["window_id"],
                    frame_start=row["frame_start"], frame_stop_exclusive=row["frame_stop_exclusive"],
                    grid_shape_zyx=[32, 32, 32], unit_scale=context["config"]["unit_scale"],
                    physical_domain_bounds=context["config"]["physical_bounds"],
                    inverse_transform="x_m=x_cell/unit_scale+lower_m; v_m_s=v_cell/unit_scale; time_s unchanged",
                    coordinate_units="native_grid_cells", velocity_units="native_grid_cells/physical_second", time_units="seconds",
                    velocity_kind="native_transport", native_velocity_scale_factor=1.0,
                    domain_information=context["config"]["domain_side_information"],
                    input_views=["xz"], observed_ids=arrays["observed_ids"].tolist(),
                    excluded_donor_ids=arrays["excluded_ids"].tolist(), probe_ids=arrays["probe_ids"].tolist(),
                    probe_policy="Evaluation only; no XYZ/velocity/probe/reference support used by reconstruction",
                    mask_policy="Unknown supervision: material & interpolation_support & ~prescribed; zero is padding, not wall",
                    cohort_policy=context["config"]["cohort_policy"], independent_cases=1,
                    split_policy="within_case_temporal; common physical group, not cross-case generalization",
                    config_sha256=context["config_sha256"], raw_sources=raw_sources,
                    per_frame_diagnostics=diagnostics,
                    native_probe_fit_rmse_m_s=float(np.sqrt(probe_error / probe_count)) if probe_count > 0 else None,
                    native_probe_fit_relative_l2=float(np.sqrt(probe_error / max(probe_energy, 1e-24))) if probe_count > 0 else None,
                    probe_diagnostic_time_weighting="trapezoidal native physical-time weights; valid probe counts pooled",
                    elapsed_seconds=time.monotonic() - started)
    observed_xyz = np.asarray(arrays["positions"][row["frame_start"]:row["frame_stop_exclusive"]])[:, observed_rows]
    payload = assemble_window(frames, observed_xyz, times, steps, arrays["observed_ids"], arrays["probe_ids"], arrays["domain_bounds"], metadata)
    path = context["output_root"] / row["path"]
    atomic_npz(path, payload)
    return validate_window(path, row, context["config_sha256"], verify_sources=False)


def validate_window(path, row, config_sha256, *, grid_size=32, n_observed=12, n_probes=32, verify_sources=True):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        require(metadata["config_sha256"] == config_sha256, f"Different prep source/config in {path}")
        require(metadata["flow_group_id"] == GROUP and metadata["case_id"] == row["case_id"]
                and metadata["window_id"] == row["window_id"], f"Window identity mismatch in {path}")
        field = archive["flow_field"]
        mask = archive["support_mask"]
        length = row["frame_stop_exclusive"] - row["frame_start"]
        require(field.shape == (length, 3, grid_size, grid_size, grid_size) and field.dtype == np.float32
                and np.isfinite(field).all(), "Invalid transient field layout or values")
        require(mask.shape == (length, grid_size, grid_size, grid_size) and mask.dtype == np.bool_, "Invalid supervision mask")
        expected = archive["material_mask"] & archive["interpolation_support_mask"] & ~archive["prescribed_mask"]
        require(np.array_equal(mask, expected) and mask.reshape(length, -1).any(axis=1).all(), "Invalid unknown material supervision")
        times = archive["observation_times"]
        require(times.dtype == np.float64 and times.shape == (length,) and np.isfinite(times).all()
                and bool((np.diff(times) > 0).all()) and np.array_equal(times, archive["velocity_times"]), "Invalid physical times")
        tracks = archive["trajectories_2d"]
        require(tracks.shape == (1, 1, n_observed, length, 2) and tracks.dtype == np.float32
                and np.isfinite(tracks).all(), "Only one XZ input camera is allowed")
        require(archive["observation_mask"].shape == tracks.shape[:-1]
                and archive["observation_mask"].dtype == np.bool_ and archive["observation_mask"].all(), "Invalid observed track validity")
        require(np.array_equal(archive["projection_matrix"], PROJECTION_XZ), "Camera differs from single XZ")
        require(np.array_equal(archive["domain_bounds"], [[0, grid_size - 1]] * 3), "Invalid canonical bounds")
        require(archive["probe_xyz"].shape == (n_probes, length, 3)
                and archive["probe_velocity"].shape == (n_probes, length, 3)
                and archive["probe_valid"].shape == (n_probes, length), "Invalid evaluation-only probe layout")
        require(archive["probe_valid"].dtype == np.bool_ and np.isfinite(archive["probe_xyz"]).all()
                and np.isfinite(archive["probe_velocity"]).all(), "Invalid probe values")
        require(not np.intersect1d(archive["particle_ids"], archive["probe_ids"]).size, "Probe observations leaked into conditioning")
        coverage = mask.mean(axis=(1, 2, 3)).tolist()
    if verify_sources:
        for record in metadata["raw_sources"]:
            require(file_stat(record["path"]) == {"bytes": record["bytes"], "mtime_ns": record["mtime_ns"]},
                    f"Raw source changed after preparation: {record['path']}")
            require(sha256(record["path"]) == record["sha256"],
                    f"Raw source checksum changed after preparation: {record['path']}")
    return dict(case_id=row["case_id"], window_id=row["window_id"], path=row["path"],
                sha256=sha256(path), bytes=path.stat().st_size, coverage_per_frame=coverage,
                native_probe_fit_rmse_m_s=metadata.get("native_probe_fit_rmse_m_s"),
                elapsed_seconds=metadata.get("elapsed_seconds"))


def eligibility_report(context):
    config = context["config"]
    # Dense arrays+five masks; compressed storage depends on actual velocities.
    per_window = FRAME_COUNT * 32**3 * (3 * 4 + 5)
    return dict(status="source_eligible_pending_grid_build", schema=SCHEMA, physical_group=GROUP,
                cohort_particles=25990, input_particles=12, heldout_probes=32, excluded_donor_particles=128,
                window_counts={key: len(value) for key, value in split_windows().items()},
                frames_to_grid=162 * FRAME_COUNT, embargo_windows=18, tail_frames=13,
                approximate_uncompressed_grid_and_masks_bytes=162 * per_window,
                config_sha256=context["config_sha256"], unit_scale=config["unit_scale"],
                full_processing_executed=False, training_ready=False,
                time_estimate="Measure the first completed windows; no unmeasured runtime promise.")


def prepare(args):
    started = time.monotonic()
    context = source_context(args.dataset_root, args.protocol_root, args.run_root)
    if args.eligibility_only:
        print(json.dumps(eligibility_report(context), indent=2))
        return
    require(bool(os.environ.get("SLURM_JOB_ID")), "Production FS preprocessing must run in an allocated Slurm job")
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "BUILD_CONFIG.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text())
            require(existing["config_sha256"] == context["config_sha256"], "Resume source/config changed; use a fresh derived root")
            require(args.resume, "Output already configured; use --resume to verify and continue")
        else:
            require(not any(output.glob("windows/*.npz")), "Unowned windows exist without BUILD_CONFIG")
            atomic_json(config_path, dict(config_sha256=context["config_sha256"], config=context["config"]))
        # During resumed verification no completion marker may claim readiness.
        (output / "COMPLETE.json").unlink(missing_ok=True)
        splits = window_rows()
        all_rows = [row for rows in splits.values() for row in rows]
        status = dict(schema="fs_transient_baseline_build_status_v1", status="building", config_sha256=context["config_sha256"],
                      completed={}, expected_windows=len(all_rows), started_at_utc=datetime.now(timezone.utc).isoformat(),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        previous_status_path = output / "BUILD_STATUS.json"
        previous_records = json.loads(previous_status_path.read_text()).get("completed", {}) if previous_status_path.exists() else {}
        pending = []
        for row in all_rows:
            path = output / row["path"]
            if path.exists():
                record = validate_window(path, row, context["config_sha256"])
                previous = previous_records.get(row["case_id"])
                require(previous is None or previous["sha256"] == record["sha256"], f"Prepared output checksum changed: {path}")
                status["completed"][row["case_id"]] = record
            else:
                pending.append(row)
        atomic_json(previous_status_path, status)
        try:
            if pending:
                with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                         initargs=(str(args.dataset_root), str(args.protocol_root), str(context["run_root"]),
                                                   context["config_sha256"], str(output))) as executor:
                    futures = {executor.submit(build_window, row): row for row in pending}
                    for future in as_completed(futures):
                        record = future.result()
                        status["completed"][record["case_id"]] = record
                        status["elapsed_seconds"] = time.monotonic() - started
                        atomic_json(previous_status_path, status)
                        print(json.dumps(dict(completed=len(status["completed"]), total=len(all_rows), **record)), flush=True)
            # Reopen every final archive and hash it before publishing the manifest.
            for row in all_rows:
                final_record = validate_window(output / row["path"], row, context["config_sha256"])
                require(final_record["sha256"] == status["completed"][row["case_id"]]["sha256"], "Window changed during final validation")
            # Also recheck source metadata and observed/probe identities after work.
            final_context = source_context(args.dataset_root, args.protocol_root, context["run_root"])
            require(final_context["config_sha256"] == context["config_sha256"], "Source data/code changed while preparing")
            manifest = dict(schema=SCHEMA, mode="within_case_temporal", status="validated",
                            flow_group_id=GROUP, independent_cases=1, splits=splits,
                            input_views=["xz"], num_particles=12, frames_per_window=20,
                            grid_shape_zyx=[32, 32, 32], unit_scale=context["config"]["unit_scale"],
                            known_boundary_information="none; prescribed masks and velocities are reference-only",
                            scientific_scope="single-case chronological conditional reconstruction; not cross-case generalization",
                            config_sha256=context["config_sha256"], source_config="BUILD_CONFIG.json",
                            source_release=str(args.dataset_root.resolve()),
                            embargo_window_ids=list(range(126, 135)) + list(range(153, 162)),
                            unused_tail_frame_indices=list(range(3600, 3613)),
                            time_units="seconds", coordinate_units="cells", velocity_units="cells/second")
            atomic_json(output / "manifest.json", manifest)
            report = dict(status="validated", windows=len(all_rows), frames=3240,
                          output_bytes=sum(record["bytes"] for record in status["completed"].values()),
                          elapsed_seconds=time.monotonic() - started,
                          worker_count=args.workers, source_eligibility=eligibility_report(context),
                          per_window=status["completed"], test_used_for_hyperparameter_selection=False)
            atomic_json(output / "build_report.json", report)
            status["status"] = "completed"
            status["elapsed_seconds"] = report["elapsed_seconds"]
            atomic_json(previous_status_path, status)
            atomic_json(output / "COMPLETE.json", dict(status="completed", verified_windows=162,
                        config_sha256=context["config_sha256"], manifest_sha256=sha256(output / "manifest.json"),
                        build_report_sha256=sha256(output / "build_report.json"),
                        completed_at_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(status="completed", manifest=str(output / "manifest.json"),
                                  elapsed_seconds=report["elapsed_seconds"], output_bytes=report["output_bytes"])), flush=True)
        except BaseException as error:
            status["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            status["error"] = str(error)
            atomic_json(previous_status_path, status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", "--release-root", dest="dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", "--run-dir", dest="run_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="Native reference gridding is CPU-only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eligibility-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    require(1 <= args.workers <= 32, "workers must be between 1 and 32")
    if args.smoke_only:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_fs_prepare.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    prepare(args)


if __name__ == "__main__":
    main()
