#!/usr/bin/env python3
"""Read-only preflight: valid experiment settings do not imply training readiness.

No solver, PyTorch import, GPU check, preprocessing, or scheduler submission runs.
The raw releases' completion records and manifest hashes are checked; this is
not a fresh bytewise checksum of every released velocity/trajectory array.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), message)


def cli_flags(path):
    """Extract declared argparse flags without executing research modules."""
    flags = set()
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flags.update(arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"))
    return flags


def validate_sources(root, settings):
    lock = read_json(root / "source_lock.json")
    require(lock["old_LBM_tensor_artifacts_allowed"] is False, "Old LBM artifacts must remain forbidden")
    paths = {}
    for item in lock["files"]:
        path = Path(item["path"])
        require(path not in paths, f"Duplicate source lock: {path}")
        require(sha256(path) == item["sha256"], f"Source hash mismatch: {path}")
        paths[path] = item
    config_file = next(path for path in paths if path.name == "config.py" and paths[path]["component"] == "baseline")
    # Loading config.py directly avoids baseline package __init__ and torch.
    spec = importlib.util.spec_from_file_location("_locked_baseline_config", config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline_count = own_count = 0
    for model in settings["models"]:
        config_path = root / model["config"]
        config = read_json(config_path)
        if model["role"] == "baseline":
            resolved = module.resolve_config(config_path)
            require(resolved == config, f"Incomplete baseline config: {model['id']}")
            require(resolved["views"] == [0], f"Baseline must use only adapter XZ view: {model['id']}")
            require(resolved["num_probes"] == settings["shared"]["heldout_probes"], "Baseline probe count differs")
            baseline_count += 1
            continue
        source_root = Path(config["source_root"])
        train_path = source_root / config["training_script"]
        sample_path = source_root / "scripts/sample_sparse_track_diffusion.py"
        require(train_path in paths and sample_path in paths, "Diffusion CLI is not source-locked")
        train_flags, sample_flags = cli_flags(train_path), cli_flags(sample_path)
        configured_train = set(config["training_arguments"]) | set(config["training_boolean_flags"]) | set(config["matrix_arguments"])
        require(configured_train <= train_flags, f"Unsupported training flags: {sorted(configured_train - train_flags)}")
        require(set(config["sampling_arguments"]) <= sample_flags, "Unsupported sampling flags")
        require(config["training_arguments"]["--prediction-type"] == "v", "Diffusion prediction must be v")
        require(config["training_arguments"]["--condition-encoder"] == "motion-residual-v2", "Repaired conditioning required")
        if model["id"] == "tensor_dit":
            require(config["tensor_artifact_required"] is True and "--tensor-artifact" in train_flags, "Tensor artifact contract missing")
        own_count += 1
    require("torch" not in sys.modules, "Validator must not import PyTorch")
    return {"locked_files": len(paths), "baseline_configs": baseline_count, "diffusion_configs": own_count}


def validate_jhtdb(config):
    raw = Path(config["root"])
    manifest = read_json(config["source_manifest"])
    complete, plan = read_json(raw / "COMPLETE.json"), read_json(raw / "plan.json")
    require(sha256(config["source_manifest"]) == config["source_manifest_sha256"], "JHTDB manifest changed")
    require(complete["status"] == config["release_state"] == "completed", "JHTDB release incomplete")
    require(complete["verified_fields"] == manifest["case_count"] == 3125, "JHTDB field count differs")
    # The downloader records the canonical JSON digest, not pretty file bytes.
    plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    require(plan_digest == complete["plan_sha256"] == manifest["plan_sha256"], "JHTDB plan hash differs")
    require(sha256(raw / "SHA256SUMS") == complete["checksum_list_sha256"], "JHTDB checksum index differs")
    plan_by_id = {case["case_id"]: case for case in plan["cases"]}
    require(len(plan_by_id) == 3125, "JHTDB plan IDs repeat")
    # One directory inventory avoids thousands of slow remote stat requests.
    # Integrity still comes from the completed release/index and pinned manifest;
    # this validator deliberately does not re-read every raw field's bytes.
    available_fields = {path.name for path in (raw / "fields").iterdir()}
    seen, blocks = set(), {}
    counts = {}
    for source_name, target_name in config["split"]["source_to_training_names"].items():
        cases = manifest["splits"][source_name]
        counts[target_name] = len(cases)
        require(len(cases) == config["split"]["counts"][target_name] == complete["split_counts"][source_name], "JHTDB split count differs")
        for case in cases:
            case_id = case["case_id"]
            require(case_id not in seen, f"JHTDB split overlap: {case_id}")
            seen.add(case_id)
            original = plan_by_id[case_id]
            for key in ("split", "macroblock_id", "flow_group_id", "origin_index_xyz"):
                require(case[key] == original[key], f"JHTDB original membership differs: {case_id}/{key}")
            require(case["split"] == source_name, "JHTDB split label differs")
            block = case["macroblock_id"]
            require(block not in blocks or blocks[block] == source_name, "JHTDB spatial block leaks across splits")
            blocks[block] = source_name
            field = raw / case["file"]
            require(field.parent == raw / "fields" and field.name in available_fields, f"Missing JHTDB case: {case_id}")
    require(seen == set(plan_by_id), "JHTDB plan/manifest memberships differ")
    generation = config["trajectory_generation"]
    # Deliberately compute the horizon bound only from train entries.
    speed = max(case["statistics"]["speed_max"] for case in manifest["splits"]["train"])
    spacing = plan["config"]["spacing"]
    close(generation["native_spacing"], spacing, "JHTDB spacing differs")
    close(generation["training_speed_bound"], speed, "JHTDB train-only speed bound differs")
    expected = 0.75 * generation["interior_initial_margin_native_cells"] * spacing / speed
    close(generation["horizon_native_time_units"], expected, "JHTDB train-only horizon formula differs")
    require(generation["boundary_mode"] == "terminate" and generation["allow_periodic_wrap"] is False, "JHTDB crop cannot wrap")
    require(generation["allow_clamp_as_valid_motion"] is False and generation["frame_count"] == 20, "JHTDB trajectory policy differs")
    require(config["split"]["preserve_original_membership"] is True, "JHTDB original split preservation required")
    require(config["geometry"]["known_boundary_mask"] == "all false", "JHTDB known-boundary input must be empty")
    return {"raw_release_verified": True, "split_counts": counts, "horizon_from_train": expected}


def validate_fs(root, config, shared):
    raw = Path(config["root"])
    manifest = read_json(config["source_manifest"])
    build = read_json(raw / "BUILD_STATUS.json")
    require(sha256(config["source_manifest"]) == config["source_manifest_sha256"], "FS manifest changed")
    require(manifest["status"] == build["status"] == config["release_state"] == "validated", "FS release not validated")
    require(build["independent_validation_passed"] is True, "FS independent validation missing")
    group = config["flow_group_id"]
    require(manifest["metadata"]["flow_group_id"] == group and manifest["counts"]["independent_cases"] == config["independent_cases"] == 1, "FS physical group differs")
    arrays = {}
    for key, info in manifest["files"].items():
        path = raw / info["path"]
        require(path.stat().st_size == info["bytes"], f"FS array size differs: {key}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        require(list(array.shape) == info["shape"] and str(array.dtype) == info["dtype"], f"FS array header differs: {key}")
        arrays[key] = array
    require(list(arrays["positions"].shape) == config["raw_track_shape"], "FS raw track shape differs")
    times = arrays["observation_times"]
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0), "FS native times invalid")
    close(float(times[0]), config["physical_time_range_s"][0], "FS initial time differs")
    close(float(times[-1]), config["physical_time_range_s"][1], "FS final time differs")
    ids = arrays["particle_ids"]
    require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(np.unique(ids)) == len(ids), "FS particle IDs invalid")
    published = read_json(raw / manifest["protocol"])
    require(published["flow_group_id"] == group and published["particle_selection_seed"] == shared["observation_seed"], "FS published ID protocol differs")
    require(published["observation_particle_counts"] == config["observations"]["counts"] == shared["particle_counts"], "FS observation counts differ")
    all_ids, previous = set(map(int, ids)), set()
    for count in shared["particle_counts"]:
        entry = published["fixed_subsets"][str(count)]
        chosen = np.load(raw / entry["path"], allow_pickle=False)
        require(chosen.ndim == 1 and np.issubdtype(chosen.dtype, np.integer), "FS conditioning ID type invalid")
        selected = set(map(int, chosen))
        require(chosen.tolist() == entry["ids"] and len(chosen) == len(selected) == count, "FS exact published IDs differ")
        require(previous <= selected <= all_ids, "FS subsets must be nested and in the native cohort")
        previous = selected
    probes = np.load(root / config["observations"]["probe_ids"], allow_pickle=False)
    probe_set = set(map(int, probes))
    require(probes.ndim == 1 and np.issubdtype(probes.dtype, np.integer), "FS probe ID type invalid")
    require(len(probes) == len(probe_set) == shared["heldout_probes"], "FS probe count differs")
    require(probe_set <= all_ids and not probe_set & previous, "FS probes overlap conditioning or leave the cohort")
    require(config["observations"]["epoch_resampling"] is False, "FS conditioning must retain published fixed IDs in every epoch")
    require(config["observations"]["preserve_native_absolute_times"] is True and config["observations"]["resample_full_process_to_20_times"] is False, "FS native time handling differs")
    require(config["target"]["reference_shared_across_N"] is True and len(previous | probe_set) == 128, "FS reference must share the fixed 128-ID exclusion union")
    require(config["target"]["masked_statistics"] is True, "FS statistics must exclude unsupported cells")
    require(config["target"]["velocity_kind"] == "native transport_velocity" and "transport_velocity_valid" in config["target"]["transport_validity_policy"], "FS native transport validity policy missing")
    require("No fallback" in config["target"]["transport_validity_policy"], "FS native transport may not silently fall back to momentum velocity")
    require(config["model_boundary_information"]["known_boundary_mask"].startswith("all false"), "FS known-boundary inputs must be empty")
    native_windows = read_json(raw / manifest["contiguous_windows"])
    split = read_json(root / config["split"]["protocol"])
    require(sha256(raw / manifest["contiguous_windows"]) == split["source_windows_sha256"], "FS published window hash differs")
    require(split["flow_group_id"] == group and split["independent_cases"] == 1 and split["no_fake_case_ids"] is True, "FS split must preserve physical group")
    require(split["frames_per_window"] == config["observations"]["frames_per_window"] == 20, "FS window length differs")
    order = ["train", "embargo_train_validation", "validation", "embargo_validation_test", "test"]
    expected_ranges = [(0, 126), (126, 135), (135, 153), (153, 162), (162, 180)]
    observed = set()
    for name, (start, stop) in zip(order, expected_ranges):
        block = split["windows"][name]
        require(block["window_ids"] == list(range(start, stop)), f"FS chronological block differs: {name}")
        require(block["window_count"] == stop - start and block["frame_start"] == 20 * start and block["frame_stop_exclusive"] == 20 * stop, "FS block/frame counts differ")
        require(not observed & set(block["window_ids"]), "FS temporal split overlap")
        observed.update(block["window_ids"])
        close(block["first_time_s"], float(times[20 * start]), "FS block initial time differs")
        close(block["last_time_s"], float(times[20 * stop - 1]), "FS block final time differs")
        for index in block["window_ids"]:
            window = native_windows["windows"][index]
            require(window["window_id"] == index and window["flow_group_id"] == group, "FS native group/window differs")
            require(window["start_index"] == 20 * index and window["stop_index_exclusive"] == 20 * (index + 1), "FS native window bounds differ")
    require(observed == set(range(180)) and split["tail_frames"] == list(range(3600, len(times))), "FS window coverage/tail differs")
    counts = config["split"]["window_counts"]
    require(counts == {"train": 126, "validation": 18, "test": 18, "embargo": 18}, "FS declared split counts differ")
    require(config["temporal_model"]["current_steady_trainer_supported"] is False and config["temporal_model"]["quasi_steady_fallback_enabled"] is False, "FS temporal adaptation must remain explicit")
    return {"raw_release_verified": True, "window_counts": counts, "physical_groups": 1, "nested_conditioning_ids_verified": True, "disjoint_probes": len(probes)}


def validate_matrix(root, settings):
    with (root / settings["matrix"]["file"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = set()
    outputs = set()
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "stage", "model", "N", "seed", "epochs"))
        require(key not in actual and row["output_dir"] not in outputs, "Matrix rows or output directories repeat")
        actual.add(key)
        outputs.add(row["output_dir"])
        require(row["status"] == "blocked_by_readiness_gates", "Matrix must remain blocked")
        require(Path(row["output_dir"]).is_relative_to(Path(settings["result_root"])), "Matrix output escapes result root")
        require(row["model"] != "pinn", "PINN is not justified for this main matrix")
        require(row["model"] != "mean" or row["seed"] == "", "Mean must not have fake training seeds")
    expected = set()
    counts, seeds = settings["shared"]["particle_counts"], settings["shared"]["training_seeds"]
    learned = settings["main_learned_models"]
    require(set(learned) == {"unet", "fno", "cnn", "deeponet", "unet_diffusion", "dit3d", "tensor_dit"}, "Main learned model set differs")
    require(next(m for m in settings["models"] if m["id"] == "pinn")["enabled"] is False, "PINN must remain excluded")
    require(settings["proposed_model"] == "tensor_dit", "Proposed model differs")
    for dataset in settings["datasets"]:
        for model in settings["pilot"]["models"]:
            expected.add((dataset, "pilot", model, str(settings["pilot"]["N"]), str(settings["pilot"]["seed"]), str(settings["pilot"]["epochs"])))
        for model, count, seed in itertools.product(learned, counts, seeds):
            expected.add((dataset, "train", model, str(count), str(seed), str(settings["shared"]["epochs"])))
        for model in ("mean", "pod"):
            expected.add((dataset, "fit_train_statistics", model, "", "", ""))
        for model in [*learned, "mean", "pod", "rbf"]:
            for count, seed in itertools.product(counts, [""] if model == "mean" else seeds):
                epoch = str(settings["shared"]["epochs"]) if model in learned else ""
                expected.add((dataset, "evaluate", model, str(count), str(seed), epoch))
    require(actual == expected, f"Matrix differs: missing={len(expected-actual)}, extra={len(actual-expected)}")
    declared = settings["matrix"]
    by_stage = Counter(row["stage"] for row in rows)
    require(len(rows) == declared["all_rows"] == 402, "Total matrix count differs")
    require(by_stage["train"] == declared["training_runs_total"] == 2 * declared["training_runs_per_dataset"], "Training matrix count differs")
    require(by_stage["pilot"] == declared["pilot_runs_total"] and by_stage["fit_train_statistics"] == 2 * declared["statistics_fits_per_dataset"] and by_stage["evaluate"] == 2 * declared["evaluation_settings_per_dataset"], "Matrix stage counts differ")
    return {"rows": len(rows), **dict(by_stage)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-ready", action="store_true", help="Exit 2 when known adapters or GPU pilot are unavailable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = {"settings_valid": False, "data_training_ready": False, "gpu_runtime_verified": False,
              "verification_scope": "Read-only settings, locked sources, raw completion/manifest metadata, array headers and protocols; not a fresh full-array checksum or GPU check."}
    try:
        settings = read_json(root / "settings.json")
        require(settings["submission_enabled"] is False and settings["jobs_submitted"] == [], "Submission must remain disabled with no submitted jobs")
        require(settings["runtime"]["gpu_runtime_verified"] is False, "GPU runtime has not been checked by this settings preparation")
        require(settings["runtime"]["login_node_training_allowed"] is False, "Login-node training must remain disabled")
        require(settings["tensor_space"]["fit_statistics_train_only"] is True and settings["tensor_space"]["reuse_old_lbm_rank8"] is False, "Train-only dataset-specific tensor fitting required")
        require(settings["tensor_space"]["fit_split"] == "train" and settings["tensor_space"]["selection_split"] == "validation", "Tensor split policy differs")
        datasets = {name: read_json(root / path) for name, path in settings["datasets"].items()}
        gates = {}
        for name, config in datasets.items():
            require(config["training_ready"] is False, f"{name} cannot yet claim training readiness")
            declared = settings["readiness_gates"][name]
            require(set(declared) == set(config["readiness_gates"]), f"Readiness gate list differs: {name}")
            require("adapted_source_locked_and_reviewed" in declared, "Adapted scientific source must be locked and reviewed before readiness")
            gates[name] = {}
            for key, gate in declared.items():
                require(gate["passed"] is False and gate["evidence"] is None, f"Unverified gate must remain false: {name}/{key}")
                gates[name][key] = False
        result["readiness_gates"] = gates
        result["sources"] = validate_sources(root, settings)
        result["datasets"] = {"jhtdb": validate_jhtdb(datasets["jhtdb"]), "fs_sph": validate_fs(root, datasets["fs_sph"], settings["shared"])}
        result["matrix"] = validate_matrix(root, settings)
        result["settings_valid"] = True
        result["submission_enabled"] = settings["submission_enabled"]
        result["unmet_gate_count"] = sum(not passed for values in gates.values() for passed in values.values())
        result["data_training_ready"] = all(all(values.values()) for values in gates.values())
    except (OSError, ValueError, KeyError, TypeError, StopIteration, SyntaxError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if not result["settings_valid"]:
        return 1
    return 2 if args.require_ready and not result["data_training_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
