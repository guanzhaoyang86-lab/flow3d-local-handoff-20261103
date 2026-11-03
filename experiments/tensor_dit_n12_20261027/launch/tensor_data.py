#!/usr/bin/env python3
"""Observation-only dataset readers and train-fitted aligned Tucker artifacts.

Production preparation runs inside Slurm, never regenerates the baseline data,
and never opens test archives. FS uses train-voxel-mean imputation ONLY outside
supervised material support. Every training window has equal mass; native-time
trapezoid weights distribute that mass over its frames, including anchor fitting.
"""
from __future__ import annotations

import argparse
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
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/src"))
from flow_observation.tensor_space import TensorSpaceCodec, spatial_tucker, align_tucker
from flow_observation.sparse_dataset import _stable_seed, _particle_count
from flow_observation.models.trajectory_encoder import TrackSetEncoder

FORMAT = "tensor_n12_artifact_v1"
DATASETS = ("jhtdb", "fs_sph")
MINIMUM_LATENT_STD = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fs_runtime():
    name = "tensor_fs_baseline_data_dependency"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "launch/fs_baseline.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def read_manifest(manifest, dataset):
    require(dataset in DATASETS, "Unknown dataset")
    path = Path(manifest).resolve(); data = json.loads(path.read_text())
    if dataset == "fs_sph":
        require(data.get("schema") == "fs_transient_baseline_manifest_v1"
                and data.get("mode") == "within_case_temporal", "FS needs explicit within-case temporal manifest")
    else:
        require(data.get("mode") == "scientific", "JHTDB needs the guarded spatial manifest")
    splits = {}; ids = set(); paths = set(); groups = {}
    for split in ("train", "validation", "test"):
        entries = data.get("splits", {}).get(split)
        require(isinstance(entries, list) and entries, f"Nonempty {split} required")
        splits[split] = []
        for entry in entries:
            row = dict(entry); target = (path.parent / row["path"]).resolve()
            require(target.is_relative_to(path.parent), "Dataset path escapes adapted release")
            require(target.is_file(), f"Missing adapted case: {target}")
            require(row["case_id"] not in ids and str(target) not in paths, "Duplicate case/path across splits")
            require(Path(row["case_id"]).name == row["case_id"], "Unsafe case ID")
            ids.add(row["case_id"]); paths.add(str(target))
            row.update(path=str(target), _split=split)
            splits[split].append(row)
        groups[split] = {row["flow_group_id"] for row in splits[split]}
    if dataset == "fs_sph":
        require(len(set.union(*groups.values())) == 1, "FS protocol preserves one physical group")
        if all("window_id" in row for rows in splits.values() for row in rows):
            edges = [(min(row["window_id"] for row in splits[s]), max(row["window_id"] for row in splits[s])) for s in splits]
            require(edges[0][1] < edges[1][0] and edges[1][1] < edges[2][0], "Temporal blocks overlap or are unordered")
    else:
        require(not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                     or groups["validation"] & groups["test"]), "Spatial macroblock leakage")
    return data, splits


def decode_metadata(archive):
    raw = archive["metadata"].item()
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def load_observation(row, dataset, seed=47, epoch=0):
    """Never accesses reference velocity, support, probe coordinates or hidden Y."""
    if dataset == "fs_sph":
        return fs_runtime().load_observation(row)
    require(dataset == "jhtdb", "Unknown dataset")
    split = row.get("_split", row.get("split", "train"))
    with np.load(row["path"], allow_pickle=False) as archive:
        info = decode_metadata(archive)
        stored = archive["trajectories_2d"]
        generator = torch.Generator().manual_seed(_stable_seed(seed, 0, split, row["case_id"], 0))
        source = int(torch.randint(stored.shape[0], (1,), generator=generator))
        count = _particle_count(archive, source)
        require(count >= 44, "Need N12 plus32 disjoint fixed probes")
        permutation = torch.randperm(count, generator=generator)
        probes, candidates = permutation[-32:], permutation[:-32]
        if split == "train" and epoch:
            rng = torch.Generator().manual_seed(_stable_seed(seed, epoch, split, row["case_id"], 0, "observations_only"))
            candidates = candidates[torch.randperm(len(candidates), generator=rng)]
        selected = candidates[:12]
        # NumPy mixes basic/advanced axes; explicit view selection preserves V,N,T,2.
        tracks = torch.from_numpy(np.array(stored[source][:1][:, selected.numpy()], copy=True)).float()
        mask = torch.from_numpy(np.array(archive["observation_mask"][source][:1][:, selected.numpy()], copy=True)).bool()
        times = torch.from_numpy(np.array(archive["observation_times"], copy=True)).double()
        projection = torch.from_numpy(np.array(archive["projection_matrix"][:1], copy=True)).float()
        bounds = torch.from_numpy(np.array(archive["domain_bounds"], copy=True)).float()
        # JHTDB geometry is declared entirely fluid; geometry shape is public.
        solid = np.array(archive["solid_mask"], copy=True)
        require(not solid.any(), "Unexpected JHTDB solid/boundary information")
        shape = tuple(solid.shape)
    scale = torch.tensor(info.get("physical_to_cell_scale_xyz", info.get("unit_scale", [])), dtype=torch.float32)
    require(scale.shape == (3,) and bool(torch.isfinite(scale).all() & (scale > 0).all()), "Missing/invalid physical unit transform")
    require(tracks.shape[:2] == (1, 12) and tracks.shape[-1] == 2 and mask.shape == tracks.shape[:-1], "Invalid XZ tracks")
    require(projection.shape == (1, 2, 3) and torch.equal(projection, torch.tensor([[[1., 0., 0.], [0., 0., 1.]]])), "Only XZ camera allowed")
    frame_weights(times)
    require(times.numel() == tracks.shape[-2] and bool(torch.isfinite(tracks[mask]).all()), "Track/time data invalid")
    return dict(tracks=torch.where(mask[..., None], tracks, 0), mask=mask, times=times, projection=projection,
                bounds=bounds, unit_scale=scale, shape=shape, particle_indices=selected,
                probe_particle_indices=probes, source_sample_index=source)


def load_reference(row, dataset):
    if dataset == "fs_sph":
        return fs_runtime().load_reference(row)
    require(dataset == "jhtdb", "Unknown dataset")
    with np.load(row["path"], allow_pickle=False) as archive:
        field = torch.from_numpy(np.array(archive["flow_field"], copy=True)).float()
        mask = ~torch.from_numpy(np.array(archive["solid_mask"], copy=True)).bool()[None]
        time0 = float(archive["observation_times"][0])
    require(field.ndim == 5 and field.shape[:2] == (1, 3) and mask.shape == (1, *field.shape[2:])
            and bool(torch.isfinite(field).all()) and bool(mask.all()), "Invalid static JHTDB field")
    return field, mask, torch.tensor([time0], dtype=torch.float64)


def frame_weights(times):
    times = torch.as_tensor(times, dtype=torch.float64)
    require(times.ndim == 1 and len(times) >= 1 and bool(torch.isfinite(times).all()), "Invalid reference times")
    if len(times) == 1:
        return torch.ones(1, dtype=torch.float32)
    require(bool((times.diff() > 0).all()), "Times must increase")
    dt = times.diff(); weights = torch.cat((dt[:1] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1:] / 2))
    return (weights / weights.sum()).float()


def normalize(field, statistics):
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float32, device=field.device).view(1, 3, 1, 1, 1)
    std = torch.as_tensor(statistics["std"], dtype=torch.float32, device=field.device).view(1, 3, 1, 1, 1)
    return (field - mean) / std


def denormalize(field, statistics):
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float32, device=field.device).view(1, 3, 1, 1, 1)
    std = torch.as_tensor(statistics["std"], dtype=torch.float32, device=field.device).view(1, 3, 1, 1, 1)
    return field * std + mean


def impute_normalized(field, support, statistics):
    fill = statistics["mean_field"].to(field.device)[None]
    return normalize(torch.where(support[:, None], field, fill), statistics)


def fit_statistics(rows, dataset):
    require(all(row.get("_split", "train") == "train" for row in rows), "Statistics must fit train only")
    total = torch.zeros(3, dtype=torch.float64); squares = total.clone(); mass = 0.
    voxel_sum = voxel_mass = None
    motion_squares = torch.zeros(4, dtype=torch.float64); motion_count = torch.zeros(4, dtype=torch.float64)
    for row in rows:
        field, mask, times = load_reference(row, dataset)
        shape = tuple(field.shape[2:])
        if voxel_sum is None:
            voxel_sum = torch.zeros((3, *shape), dtype=torch.float64); voxel_mass = torch.zeros(shape, dtype=torch.float64)
        require(tuple(voxel_mass.shape) == shape, "Spatial shapes differ")
        weights = mask.double() * frame_weights(times).double()[:, None, None, None]
        safe = torch.where(mask[:, None], field, 0).double()
        total += (safe * weights[:, None]).sum((0, 2, 3, 4))
        squares += (safe.square() * weights[:, None]).sum((0, 2, 3, 4)); mass += float(weights.sum())
        voxel_sum += (safe * weights[:, None]).sum(0); voxel_mass += weights.sum(0)
        obs = load_observation(row, dataset)
        features, _ = TrackSetEncoder.token_features(obs["tracks"][None], obs["mask"][None], obs["projection"], obs["times"].float(), obs["bounds"])
        pair = obs["mask"][..., 1:] & obs["mask"][..., :-1]
        active = torch.stack((obs["mask"], obs["mask"], F.pad(pair, (1, 0)), F.pad(pair, (1, 0))), -1)[None]
        motion_squares += torch.where(active, features[..., 2:6].double().square(), 0).reshape(-1, 4).sum(0)
        motion_count += active.reshape(-1, 4).sum(0)
    require(mass > 0, "No supported training values")
    mean = total / mass; std = (squares / mass - mean.square()).clamp_min(1e-12).sqrt()
    fill = voxel_sum / voxel_mass.clamp_min(1e-12)[None]
    fill = torch.where((voxel_mass > 0)[None], fill, mean[:, None, None, None]).float()
    return dict(mean=mean.float().tolist(), std=std.float().tolist(), mean_field=fill,
                train_support=voxel_mass > 0, shape=list(voxel_mass.shape), fit_split="train",
                motion_scales=(motion_squares / motion_count.clamp_min(1)).sqrt().clamp_min(1e-6).float().tolist(),
                weighted_mass=mass, imputation="train_only_per_voxel_mean; global_train_component_mean_if_never_observed")


def weighted_medoid_anchors(factors, weights):
    require(len(factors) and len(weights) == len(factors), "Empty/mismatched train factors")
    weights = torch.as_tensor(weights, dtype=torch.float64)
    require(bool((weights >= 0).all()) and bool(weights.sum() > 0), "Invalid medoid frame weights")
    weights = weights / weights.sum(); anchors = []; indices = []
    for axis in range(3):
        stack = torch.stack([triplet[axis].detach().cpu().double() for triplet in factors])
        projectors = stack @ stack.transpose(-1, -2)
        centroid = torch.einsum("f,fij->ij", weights, projectors)
        distances = (projectors - centroid).square().sum((-1, -2))
        index = int(distances.argmin()); anchors.append(stack[index].float()); indices.append(index)
    return anchors, indices


def source_provenance(manifest, splits):
    # No test archive is opened, hashed or used for rank/statistics fitting.
    return {split: {row["case_id"]: dict(path=row["path"], sha256=sha256(row["path"]),
                                        flow_group_id=row["flow_group_id"]) for row in splits[split]}
            for split in ("train", "validation")}


def primitives(row, dataset, statistics, rank, hooi, device, cache_path, config_sha, source_sha, resume):
    if cache_path.exists():
        require(resume, "Primitive cache exists; explicit resume required")
        state = torch.load(cache_path, map_location="cpu", weights_only=True)
        require(state["config_sha256"] == config_sha and state["source_sha256"] == source_sha
                and state["rank"] == rank, "Primitive source/config changed")
        return state
    field, support, times = load_reference(row, dataset)
    target = impute_normalized(field, support, statistics).to(device)
    cores = []; factors = [[], [], []]
    for frame in target:
        core, triplet = spatial_tucker(frame, rank, hooi_iterations=hooi)
        cores.append(core.cpu())
        for axis in range(3): factors[axis].append(triplet[axis].cpu())
    state = dict(config_sha256=config_sha, source_sha256=source_sha, rank=rank,
                 core=torch.stack(cores), factors=[torch.stack(values) for values in factors],
                 frame_weights=frame_weights(times), case_id=row["case_id"])
    atomic_save(cache_path, state)
    return state


def reconstruction_score(decoded, field, support, times, unit_scale):
    scale = unit_scale.to(decoded.device).view(1, 3, 1, 1, 1)
    valid = support[:, None].expand_as(field)
    error = torch.where(valid, (decoded - field) / scale, 0).double()
    truth = torch.where(valid, field / scale, 0).double()
    weights = frame_weights(times).to(decoded.device).double()
    errors = error.square().sum((1, 2, 3, 4)); energies = truth.square().sum((1, 2, 3, 4))
    require(bool(torch.isfinite(errors).all() & torch.isfinite(energies).all()), "Nonfinite reconstruction score")
    energy = float((weights * energies).sum()); numerator = float((weights * errors).sum())
    if energy <= 1e-24:
        return 0.0 if numerator <= 1e-24 else float("inf")
    return (numerator / energy) ** .5


def build_rank(splits, dataset, statistics, rank, hooi, device, output, config_sha, provenance, resume):
    rank_root = output / f"rank_{rank}"; states = {"train": {}, "validation": {}}
    train_factors = []; train_weights = []; pointers = []
    for split in states:
        for row in splits[split]:
            state = primitives(row, dataset, statistics, rank, hooi, device,
                               rank_root / "primitives" / split / (row["case_id"] + ".pt"), config_sha,
                               provenance[split][row["case_id"]]["sha256"], resume)
            states[split][row["case_id"]] = state
            if split == "train":
                for index in range(len(state["core"])):
                    train_factors.append([state["factors"][axis][index] for axis in range(3)])
                    train_weights.append(float(state["frame_weights"][index]))
                    pointers.append(dict(case_id=row["case_id"], frame_index=index))
    anchors, indices = weighted_medoid_anchors(train_factors, train_weights)
    codec = TensorSpaceCodec(statistics["shape"], rank, anchors, hooi_iterations=hooi).to(device)
    raw = {"train": {}, "validation": {}}; weights_cache = {"train": {}, "validation": {}}
    total = torch.zeros(codec.latent_dim, dtype=torch.float64); square = total.clone(); mass = 0.
    minimum_overlap = 1.
    for split, mapping in states.items():
        for case_id, state in mapping.items():
            encoded = []
            for index, core in enumerate(state["core"]):
                core, factors, overlaps = align_tucker(core.to(device), [value[index].to(device) for value in state["factors"]], codec.anchors)
                minimum_overlap = min(minimum_overlap, min(overlaps))
                encoded.append(codec.pack(core, factors).cpu())
            values = torch.stack(encoded); raw[split][case_id] = values
            weights = state["frame_weights"].double(); weights_cache[split][case_id] = weights.float()
            if split == "train":
                total += (values.double() * weights[:, None]).sum(0)
                square += (values.double().square() * weights[:, None]).sum(0); mass += float(weights.sum())
    mean = total / mass; std = (square / mass - mean.square()).clamp_min(0).sqrt().clamp_min(MINIMUM_LATENT_STD)
    codec.latent_mean.copy_(mean.float().to(device)); codec.latent_std.copy_(std.float().to(device))
    latents = {split: {key: ((value - mean.float()) / std.float()).float() for key, value in values.items()} for split, values in raw.items()}
    scores = {}
    for row in splits["validation"]:
        field, support, times = load_reference(row, dataset)
        observation = load_observation(row, dataset)
        with torch.no_grad(): decoded = denormalize(codec.decode(latents["validation"][row["case_id"]].to(device)), statistics)
        score = reconstruction_score(decoded, field.to(device), support.to(device), times, observation["unit_scale"])
        require(np.isfinite(score), "Undefined/failed validation rank reconstruction")
        scores[row["case_id"]] = score
    mean_score = sum(scores.values()) / len(scores); maximum = max(scores.values())
    report = dict(rank=rank, validation_mean_relative_l2=mean_score, validation_max_relative_l2=maximum,
                  per_window_physical_relative_l2=scores, selection_split="validation", test_fields_read=False,
                  score_definition="Per-window time-weighted supported physical-velocity relativeL2; mean/max across windows",
                  passed=mean_score <= .05 and maximum <= .20,
                  anchor_train_frames=[pointers[index] for index in indices], minimum_factor_anchor_singular_value=minimum_overlap)
    artifact = dict(format=FORMAT, dataset=dataset, rank=rank, latent_dim=codec.latent_dim,
                    manifest_sha256=None,  # Bound by the caller before any artifact is published.
                    statistics=statistics, codec=codec.to_artifact(), latents=latents,
                    frame_weights=weights_cache, latent_mean=codec.latent_mean.detach().cpu(),
                    latent_std=codec.latent_std.detach().cpu(), rank_report=report,
                    provenance=dict(fitted_split="train", rank_selection_split="validation", test_fields_read=False,
                                    frame_weighting="Equal windows, native trapezoidal frame weights",
                                    anchor_train_frames=report["anchor_train_frames"], source_files=provenance,
                                    config_sha256=config_sha, minimum_latent_std=MINIMUM_LATENT_STD))
    return artifact, report


def load_artifact(path, manifest, dataset, verify_files=True):
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(value.get("format") == FORMAT and value.get("dataset") == dataset, "Wrong tensor artifact/dataset")
    require(value.get("manifest_sha256") == sha256(manifest), "Artifact manifest changed")
    require(value["statistics"]["fit_split"] == "train" and value["provenance"]["fitted_split"] == "train"
            and value["provenance"]["rank_selection_split"] == "validation"
            and value["rank_report"]["passed"], "Invalid train-only fit or rank gate")
    _, splits = read_manifest(manifest, dataset)
    codec = TensorSpaceCodec.from_artifact(value["codec"])
    require(codec.rank == value["rank"] and codec.latent_dim == value["latent_dim"], "Codec shape mismatch")
    for split in ("train", "validation"):
        require(set(value["latents"][split]) == {row["case_id"] for row in splits[split]}, "Latent case membership changed")
        for row in splits[split]:
            latent = value["latents"][split][row["case_id"]]
            require(latent.ndim == 2 and latent.shape[1] == codec.latent_dim and bool(torch.isfinite(latent).all()), "Invalid cached latent")
            weights = value["frame_weights"][split][row["case_id"]]
            require(weights.shape == (len(latent),) and bool((weights > 0).all())
                    and bool(torch.isclose(weights.sum(), torch.tensor(1.))), "Invalid native frame weights")
            if verify_files:
                require(sha256(row["path"]) == value["provenance"]["source_files"][split][row["case_id"]]["sha256"], "Artifact source archive changed")
    return value


def prepare_artifacts(manifest, dataset, output_dir, *, ranks=(4, 8, 12, 16), hooi_iterations=2, device="cpu", resume=False):
    manifest = Path(manifest).resolve(); output = Path(output_dir).resolve(); _, splits = read_manifest(manifest, dataset)
    require(list(ranks) == sorted(set(ranks)) and ranks and min(ranks) >= 1, "Ranks must be unique ascending positive integers")
    require(hooi_iterations >= 0, "HOOI iterations must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = source_provenance(manifest, splits)
        config = dict(format=FORMAT, dataset=dataset, manifest_sha256=sha256(manifest), ranks=list(ranks),
                      hooi_iterations=hooi_iterations, minimum_latent_std=MINIMUM_LATENT_STD, source_files=provenance,
                      source_code_sha256={str(path): sha256(path) for path in
                                          (Path(__file__).resolve(), ROOT / "launch/fs_baseline.py",
                                           ROOT / "source/src/flow_observation/tensor_space.py")})
        config_sha = stable_hash(config); config_path = output / "PREPARATION.json"
        if config_path.exists():
            require(resume and json.loads(config_path.read_text())["config_sha256"] == config_sha,
                    "Existing preparation differs or explicit resume missing")
        else:
            atomic_json(config_path, dict(config=config, config_sha256=config_sha))
        complete = output / "COMPLETE.json"
        if complete.exists():
            marker = json.loads(complete.read_text())
            require(marker["selected_artifact_sha256"] == sha256(output / "selected.pt"), "Selected artifact hash changed")
            load_artifact(output / "selected.pt", manifest, dataset)
            return marker
        stats_path = output / "statistics.pt"
        if stats_path.exists():
            saved = torch.load(stats_path, map_location="cpu", weights_only=True)
            require(saved["config_sha256"] == config_sha, "Statistics source/config changed")
            statistics = saved["statistics"]
        else:
            statistics = fit_statistics(splits["train"], dataset)
            atomic_save(stats_path, dict(config_sha256=config_sha, statistics=statistics))
        require(max(ranks) <= min(statistics["shape"]), "Candidate rank exceeds spatial dimensions")
        reports = {}; started = time.monotonic()
        for rank in ranks:
            rank_path = output / f"rank_{rank}" / "artifact.pt"
            if rank_path.exists():
                require(resume, "Rank artifact exists; explicit resume required")
                artifact = torch.load(rank_path, map_location="cpu", weights_only=True)
                require(artifact["provenance"]["config_sha256"] == config_sha, "Rank source/config changed")
                report = artifact["rank_report"]
            else:
                artifact, report = build_rank(splits, dataset, statistics, rank, hooi_iterations, torch.device(device),
                                               output, config_sha, provenance, resume)
                artifact["manifest_sha256"] = config["manifest_sha256"]
                atomic_save(rank_path, artifact)
            reports[str(rank)] = report
            atomic_json(output / "rank_report.json", dict(dataset=dataset, ranks=reports, test_fields_read=False,
                        candidate_ranks=list(ranks), elapsed_seconds=time.monotonic() - started))
            print(json.dumps(dict(rank=rank, mean_relative_l2=report["validation_mean_relative_l2"],
                                  maximum_relative_l2=report["validation_max_relative_l2"], passed=report["passed"])), flush=True)
            if report["passed"]:
                # Re-hash every train/validation source after the long GPU decomposition.
                require(source_provenance(manifest, splits) == provenance, "Source fields changed during Tucker fit")
                atomic_save(output / "selected.pt", artifact)
                load_artifact(output / "selected.pt", manifest, dataset)
                marker = dict(status="completed", manifest_sha256=config["manifest_sha256"], selected_rank=rank,
                              selected_artifact_sha256=sha256(output / "selected.pt"), config_sha256=config_sha,
                              train_frames=sum(len(value) for value in artifact["latents"]["train"].values()),
                              validation_frames=sum(len(value) for value in artifact["latents"]["validation"].values()),
                              test_fields_read=False, elapsed_seconds=time.monotonic() - started)
                atomic_json(complete, marker)
                return marker
        atomic_json(output / "FAILED_RANK_GATE.json", dict(status="failed", reason="No rank meets validation mean<=0.05 and maximum<=0.20",
                    manifest_sha256=config["manifest_sha256"], ranks=reports, test_fields_read=False))
        raise ValueError("No candidate Tucker rank passes validation; no COMPLETE marker published")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--ranks", default="4,8,12,16")
    prepare.add_argument("--hooi-iterations", type=int, default=2)
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    prepare.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    require(bool(os.environ.get("SLURM_JOB_ID")), "Production Tucker preparation requires a Slurm allocation")
    _, splits = read_manifest(args.manifest, args.dataset)
    expected = (2500, 312) if args.dataset == "jhtdb" else (126, 18)
    require((len(splits["train"]), len(splits["validation"])) == expected, "Unexpected production train/validation case counts")
    marker = prepare_artifacts(args.manifest, args.dataset, args.output_dir, ranks=[int(value) for value in args.ranks.split(",")],
                               hooi_iterations=args.hooi_iterations, device=args.device, resume=args.resume)
    print(json.dumps(marker), flush=True)


if __name__ == "__main__":
    main()
