#!/usr/bin/env python3
"""Time-conditioned FS-SPH baseline adaptations, not unchanged paper replicas.

Each learned method maps the same twelve XZ tracks to spatial fields queried at
native absolute times. One trapezoid-weighted random frame per training window
gives an unbiased estimate of the window's time-averaged masked frame MSE.
Validation evaluates every frame. The shared residual time adapter retains
absolute seconds, window start/duration, relative query time, and cell/m scales.

Diffusion is partially supervised: missing target latents are independent unit
Gaussians, so their forward marginals remain N(0,1). Supported v/epsilon targets
give the physical loss. A separate weight-one analytic nuisance score loss
targets v=0 or epsilon=sqrt(1-alpha_bar)*x_t at unsupported cells, constraining
their stationary Gaussian reverse marginal. This is an artificial latent prior,
never a zero physical-velocity target. The reference support mask is never a
denoiser input or inference-time projection.

POD uses train-only masked mean imputation and randomized spatial POD; inverse
POD/RBF coefficients vary at every native frame and are fit only to XZ tracks.
All results are held-out times of ONE material process, not independent flows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "src"))
from flow_baselines.config import resolve_config
from flow_baselines.models import build_decoder
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.unet3d import ConditionalUNet3D
from flow_observation.diffusion import GaussianDiffusion
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.advection import advect_particles
from flow_observation.interpolation import sample_velocity
from flow_baselines.linear import RBFField

LEARNED = ("unet", "fno", "cnn", "deeponet", "unet_diffusion")
MODELS = ("mean", "pod", *LEARNED, "rbf")
FORMAT = "fs-transient-baseline-v1"
SCOPE = "one_physical_material_case_chronological_holdout"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temp); temp.replace(path)


def read_manifest(path):
    path = Path(path).resolve(); data = json.loads(path.read_text())
    schema = data.get("schema", data.get("format"))
    if schema != "fs_transient_baseline_manifest_v1" or data.get("mode") != "within_case_temporal":
        raise ValueError("FS manifest must explicitly declare within_case_temporal and the transient schema")
    source = data.get("splits", {}); splits = {}
    ids, paths, groups = set(), set(), set()
    for split in ("train", "validation", "test"):
        rows = source.get(split, source.get("val") if split == "validation" else None)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Nonempty {split} split required")
        splits[split] = []
        for original in rows:
            row = dict(original)
            row["path"] = str((path.parent / row["path"]).resolve())
            if row["case_id"] in ids or row["path"] in paths:
                raise ValueError("Duplicate/overlapping temporal window")
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(row["path"])
            ids.add(row["case_id"]); paths.add(row["path"]); groups.add(row["flow_group_id"])
            splits[split].append(row)
    if len(groups) != 1:
        raise ValueError("This protocol requires exactly one declared physical case")
    # Actual window order/embargo is part of the externally frozen manifest.
    if all("window_id" in r for rows in splits.values() for r in rows):
        bounds = [(min(int(r["window_id"]) for r in splits[s]),
                   max(int(r["window_id"]) for r in splits[s])) for s in splits]
        if not (bounds[0][1] < bounds[1][0] and bounds[1][1] < bounds[2][0]):
            raise ValueError("Temporal train/validation/test blocks must be ordered and disjoint")
    return data, splits


def metadata(archive):
    if "metadata" not in archive:
        return {}
    value = archive["metadata"].item()
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))


def frame_weights(times):
    times = torch.as_tensor(times, dtype=torch.float64)
    if times.ndim != 1 or len(times) < 2 or not bool(torch.isfinite(times).all()) or not bool((times.diff() > 0).all()):
        raise ValueError("Need at least two strictly increasing native physical times")
    dt = times.diff(); w = torch.cat((dt[:1] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1:] / 2))
    return (w / w.sum()).float()


def load_observation(row, device="cpu"):
    """This reader does not access flow_field, support_mask, or probe truth."""
    with np.load(row["path"], allow_pickle=False) as a:
        info = metadata(a)
        tracks = torch.from_numpy(np.array(a["trajectories_2d"][0], copy=True)).float()
        mask = torch.from_numpy(np.array(a["observation_mask"][0], copy=True)).bool()
        times = torch.from_numpy(np.array(a["observation_times"], copy=True)).double()
        projection = torch.from_numpy(np.array(a["projection_matrix"], copy=True)).float()
        bounds = torch.from_numpy(np.array(a["domain_bounds"], copy=True)).float()
    if tracks.ndim != 4 or tracks.shape[:2] != (1, 12) or tracks.shape[-1] != 2 or tuple(mask.shape) != tuple(tracks.shape[:-1]):
        raise ValueError("Exactly twelve published XZ tracks [1,12,T,2] required")
    expected = torch.tensor([[[1., 0., 0.], [0., 0., 1.]]])
    if projection.shape != (1, 2, 3) or not torch.allclose(projection, expected):
        raise ValueError("Only the declared XZ orthographic camera is supported")
    if times.numel() != tracks.shape[-2] or not bool(mask[..., 0].all()):
        raise ValueError("Track/time mismatch or missing initial observations")
    frame_weights(times)
    if not bool(torch.isfinite(tracks[mask]).all()) or bounds.shape != (3, 2) or not bool((bounds[:, 1] > bounds[:, 0]).all()):
        raise ValueError("Invalid observed coordinates or domain")
    if "unit_scale" not in info and "unit_scale_xyz" not in info:
        raise ValueError("Explicit physical-to-cell XYZ unit scales are required")
    scale = torch.tensor(info.get("unit_scale", info.get("unit_scale_xyz")), dtype=torch.float32)
    if scale.shape != (3,) or not bool(torch.isfinite(scale).all() & (scale > 0).all()):
        raise ValueError("metadata.unit_scale must be positive XYZ cell/m scales")
    result = {"tracks": torch.where(mask[..., None], tracks, 0), "mask": mask,
              "times": times, "projection": projection, "bounds": bounds, "unit_scale": scale,
              "shape": tuple(info.get("grid_shape_zyx", (32, 32, 32)))}
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}


def load_reference(row, device="cpu"):
    with np.load(row["path"], allow_pickle=False) as a:
        field = torch.from_numpy(np.array(a["flow_field"], copy=True)).float()
        mask = torch.from_numpy(np.array(a["support_mask"], copy=True)).bool()
        times = torch.from_numpy(np.array(a["velocity_times"], copy=True)).double()
    if field.ndim != 5 or field.shape[1] != 3 or tuple(mask.shape) != (field.shape[0], *field.shape[2:]):
        raise ValueError("Transient fields/support must be [T,3,D,H,W]/[T,D,H,W]")
    if len(times) != len(field) or not bool(mask.flatten(1).any(1).all()):
        raise ValueError("Every reference frame needs supported material samples")
    expanded = mask[:, None].expand_as(field)
    if not bool(torch.isfinite(field[expanded]).all()):
        raise ValueError("Supported targets must be finite")
    return torch.where(expanded, field, 0).to(device), mask.to(device), times.to(device)


def normalize(field, stats):
    shape = (1, 3, 1, 1, 1)
    return (field - torch.as_tensor(stats["mean"], device=field.device).view(shape)) / torch.as_tensor(stats["std"], device=field.device).view(shape)


def denormalize(field, stats):
    shape = (1, 3, 1, 1, 1)
    return field * torch.as_tensor(stats["std"], device=field.device).view(shape) + torch.as_tensor(stats["mean"], device=field.device).view(shape)


def fit_statistics(rows):
    total = torch.zeros(3, dtype=torch.float64); squares = total.clone(); mass = 0.
    mean_sum = mean_mass = None; motion_squares = torch.zeros(4, dtype=torch.float64); motion_count = torch.zeros(4, dtype=torch.float64)
    shape = None
    for row in rows:
        field, mask, times = load_reference(row); w = frame_weights(times).double()
        shape_now = tuple(field.shape[2:])
        if shape is None:
            shape = shape_now; mean_sum = torch.zeros((3, *shape), dtype=torch.float64); mean_mass = torch.zeros(shape, dtype=torch.float64)
        if shape_now != shape:
            raise ValueError("All windows require one common spatial grid")
        weights = mask.double() * w[:, None, None, None]
        total += (field.double() * weights[:, None]).sum((0, 2, 3, 4))
        squares += (field.double().square() * weights[:, None]).sum((0, 2, 3, 4)); mass += float(weights.sum())
        mean_sum += (field.double() * weights[:, None]).sum(0); mean_mass += weights.sum(0)
        obs = load_observation(row)
        features, valid = TrackSetEncoder.token_features(obs["tracks"][None], obs["mask"][None], obs["projection"], obs["times"].float(), obs["bounds"])
        pair = obs["mask"][..., 1:] & obs["mask"][..., :-1]
        active = torch.stack((obs["mask"], obs["mask"], F.pad(pair, (1, 0)), F.pad(pair, (1, 0))), -1)[None]
        motion_squares += torch.where(active, features[..., 2:6].double().square(), 0).reshape(-1, 4).sum(0)
        motion_count += active.reshape(-1, 4).sum(0)
    if mass <= 0:
        raise ValueError("No training supervision")
    mean = total / mass; std = (squares / mass - mean.square()).clamp_min(1e-12).sqrt()
    field_mean = mean_sum / mean_mass.clamp_min(1e-12)[None]
    # Never-observed locations have no target: fill their prior mean only, not zero velocity.
    field_mean = torch.where((mean_mass > 0)[None], field_mean, mean[:, None, None, None])
    return {"mean": mean.float().tolist(), "std": std.float().tolist(), "shape": list(shape),
            "motion_scales": (motion_squares / motion_count.clamp_min(1)).sqrt().clamp_min(1e-6).float().tolist(),
            "mean_field": field_mean.float(), "train_support": mean_mass > 0,
            "fit_split": "train", "weighted_mass": mass}


def masked_mse(prediction, target, support):
    valid = support[:, None].expand_as(target)
    if not bool(valid.any()):
        raise ValueError("No supported target samples")
    # Index first so arbitrary/NaN padding cannot affect loss or its gradient.
    return (prediction.float()[valid] - target.float()[valid]).square().mean()


def physical_time_features(times, indices, scale):
    times = times.double(); query = times[indices]; duration = times[-1] - times[0]
    if not bool(duration > 0):
        raise ValueError("Positive physical duration required")
    relative = (query - times[0]) / duration
    return torch.cat((torch.stack((query, times[0].expand_as(query), duration.expand_as(query), relative), -1).float(),
                      scale.log().expand(len(indices), -1)), -1)


class TemporalSystem(nn.Module):
    def __init__(self, kind, config, stats):
        super().__init__(); self.kind = kind
        self.encoder = TrackSetEncoder(config["condition_dim"], variant="motion-residual-v2",
                                       motion_scales=stats["motion_scales"], condition_flow_grid=config["condition_flow_grid"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"] + 104729)
            self.time_adapter = nn.Sequential(nn.Linear(7, config["condition_dim"]), nn.SiLU(), nn.Linear(config["condition_dim"], config["condition_dim"]))
            if kind == "unet_diffusion":
                self.decoder = ConditionalUNet3D(condition_dim=config["condition_dim"], **config[kind])
            else:
                self.decoder = build_decoder(kind, stats["shape"], condition_dim=config["condition_dim"], **config[kind])

    def condition(self, observation, indices):
        condition = self.encoder(observation["tracks"][None], observation["mask"][None], observation["projection"], observation["times"].float(), observation["bounds"])
        features = physical_time_features(observation["times"], indices, observation["unit_scale"])
        with torch.autocast(device_type=condition.device.type, enabled=False):
            return condition.expand(len(indices), -1).float() + self.time_adapter(features.float())


def nuisance_score_target(diffusion, noisy, timestep):
    """Conditional Bayes score target for an artificial unit-Gaussian latent.

    When x0 and epsilon are iid N(0,1), E[v|x_t]=0 and
    E[epsilon|x_t]=sqrt(1-alpha_bar)*x_t. Neither target means u_physical=0.
    """
    if diffusion.prediction_type == "v":
        return torch.zeros_like(noisy)
    sigma = diffusion.sqrt_one_minus_alpha_bars[timestep].view(-1, 1, 1, 1, 1)
    return sigma * noisy


def partial_score_loss(output, physical_target, noisy, timestep, diffusion, support):
    physical = masked_mse(output, physical_target, support)
    missing = (~support[:, None]).expand_as(output)
    if bool(missing.any()):
        nuisance = nuisance_score_target(diffusion, noisy, timestep).detach()
        latent_prior = (output.float()[missing] - nuisance.float()[missing]).square().mean()
        return physical + latent_prior  # Independently normalized; fixed weight one.
    return physical


def partial_diffusion_loss(system, diffusion, target, support, condition, *, generator=None, training=True, dropout=0.):
    shape = target.shape
    missing = torch.randn(shape, device=target.device, generator=generator)
    clean = torch.where(support[:, None], target, missing)
    noise = torch.randn(shape, device=target.device, generator=generator)
    timestep = torch.randint(diffusion.num_steps, (len(target),), device=target.device, generator=generator)
    noisy = diffusion.q_sample(clean, timestep, noise=noise)
    if training and dropout:
        drop = torch.rand((len(condition), 1), device=condition.device, generator=generator) < dropout
        condition = torch.where(drop, 0, condition)
    output = system.decoder(noisy, timestep, condition)
    if diffusion.prediction_type == "epsilon":
        truth = noise
    else:
        a = diffusion.sqrt_alpha_bars[timestep].view(-1, 1, 1, 1, 1)
        s = diffusion.sqrt_one_minus_alpha_bars[timestep].view(-1, 1, 1, 1, 1)
        truth = a * noise - s * clean
    return partial_score_loss(output, truth, noisy, timestep, diffusion, support)


def coarse_loss(system, condition, target, support):
    prediction = system.encoder.predict_condition_flow(condition)
    grid = prediction.shape[-1]; valid = support[:, None]
    mass = F.adaptive_avg_pool3d(valid.float(), grid)
    expected = F.adaptive_avg_pool3d(torch.where(valid, target.float(), 0), grid) / mass.clamp_min(1e-12)
    selected = (mass > 0).expand_as(prediction)
    return (prediction[selected] - expected[selected]).square().mean()


def loss_frame(system, diffusion, observation, field, mask, index, stats, config, *, generator=None, training=True):
    indices = torch.tensor([index], device=field.device)
    condition = system.condition(observation, indices)
    target = normalize(field[index:index + 1], stats); support = mask[index:index + 1]
    if diffusion is None:
        loss = masked_mse(system.decoder(condition), target, support)
    else:
        loss = partial_diffusion_loss(system, diffusion, target, support, condition, generator=generator,
                                      training=training, dropout=config["condition_dropout"])
    if config["condition_flow_weight"]:
        loss = loss + config["condition_flow_weight"] * coarse_loss(system, condition, target, support)
    return loss


def config_from(args):
    value = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else {}
    if "fs_transient" in value:
        value.pop("fs_transient")  # Temporal protocol is explicit/fixed by this runtime, not passed to baseline schema.
    for key in ("seed", "epochs", "num_particles"):
        if getattr(args, key, None) is not None:
            value[key] = getattr(args, key)
    value.setdefault("num_particles", 12); value.setdefault("views", [0]); value.setdefault("num_probes", 32)
    config = resolve_config(overrides=value)
    if config["num_particles"] != 12 or config["views"] != [0] or config["boundary_projection"] != "none":
        raise ValueError("FS protocol requires N12, XZ only, and no known velocity-boundary conditioning")
    if config["batch_size"] != 1:
        raise ValueError("This low-memory per-window temporal implementation requires batch_size=1")
    return config


def make_diffusion(config, device):
    return GaussianDiffusion(config["diffusion_steps"], schedule="cosine", prediction_type=config["prediction_type"],
                             clip_x0=config["clip_x0"]).to(device)


def fit_pod(rows, stats, rank, device, seed):
    columns = []
    mean = stats["mean_field"]
    for row in rows:
        field, support, times = load_reference(row)
        # Missing reference samples are imputed to the train-only voxel mean.
        # Multiplying centered snapshots by sqrt(time weight) is weighted POD.
        centered = torch.where(support[:, None], field - mean[None], 0)
        columns.append((centered * frame_weights(times).sqrt()[:, None, None, None, None]).flatten(1))
    matrix = torch.cat(columns).to(device)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("Nonfinite masked POD training matrix")
    q = min(int(rank) + 8, min(matrix.shape))
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if torch.device(device).type == "cuda" else []):
        torch.manual_seed(seed)
        if torch.device(device).type == "cuda": torch.cuda.manual_seed_all(seed)
        if q == min(matrix.shape):
            _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
            vectors = vh.T
        else:
            _, singular, vectors = torch.pca_lowrank(matrix, q=q, center=False, niter=2)
    tolerance = torch.finfo(matrix.dtype).eps * max(matrix.shape) * singular[0]
    effective = min(int(rank), int((singular > tolerance).sum()))
    basis = vectors[:, :effective].T.reshape(effective, 3, *stats["shape"]).cpu()
    return {"mean": mean, "basis": basis, "variance": (singular[:effective].square() / len(rows)).cpu(),
            "requested_rank": int(rank), "effective_rank": effective,
            "method": "train_only_masked_mean_imputation_time_weighted_randomized_POD", "imputation_iterations": 0}


def train(args):
    if args.model not in (*LEARNED, "mean", "pod"):
        raise ValueError("RBF fits each case through evaluate; PINN is excluded")
    manifest = Path(args.manifest).resolve(); _, splits = read_manifest(manifest)
    config = config_from(args); out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.resume is None and any((out / name).exists() for name in ("latest.pt", "run.json", "history.jsonl")):
        raise FileExistsError("Use an empty output directory or explicit --resume")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.manual_seed(config["seed"])
    if device.type == "cuda": torch.cuda.manual_seed_all(config["seed"])
    old = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    identity = {"format": FORMAT, "model": args.model, "manifest_sha256": digest(manifest),
                "runtime_sha256": digest(__file__), "scope": SCOPE, "observation_ids": "published_N12_seed47_fixed_all_epochs",
                "frame_objective": "one_frame_per_window_drawn_by_native_trapezoid_weights",
                "diffusion_missing_target_policy": "independent_N01_clean_latents_with_separately_normalized_analytic_nuisance_score_loss_weight1",
                "diffusion_nuisance_target": "v=0_or_epsilon=sqrt(1-alpha_bar)*x_t; artificial_latent_prior_not_zero_physical_velocity",
                "reference_support_input_at_inference": False}
    if old:
        for k, v in identity.items():
            if old.get(k) != v: raise ValueError(f"Resume identity mismatch: {k}")
        if {k:v for k,v in config.items() if k != "epochs"} != {k:v for k,v in old["config"].items() if k != "epochs"}:
            raise ValueError("Resume config differs")
        stats = old["statistics"]
    else:
        stats = fit_statistics(splits["train"])
    state = {**identity, "config": config, "statistics": stats,
             "manifest": str(manifest), "train_case_ids": [r["case_id"] for r in splits["train"]]}
    if args.model in ("mean", "pod"):
        if old: raise ValueError("Statistical fits do not resume")
        if args.model == "pod": state["pod"] = fit_pod(splits["train"], stats, config["pod"]["rank"], device, config["seed"])
        state["epoch"] = 0; save(out / "latest.pt", state)
        write_json(out / "run.json", {**identity, "status": "completed", "epoch": 0, "checkpoint_sha256": digest(out / "latest.pt")})
        return state
    system = TemporalSystem(args.model, config, stats).to(device); ema = copy.deepcopy(system).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(system.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    diffusion = make_diffusion(config, device) if args.model == "unet_diffusion" else None
    shuffle = torch.Generator().manual_seed(config["seed"])
    start, updates, skipped, best = 1, 0, 0, math.inf
    history = []
    if old:
        system.load_state_dict(old["system"]); ema.load_state_dict(old["ema_system"])
        optimizer.load_state_dict(old["optimizer"]); scaler.load_state_dict(old["scaler"])
        torch.set_rng_state(old["rng_state"]); shuffle.set_state(old["shuffle_state"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(old["cuda_rng_state"])
        start, updates, skipped, best = old["epoch"] + 1, old["optimizer_steps"], old["amp_skips"], old["best_validation_loss"]
        if start > config["epochs"]: raise ValueError("Checkpoint already reached final epoch")
        history = old["history"]
        (out / "history.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
    else:
        torch.manual_seed(config["seed"] + 27183)
        if device.type == "cuda": torch.cuda.manual_seed_all(config["seed"] + 27183)
    write_json(out / "run.json", {**identity, "status": "running", "epochs": config["epochs"], "config": config})
    for epoch in range(start, config["epochs"] + 1):
        began = time.monotonic(); system.train(); train_losses = []
        for i in torch.randperm(len(splits["train"]), generator=shuffle).tolist():
            row = splits["train"][i]; obs = load_observation(row, device); field, mask, times = load_reference(row, device)
            if tuple(field.shape[2:]) != tuple(stats["shape"]) or not torch.allclose(times, obs["times"]):
                raise ValueError("Reference/observation geometry or times differ")
            index = int(torch.multinomial(frame_weights(times).to(device), 1).item())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_frame(system, diffusion, obs, field, mask, index, stats, config)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite training loss")
            before = scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), config["gradient_clip"])
            if not use_amp and not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Nonfinite full-precision gradients")
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < before:
                skipped += 1
            else:
                updates += 1
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), system.parameters()): a.lerp_(b, 1 - config["ema_decay"])
                    for a, b in zip(ema.buffers(), system.buffers()): a.copy_(b)
            train_losses.append(float(loss.detach()))
        ema.eval(); validation = []
        # All validation frames with fixed diffusion noise; no consumption of training RNG.
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(config["seed"] + 100000)
            for row in splits["validation"]:
                obs = load_observation(row, device); field, mask, times = load_reference(row, device)
                losses = [loss_frame(ema, diffusion, obs, field, mask, j, stats, config, generator=generator, training=False) for j in range(len(times))]
                validation.append(float((torch.stack(losses) * frame_weights(times).to(device)).sum()))
        value = sum(validation) / len(validation)
        if not math.isfinite(value): raise RuntimeError("Nonfinite validation loss")
        improved = value < best; best = min(best, value)
        record = {"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses), "validation_loss": value,
                  "optimizer_steps": updates, "amp_skips": skipped, "elapsed_seconds": time.monotonic() - began}
        history.append(record)
        state.update(epoch=epoch, system=system.state_dict(), ema_system=ema.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     shuffle_state=shuffle.get_state(), optimizer_steps=updates, amp_skips=skipped, best_validation_loss=best, history=history,
                     trainable_parameters=sum(p.numel() for p in system.parameters()))
        save(out / "latest.pt", state)
        if improved: save(out / "best.pt", state)
        if epoch % 10 == 0 or epoch == config["epochs"]: save(out / f"epoch_{epoch:04d}.pt", state)
        with (out / "history.jsonl").open("a") as f: f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    if updates < 1: raise RuntimeError("No successful optimizer updates")
    write_json(out / "run.json", {**identity, "status": "completed", "epoch": state["epoch"], "optimizer_steps": updates,
                                  "checkpoint_sha256": digest(out / "latest.pt")})
    return state


class TransientLikelihood(ProjectedTrajectoryLikelihood):
    """Time-dependent Euler-transition NLL with bounded latent initial depth."""
    def replay(self, field, depth):
        if field.ndim != 5 or len(field) != len(self.times) or field.shape[1] != 3:
            raise ValueError("One velocity volume per physical observation time required")
        return advect_particles(field, self.initial_positions(depth)[None], self.times,
                                integrator=self.integrator, num_substeps=self.num_substeps, domain_bounds=self.bounds,
                                boundary_mode="clamp", velocity_times=self.times, return_validity=True)

    def residual(self, field, depth):
        positions, validity = self.replay(field, depth)
        n, t = positions.shape[1:3]
        velocity = sample_velocity(field, positions[:, :, :-1].reshape(1, -1, 3),
                                   self.times[:-1].expand(n, t - 1).reshape(1, -1), self.bounds,
                                   "clamp", velocity_times=self.times).reshape(n, t - 1, 3)
        projected = torch.einsum("ntc,voc->vnto", velocity, self.projections)
        residual = self.tracks.diff(dim=-2) - projected * self.times.diff()[None, None, :, None]
        mask = self.mask[..., :-1] & self.mask[..., 1:]
        return torch.where(mask[..., None], residual, 0), mask, validity


def temporal_coefficient_penalty(coeff, times):
    # Native times retained; regularization is relative to the window duration.
    relative = (times - times[0]) / (times[-1] - times[0])
    return ((coeff[1:] - coeff[:-1]) / relative.diff().view(-1, *([1] * (coeff.ndim - 1)))).square().mean()


def inverse_fit(kind, observation, config, pod=None):
    device = observation["tracks"].device; t = len(observation["times"])
    likelihood = TransientLikelihood(observation["tracks"], observation["projection"], observation["times"].float(), observation["bounds"],
                                     sigma=config["sigma"], mask=observation["mask"], integrator="rk4")
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    depth = nn.Parameter(likelihood.initialize_depth(generator))
    if kind == "pod":
        basis = pod["basis"].to(device); mean = pod["mean"].to(device); variance = pod["variance"].to(device)
        coeff = nn.Parameter(torch.zeros((t, len(basis)), device=device))
        def decode(): return mean[None] + torch.einsum("tk,kcdhw->tcdhw", coeff, basis)
        def prior():
            if coeff.numel() == 0: return depth.sum() * 0
            return config["coefficient_weight"] * (coeff.square() / variance.clamp_min(1e-12)).mean()
    else:
        model = RBFField(observation["bounds"], observation["shape"], **config["rbf"]).to(device)
        design = model.design.detach(); shape = observation["shape"]
        coeff = nn.Parameter(torch.zeros((t, 3, design.shape[0]), device=device))
        def decode(): return (coeff @ design).reshape(t, 3, *shape)
        def prior(): return config["rbf"]["ridge"] * coeff[:, :, 1:].square().mean()
    optimizer = torch.optim.Adam([{"params": [coeff], "lr": config["fit_learning_rate"]},
                                  {"params": [depth], "lr": config["depth_learning_rate"]}])
    best, best_field, best_depth = math.inf, None, None
    for iteration in range(config["fit_steps"] + 1):
        optimizer.zero_grad(); field = decode()
        residual, mask, _ = likelihood.residual(field, depth)
        objective = residual.square().sum() / (2 * config["sigma"] ** 2 * (2 * mask.sum()).clamp_min(1)) + prior()
        if coeff.numel(): objective = objective + 1e-4 * temporal_coefficient_penalty(coeff, likelihood.times)
        if not bool(torch.isfinite(objective)): raise RuntimeError("Nonfinite inverse objective")
        value = float(objective.detach())
        if value < best:
            best, best_field, best_depth = value, field.detach().clone(), depth.detach().clone()
        if iteration == config["fit_steps"]: break
        objective.backward()
        torch.nn.utils.clip_grad_norm_([coeff, depth], config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad(): depth.copy_(likelihood.reflect_depth(depth))
    return best_field, best_depth, {"steps": config["fit_steps"], "objective": best, "selection": "minimum_observation_objective_within_fixed_budget",
            "temporal_coefficients": True, "temporal_derivative_weight": 1e-4, "reference_used": False}


def fit_fixed_depth(field, observation, seed, steps=100, lr=.01):
    likelihood = TransientLikelihood(observation["tracks"], observation["projection"], observation["times"].float(), observation["bounds"],
                                     sigma=.05, mask=observation["mask"], integrator="rk4")
    depth = nn.Parameter(likelihood.initialize_depth(torch.Generator(device=field.device).manual_seed(seed)))
    optimizer = torch.optim.Adam([depth], lr=lr)
    best, selected = math.inf, depth.detach().clone()
    for iteration in range(steps + 1):
        optimizer.zero_grad(); objective = likelihood.nll(field.detach(), depth)
        if not bool(torch.isfinite(objective)): raise RuntimeError("Nonfinite depth diagnostic")
        if float(objective.detach()) < best:
            best, selected = float(objective.detach()), depth.detach().clone()
        if iteration == steps: break
        objective.backward(); optimizer.step()
        with torch.no_grad(): depth.copy_(likelihood.reflect_depth(depth))
    return selected, likelihood


def reconstruct(observation, kind, config, saved=None):
    """No archive/reference/probe argument is accepted by this function."""
    device = observation["tracks"].device; t = len(observation["times"])
    if kind == "mean":
        return saved["statistics"]["mean_field"].to(device)[None].expand(t, -1, -1, -1, -1).clone(), None
    if kind in ("pod", "rbf"):
        field, depth, info = inverse_fit(kind, observation, config, saved.get("pod") if saved else None)
        return field, {"depth": depth, "fitting": info}
    system = TemporalSystem(kind, saved["config"], saved["statistics"]).to(device)
    system.load_state_dict(saved["ema_system"]); system.eval().requires_grad_(False)
    diffusion = make_diffusion(config, device) if kind == "unet_diffusion" else None
    draws = []
    with torch.no_grad():
        conditions = [system.condition(observation, torch.tensor([j], device=device)) for j in range(t)]
        for sample in range(config["num_samples"] if diffusion is not None else 1):
            fields = []
            for j, condition in enumerate(conditions):
                if diffusion is None:
                    result = system.decoder(condition)
                else:
                    generator = torch.Generator(device=device).manual_seed(config["sampling_seed"] + sample * 1000000007 + 100003 * j)
                    result = diffusion.ddim_sample(system.decoder, (1, 3, *saved["statistics"]["shape"]), condition,
                                                   sampling_steps=config["sampling_steps"], eta=config["sampling_eta"],
                                                   guidance_scale=config["cfg_scale"], generator=generator, device=device)
                fields.append(denormalize(result.float(), saved["statistics"])[0])
            draws.append(torch.stack(fields))
    if diffusion is None:
        return draws[0], None
    samples = torch.stack(draws)
    return samples.mean(0), {"samples": samples}


def evaluate_field(field, row, observation):
    """Posthoc reference/probe read, called only after reconstruction is fixed."""
    reference, support, times = load_reference(row, field.device)
    if field.shape != reference.shape or not torch.allclose(times, observation["times"]):
        raise ValueError("Predicted/reference physical times or grid shapes differ")
    scale = observation["unit_scale"][None, :, None, None, None]
    prediction, target = field / scale, reference / scale
    weights = frame_weights(times).to(field.device)
    numerator = denominator = 0.; frame_rmse = []; component_sums = torch.zeros(3, device=field.device)
    total_mass = 0.
    for j, weight in enumerate(weights):
        error = prediction[j, :, support[j]] - target[j, :, support[j]]
        truth = target[j, :, support[j]]
        numerator += float(weight * error.square().sum()); denominator += float(weight * truth.square().sum())
        frame_rmse.append(error.square().sum(0).mean().sqrt())
        component_sums += weight * error.square().sum(1); total_mass += float(weight * support[j].sum())
    result = {"physical_velocity_relative_l2": math.sqrt(numerator / denominator) if denominator > 0 else None,
              "physical_velocity_vector_rmse": math.sqrt(numerator / total_mass),
              "physical_per_component_rmse": (component_sums / total_mass).sqrt().tolist(),
              "coverage": float(support.float().mean()), "support_is_evaluation_only": True}
    with np.load(row["path"], allow_pickle=False) as a:
        if all(k in a for k in ("probe_xyz", "probe_velocity", "probe_valid")):
            xyz = torch.from_numpy(np.array(a["probe_xyz"], copy=True)).to(field.device).float()
            velocity = torch.from_numpy(np.array(a["probe_velocity"], copy=True)).to(field.device).float()
            valid = torch.from_numpy(np.array(a["probe_valid"], copy=True)).to(field.device).bool()
        else:
            xyz = velocity = valid = None
    if xyz is not None:
        if tuple(xyz.shape) != tuple(velocity.shape) or tuple(valid.shape) != tuple(xyz.shape[:-1]):
            raise ValueError("Evaluation probe shapes disagree")
        if not bool(torch.isfinite(xyz[valid]).all() & torch.isfinite(velocity[valid]).all()):
            raise ValueError("Valid native probes must have finite coordinates and velocities")
        safe_xyz = torch.where(valid[..., None], xyz, observation["bounds"].mean(-1))
        query_times = times[None].expand(xyz.shape[0], -1).reshape(1, -1).float()
        sampled = sample_velocity(field, safe_xyz.reshape(1, -1, 3), query_times, observation["bounds"], "clamp", velocity_times=times.float()).reshape_as(xyz)
        si_error = (sampled - velocity) / observation["unit_scale"]
        weighted = valid * weights[None]; mass = weighted.sum()
        result["independent_probe_velocity_vector_rmse_m_s"] = float((torch.where(valid[..., None], si_error.square(), 0).sum(-1) * weights[None]).sum().div(mass).sqrt()) if bool(mass > 0) else None
        result["independent_probe_count"] = int(valid.any(-1).sum())
        result["probe_positions_used_only_after_prediction"] = True
    return result


def case_indices(value, count):
    result = list(range(count)) if value == "all" else [int(v) for v in value.split(",")]
    if not result or len(result) != len(set(result)) or min(result) < 0 or max(result) >= count:
        raise ValueError("Invalid/duplicate case indices")
    return result


def evaluate(args):
    manifest = Path(args.manifest).resolve(); _, splits = read_manifest(manifest)
    config = config_from(args); saved = None; kind = args.model
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or saved.get("manifest_sha256") != digest(manifest) or saved.get("runtime_sha256") != digest(__file__):
            raise ValueError("Checkpoint source or manifest mismatch")
        if kind is not None and kind != saved["model"]: raise ValueError("Checkpoint model mismatch")
        kind = saved["model"]
        supplied = config_from(args)
        config = copy.deepcopy(saved["config"])
        for key in ("seed", "sampling_seed", "num_probes", "fit_steps", "fit_learning_rate", "sigma", "depth_learning_rate"):
            if args.config or getattr(args, key, None) is not None: config[key] = supplied[key]
        if kind in LEARNED and saved["epoch"] != saved["config"]["epochs"]:
            raise ValueError("Main evaluation requires final-epoch checkpoint")
    elif kind != "rbf":
        raise ValueError("Fitted checkpoint required except RBF")
    indices = case_indices(args.case_indices, len(splits[args.split])); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    request = {"manifest_sha256": digest(manifest), "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
               "model": kind, "config": config, "split": args.split, "case_indices": indices, "runtime_sha256": digest(__file__)}
    progress_path = out / "progress.json"
    if progress_path.exists():
        if not args.resume or json.loads(progress_path.read_text())["request"] != request: raise ValueError("Existing evaluation request differs or --resume missing")
    rows = []
    for index in indices:
        row = splits[args.split][index]; target = out / f"case_{index:06d}.json"; npz = target.with_suffix(".npz")
        if target.exists() or npz.exists():
            if not args.resume or not target.is_file() or not npz.is_file(): raise ValueError("Conflicting or partial case output")
            record = json.loads(target.read_text())
            if record["request"] != request or record["source_case_sha256"] != digest(row["path"]) or record["output_sha256"] != digest(npz):
                raise ValueError("Existing case provenance differs")
            rows.append(record); continue
        began = time.monotonic(); obs = load_observation(row, args.device)
        field, extra = reconstruct(obs, kind, config, saved)
        if not bool(torch.isfinite(field).all()): raise RuntimeError("Nonfinite predicted fields")
        if extra is None or "depth" not in extra:
            depth, likelihood = fit_fixed_depth(field, obs, config["sampling_seed"], lr=config["depth_learning_rate"])
        else:
            depth = extra["depth"]; likelihood = TransientLikelihood(obs["tracks"], obs["projection"], obs["times"].float(), obs["bounds"], sigma=config["sigma"], mask=obs["mask"], integrator="rk4")
        with torch.no_grad(): diagnostics = likelihood.diagnostics(field, depth)
        metrics = evaluate_field(field, row, obs)
        for name in ("physical_velocity_relative_l2", "physical_velocity_vector_rmse"):
            if metrics[name] is None or not math.isfinite(metrics[name]):
                raise RuntimeError(f"Required posthoc metric is undefined/nonfinite: {name}")
        if extra is not None and "samples" in extra:
            draw_metrics = [evaluate_field(sample, row, obs) for sample in extra["samples"]]
            metrics["mean_per_draw_physical_velocity_relative_l2"] = sum(m["physical_velocity_relative_l2"] for m in draw_metrics) / len(draw_metrics)
            metrics["sample_mean_error_distinct_from_mean_draw_error"] = True
        temporary = npz.with_name(npz.name + ".tmp")
        with temporary.open("wb") as f:
            payload = dict(prediction=field.cpu().numpy(), initial_depth=depth.cpu().numpy(), observation_times=obs["times"].cpu().numpy(),
                           unit_scale=obs["unit_scale"].cpu().numpy())
            if extra is not None and "samples" in extra:
                payload.update(samples=extra["samples"].cpu().numpy(), sample_variance=extra["samples"].var(0, unbiased=False).cpu().numpy())
            np.savez_compressed(f, **payload)
        temporary.replace(npz)
        record = {"status": "completed", "scope": SCOPE, "request": request, "case_index": index, "case_id": row["case_id"],
                  "flow_group_id": row["flow_group_id"], "source_case_sha256": digest(row["path"]), "output_sha256": digest(npz),
                  "metrics": metrics, "diagnostics": diagnostics, "fitting": extra.get("fitting") if extra else None,
                  "reference_used_during_inference": False, "reference_masks_used_during_inference": False,
                  "elapsed_seconds": time.monotonic() - began, "model_adaptation": "physical_time_conditioned_spatial_baseline",
                  "training_seed": saved["config"]["seed"] if kind in LEARNED else None,
                  "fit_seed": config["seed"] if kind in ("pod", "rbf") else None,
                  "checkpoint_epoch": saved["epoch"] if saved else None,
                  "diffusion_temporal_law": "independent_per_frame_conditional_marginals" if kind == "unet_diffusion" else None}
        write_json(target, record); rows.append(record)
        write_json(progress_path, {"status": "running", "request": request, "completed_cases": len(rows), "total_cases": len(indices)})
    names = ("physical_velocity_relative_l2", "physical_velocity_vector_rmse", "independent_probe_velocity_vector_rmse_m_s")
    summary = {}
    for name in names:
        values = [r["metrics"].get(name) for r in rows]
        if any(v is None or not math.isfinite(v) for v in values):
            if name.startswith("independent_probe"):
                summary[name] = None; continue
            raise RuntimeError(f"Cannot summarize missing/nonfinite {name}")
        summary[name] = sum(values) / len(values)
    summary["invalid_replay_case_count"] = sum(not r["diagnostics"]["all_replay_particles_valid"] for r in rows)
    result = {"status": "completed", "request": request, "scope": SCOPE, "completed_cases": len(rows), "total_cases": len(indices),
              "summary": summary}
    write_json(out / "summary.json", result)
    write_json(progress_path, result); return result


def parser():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        s = sub.add_parser(command); s.add_argument("--manifest", required=True); s.add_argument("--output-dir", required=True)
        s.add_argument("--model", choices=MODELS, required=command == "train"); s.add_argument("--config")
        s.add_argument("--seed", type=int); s.add_argument("--num-particles", type=int, default=12); s.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if command == "train":
            s.add_argument("--epochs", type=int); s.add_argument("--resume")
        else:
            s.add_argument("--checkpoint"); s.add_argument("--split", choices=("validation", "test"), default="test")
            s.add_argument("--case-indices", default="all"); s.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = train(args) if args.command == "train" else evaluate(args)
    print(json.dumps({"status": "completed", "command": args.command, "epoch": result.get("epoch")}), flush=True)


if __name__ == "__main__":
    main()
