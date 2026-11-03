#!/usr/bin/env python3
"""N12 Tensor-DiT training and observation-only reconstruction.

JHTDB uses the repaired conditional Tucker diffusion model on frozen snapshots.
FS uses physical-time-conditioned frame marginals, not a joint temporal prior.
Its Tucker latent targets encode a train-mean-imputed proxy complete volume;
unknown cells are NOT measured zero velocities. Low-noise decoded-field loss
and coarse condition supervision use only actual supported material targets.
No reference mask, XYZ trajectory, or reference velocity enters inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_baseline as fs
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.dit import TensorDiT
from flow_observation.tensor_space import TensorSpaceCodec
from flow_observation.diffusion import GaussianDiffusion
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.interpolation import sample_velocity
from flow_observation.advection import advect_particles

FORMAT = "tensor-dit-n12-v1"
MODEL = "tensor_dit"


def data_api():
    return importlib.import_module("tensor_data")


def default_config(dataset, seed=31):
    if dataset not in ("jhtdb", "fs_sph"):
        raise ValueError("dataset must be jhtdb or fs_sph")
    return {"schema": "tensor_dit_n12_config_v1", "model": MODEL, "dataset": dataset,
            "seed": int(seed), "observation_seed": 47, "sampling_seed": 47, "num_particles": 12,
            "epochs": 100, "batch_size": 1, "learning_rate": 2e-4, "weight_decay": 1e-4,
            "gradient_clip": 1., "ema_decay": .999, "amp": True,
            "dit_hidden_dim": 192, "dit_depth": 4, "dit_heads": 6, "condition_dim": 128,
            "condition_encoder": "motion-residual-v2", "condition_flow_grid": 4,
            "condition_flow_weight": .1, "condition_dropout": .1, "dropout": 0.,
            "diffusion_steps": 1000, "schedule": "cosine", "prediction_type": "v", "clip_x0": None,
            "decoded_field_weight": .1 if dataset == "fs_sph" else 0., "decoded_auxiliary_max_t_fraction": .2,
            "physical_time_adapter": dataset == "fs_sph", "sampling_steps": 50, "sampling_eta": 1.,
            "cfg_scale": 1., "num_samples": 2, "depth_fit_steps": 100, "depth_learning_rate": .01,
            "sigma": .05, "save_every": 10}


def resolve_config(dataset, path=None, overrides=None):
    config = default_config(dataset)
    changes = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(changes, dict) or set(changes) - set(config):
        raise ValueError("Unknown Tensor-DiT config keys")
    config.update(changes)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if config["model"] != MODEL or config["dataset"] != dataset or config["condition_encoder"] != "motion-residual-v2":
        raise ValueError("Model/dataset/encoder config mismatch")
    if config["num_particles"] != 12 or config["batch_size"] != 1:
        raise ValueError("This implementation requires N12 and batch1")
    if config["schedule"] != "cosine" or config["prediction_type"] != "v" or config["clip_x0"] is not None:
        raise ValueError("Frozen Tensor recipe requires cosine/v and no clipping of Tucker factors")
    if config["physical_time_adapter"] != (dataset == "fs_sph"):
        raise ValueError("Physical-time adapter is required exactly for the FS temporal adaptation")
    for key in ("epochs", "condition_dim", "condition_flow_grid", "dit_hidden_dim", "dit_depth", "dit_heads",
                "diffusion_steps", "sampling_steps", "num_samples", "save_every"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["diffusion_steps"] < 2 or config["sampling_steps"] > config["diffusion_steps"] or config["dit_hidden_dim"] % config["dit_heads"]:
        raise ValueError("Invalid diffusion steps or Transformer dimensions")
    for key in ("seed", "observation_seed", "sampling_seed", "depth_fit_steps"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"Invalid nonnegative integer: {key}")
    for key in ("learning_rate", "gradient_clip", "depth_learning_rate", "sigma"):
        if not math.isfinite(config[key]) or config[key] <= 0: raise ValueError(f"Invalid {key}")
    for key in ("weight_decay", "condition_flow_weight", "decoded_field_weight", "cfg_scale"):
        if not math.isfinite(config[key]) or config[key] < 0: raise ValueError(f"Invalid {key}")
    for key in ("ema_decay", "condition_dropout", "dropout"):
        if not 0 <= config[key] < 1: raise ValueError(f"Invalid {key}")
    for key in ("decoded_auxiliary_max_t_fraction", "sampling_eta"):
        if not 0 <= config[key] <= 1: raise ValueError(f"Invalid {key}")
    if not isinstance(config["amp"], bool): raise ValueError("amp must be boolean")
    return config


def source_identity():
    files = [Path(__file__), Path(fs.__file__), Path(__file__).with_name("tensor_data.py")]
    return {p.name: fs.digest(p) for p in files if p.is_file()}


def to_device(observation, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in observation.items()}


class TensorSystem(nn.Module):
    def __init__(self, config, statistics, codec):
        super().__init__(); self.config = config
        self.encoder = TrackSetEncoder(config["condition_dim"], variant="motion-residual-v2",
                                       motion_scales=statistics["motion_scales"], condition_flow_grid=config["condition_flow_grid"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"] + 104729)
            self.denoiser = TensorDiT(spatial_shape=statistics["shape"], rank=codec.rank,
                                     hidden_dim=config["dit_hidden_dim"], depth=config["dit_depth"],
                                     num_heads=config["dit_heads"], condition_dim=config["condition_dim"], dropout=config["dropout"])
        self.time_adapter = None
        if config["physical_time_adapter"]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["seed"] + 524287)
                self.time_adapter = nn.Sequential(nn.Linear(7, config["condition_dim"]), nn.SiLU(), nn.Linear(config["condition_dim"], config["condition_dim"]))

    def condition(self, observation, index):
        value = self.encoder(observation["tracks"][None].float(), observation["mask"][None],
                             observation["projection"].float(), observation["times"].float(), observation["bounds"].float())
        if self.time_adapter is not None:
            feature = fs.physical_time_features(observation["times"], torch.tensor([index], device=value.device), observation["unit_scale"])
            with torch.autocast(device_type=value.device.type, enabled=False):
                value = value.float() + self.time_adapter(feature.float())
        return value


def make_diffusion(config, device):
    return GaussianDiffusion(config["diffusion_steps"], schedule=config["schedule"], prediction_type=config["prediction_type"], clip_x0=None).to(device)


def checked_artifact(path, manifest, dataset, splits):
    artifact = data_api().load_artifact(path, manifest, dataset)
    if artifact.get("format") != "tensor_n12_artifact_v1" or artifact.get("dataset") != dataset or artifact.get("manifest_sha256") != fs.digest(manifest):
        raise ValueError("Artifact identity mismatch")
    if artifact["statistics"].get("fit_split") != "train": raise ValueError("Normalization must fit train only")
    codec = TensorSpaceCodec.from_artifact(artifact["codec"])
    if codec.rank != artifact["rank"] or codec.latent_dim != artifact["latent_dim"]:
        raise ValueError("Artifact rank/latent dimension mismatch")
    for split in ("train", "validation"):
        if set(artifact["latents"][split]) != {r["case_id"] for r in splits[split]}:
            raise ValueError("Artifact latent cache differs from manifest cases")
        for row in splits[split]:
            cached = artifact["latents"][split][row["case_id"]]
            weights = artifact["frame_weights"][split][row["case_id"]]
            if cached.ndim != 2 or cached.shape[1] != codec.latent_dim or len(weights) != len(cached) or not torch.isfinite(cached).all():
                raise ValueError("Invalid standardized latent targets")
            if not torch.isfinite(weights).all() or (weights <= 0).any() or not torch.allclose(weights.sum(), torch.tensor(1., dtype=weights.dtype), atol=1e-6):
                raise ValueError("Invalid physical-time frame weights")
    return artifact, codec


def frame_loss(system, codec, diffusion, observation, latent, field, support, index, statistics, config, *, generator=None, training=True):
    condition = system.condition(observation, index)
    used = condition
    if training and config["condition_dropout"]:
        dropped = torch.rand((1, 1), device=condition.device, generator=generator) < config["condition_dropout"]
        used = torch.where(dropped, 0, condition)
    timestep = torch.randint(diffusion.num_steps, (1,), device=latent.device, generator=generator)
    noise = torch.randn(latent.shape, device=latent.device, generator=generator)
    result = diffusion.training_loss(system.denoiser, latent, used, timesteps=timestep, noise=noise)
    loss = result["loss"]
    normalized = fs.normalize(field[index:index + 1], statistics)
    supported = support[index:index + 1]
    if config["condition_flow_weight"]:
        # This head is trained on raw supported normalized targets, not the imputed proxy.
        loss = loss + config["condition_flow_weight"] * fs.coarse_loss(system, condition, normalized, supported)
    if config["decoded_field_weight"] and int(timestep.item()) <= config["decoded_auxiliary_max_t_fraction"] * (diffusion.num_steps - 1):
        decoded = codec.decode(result["predicted_x0"].float())
        loss = loss + config["decoded_field_weight"] * fs.masked_mse(decoded, normalized, supported)
    return loss


def subset(rows, maximum):
    if maximum is None: return rows
    if maximum < 1: raise ValueError("Subset limits must be positive")
    return rows[:maximum]


def train(args):
    dataset = args.dataset; manifest = Path(args.manifest).resolve(); api = data_api()
    _, splits = api.read_manifest(manifest, dataset)
    config = resolve_config(dataset, args.config, {"epochs": args.epochs, "seed": args.seed})
    artifact, codec = checked_artifact(args.artifact, manifest, dataset, splits)
    statistics = artifact["statistics"]
    rows = subset(splits["train"], args.max_train_cases); validation_rows = subset(splits["validation"], args.max_validation_cases)
    identity = {"format": FORMAT, "model": MODEL, "dataset": dataset, "manifest_sha256": fs.digest(manifest),
                "artifact_sha256": fs.digest(args.artifact), "runtime_sources": source_identity(),
                "max_train_cases": args.max_train_cases, "max_validation_cases": args.max_validation_cases,
                "training_case_ids": [r["case_id"] for r in rows], "validation_case_ids": [r["case_id"] for r in validation_rows],
                "pilot_subset": args.max_train_cases is not None or args.max_validation_cases is not None,
                "unknown_target_policy": "train_mean_imputed_proxy_Tucker_latents_with_masked_physical_auxiliaries" if dataset == "fs_sph" else "fully_observed_frozen_snapshot",
                "temporal_law": "physical_time_conditioned_frame_marginals_not_joint_temporal_prior" if dataset == "fs_sph" else "steady_frozen_snapshot",
                "reference_truth_used_at_inference": False, "clip_x0": None}
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / p).exists() for p in ("latest.pt", "run.json", "history.jsonl")):
        raise FileExistsError("Output exists: use an empty directory or explicit --resume")
    old = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if old:
        for key, value in identity.items():
            if old.get(key) != value: raise ValueError(f"Resume identity changed: {key}")
        if {k:v for k,v in old["config"].items() if k != "epochs"} != {k:v for k,v in config.items() if k != "epochs"}:
            raise ValueError("Resume model/training config changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.manual_seed(config["seed"])
    if device.type == "cuda": torch.cuda.manual_seed_all(config["seed"])
    codec = codec.to(device).eval().requires_grad_(False)
    system = TensorSystem(config, statistics, codec).to(device)
    ema = copy.deepcopy(system).eval().requires_grad_(False)
    diffusion = make_diffusion(config, device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp = config["amp"] and device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    shuffle = torch.Generator().manual_seed(config["seed"])
    start, updates, skips, best, history = 1, 0, 0, math.inf, []
    if old:
        system.load_state_dict(old["system"]); ema.load_state_dict(old["ema_system"])
        optimizer.load_state_dict(old["optimizer"]); scaler.load_state_dict(old["scaler"])
        torch.set_rng_state(old["rng_state"]); shuffle.set_state(old["shuffle_rng_state"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(old["cuda_rng_state"])
        start, updates, skips, best, history = old["epoch"] + 1, old["optimizer_steps"], old["amp_skips"], old["best_validation_loss"], old["history"]
        if start > config["epochs"]: raise ValueError("Checkpoint already reached final epoch")
        (out / "history.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
    else:
        torch.manual_seed(config["seed"] + 27183)
        if device.type == "cuda": torch.cuda.manual_seed_all(config["seed"] + 27183)
    state = {**identity, "config": config, "statistics": statistics, "codec": artifact["codec"],
             "rank": artifact["rank"], "latent_dim": artifact["latent_dim"], "manifest": str(manifest),
             "trainable_parameters": sum(p.numel() for p in system.parameters())}
    fs.write_json(out / "run.json", {**identity, "status": "running", "config": config})
    for epoch in range(start, config["epochs"] + 1):
        began = time.monotonic(); system.train(); train_losses = []
        for position in torch.randperm(len(rows), generator=shuffle).tolist():
            row = rows[position]
            observation = to_device(api.load_observation(row, dataset, seed=config["observation_seed"], epoch=epoch - 1), device)
            field, support, times = api.load_reference(row, dataset); field, support = field.to(device), support.to(device)
            cached = artifact["latents"]["train"][row["case_id"]]
            index = int(torch.multinomial(artifact["frame_weights"]["train"][row["case_id"]].to(device), 1).item())
            if len(field) != len(cached): raise ValueError("Cached frame count changed")
            latent = cached[index:index + 1].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = frame_loss(system, codec, diffusion, observation, latent, field, support, index, statistics, config)
            if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite training objective")
            before = scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(system.parameters(), config["gradient_clip"])
            if not amp and not bool(torch.isfinite(norm)): raise RuntimeError("Nonfinite gradients")
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < before:
                skips += 1
            else:
                updates += 1
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), system.parameters()): target.lerp_(source, 1 - config["ema_decay"])
                    for target, source in zip(ema.buffers(), system.buffers()): target.copy_(source)
            train_losses.append(float(loss.detach()))
        ema.eval(); validation = []
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(config["seed"] + 100000)
            for row in validation_rows:
                observation = to_device(api.load_observation(row, dataset, seed=config["observation_seed"], epoch=0), device)
                field, support, _ = api.load_reference(row, dataset); field, support = field.to(device), support.to(device)
                cached = artifact["latents"]["validation"][row["case_id"]]
                losses = [frame_loss(ema, codec, diffusion, observation, cached[j:j + 1].to(device), field, support, j, statistics, config,
                                     generator=generator, training=False) for j in range(len(cached))]
                weights = artifact["frame_weights"]["validation"][row["case_id"]].to(device)
                validation.append(float((torch.stack(losses) * weights).sum()))
        val = sum(validation) / len(validation)
        if not math.isfinite(val): raise RuntimeError("Nonfinite validation objective")
        improved = val < best; best = min(best, val)
        record = {"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses), "validation_loss": val,
                  "optimizer_steps": updates, "amp_skips": skips, "elapsed_seconds": time.monotonic() - began}
        history.append(record)
        state.update(epoch=epoch, system=system.state_dict(), ema_system=ema.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     shuffle_rng_state=shuffle.get_state(), optimizer_steps=updates, amp_skips=skips, best_validation_loss=best, history=history)
        fs.save(out / "latest.pt", state)
        if improved: fs.save(out / "best.pt", state)
        if epoch % config["save_every"] == 0 or epoch == config["epochs"]: fs.save(out / f"epoch_{epoch:04d}.pt", state)
        with (out / "history.jsonl").open("a") as f: f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
    if updates < 1: raise RuntimeError("No successful optimizer updates")
    fs.write_json(out / "run.json", {**identity, "status": "completed", "epoch": state["epoch"], "optimizer_steps": updates,
                                     "trainable_parameters": state["trainable_parameters"], "config": config,
                                     "checkpoint_sha256": fs.digest(out / "latest.pt")})
    return state


def reconstruct(observation, saved, *, case_index=0):
    """Accepts observations and trained state only; no row/reference interface."""
    config = saved["config"]; device = observation["tracks"].device
    codec = TensorSpaceCodec.from_artifact(saved["codec"]).to(device).eval().requires_grad_(False)
    system = TensorSystem(config, saved["statistics"], codec).to(device)
    system.load_state_dict(saved["ema_system"]); system.eval().requires_grad_(False)
    diffusion = make_diffusion(config, device)
    frame_count = len(observation["times"]) if saved["dataset"] == "fs_sph" else 1
    samples = []
    with torch.no_grad():
        conditions = [system.condition(observation, j) for j in range(frame_count)]
        for sample in range(config["num_samples"]):
            fields = []
            for j, condition in enumerate(conditions):
                generator = torch.Generator(device=device).manual_seed(config["sampling_seed"] + 1000000007 * sample + 100003 * case_index + 104729 * j)
                latent = diffusion.ddim_sample(system.denoiser, (1, codec.latent_dim), condition, sampling_steps=config["sampling_steps"],
                                               eta=config["sampling_eta"], guidance_scale=config["cfg_scale"], generator=generator, device=device)
                field = fs.denormalize(codec.decode(latent.float()), saved["statistics"])[0]
                if not bool(torch.isfinite(field).all()): raise RuntimeError("Nonfinite generated field")
                fields.append(field)
            samples.append(torch.stack(fields))
    return torch.stack(samples)


def likelihood_for(observation, dataset, sigma=.05):
    cls = fs.TransientLikelihood if dataset == "fs_sph" else ProjectedTrajectoryLikelihood
    return cls(observation["tracks"], observation["projection"], observation["times"].float(), observation["bounds"],
               sigma=sigma, mask=observation["mask"], integrator="rk4", num_substeps=1)


def depth_diagnostics(field, observation, dataset, config):
    likelihood = likelihood_for(observation, dataset, config["sigma"])
    generator = torch.Generator(device=field.device).manual_seed(config["sampling_seed"])
    depth = nn.Parameter(likelihood.initialize_depth(generator)); optimizer = torch.optim.Adam([depth], lr=config["depth_learning_rate"])
    best, selected = math.inf, depth.detach().clone()
    for step in range(config["depth_fit_steps"] + 1):
        optimizer.zero_grad(); loss = likelihood.nll(field.detach(), depth)
        if not bool(torch.isfinite(loss)): raise RuntimeError("Nonfinite depth diagnostic")
        if float(loss.detach()) < best: best, selected = float(loss.detach()), depth.detach().clone()
        if step == config["depth_fit_steps"]: break
        loss.backward(); optimizer.step()
        with torch.no_grad(): depth.copy_(likelihood.reflect_depth(depth))
    with torch.no_grad(): diagnostics = likelihood.diagnostics(field, selected)
    return selected, diagnostics


def physical_metrics(field, row, observation, dataset):
    if dataset == "fs_sph":
        return fs.evaluate_field(field, row, observation)
    reference, support, times = data_api().load_reference(row, dataset)
    reference, support = reference.to(field.device), support.to(field.device)
    if field.shape != reference.shape: raise ValueError("Reference/predicted field shape mismatch")
    scale = observation["unit_scale"][None, :, None, None, None]
    valid = support[:, None].expand_as(field)
    error = ((field - reference) / scale)[valid].double()
    truth = (reference / scale)[valid].double()
    if not bool(torch.isfinite(error).all() & torch.isfinite(truth).all()) or not bool(truth.square().sum() > 0):
        raise ValueError("Invalid physical field evaluation")
    result = {"physical_velocity_relative_l2": float(error.norm() / truth.norm()),
              "physical_velocity_vector_rmse": float((error.square().sum() / support.sum()).sqrt()),
              "physical_per_component_rmse": [float((((field[:, c] - reference[:, c]) / observation["unit_scale"][c])[support]).double().square().mean().sqrt()) for c in range(3)],
              "coverage": float(support.float().mean()), "support_is_evaluation_only": True}
    # Optional probe XYZ is opened only here, after every predicted draw is fixed.
    if "probe_particle_indices" in observation:
        with np.load(row["path"], allow_pickle=False) as a:
            if "trajectories_3d" not in a: return result
            source = int(observation.get("source_sample_index", 0))
            indices = observation["probe_particle_indices"].cpu().numpy()
            xyz = torch.from_numpy(np.array(a["trajectories_3d"][source, indices], copy=True)).to(field.device).float()
            valid_probe = torch.from_numpy(np.array(a["observation_mask_3d"][source, indices], copy=True)).to(field.device).bool()
        prediction, validity = advect_particles(field, xyz[:, 0][None], observation["times"].float(), integrator="rk4", num_substeps=4,
                                               domain_bounds=observation["bounds"], boundary_mode="terminate", return_validity=True)
        projected_error = (prediction[0, ..., [0, 2]] - xyz[..., [0, 2]]) / observation["unit_scale"][[0, 2]]
        result["heldout_projected_track_vector_rmse_native"] = float(projected_error[valid_probe].double().square().sum(-1).mean().sqrt())
        result["heldout_probe_replay_valid_fraction"] = float(validity.float().mean())
        result["probe_initial_xyz_is_posthoc_only"] = True
    return result


def evaluate(args):
    dataset = args.dataset; manifest = Path(args.manifest).resolve(); _, splits = data_api().read_manifest(manifest, dataset)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = {"format": FORMAT, "model": MODEL, "dataset": dataset, "manifest_sha256": fs.digest(manifest), "runtime_sources": source_identity()}
    for key, value in expected.items():
        if saved.get(key) != value: raise ValueError(f"Evaluation checkpoint identity differs: {key}")
    if saved["epoch"] != saved["config"]["epochs"] or saved["optimizer_steps"] < 1: raise ValueError("Final successful epoch checkpoint required")
    if args.config:
        requested = resolve_config(dataset, args.config)
        # Sampling/depth diagnostics may change without changing fitted weights.
        allowed = {"sampling_steps", "sampling_eta", "cfg_scale", "num_samples", "sampling_seed", "depth_fit_steps", "depth_learning_rate", "sigma", "epochs"}
        for key in requested:
            if requested[key] != saved["config"][key] and key not in allowed: raise ValueError(f"Cannot change fitted config: {key}")
        saved["config"] = {**saved["config"], **{k: requested[k] for k in allowed if k != "epochs"}}
    config = saved["config"]; indices = fs.case_indices(args.case_indices, len(splits[args.split])); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    request = {"dataset": dataset, "manifest_sha256": fs.digest(manifest), "checkpoint_sha256": fs.digest(args.checkpoint),
               "config": config, "split": args.split, "case_indices": indices, "runtime_sources": source_identity()}
    progress = out / "progress.json"
    if progress.exists() and (not args.resume or json.loads(progress.read_text())["request"] != request):
        raise ValueError("Existing evaluation request differs or explicit --resume missing")
    records = []
    for index in indices:
        row = splits[args.split][index]; record_path = out / f"case_{index:06d}.json"; npz = record_path.with_suffix(".npz")
        if record_path.exists() or npz.exists():
            if not args.resume or not record_path.is_file() or not npz.is_file(): raise ValueError("Existing/partial case prediction")
            record = json.loads(record_path.read_text())
            if record["request"] != request or record["source_case_sha256"] != fs.digest(row["path"]) or record["output_sha256"] != fs.digest(npz):
                raise ValueError("Existing case checksum/protocol mismatch")
            records.append(record); continue
        began = time.monotonic()
        observation = to_device(data_api().load_observation(row, dataset, seed=config["observation_seed"], epoch=0), args.device)
        samples = reconstruct(observation, saved, case_index=index)
        mean = samples.mean(0)
        depth, diagnostics = depth_diagnostics(mean, observation, dataset, config)
        # All generated draws are fixed before reference/probe metrics are read.
        metric = physical_metrics(mean, row, observation, dataset)
        per_draw = [physical_metrics(draw, row, observation, dataset) for draw in samples]
        metric["mean_per_draw_physical_velocity_relative_l2"] = sum(m["physical_velocity_relative_l2"] for m in per_draw) / len(per_draw)
        for name in ("physical_velocity_relative_l2", "physical_velocity_vector_rmse", "mean_per_draw_physical_velocity_relative_l2"):
            if metric[name] is None or not math.isfinite(metric[name]): raise RuntimeError(f"Missing/nonfinite metric {name}")
        temporary = npz.with_name(npz.name + ".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, samples=samples.cpu().numpy(), prediction_mean=mean.cpu().numpy(), sample_variance=samples.var(0, unbiased=False).cpu().numpy(),
                                initial_depth=depth.cpu().numpy(), observation_times=observation["times"].cpu().numpy(), unit_scale=observation["unit_scale"].cpu().numpy())
        temporary.replace(npz)
        record = {"status": "completed", "model": MODEL, "dataset": dataset, "case_id": row["case_id"], "case_index": index,
                  "request": request, "source_case_sha256": fs.digest(row["path"]), "output_sha256": fs.digest(npz),
                  "metrics": metric, "diagnostics": diagnostics, "reference_used_during_inference": False,
                  "support_mask_used_during_inference": False, "checkpoint_epoch": saved["epoch"], "training_seed": config["seed"],
                  "pilot_subset_checkpoint": saved["pilot_subset"], "elapsed_seconds": time.monotonic() - began,
                  "unknown_target_policy": saved["unknown_target_policy"], "temporal_law": saved["temporal_law"]}
        fs.write_json(record_path, record); records.append(record)
        fs.write_json(progress, {"status": "running", "request": request, "completed_cases": len(records), "total_cases": len(indices)})
    names = ("physical_velocity_relative_l2", "physical_velocity_vector_rmse", "mean_per_draw_physical_velocity_relative_l2")
    summary = {name: sum(r["metrics"][name] for r in records) / len(records) for name in names}
    summary["invalid_replay_case_count"] = sum(not r["diagnostics"]["all_replay_particles_valid"] for r in records)
    result = {"status": "completed", "request": request, "summary": summary, "completed_cases": len(records), "total_cases": len(indices)}
    fs.write_json(out / "summary.json", result); fs.write_json(progress, result); return result


def parser():
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        s = commands.add_parser(command); s.add_argument("--dataset", choices=("jhtdb", "fs_sph"), required=True)
        s.add_argument("--manifest", required=True); s.add_argument("--config"); s.add_argument("--output-dir", required=True)
        s.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if command == "train":
            s.add_argument("--artifact", "--tensor-artifact", required=True); s.add_argument("--epochs", type=int); s.add_argument("--seed", type=int)
            s.add_argument("--resume"); s.add_argument("--max-train-cases", type=int); s.add_argument("--max-validation-cases", type=int)
        else:
            s.add_argument("--checkpoint", required=True); s.add_argument("--split", choices=("validation", "test"), default="validation")
            s.add_argument("--case-indices", default="all"); s.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv); result = train(args) if args.command == "train" else evaluate(args)
    print(json.dumps({"status": "completed", "command": args.command, "epoch": result.get("epoch")}), flush=True)


if __name__ == "__main__": main()
