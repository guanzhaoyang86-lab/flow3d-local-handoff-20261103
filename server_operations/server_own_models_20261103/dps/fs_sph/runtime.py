#!/usr/bin/env python3
"""Unconditional frame-prior DPS with the FS-SPH native-time likelihood."""
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

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
SOURCE = PACKAGE / "fs_source"
sys.path.insert(0, str(SOURCE / "src"))
import numpy as np
import torch
from flow_observation.diffusion import GaussianDiffusion
from flow_observation.models.factory import build_denoiser
from flow_observation.dps.training import _optimizer_step
from flow_observation.dps.sampler import dps_sample
import physics


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary); temporary.replace(path)


def verify_inputs(require_fingerprint=True):
    config = read_json(HERE / "config.json")
    frozen = HERE / "frozen_manifest.json"
    lock = read_json(frozen)
    if require_fingerprint and os.environ.get("FLOW3D_DPS_FS_FROZEN_SHA256") != sha256(frozen):
        raise RuntimeError("Missing or incorrect submitted FS source fingerprint")
    for row in lock["files"]:
        if sha256(PACKAGE / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Frozen FS source changed: {row['path']}")
    if sha256(config["manifest"]) != config["manifest_sha256"] or lock["manifest_sha256"] != config["manifest_sha256"]:
        raise RuntimeError("FS scientific manifest changed")
    manifest, splits = physics.read_manifest(config["manifest"])
    if {k: len(v) for k, v in splits.items()} != config["expected_split_counts"]:
        raise RuntimeError("FS split counts differ from the frozen protocol")
    return config, lock, manifest, splits


def cuda_check():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production FS stages require a Slurm allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    x = torch.ones(3, device="cuda", requires_grad=True)
    x.square().sum().backward()
    if not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError("CUDA forward/backward preflight failed")


def fit_statistics(rows):
    """Read only training reference velocity, its support and native times."""
    total = torch.zeros(3, dtype=torch.float64)
    squares = total.clone(); mass = 0.; shape = None
    for row in rows:
        field, support, times = physics.load_reference(row)
        current = list(field.shape[2:])
        if shape is not None and shape != current:
            raise ValueError("All prior windows need one spatial grid")
        shape = current
        weights = support.double() * physics.frame_weights(times).double()[:, None, None, None]
        total += (field.double() * weights[:, None]).sum((0, 2, 3, 4))
        squares += (field.double().square() * weights[:, None]).sum((0, 2, 3, 4))
        mass += float(weights.sum())
    if not mass > 0:
        raise RuntimeError("No supported training velocities")
    mean = total / mass
    std = (squares / mass - mean.square()).clamp_min(1e-12).sqrt()
    return {"mean": mean.float().tolist(), "std": std.float().tolist(), "shape": shape,
            "fit_split": "train", "weighted_mass": mass,
            "training_case_ids": [row["case_id"] for row in rows]}


def partial_loss(model, diffusion, target, support, generator=None):
    """Independent missing latents plus the analytic stationary nuisance score."""
    missing = torch.randn(target.shape, device=target.device, generator=generator)
    clean = torch.where(support[:, None], target, missing)
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    timestep = torch.randint(diffusion.num_steps, (len(target),), device=target.device, generator=generator)
    noisy = diffusion.q_sample(clean, timestep, noise=noise)
    output = model(noisy, timestep, None)
    if diffusion.prediction_type == "epsilon":
        truth = noise
    else:
        a = diffusion.sqrt_alpha_bars[timestep].view(-1, 1, 1, 1, 1)
        s = diffusion.sqrt_one_minus_alpha_bars[timestep].view(-1, 1, 1, 1, 1)
        truth = a * noise - s * clean
    return physics.partial_score_loss(output, truth, noisy, timestep, diffusion, support)


def model_configuration(training, shape):
    return {"architecture": "dit3d", "condition_dim": 1, "dropout": 0.,
            "spatial_shape": list(shape), "dit_hidden_dim": training["dit_hidden_dim"],
            "dit_depth": training["dit_depth"], "dit_heads": training["dit_heads"],
            "patch_size": training["patch_size"]}


def validation_loss(model, diffusion, rows, statistics, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval(); total = 0.
    with torch.no_grad():
        for row in rows:
            field, support, times = physics.load_reference(row, device)
            weighted = 0.
            for index, weight in enumerate(physics.frame_weights(times)):
                loss = partial_loss(model, diffusion, physics.normalize(field[index:index + 1], statistics),
                                    support[index:index + 1], generator)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite prior validation score")
                weighted += float(weight) * float(loss)
            total += weighted
    return total / len(rows)


def verify_prior(config, lock, require_completion=True):
    path = Path(config["result_root"]) / "prior_s31" / "latest.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("format") != "fs-frame-factorized-unconditional-prior-v1" or
            state.get("epoch") != config["training"]["epochs"] or
            state.get("conditioning") != "none" or state.get("normalization_fit_split") != "train" or
            state.get("manifest_sha256") != config["manifest_sha256"] or
            state.get("frozen_sha256") != sha256(HERE / "frozen_manifest.json") or
            state.get("train_config") != config["training"] or state.get("successful_optimizer_steps", 0) < 1):
        raise RuntimeError("FS prior completion/provenance verification failed")
    if not all(bool(torch.isfinite(value).all()) for value in state["ema_model"].values()):
        raise RuntimeError("FS prior contains nonfinite EMA weights")
    directory = path.parent
    if require_completion:
        run = read_json(directory / "run.json")
        if run.get("status") != "completed" or run.get("checkpoint_sha256") != sha256(path):
            raise RuntimeError("FS prior completion marker is missing or inconsistent")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line.strip()]
    if [r["epoch"] for r in history] != list(range(1, config["training"]["epochs"] + 1)):
        raise RuntimeError("FS prior history does not cover every requested epoch exactly once")
    if not all(math.isfinite(r[k]) for r in history for k in ("train_loss", "validation_ema_loss")):
        raise RuntimeError("FS prior history contains a nonfinite score")
    return path, state


def train(config, lock, splits):
    training = config["training"]; output = Path(config["result_root"]) / "prior_s31"
    output.mkdir(parents=True, exist_ok=True); checkpoint = output / "latest.pt"
    torch.manual_seed(training["seed"]); torch.cuda.manual_seed_all(training["seed"])
    old = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if old and (old.get("frozen_sha256") != sha256(HERE / "frozen_manifest.json") or
                old.get("manifest_sha256") != config["manifest_sha256"] or old.get("train_config") != training):
        raise RuntimeError("Resume requires the identical FS source, manifest and training protocol")
    if old and old["epoch"] == training["epochs"]:
        verify_prior(config, lock, require_completion=False)
        write_json(output / "run.json", {"status": "completed", "audit": "verified",
            "epoch": old["epoch"], "checkpoint_sha256": sha256(checkpoint),
            "normalization_fit_split": "train", "conditioning": "none",
            "prior_semantics": config["prior_semantics"],
            "successful_optimizer_steps": old["successful_optimizer_steps"],
            "skipped_amp_steps": old["skipped_amp_steps"], "config": config})
        verify_prior(config, lock); return
    history_path = output / "history.jsonl"
    if history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        committed = old["epoch"] if old else 0
        kept = [row for row in rows if row["epoch"] <= committed]
        if [r["epoch"] for r in kept] != list(range(1, committed + 1)):
            raise RuntimeError("Checkpoint history is incomplete or duplicated")
        if rows != kept:
            history_path.rename(output / f"history_before_resume_{os.environ['SLURM_JOB_ID']}.jsonl")
            history_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in kept))
    elif old:
        raise RuntimeError("Refusing resume without committed epoch history")
    statistics = old["normalization"] if old else fit_statistics(splits["train"])
    if statistics["training_case_ids"] != [r["case_id"] for r in splits["train"]]:
        raise RuntimeError("Prior normalization is not train-only")
    model_config = model_configuration(training, statistics["shape"])
    diffusion_config = {"num_steps": training["diffusion_steps"], "prediction_type": training["prediction_type"],
                        "schedule": "cosine", "clip_x0": None}
    model = build_denoiser(model_config).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(**diffusion_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=training["amp"])
    shuffle = torch.Generator().manual_seed(training["seed"])
    start = 1; best = math.inf; successful = skipped = 0
    if old:
        model.load_state_dict(old["model"]); ema.load_state_dict(old["ema_model"])
        optimizer.load_state_dict(old["optimizer"]); scaler.load_state_dict(old["scaler"])
        torch.set_rng_state(old["rng_state"]); torch.cuda.set_rng_state_all(old["cuda_rng_state"])
        shuffle.set_state(old["shuffle_rng_state"])
        start = old["epoch"] + 1; best = old["best_validation_loss"]
        successful = old["successful_optimizer_steps"]; skipped = old["skipped_amp_steps"]
    write_json(output / "run.json", {"status": "running", "conditioning": "none", "config": config,
                                     "normalization_fit_split": "train", "prior_semantics": config["prior_semantics"]})
    for epoch in range(start, training["epochs"] + 1):
        began = time.monotonic(); model.train(); total = 0.; updated_this_epoch = skips_this_epoch = 0
        for row_index in torch.randperm(len(splits["train"]), generator=shuffle).tolist():
            field, support, times = physics.load_reference(splits["train"][row_index])
            index = int(torch.multinomial(physics.frame_weights(times), 1, generator=shuffle))
            target = physics.normalize(field[index:index + 1].cuda(), statistics)
            valid = support[index:index + 1].cuda(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=training["amp"]):
                loss = partial_loss(model, diffusion, target, valid)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite FS prior training loss")
            updated = _optimizer_step(loss, model, optimizer, scaler, training["gradient_clip"])
            if updated:
                successful += 1; updated_this_epoch += 1
                with torch.no_grad():
                    for target_parameter, parameter in zip(ema.parameters(), model.parameters()):
                        target_parameter.lerp_(parameter, 1 - training["ema_decay"])
                    for target_buffer, buffer in zip(ema.buffers(), model.buffers()):
                        target_buffer.copy_(buffer)
            else:
                skipped += 1; skips_this_epoch += 1
            total += float(loss.detach())
        validation = validation_loss(ema, diffusion, splits["validation"], statistics, "cuda", training["seed"] + 100000)
        improved = validation < best; best = min(best, validation)
        row = {"epoch": epoch, "train_loss": total / len(splits["train"]), "validation_ema_loss": validation,
               "successful_optimizer_steps": successful, "skipped_amp_steps": skipped,
               "optimizer_steps_this_epoch": updated_this_epoch, "amp_skipped_steps_this_epoch": skips_this_epoch,
               "elapsed_seconds": time.monotonic() - began}
        with history_path.open("a") as stream: stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        state = {"format": "fs-frame-factorized-unconditional-prior-v1", "conditioning": "none", "epoch": epoch,
                 "prior_semantics": config["prior_semantics"], "manifest_sha256": config["manifest_sha256"],
                 "frozen_sha256": sha256(HERE / "frozen_manifest.json"), "train_config": training,
                 "model_config": model_config, "diffusion_config": diffusion_config,
                 "normalization": statistics, "normalization_fit_split": "train",
                 "train_case_ids": [r["case_id"] for r in splits["train"]],
                 "model": model.state_dict(), "ema_model": ema.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_validation_loss": best,
                 "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                 "shuffle_rng_state": shuffle.get_state(), "successful_optimizer_steps": successful, "skipped_amp_steps": skipped}
        save(checkpoint, state)
        if improved: save(output / "best.pt", state)
        if epoch % training["save_every"] == 0 or epoch == training["epochs"]:
            save(output / f"epoch_{epoch:04d}.pt", state)
    write_json(output / "run.json", {"status": "completed", "audit": "verified", "epoch": training["epochs"],
                                     "checkpoint_sha256": sha256(checkpoint), "normalization_fit_split": "train",
                                     "conditioning": "none", "prior_semantics": config["prior_semantics"],
                                     "successful_optimizer_steps": successful, "skipped_amp_steps": skipped, "config": config})
    verify_prior(config, lock)


def make_likelihood(observation, sigma, sampling):
    result = physics.TransientLikelihood(observation["tracks"], observation["projection"], observation["times"],
                observation["bounds"], sigma=sigma, mask=observation["mask"], integrator=sampling["integrator"],
                num_substeps=sampling["trajectory_substeps"], likelihood=sampling["likelihood"])
    # Preserve the native absolute time grid instead of the base static reader's
    # track-dtype copy. The interpolation operators handle field-dtype casting.
    result.times = observation["times"]
    return result


def reconstruct(model, diffusion, statistics, observation, sampling, sigma, case_index):
    """Only observed XZ trajectories and known geometry are accepted here."""
    likelihood = make_likelihood(observation, sigma, sampling)
    fields = []; depths = []; traces = []; validity = []
    device = observation["tracks"].device
    for sample_index in range(sampling["num_samples"]):
        seed = sampling["seed"] + sample_index + case_index * 100003
        initial_generator = torch.Generator(device=device).manual_seed(seed)
        update_generator = torch.Generator(device=device).manual_seed(seed + 1000000007)
        depth_generator = torch.Generator(device=device).manual_seed(seed + 2000000011)
        shape = (len(observation["times"]), 3, *observation["shape"])
        result = dps_sample(model, diffusion, shape, likelihood,
            decode=lambda field: physics.denormalize(field, statistics), method=sampling["sampler"],
            sampling_steps=sampling["sampling_steps"], eta=sampling["eta"], guidance_step=sampling["guidance_step"],
            objective=sampling["gradient_objective"], max_update_rms=sampling["max_update_rms"],
            depth_step=sampling["depth_step"], depth_mode=sampling["depth_mode"],
            initial_depth=likelihood.initialize_depth(depth_generator),
            initial_noise=torch.randn(shape, device=device, generator=initial_generator),
            generator=update_generator, device=device)
        field = physics.denormalize(result["sample"], statistics).detach()
        with torch.no_grad():
            _, valid = likelihood.replay(field, result["depth"])
        if not bool(torch.isfinite(field).all()):
            raise RuntimeError("Nonfinite FS posterior field")
        fields.append(field); depths.append(result["depth"]); traces.append(result["trace"])
        validity.append(bool(valid.all()))
    return torch.stack(fields), torch.stack(depths), traces, validity


def verify_prediction(path, config, checkpoint_sha, split, index, sigma):
    row = read_json(path.with_suffix(".json"))
    expected = {"manifest_sha256": config["manifest_sha256"], "checkpoint_sha256": checkpoint_sha,
                "frozen_sha256": sha256(HERE / "frozen_manifest.json"), "sampling": config["sampling"],
                "split": split, "case_index": index, "sigma": sigma}
    if row.get("status") != "completed" or row.get("output_sha256") != sha256(path):
        raise RuntimeError("Incomplete or changed FS posterior artifact")
    if any(row.get(k) != value for k, value in expected.items()) or row.get("reference_used_for_sampling") is not False:
        raise RuntimeError("FS posterior protocol/provenance mismatch")
    with np.load(path, allow_pickle=False) as data:
        if data["samples"].shape[:3] != (config["sampling"]["num_samples"], 20, 3):
            raise RuntimeError("FS posterior shape or sample count differs")
        if not np.isfinite(data["samples"]).all() or not np.isfinite(data["initial_depth"]).all():
            raise RuntimeError("FS posterior contains nonfinite values")
    return row


def predict(config, state, checkpoint_sha, splits, split, index, sigma, output, model, diffusion):
    if output.exists():
        return verify_prediction(output, config, checkpoint_sha, split, index, sigma)
    if output.with_suffix(".json").exists():
        raise RuntimeError("Refusing to overwrite orphan posterior metadata")
    row = splits[split][index]
    observation = physics.load_observation(row, "cuda")
    # This completes every sample before any reference/support/probe is opened.
    fields, depths, traces, validity = reconstruct(model, diffusion, state["normalization"], observation,
                                                   config["sampling"], sigma, index)
    fixed = fields.detach(); mean = fixed.mean(0)
    with torch.no_grad():
        per_sample = [physics.evaluate_field(field, row, observation) for field in fixed]
        mean_scores = physics.evaluate_field(mean, row, observation)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, samples=fixed.cpu().numpy(), posterior_mean=mean.cpu().numpy(),
                            initial_depth=depths.cpu().numpy(), velocity_times=observation["times"].cpu().numpy())
    temporary.replace(output)
    metadata = {"status": "completed", "case_id": row["case_id"], "split": split, "case_index": index,
                "manifest_sha256": config["manifest_sha256"], "checkpoint_sha256": checkpoint_sha,
                "frozen_sha256": sha256(HERE / "frozen_manifest.json"), "output_sha256": sha256(output),
                "sampling": config["sampling"], "sigma": sigma, "reference_used_for_sampling": False,
                "prior_semantics": config["prior_semantics"], "scope": config["scope"],
                "trajectory_valid_samples": validity, "valid_sample_count": sum(validity),
                "posthoc_evaluation": {"posterior_mean": mean_scores, "per_sample": per_sample}, "trace": traces}
    write_json(output.with_suffix(".json"), metadata)
    return verify_prediction(output, config, checkpoint_sha, split, index, sigma)


def load_model(config, lock):
    checkpoint, state = verify_prior(config, lock)
    model = build_denoiser(state["model_config"]).cuda().eval().requires_grad_(False)
    model.load_state_dict(state["ema_model"])
    diffusion = GaussianDiffusion(**state["diffusion_config"]).cuda()
    return checkpoint, state, model, diffusion


def calibrate(config, lock, splits):
    checkpoint, state, model, diffusion = load_model(config, lock)
    digest = sha256(checkpoint); result = Path(config["result_root"])
    selected_path = result / "selected_config.json"
    if selected_path.exists() and (result / "CALIBRATION_COMPLETE.json").exists():
        selection = read_json(selected_path); marker = read_json(result / "CALIBRATION_COMPLETE.json")
        if marker["selected_config_sha256"] != sha256(selected_path) or selection["checkpoint_sha256"] != digest:
            raise RuntimeError("Existing FS selection changed")
        return
    records = []; means = []
    for sigma in sorted(config["calibration"]["sigma_candidates"]):
        scores = []
        for index in config["calibration"]["case_indices"]:
            output = result / "validation_calibration" / f"sigma_{sigma:g}" / f"case_{index:04d}.npz"
            row = predict(config, state, digest, splits, "validation", index, sigma, output, model, diffusion)
            score = row["posthoc_evaluation"]["posterior_mean"]["physical_velocity_relative_l2"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("FS calibration metric is undefined/nonfinite")
            scores.append(score)
            records.append({"sigma": sigma, "case_index": index, "metric": score, "output_sha256": row["output_sha256"]})
        means.append((sum(scores) / len(scores), sigma))
    score, sigma = min(means)
    selection = {"status": "frozen", "sigma": sigma, "validation_metric": score, "selection_records": records,
                 "manifest_sha256": config["manifest_sha256"], "checkpoint_sha256": digest,
                 "frozen_sha256": sha256(HERE / "frozen_manifest.json"), "sampling": config["sampling"],
                 "selection_protocol": config["calibration"], "test_used": False}
    write_json(selected_path, selection)
    write_json(result / "CALIBRATION_COMPLETE.json", {"status": "completed", "test_used": False,
                                                       "selected_config_sha256": sha256(selected_path)})


def evaluate_test(config, lock, splits):
    checkpoint, state, model, diffusion = load_model(config, lock)
    digest = sha256(checkpoint); result = Path(config["result_root"])
    selected_path = result / "selected_config.json"; selection = read_json(selected_path)
    marker = read_json(result / "CALIBRATION_COMPLETE.json")
    if (marker["selected_config_sha256"] != sha256(selected_path) or selection["checkpoint_sha256"] != digest or
            selection["frozen_sha256"] != sha256(HERE / "frozen_manifest.json") or
            selection["manifest_sha256"] != config["manifest_sha256"] or selection["sampling"] != config["sampling"] or
            selection["selection_protocol"] != config["calibration"] or selection["test_used"] is not False or
            selection.get("status") != "frozen" or selection["sigma"] not in config["calibration"]["sigma_candidates"]):
        raise RuntimeError("FS test requires the unchanged validation-only selection")
    scores = []; records = []; valid = 0
    for index in range(len(splits["test"])):
        output = result / "test" / f"case_{index:04d}.npz"
        row = predict(config, state, digest, splits, "test", index, selection["sigma"], output, model, diffusion)
        score = row["posthoc_evaluation"]["posterior_mean"]["physical_velocity_relative_l2"]
        if score is None or not math.isfinite(score): raise RuntimeError("Undefined FS test score")
        scores.append(score); valid += row["valid_sample_count"]
        records.append({"case_index": index, "output_sha256": row["output_sha256"], "metric": score})
    write_json(result / "TEST_COMPLETE.json", {"status": "completed", "case_count": len(scores),
        "expected_case_count": len(splits["test"]), "finite_metric_case_count": len(scores),
        "mean_posterior_mean_physical_velocity_relative_l2": sum(scores) / len(scores),
        "valid_sample_count": valid, "requested_sample_count": len(scores) * config["sampling"]["num_samples"],
        "selected_config_sha256": sha256(selected_path), "sigma": selection["sigma"],
        "scope": config["scope"], "prior_semantics": config["prior_semantics"], "outputs": records})


def preflight(config, splits):
    prior, support, times = physics.load_reference(splits["train"][0])
    observation = physics.load_observation(splits["validation"][0])
    if prior.shape != (20, 3, 32, 32, 32) or observation["tracks"].shape != (1, 12, 20, 2):
        raise RuntimeError("FS native field or observation shapes differ")
    return {"status": "passed_cpu_data_preflight", "split_counts": {k: len(v) for k, v in splits.items()},
            "native_times": times.tolist(), "native_time_step_range": [float(times.diff().min()), float(times.diff().max())],
            "prior_shape": list(prior.shape), "observation_shape": list(observation["tracks"].shape),
            "train_reference_support_fraction": float(support.float().mean()),
            "prior_semantics": config["prior_semantics"], "production_gpu_validation": "pending_slurm",
            "production_training_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "train", "calibrate", "test"])
    args = parser.parse_args(); config, lock, manifest, splits = verify_inputs(args.stage != "preflight")
    if args.stage == "preflight":
        print(json.dumps(preflight(config, splits), indent=2, allow_nan=False)); return
    cuda_check(); root = Path(config["result_root"])
    status = root / f"{args.stage}_status_{os.environ['SLURM_JOB_ID']}.json"
    write_json(status, {"status": "running", "stage": args.stage, "job_id": os.environ["SLURM_JOB_ID"]})
    try:
        {"train": train, "calibrate": calibrate, "test": evaluate_test}[args.stage](config, lock, splits)
    except BaseException as error:
        write_json(status, {"status": "failed", "stage": args.stage, "error": repr(error)})
        raise
    write_json(status, {"status": "completed", "stage": args.stage})


if __name__ == "__main__":
    main()
