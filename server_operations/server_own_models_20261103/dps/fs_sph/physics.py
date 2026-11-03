"""Frozen FS physical adapter; no baseline model or trajectory encoder."""
from pathlib import Path
import json
import math
import numpy as np
import torch
from flow_observation.dps.observations import ProjectedTrajectoryLikelihood
from flow_observation.advection import advect_particles
from flow_observation.interpolation import sample_velocity

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

def masked_mse(prediction, target, support):
    valid = support[:, None].expand_as(target)
    if not bool(valid.any()):
        raise ValueError("No supported target samples")
    # Index first so arbitrary/NaN padding cannot affect loss or its gradient.
    return (prediction.float()[valid] - target.float()[valid]).square().mean()

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
