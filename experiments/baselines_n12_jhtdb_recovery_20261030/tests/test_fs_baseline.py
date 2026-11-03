"""Small numerical tests of the temporal adaptation, not GPU performance tests."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

RUNTIME = Path(__file__).resolve().parents[1] / "launch" / "fs_baseline.py"
spec = importlib.util.spec_from_file_location("fs_baseline", RUNTIME)
fs = importlib.util.module_from_spec(spec); spec.loader.exec_module(fs)
torch.set_num_threads(1)


def observation(times=None):
    times = torch.tensor([1., 1.1, 1.3], dtype=torch.float64) if times is None else times
    base = torch.linspace(.5, 1.5, 12)
    tracks = torch.zeros(1, 12, 3, 2)
    tracks[0, :, :, 0] = base[:, None] + torch.tensor([0., .03, .13])[None]
    tracks[0, :, :, 1] = .7
    return {"tracks": tracks, "mask": torch.ones(1, 12, 3, dtype=torch.bool), "times": times,
            "projection": torch.tensor([[[1., 0., 0.], [0., 0., 1.]]]),
            "bounds": torch.tensor([[0., 3.], [0., 3.], [0., 3.]]), "unit_scale": torch.tensor([2., 3., 4.]), "shape": (4, 4, 4)}


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i in range(4):
        obs = observation(); obs["times"] += i * .5
        field = np.zeros((3, 3, 4, 4, 4), np.float32)
        field[:, 0] = np.array([.2, .4, .6])[:, None, None, None]
        field[:, 1] = .01
        support = np.ones((3, 4, 4, 4), bool); support[:, :, :, 0] = False
        field[:, :, :, :, 0] = 10000
        probe_xyz = np.ones((32, 3, 3), np.float32)
        probe_xyz[:, :, 0] += np.array([0, .03, .13])[None]
        probe_velocity = np.zeros_like(probe_xyz); probe_velocity[:, :, 0] = np.array([.2, .4, .6])[None]
        probe_velocity[:, :, 1] = .01
        path = tmp_path / f"window_{i}.npz"
        np.savez(path, flow_field=field, support_mask=support, velocity_times=obs["times"].numpy(),
                 trajectories_2d=obs["tracks"].numpy()[None], observation_mask=obs["mask"].numpy()[None],
                 observation_times=obs["times"].numpy(), projection_matrix=obs["projection"].numpy(), domain_bounds=obs["bounds"].numpy(),
                 metadata=np.array(json.dumps({"unit_scale": [2., 3., 4.], "grid_shape_zyx": [4, 4, 4]})),
                 probe_xyz=probe_xyz, probe_velocity=probe_velocity, probe_valid=np.ones((32, 3), bool))
        rows.append({"case_id": f"window_{i}", "path": str(path), "flow_group_id": "fs_sph_3429811", "window_id": i})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "fs_transient_baseline_manifest_v1", "mode": "within_case_temporal",
                                    "splits": {"train": rows[:2], "validation": rows[2:3], "test": rows[3:]}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"condition_dim": 8, "condition_flow_grid": 2, "cnn": {"width": 4, "depth": 1},
                                  "amp": False, "condition_flow_weight": .1, "fit_steps": 2,
                                  "pod": {"rank": 2}, "rbf": {"centers_per_axis": 2}, "epochs": 2}))
    return manifest, config, rows


def train_args(dataset, out, epochs, resume=None, model="cnn"):
    manifest, config, _ = dataset
    return SimpleNamespace(manifest=str(manifest), config=str(config), output_dir=str(out), epochs=epochs, seed=31,
                           num_particles=12, model=model, device="cpu", resume=resume)


def test_time_features_retain_absolute_seconds_and_irregular_weights():
    times = torch.tensor([1., 1.1, 1.3], dtype=torch.float64)
    w = fs.frame_weights(times)
    assert torch.allclose(w, torch.tensor([1 / 6, .5, 1 / 3]))
    a = fs.physical_time_features(times, torch.tensor([1]), torch.ones(3))
    b = fs.physical_time_features(times + 4, torch.tensor([1]), torch.ones(3))
    assert torch.allclose(b[:, :2] - a[:, :2], torch.full((1, 2), 4.))
    assert torch.equal(a[:, 2:], b[:, 2:])


def test_masked_mse_ignores_nan_padding_and_gradients():
    prediction = torch.arange(24.).reshape(1, 3, 2, 2, 2).requires_grad_(True)
    target = torch.ones_like(prediction); mask = torch.ones((1, 2, 2, 2), dtype=torch.bool); mask[..., 0] = False
    target[~mask[:, None].expand_as(target)] = torch.nan
    loss = fs.masked_mse(prediction, target, mask); loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(prediction.grad[~mask[:, None].expand_as(prediction)], torch.zeros(12))


def test_partial_diffusion_padding_truth_is_irrelevant():
    class Toy(nn.Module):
        def __init__(self): super().__init__(); self.scale = nn.Parameter(torch.tensor(.2))
        def forward(self, x, step, cond): return self.scale * x
    system = SimpleNamespace(decoder=Toy())
    diffusion = fs.GaussianDiffusion(8, prediction_type="v")
    clean = torch.ones(1, 3, 2, 2, 2); mask = torch.ones(1, 2, 2, 2, dtype=torch.bool); mask[..., 0] = False
    changed = clean.clone(); changed[~mask[:, None].expand_as(clean)] = torch.nan
    a = fs.partial_diffusion_loss(system, diffusion, clean, mask, torch.zeros(1, 8), generator=torch.Generator().manual_seed(7))
    b = fs.partial_diffusion_loss(system, diffusion, changed, mask, torch.zeros(1, 8), generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    b.backward(); assert torch.isfinite(system.decoder.scale.grad)


@pytest.mark.parametrize("prediction_type", ["v", "epsilon"])
def test_analytic_missing_latent_score_and_finite_unsupported_gradients(prediction_type):
    diffusion = fs.GaussianDiffusion(8, prediction_type=prediction_type)
    noisy = torch.linspace(-2, 2, 24).reshape(1, 3, 2, 2, 2)
    step = torch.tensor([3])
    expected = torch.zeros_like(noisy) if prediction_type == "v" else diffusion.sqrt_one_minus_alpha_bars[3] * noisy
    assert torch.equal(fs.nuisance_score_target(diffusion, noisy, step), expected)
    support = torch.ones(1, 2, 2, 2, dtype=torch.bool); support[..., 0] = False
    # Physical supported targets are independent of this artificial nuisance prior.
    physical = torch.full_like(noisy, 7.)
    optimal = torch.where(support[:, None], physical, expected)
    assert fs.partial_score_loss(optimal, physical, noisy, step, diffusion, support) == 0
    perturbed = optimal.clone().requires_grad_(True)
    missing = (~support[:, None]).expand_as(perturbed)
    loss = fs.partial_score_loss(perturbed + missing.float(), physical, noisy, step, diffusion, support)
    loss.backward()
    assert torch.isfinite(perturbed.grad).all()
    assert torch.all(perturbed.grad[missing] > 0)
    assert torch.equal(perturbed.grad[~missing], torch.zeros_like(perturbed.grad[~missing]))


def test_transient_rk4_and_gradient_through_time_varying_field():
    obs = observation()
    likelihood = fs.TransientLikelihood(obs["tracks"], obs["projection"], obs["times"].float(), obs["bounds"], sigma=.05,
                                        mask=obs["mask"], integrator="rk4")
    field = torch.zeros(3, 3, 4, 4, 4); field[:, 0] = torch.tensor([.2, .4, .6])[:, None, None, None]
    field.requires_grad_(); depth = likelihood.initialize_depth(random=False)
    positions, valid = likelihood.replay(field, depth)
    assert valid.all()
    assert torch.allclose(positions[0, :, :, 0] - positions[0, :, :1, 0], torch.tensor([0., .03, .13])[None], atol=2e-6)
    residual, mask, _ = likelihood.residual(field, depth)
    expected = torch.tensor([.01, .02])
    assert torch.allclose(residual[0, 0, :, 0], expected, atol=2e-6)
    residual.square().sum().backward(); assert torch.isfinite(field.grad).all() and field.grad.abs().sum() > 0


def test_train_stats_masked_and_time_query_changes_output(dataset):
    _, splits = fs.read_manifest(dataset[0]); stats = fs.fit_statistics(splits["train"])
    assert stats["mean"][0] < 1 and stats["mean"][1] < 1
    config = fs.config_from(train_args(dataset, Path("unused"), 1))
    system = fs.TemporalSystem("cnn", config, stats)
    obs = fs.load_observation(splits["train"][0])
    condition = system.condition(obs, torch.tensor([0, 2]))
    assert not torch.equal(condition[0], condition[1])
    prediction = system.decoder(condition)
    assert not torch.equal(prediction[0], prediction[1])


def test_resume_matches_uninterrupted_and_observation_reader_isolates_truth(dataset, tmp_path):
    a = fs.train(train_args(dataset, tmp_path / "full", 2))
    fs.train(train_args(dataset, tmp_path / "split", 1))
    b = fs.train(train_args(dataset, tmp_path / "split", 2, str(tmp_path / "split" / "latest.pt")))
    for group in ("system", "ema_system"):
        for key in a[group]: assert torch.equal(a[group][key], b[group][key]), key
    assert a["optimizer_steps"] == b["optimizer_steps"] == 4
    _, splits = fs.read_manifest(dataset[0]); row = splits["test"][0]
    obs = fs.load_observation(row)
    before, _ = fs.reconstruct(obs, "cnn", a["config"], a)
    path = Path(row["path"])
    with np.load(path) as archive: payload = {k: archive[k] for k in archive.files}
    payload["flow_field"] = payload["flow_field"] * -200
    payload["support_mask"] = ~payload["support_mask"]
    payload["probe_xyz"] = payload["probe_xyz"] * 100
    np.savez(path, **payload)
    after, _ = fs.reconstruct(fs.load_observation(row), "cnn", a["config"], a)
    assert torch.equal(before, after)
    assert (tmp_path / "full" / "epoch_0002.pt").is_file()
    assert json.loads((tmp_path / "split" / "run.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("kind", ["pod", "rbf"])
def test_inverse_temporal_coefficients_only_use_observations(dataset, tmp_path, kind):
    args = train_args(dataset, tmp_path / "fit", 1, model="pod")
    saved = fs.train(args) if kind == "pod" else None
    obs = fs.load_observation(dataset[2][2]); config = fs.config_from(args)
    field, extra = fs.reconstruct(obs, kind, config, saved)
    assert field.shape == (3, 3, 4, 4, 4) and torch.isfinite(field).all()
    assert extra["fitting"]["temporal_coefficients"] and not extra["fitting"]["reference_used"]


@pytest.mark.parametrize("kind", ["unet", "fno", "deeponet", "unet_diffusion"])
def test_each_enabled_learned_architecture_has_finite_temporal_gradients(dataset, kind):
    _, splits = fs.read_manifest(dataset[0]); stats = fs.fit_statistics(splits["train"])
    config = fs.config_from(train_args(dataset, Path("unused"), 1))
    config["diffusion_steps"] = 8
    config["unet"]["base_channels"] = 4
    config["unet_diffusion"]["base_channels"] = 4
    config["unet_diffusion"]["channel_multipliers"] = [1]
    config["fno"].update(width=4, depth=1, modes=2)
    config["deeponet"].update(hidden=8, latent=8, layers=1)
    system = fs.TemporalSystem(kind, config, stats)
    diffusion = fs.make_diffusion(config, "cpu") if kind == "unet_diffusion" else None
    obs = fs.load_observation(splits["train"][0]); field, mask, _ = fs.load_reference(splits["train"][0])
    loss = fs.loss_frame(system, diffusion, obs, field, mask, 1, stats, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)


def test_mean_evaluation_summary_and_case_resume(dataset, tmp_path):
    args = train_args(dataset, tmp_path / "mean", 1, model="mean")
    fs.train(args)
    evaluate = SimpleNamespace(manifest=str(dataset[0]), config=str(dataset[1]), seed=31, num_particles=12,
                               checkpoint=str(tmp_path / "mean" / "latest.pt"), model="mean", device="cpu",
                               output_dir=str(tmp_path / "eval"), split="validation", case_indices="0", resume=False)
    result = fs.evaluate(evaluate)
    assert result["status"] == "completed" and result["completed_cases"] == 1
    assert result["summary"]["physical_velocity_relative_l2"] >= 0
    assert result["summary"]["independent_probe_velocity_vector_rmse_m_s"] >= 0
    assert (tmp_path / "eval" / "summary.json").is_file()
    evaluate.resume = True
    assert fs.evaluate(evaluate)["summary"] == result["summary"]
