"""Numerical CPU checks of Tensor-DiT runtime, independent of production prep."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "launch" / "tensor_model.py"
spec = importlib.util.spec_from_file_location("tensor_model", PATH)
model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)
torch.set_num_threads(1)


@pytest.fixture(params=["jhtdb", "fs_sph"])
def prepared(request, tmp_path, monkeypatch):
    dataset = request.param; row_list = []; observations = {}; references = {}
    shape = (4, 4, 4); frames = 3 if dataset == "fs_sph" else 1
    for case in range(4):
        path = tmp_path / f"window_{case}.npz"
        times = torch.tensor([1., 1.1, 1.3], dtype=torch.float64) + case * .5
        tracks = torch.zeros(1, 12, 3, 2); tracks[0, :, :, 0] = torch.linspace(.5, 1.5, 12)[:, None] + torch.tensor([0., .01, .03])[None]; tracks[..., 1] = .7
        field = torch.zeros(frames, 3, *shape)
        for j in range(frames): field[j, 0] = .2 + .1 * case + .05 * j
        field[:, 1] = .01; field[:, 2] = .03
        support = torch.ones(frames, *shape, dtype=torch.bool)
        if dataset == "fs_sph": support[..., 0] = False
        reference_times = times if frames > 1 else times[:1]
        info = {"unit_scale": [2., 3., 4.], "grid_shape_zyx": list(shape)}
        np.savez(path, flow_field=field.numpy(), support_mask=support.numpy(), velocity_times=reference_times.numpy(),
                 trajectories_2d=tracks.numpy()[None], observation_mask=np.ones((1, 1, 12, 3), bool),
                 observation_times=times.numpy(), projection_matrix=np.array([[[1., 0., 0.], [0., 0., 1.]]]),
                 domain_bounds=np.array([[0., 3.]] * 3), metadata=np.array(json.dumps(info)))
        row = {"case_id": f"case_{case}", "path": str(path), "flow_group_id": "one_process" if dataset == "fs_sph" else f"block_{case}", "window_id": case}
        row_list.append(row)
        observations[row["case_id"]] = {"tracks": tracks, "mask": torch.ones(1, 12, 3, dtype=torch.bool), "times": times,
                                        "projection": torch.tensor([[[1., 0., 0.], [0., 0., 1.]]]), "bounds": torch.tensor([[0., 3.]] * 3),
                                        "unit_scale": torch.tensor([2., 3., 4.]), "shape": shape}
        references[row["case_id"]] = (field, support, reference_times)
    splits = {"train": row_list[:2], "validation": row_list[2:3], "test": row_list[3:]}
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"dataset": dataset, "splits": splits}))
    anchors = [torch.linalg.qr(torch.arange(1., 9.).reshape(4, 2) + torch.eye(4, 2) * 3).Q for _ in range(3)]
    codec = model.TensorSpaceCodec(shape, 2, anchors, hooi_iterations=0)
    statistics = {"mean": [0., 0., 0.], "std": [1., 1., 1.], "shape": list(shape), "motion_scales": [.1, .1, .1, .1],
                  "mean_field": torch.full((3, *shape), .2), "train_support": torch.ones(shape, dtype=torch.bool), "fit_split": "train"}
    latents = {}; weights = {}
    for split in ("train", "validation"):
        latents[split] = {}; weights[split] = {}
        for row in splits[split]:
            field, mask, times = references[row["case_id"]]
            filled = torch.where(mask[:, None], field, statistics["mean_field"][None])
            latents[split][row["case_id"]] = codec.encode(filled)
            weights[split][row["case_id"]] = model.fs.frame_weights(times) if len(times) > 1 else torch.ones(1)
    artifact = {"format": "tensor_n12_artifact_v1", "dataset": dataset, "manifest_sha256": model.fs.digest(manifest), "codec": codec.to_artifact(),
                "statistics": statistics, "latents": latents, "frame_weights": weights, "rank": 2, "latent_dim": codec.latent_dim}
    artifact_path = tmp_path / "selected.pt"; torch.save(artifact, artifact_path)
    api = SimpleNamespace(read_manifest=lambda path, ds: ({"dataset": ds}, splits),
                          load_observation=lambda row, ds, seed=47, epoch=0: copy_observation(observations[row["case_id"]]),
                          load_reference=lambda row, ds: references[row["case_id"]],
                          load_artifact=lambda path, manifest, ds: torch.load(path, weights_only=True))
    monkeypatch.setattr(model, "data_api", lambda: api)
    config = model.default_config(dataset)
    config.update(amp=False, dit_hidden_dim=12, dit_depth=1, dit_heads=3, condition_dim=8, condition_flow_grid=2,
                  diffusion_steps=8, sampling_steps=2, num_samples=2, decoded_auxiliary_max_t_fraction=1., depth_fit_steps=2)
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    return SimpleNamespace(dataset=dataset, rows=row_list, splits=splits, manifest=manifest, artifact=artifact, artifact_path=artifact_path,
                           config=config, config_path=config_path, statistics=statistics, codec=codec, references=references, observations=observations)


def copy_observation(obs):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}


def args(prepared, out, epochs=1, resume=None):
    return SimpleNamespace(dataset=prepared.dataset, manifest=str(prepared.manifest), artifact=str(prepared.artifact_path), config=str(prepared.config_path),
                           output_dir=str(out), epochs=epochs, seed=31, device="cpu", resume=resume, max_train_cases=None, max_validation_cases=None)


def test_default_recipe_and_no_factor_clipping():
    j = model.default_config("jhtdb"); f = model.default_config("fs_sph")
    assert (j["dit_hidden_dim"], j["dit_depth"], j["dit_heads"]) == (192, 4, 6)
    assert j["prediction_type"] == "v" and j["clip_x0"] is None and j["decoded_field_weight"] == 0
    assert f["decoded_field_weight"] == .1 and f["physical_time_adapter"]
    assert model.make_diffusion(j, "cpu").clip_x0 is None


def test_tensor_latent_loss_masked_aux_and_finite_gradients(prepared):
    c = prepared.config; system = model.TensorSystem(c, prepared.statistics, prepared.codec)
    diffusion = model.make_diffusion(c, "cpu"); row = prepared.rows[0]
    field, support, times = prepared.references[row["case_id"]]
    latent = prepared.artifact["latents"]["train"][row["case_id"]][:1]
    obs = prepared.observations[row["case_id"]]
    loss = model.frame_loss(system, prepared.codec, diffusion, obs, latent, field, support, 0, prepared.statistics, c,
                             generator=torch.Generator().manual_seed(3))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None)
    if prepared.dataset == "fs_sph":
        changed = field.clone(); changed[~support[:, None].expand_as(field)] = torch.nan
        before = model.frame_loss(system, prepared.codec, diffusion, obs, latent, field, support, 0, prepared.statistics, c,
                                   generator=torch.Generator().manual_seed(4))
        after = model.frame_loss(system, prepared.codec, diffusion, obs, latent, changed, support, 0, prepared.statistics, c,
                                  generator=torch.Generator().manual_seed(4))
        assert torch.equal(before, after)
        a, b = system.condition(obs, 0), system.condition(obs, 2)
        assert not torch.equal(a, b)


def test_tensor_exact_resume_and_observation_only_sampling(prepared, tmp_path):
    full = model.train(args(prepared, tmp_path / "full", 2))
    model.train(args(prepared, tmp_path / "resume", 1))
    resumed = model.train(args(prepared, tmp_path / "resume", 2, str(tmp_path / "resume" / "latest.pt")))
    for group in ("system", "ema_system"):
        for key in full[group]: assert torch.equal(full[group][key], resumed[group][key]), key
    assert full["epoch"] == 2 and full["optimizer_steps"] == resumed["optimizer_steps"] == 4
    obs = copy_observation(prepared.observations[prepared.rows[-1]["case_id"]])
    before = model.reconstruct(obs, full)
    field, support, times = prepared.references[prepared.rows[-1]["case_id"]]
    field[:] = 10000; support[:] = False
    after = model.reconstruct(obs, full)
    assert torch.equal(before, after) and torch.isfinite(after).all()
    assert after.shape[:3] == (2, 3 if prepared.dataset == "fs_sph" else 1, 3)
    assert json.loads((tmp_path / "full" / "run.json").read_text())["status"] == "completed"
    # Resume may extend final epochs, but cannot change the sampled training subset.
    changed_args = args(prepared, tmp_path / "resume", 3, str(tmp_path / "resume" / "latest.pt")); changed_args.max_train_cases = 1
    with pytest.raises(ValueError, match="identity"):
        model.train(changed_args)


def test_pilot_subset_checkpoint_and_validation_sample(prepared, tmp_path):
    pilot_args = args(prepared, tmp_path / "pilot"); pilot_args.max_train_cases = 1; pilot_args.max_validation_cases = 1
    saved = model.train(pilot_args)
    assert saved["pilot_subset"] and saved["max_train_cases"] == 1 and saved["config"]["epochs"] == 1
    evaluate = SimpleNamespace(dataset=prepared.dataset, manifest=str(prepared.manifest), checkpoint=str(tmp_path / "pilot" / "latest.pt"),
                               config=None, output_dir=str(tmp_path / "evaluation"), split="validation", case_indices="0", device="cpu", resume=False)
    result = model.evaluate(evaluate)
    assert result["status"] == "completed" and result["summary"]["physical_velocity_relative_l2"] >= 0
    record = json.loads((tmp_path / "evaluation" / "case_000000.json").read_text())
    assert not record["reference_used_during_inference"] and not record["support_mask_used_during_inference"]
    assert record["pilot_subset_checkpoint"]
    evaluate.resume = True
    assert model.evaluate(evaluate)["summary"] == result["summary"]


@pytest.mark.parametrize("dataset", ["jhtdb", "fs_sph"])
def test_actual_data_prepare_train_and_validation_evaluation(dataset, tmp_path, monkeypatch):
    """Exercise real archive/schema, Tucker preparation, runtime and posthoc probes.

    The held-out test file remains deliberately unreadable throughout: neither
    rank selection, pilot fitting, nor validation sampling may open its arrays.
    """
    fixture_spec = importlib.util.spec_from_file_location("tensor_fixture_helpers", Path(__file__).with_name("test_tensor_data.py"))
    helpers = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(helpers)
    manifest = helpers.fixture_manifest(tmp_path, dataset)
    api = helpers.data
    # Use the real loader and artifact implementation rather than the mock fixture.
    monkeypatch.setattr(model, "data_api", lambda: api)
    _, splits = api.read_manifest(manifest, dataset)
    for split in ("train", "validation"):
        for row in splits[split]:
            path = Path(row["path"])
            with np.load(path) as archive: values = {k: archive[k] for k in archive.files}
            if dataset == "fs_sph":
                support = values["support_mask"]
                values["flow_field"][~np.broadcast_to(support[:, None], values["flow_field"].shape)] = np.nan
                times = values["observation_times"]; count = len(times)
                xyz = torch.ones(32, count, 3)
                safe_field = torch.from_numpy(np.where(support[:, None], values["flow_field"], 0)).float()
                velocity = model.sample_velocity(safe_field, xyz.reshape(1, -1, 3),
                                                  torch.tensor(times, dtype=torch.float32)[None].expand(32, -1).reshape(1, -1),
                                                  torch.tensor(values["domain_bounds"], dtype=torch.float32), "clamp",
                                                  velocity_times=torch.tensor(times, dtype=torch.float32)).reshape_as(xyz)
                values.update(probe_xyz=xyz.numpy(), probe_velocity=velocity.numpy(), probe_valid=np.ones((32, count), bool))
            else:
                tracks = values["trajectories_2d"][0, 0]
                xyz = np.ones((1, tracks.shape[0], tracks.shape[1], 3), np.float32)
                xyz[0, ..., 0] = tracks[..., 0]; xyz[0, ..., 2] = tracks[..., 1]
                values.update(trajectories_3d=xyz, observation_mask_3d=np.ones(xyz.shape[:-1], bool))
            np.savez(path, **values)
    prep = tmp_path / "artifacts"
    marker = api.prepare_artifacts(manifest, dataset, prep, ranks=[4], hooi_iterations=2)
    assert marker["status"] == "completed" and marker["test_fields_read"] is False
    config = model.default_config(dataset)
    config.update(epochs=1, amp=False, dit_hidden_dim=12, dit_depth=1, dit_heads=3, condition_dim=8, condition_flow_grid=2,
                  diffusion_steps=8, sampling_steps=2, depth_fit_steps=2)
    config_path = tmp_path / "real_config.json"; config_path.write_text(json.dumps(config))
    train = SimpleNamespace(dataset=dataset, manifest=str(manifest), artifact=str(prep / "selected.pt"), config=str(config_path),
                            output_dir=str(tmp_path / "pilot"), epochs=1, seed=31, device="cpu", resume=None,
                            max_train_cases=1, max_validation_cases=1)
    saved = model.train(train)
    assert saved["model"] == "tensor_dit" and saved["optimizer_steps"] == 1
    evaluate = SimpleNamespace(dataset=dataset, manifest=str(manifest), checkpoint=str(tmp_path / "pilot" / "latest.pt"), config=None,
                               output_dir=str(tmp_path / "validation"), split="validation", case_indices="0", device="cpu", resume=False)
    result = model.evaluate(evaluate)
    assert result["status"] == "completed" and result["completed_cases"] == 1
    record = json.loads((tmp_path / "validation" / "case_000000.json").read_text())
    assert not record["reference_used_during_inference"] and not record["support_mask_used_during_inference"]
    if dataset == "fs_sph":
        assert np.isfinite(record["metrics"]["independent_probe_velocity_vector_rmse_m_s"])
        assert record["metrics"]["probe_positions_used_only_after_prediction"]
    else:
        assert record["metrics"]["probe_initial_xyz_is_posthoc_only"]
        assert np.isfinite(record["metrics"]["heldout_projected_track_vector_rmse_native"])
    assert Path(splits["test"][0]["path"]).read_bytes() == b"TEST MUST NOT BE OPENED FOR TUCKER FIT"
    evaluate.resume = True
    assert model.evaluate(evaluate)["summary"] == result["summary"]
