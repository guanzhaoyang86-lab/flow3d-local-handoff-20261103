"""Small analytic Tucker/data tests; no real data preparation or GPU work."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tensor_data_tests", ROOT / "launch/tensor_data.py")
data = importlib.util.module_from_spec(spec); spec.loader.exec_module(data)
torch.set_num_threads(1)


def fixture_manifest(tmp_path, dataset="fs_sph", random_fields=False):
    splits = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(45)
    z, y, x = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij")
    base = np.stack((x + .4 * y + .2, y - .1 * z + .3, z + .2 * x - .1)).astype(np.float32)
    projection = np.array([[[1, 0, 0], [0, 0, 1]]], np.float32)
    for index, split in enumerate(("train", "train", "validation", "test")):
        path = tmp_path / f"case_{index}.npz"
        row = dict(case_id=f"case_{index}", path=path.name, flow_group_id="fs_sph_3429811" if dataset == "fs_sph" else f"block_{index}", window_id=index)
        splits[split].append(row)
        if split == "test":
            path.write_bytes(b"TEST MUST NOT BE OPENED FOR TUCKER FIT")
            continue
        times = np.array([1., 1.01, 1.1]) + index if dataset == "fs_sph" else np.linspace(0, .005, 20)
        frames = 3 if dataset == "fs_sph" else 1
        field = np.stack([base * (1 + .1 * index + .03 * j) for j in range(frames)])
        if random_fields: field = rng.normal(size=field.shape).astype(np.float32)
        support = np.ones((frames, 4, 4, 4), bool)
        if dataset == "fs_sph":
            support[:, :, :, 0] = False
            field[:, :, :, :, 0] = 1e12  # Masked storage must never donate statistics/latents.
        n = 12 if dataset == "fs_sph" else 160
        tracks = rng.uniform(.5, 2., size=(1, 1, n, len(times), 2)).astype(np.float32)
        payload = dict(flow_field=field, support_mask=support, velocity_times=times[:frames],
                       trajectories_2d=tracks, observation_mask=np.ones(tracks.shape[:-1], bool),
                       projection_matrix=projection, observation_times=times, domain_bounds=np.array([[0., 3.]] * 3),
                       solid_mask=np.zeros((4, 4, 4), bool), particle_counts=np.array([n]),
                       metadata=np.array(json.dumps({"unit_scale": [2., 3., 4.], "physical_to_cell_scale_xyz": [2., 3., 4.], "grid_shape_zyx": [4] * 3})))
        np.savez(path, **payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema="fs_transient_baseline_manifest_v1", mode="within_case_temporal" if dataset == "fs_sph" else "scientific", splits=splits)))
    return manifest


def test_jhtdb_observation_selection_exactly_matches_baseline_and_reserves_probes(tmp_path):
    from flow_observation.dps.data import load_observation_case
    from flow_observation.sparse_dataset import SparseFlowCase
    manifest = fixture_manifest(tmp_path, "jhtdb")
    _, splits = data.read_manifest(manifest, "jhtdb"); row = splits["train"][0]
    actual = data.load_observation(row, "jhtdb", seed=47, epoch=0)
    expected = load_observation_case(SparseFlowCase(row["case_id"], Path(row["path"]), row["flow_group_id"]),
                                     "train", 12, observation_seed=47, num_probes=32, views=[0])
    assert torch.equal(actual["tracks"], expected["tracks"])
    assert torch.equal(actual["particle_indices"], expected["particle_indices"])
    assert torch.equal(actual["probe_particle_indices"], expected["probe_particle_indices"])
    changed = data.load_observation(row, "jhtdb", seed=47, epoch=1)
    assert torch.equal(actual["probe_particle_indices"], changed["probe_particle_indices"])
    assert not torch.equal(actual["particle_indices"], changed["particle_indices"])
    assert not set(changed["particle_indices"].tolist()) & set(changed["probe_particle_indices"].tolist())
    from flow_baselines.common import TrainingDataset
    from flow_baselines.config import resolve_config
    from flow_observation.sparse_dataset import FlowNormalizationStats
    config = resolve_config(overrides=dict(num_particles=12, num_probes=32, views=[0], observation_seed=47))
    training = TrainingDataset(manifest, "train", config,
                               FlowNormalizationStats(torch.zeros(3), torch.ones(3), count=1))
    for epoch in (0, 1):
        training.set_epoch(epoch); baseline = training[0]
        tensor = data.load_observation(row, "jhtdb", seed=47, epoch=epoch)
        assert torch.equal(tensor["particle_indices"], baseline["particle_indices"])
        assert torch.equal(tensor["probe_particle_indices"], baseline["probe_particle_indices"])
        assert torch.equal(tensor["tracks"], baseline["tracks"])
    field, support, times = data.load_reference(row, "jhtdb")
    assert field.shape == (1, 3, 4, 4, 4) and support.all() and times.shape == (1,)


def test_train_only_statistics_mask_imputation_and_time_weights(tmp_path):
    manifest = fixture_manifest(tmp_path); _, splits = data.read_manifest(manifest, "fs_sph")
    stats = data.fit_statistics(splits["train"], "fs_sph")
    assert max(abs(x) for x in stats["mean"]) < 10
    before = stats["mean_field"].clone()
    row = splits["validation"][0]
    with np.load(row["path"]) as archive: arrays = {key: archive[key] for key in archive.files}
    arrays["flow_field"] *= -10000; np.savez(row["path"], **arrays)
    after = data.fit_statistics(splits["train"], "fs_sph")
    assert torch.equal(before, after["mean_field"])
    field, support, _ = data.load_reference(splits["train"][0], "fs_sph")
    changed = field.clone(); changed[~support[:, None].expand_as(field)] = torch.nan
    filled = data.impute_normalized(changed, support, stats)
    assert torch.isfinite(filled).all()
    assert torch.allclose(data.frame_weights([0., .1, 1.]), torch.tensor([.05, .5, .45]))
    with pytest.raises(ValueError, match="train only"):
        data.fit_statistics(splits["validation"], "fs_sph")


def test_weighted_medoid_respects_frame_time_mass():
    a = torch.tensor([[1.], [0.]])
    b = torch.tensor([[0.], [1.]])
    factors = [[a, a, a], [a, a, a], [b, b, b]]
    anchors, indices = data.weighted_medoid_anchors(factors, [.01, .01, .98])
    assert indices == [2, 2, 2]
    assert all(torch.equal(value, b) for value in anchors)


def test_rank_gate_pass_artifact_selfcontained_resume_and_source_tampering(tmp_path):
    manifest = fixture_manifest(tmp_path)
    output = tmp_path / "tucker"
    marker = data.prepare_artifacts(manifest, "fs_sph", output, ranks=[4], hooi_iterations=2)
    assert marker["status"] == "completed" and marker["selected_rank"] == 4
    assert marker["train_frames"] == 6 and marker["validation_frames"] == 3
    artifact = data.load_artifact(output / "selected.pt", manifest, "fs_sph")
    assert set(artifact["latents"]) == {"train", "validation"}
    assert artifact["statistics"]["fit_split"] == "train"
    assert artifact["provenance"]["test_fields_read"] is False
    assert artifact["provenance"]["minimum_latent_std"] == 1e-4
    assert bool((artifact["latent_std"] >= 1e-4).all())
    codec = data.TensorSpaceCodec.from_artifact(artifact["codec"])
    assert codec.latent_dim == artifact["latent_dim"]
    assert data.prepare_artifacts(manifest, "fs_sph", output, ranks=[4], resume=True) == marker
    _, splits = data.read_manifest(manifest, "fs_sph")
    with Path(splits["train"][0]["path"]).open("ab") as stream: stream.write(b"source changed")
    with pytest.raises(ValueError, match="source archive changed"):
        data.load_artifact(output / "selected.pt", manifest, "fs_sph")
    with pytest.raises(ValueError, match="differs"):
        data.prepare_artifacts(manifest, "fs_sph", output, ranks=[4], resume=True)


def test_all_candidate_ranks_fail_without_publishing_complete(tmp_path):
    manifest = fixture_manifest(tmp_path, random_fields=True); output = tmp_path / "failed"
    with pytest.raises(ValueError, match="No candidate Tucker rank"):
        data.prepare_artifacts(manifest, "fs_sph", output, ranks=[1], hooi_iterations=2)
    assert not (output / "COMPLETE.json").exists()
    assert not (output / "selected.pt").exists()
    assert (output / "FAILED_RANK_GATE.json").is_file()
    with pytest.raises(ValueError, match="No candidate Tucker rank"):
        data.prepare_artifacts(manifest, "fs_sph", output, ranks=[1], hooi_iterations=2, resume=True)


def test_physical_reconstruction_score_uses_component_inverse_units():
    target = torch.ones(2, 3, 2, 2, 2) * torch.tensor([2., 6., 12.]).view(1, 3, 1, 1, 1)
    predicted = target.clone(); predicted[:, 0] += 2
    score = data.reconstruction_score(predicted, target, torch.ones(2, 2, 2, 2, dtype=torch.bool),
                                      [10., 11.], torch.tensor([2., 3., 4.]))
    assert abs(score - 1 / (14 ** .5)) < 1e-6
