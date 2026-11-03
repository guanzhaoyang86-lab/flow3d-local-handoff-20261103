"""Batch orchestration tests use tiny saved arrays and a mocked predictor."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow_baselines.config import resolve_config


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baseline.py"
    spec = importlib.util.spec_from_file_location("evaluate_baseline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch(tmp_path, monkeypatch):
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    cases = []
    for index in range(3):
        path = tmp_path / f"source_{index}.npz"
        path.write_bytes(f"source {index}".encode())
        cases.append(SimpleNamespace(case_id=f"case_{index}", flow_group_id=f"group_{index}", path=path))
    monkeypatch.setattr(module, "load_sparse_flow_manifest", lambda _: ({}, {"validation": cases, "test": cases}))
    calls = []

    def predict(manifest, output, *, kind, checkpoint, config, split, case_index, device, evaluate):
        assert evaluate is True
        calls.append(case_index)
        cfg = resolve_config(overrides=config)
        if checkpoint:
            saved = torch.load(checkpoint, weights_only=True)
            cfg = resolve_config(overrides={**saved["config"], **config})
        case = cases[case_index]
        metadata = {"format": "flow3d-baseline-prediction-v1", "status": "completed", "model": kind,
                    "reference_used_during_inference": False, "actual_predictions": 1,
                    "manifest_sha256": module.sha256(manifest),
                    "source_case_sha256": module.sha256(case.path),
                    "checkpoint_sha256": module.sha256(checkpoint) if checkpoint else None,
                    "config": cfg, "split": split, "case_id": case.case_id, "flow_group_id": case.flow_group_id,
                    "evaluation": {"field_l2": 0.2}}
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, predictions=np.zeros((1, 3, 2, 2, 2), dtype=np.float32), metadata=np.array(json.dumps(metadata)))
        output.with_suffix(".json").write_text(json.dumps({**metadata, "output_sha256": module.sha256(output)}))
        return output

    monkeypatch.setattr(module, "predict", predict)
    args = module.parser().parse_args(["--manifest", str(manifest), "--output-dir", str(tmp_path / "output"),
                                      "--model", "rbf", "--device", "cpu"])
    return module, args, calls, cases


def test_case_selection_and_completed_output(batch):
    module, args, calls, _ = batch
    args.case_indices = "2,0"
    progress = module.run_batch(args)
    assert calls == [2, 0]
    assert progress["status"] == "completed"
    assert progress["completed_cases"] == 2
    assert progress["failed_cases"] == 0
    assert (Path(args.output_dir) / "case_000002.npz").is_file()
    assert module.parse_case_indices("all", 3) == [0, 1, 2]


@pytest.mark.parametrize("text", ["", "0,", "0,0", "-1", "3", "abc"])
def test_invalid_case_selection(text):
    with pytest.raises(ValueError):
        _module().parse_case_indices(text, 3)


def test_failed_case_does_not_mark_batch_completed_and_can_resume(batch, monkeypatch):
    module, args, calls, _ = batch
    original = module.predict

    def failing(*positional, **keywords):
        if keywords["case_index"] == 1:
            raise RuntimeError("synthetic failure")
        return original(*positional, **keywords)

    monkeypatch.setattr(module, "predict", failing)
    progress = module.run_batch(args)
    assert calls == [0, 2]
    assert progress["status"] == "failed"
    assert progress["completed_cases"] == 2
    assert progress["failed_cases"] == 1
    assert progress["cases"][1]["error"]["message"] == "synthetic failure"
    monkeypatch.setattr(module, "predict", original)
    args.resume = True
    progress = module.run_batch(args)
    assert calls == [0, 2, 1]
    assert progress["status"] == "completed"
    assert [row["resumed"] for row in progress["cases"]] == [True, False, True]


def test_no_resume_refuses_existing_outputs(batch):
    module, args, calls, _ = batch
    module.run_batch(args)
    with pytest.raises(FileExistsError, match="resume"):
        module.run_batch(args)
    assert calls == [0, 1, 2]


@pytest.mark.parametrize("change", ["seed", "particles", "config", "source", "manifest"])
def test_resume_rejects_changed_request_or_source(batch, tmp_path, change):
    module, args, calls, cases = batch
    module.run_batch(args)
    args.resume = True
    if change == "seed":
        args.seed = 99
    elif change == "particles":
        args.num_particles = 4
    elif change == "config":
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"sigma": 0.2}))
        args.config = str(path)
    elif change == "source":
        cases[0].path.write_bytes(b"changed source")
    elif change == "manifest":
        Path(args.manifest).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs"):
        module.run_batch(args)
    assert calls == [0, 1, 2]


def test_resume_rejects_corruption_before_new_predictions(batch):
    module, args, calls, _ = batch
    module.run_batch(args)
    args.resume = True
    (Path(args.output_dir) / "case_000001.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        module.run_batch(args)
    assert calls == [0, 1, 2]


def test_resume_rejects_partial_npz_pair(batch):
    module, args, calls, _ = batch
    output = Path(args.output_dir)
    output.mkdir()
    (output / "case_000000.npz").write_bytes(b"partial")
    args.resume = True
    with pytest.raises(ValueError, match="Incomplete"):
        module.run_batch(args)
    assert calls == []
    assert not (output / "progress.json").exists()


def test_resume_without_progress_still_checks_case_identity(batch):
    module, args, calls, _ = batch
    module.run_batch(args)
    args.resume = True
    output_dir = Path(args.output_dir)
    (output_dir / "progress.json").unlink()
    archive = output_dir / "case_000000.npz"
    with np.load(archive, allow_pickle=False) as payload:
        predictions = payload["predictions"].copy()
        metadata = json.loads(str(payload["metadata"].item()))
    metadata["case_id"] = "unrelated_case"
    np.savez_compressed(archive, predictions=predictions, metadata=np.array(json.dumps(metadata)))
    archive.with_suffix(".json").write_text(json.dumps({**metadata, "output_sha256": module.sha256(archive)}))
    with pytest.raises(ValueError, match="case_id"):
        module.run_batch(args)
    assert calls == [0, 1, 2]


def test_resume_checks_checkpoint_weights(batch, tmp_path):
    module, args, calls, _ = batch
    checkpoint = tmp_path / "mean.pt"
    saved = {"format": "flow3d-baseline-v1", "manifest_sha256": module.sha256(args.manifest),
             "model": "mean", "config": resolve_config(), "mean": torch.zeros(1, 3, 2, 2, 2)}
    torch.save(saved, checkpoint)
    args.model, args.checkpoint = "mean", str(checkpoint)
    module.run_batch(args)
    args.resume = True
    saved["mean"] += 1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="differs"):
        module.run_batch(args)
    assert calls == [0, 1, 2]


def test_cli_overrides_partial_pinn_config(batch, tmp_path):
    module, args, _, _ = batch
    config = tmp_path / "pinn.json"
    config.write_text(json.dumps({"pinn": {"hidden": 16, "viscosity": 0.2}, "seed": 32}))
    args.model, args.config = "pinn", str(config)
    args.viscosity, args.seed, args.num_particles = 0.03, 33, 6
    _, _, kind, effective, overrides = module._request_config(args)
    assert kind == "pinn"
    assert effective["pinn"]["viscosity"] == overrides["pinn"]["viscosity"] == 0.03
    assert effective["pinn"]["hidden"] == 16
    assert effective["seed"] == 33
    assert effective["num_particles"] == 6
