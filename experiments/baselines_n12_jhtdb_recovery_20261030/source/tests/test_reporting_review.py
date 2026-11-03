"""Independent statistical/provenance counterexamples found during code review."""
import hashlib
import json

import numpy as np
import pytest

from flow_baselines.reporting import summarize


def _prediction(tmp_path, case="a", seed=31, **changes):
    record = {
        "format": "flow3d-baseline-prediction-v1", "status": "completed", "model": "unet",
        "reference_used_during_inference": False, "actual_predictions": 1,
        "case_id": case, "training_seed": seed, "checkpoint_sha256": f"weights-{seed}",
        "source_case_sha256": f"source-{case}", "split": "test", "manifest_sha256": "same",
        "config": {"seed": seed, "num_particles": 2, "sampling_seed": 47},
        "evaluation": {"per_sample_field_relative_l2_unknown_interior": [1.]},
    }
    record.update(changes)
    path = tmp_path / f"{case}_{seed}_{len(list(tmp_path.glob('*.npz')))}.npz"
    values = np.zeros((record["actual_predictions"], 3, 2, 2, 2), np.float32)
    np.savez(path, predictions=values, metadata=np.array(json.dumps(record)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({**record, "output_sha256": digest}))
    return path


def test_sampling_seed_is_part_of_comparison_protocol(tmp_path):
    first = _prediction(tmp_path)
    other = _prediction(tmp_path, seed=32, config={"seed": 32, "num_particles": 2, "sampling_seed": 99})
    report = summarize([first.with_suffix(".json"), other.with_suffix(".json")], tmp_path / "summary.json")
    assert len(report["groups"]) == 2


@pytest.mark.parametrize("level", ("sample", "case", "seed"))
def test_partial_null_metrics_cannot_change_averaging_population(tmp_path, level):
    missing = {"per_sample_field_relative_l2_unknown_interior": [None]}
    if level == "sample":
        paths = [_prediction(tmp_path, actual_predictions=2,
                             evaluation={"per_sample_field_relative_l2_unknown_interior": [1., None]})]
    elif level == "case":
        paths = [_prediction(tmp_path, "a"), _prediction(tmp_path, "b", evaluation=missing)]
    else:
        paths = [_prediction(tmp_path), _prediction(tmp_path, seed=32, evaluation=missing)]
    with pytest.raises(ValueError, match="Incomplete metric coverage"):
        summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")


def test_all_unavailable_optional_metrics_remain_explicit_null(tmp_path):
    paths = [_prediction(tmp_path, seed=seed) for seed in (31, 32)]
    metrics = summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")["groups"][0]["metrics"]
    assert metrics["probe_rmse"]["mean"] is None
    assert metrics["probe_rmse"]["seed_count"] == 0


def test_summary_checks_artifact_integrity_before_using_sidecar_metrics(tmp_path):
    path = _prediction(tmp_path)
    path.write_bytes(b"corrupted prediction")
    with pytest.raises(ValueError, match="checksum"):
        summarize([path.with_suffix(".json")], tmp_path / "summary.json")


def test_one_seed_cannot_mix_checkpoint_weights_across_cases(tmp_path):
    paths = [_prediction(tmp_path, "a"), _prediction(tmp_path, "b", checkpoint_sha256="other-weights")]
    with pytest.raises(ValueError, match="checkpoint"):
        summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")


def test_training_seed_weights_may_differ_but_case_source_must_match(tmp_path):
    paths = [_prediction(tmp_path, seed=seed) for seed in (31, 32)]
    group = summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")["groups"][0]
    assert group["replicate_count"] == 2
    other = _prediction(tmp_path, seed=33, source_case_sha256="changed-source")
    with pytest.raises(ValueError, match="source"):
        summarize([paths[0].with_suffix(".json"), other.with_suffix(".json")], tmp_path / "summary.json")


def test_training_boundary_information_and_epoch_remain_protocol_keys(tmp_path):
    paths = [_prediction(tmp_path, training_boundary_projection=boundary, checkpoint_epoch=epoch)
             for boundary, epoch in (("none", 100), ("known", 100), ("none", 50))]
    report = summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")
    assert len(report["groups"]) == 3


def test_summary_accepts_verified_npz_inputs_and_reports_provenance(tmp_path):
    path = _prediction(tmp_path)
    group = summarize([path], tmp_path / "summary.json")["groups"][0]
    assert group["provenance_complete"] is True
    assert group["seed_case_means"][0]["checkpoint_sha256"] == "weights-31"


def test_legacy_missing_provenance_is_explicit_not_fabricated(tmp_path):
    path = _prediction(tmp_path, checkpoint_sha256=None, source_case_sha256=None)
    group = summarize([path.with_suffix(".json")], tmp_path / "summary.json")["groups"][0]
    assert group["provenance_complete"] is False


def test_explicit_json_glob_can_include_batch_progress(tmp_path):
    path = _prediction(tmp_path)
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"format": "flow3d-baseline-batch-v1", "status": "completed"}))
    report = summarize([path.with_suffix(".json"), progress], tmp_path / "summary.json")
    assert report["input_record_count"] == 1
    assert report["ignored_nonprediction_json_count"] == 1


def test_best_validation_checkpoints_may_select_different_epochs_across_seeds(tmp_path):
    paths = [_prediction(tmp_path, seed=seed, checkpoint_epoch=epoch, checkpoint_selection="best_validation")
             for seed, epoch in ((31, 63), (32, 84))]
    assert len(summarize([path.with_suffix(".json") for path in paths], tmp_path / "summary.json")["groups"]) == 1
