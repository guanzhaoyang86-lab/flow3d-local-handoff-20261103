"""Regression checks for configuration values that can silently alter runs."""

import json

import pytest

from flow_baselines.config import resolve_config


def test_tuple_view_indices_are_normalized_without_mutating_input():
    views = (2, 0)
    result = resolve_config(overrides={"views": views})
    assert result["views"] == [2, 0]
    assert isinstance(result["views"], list)
    assert views == (2, 0)


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_amp_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match="amp"):
        resolve_config(overrides={"amp": value})


@pytest.mark.parametrize("value", [[0.5], [True], [0, 0], [-1], [], "0,1"])
def test_invalid_view_indices_are_rejected(value):
    with pytest.raises(ValueError, match="views"):
        resolve_config(overrides={"views": value})


@pytest.mark.parametrize("key", ["seed", "sampling_seed", "observation_seed"])
@pytest.mark.parametrize("value", [31.0, True, -1, 2**63])
def test_all_seed_fields_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match=key):
        resolve_config(overrides={key: value})


@pytest.mark.parametrize("value", [0, 2**63 - 1])
def test_seed_range_endpoints_are_valid(value):
    values = {key: value for key in ("seed", "sampling_seed", "observation_seed")}
    result = resolve_config(overrides=values)
    assert all(result[key] == value for key in values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, True, "8"])
def test_invalid_x0_clipping_is_rejected(value):
    with pytest.raises(ValueError, match="clip_x0"):
        resolve_config(overrides={"clip_x0": value})


@pytest.mark.parametrize("value", [None, 0.5])
def test_json_x0_clip_can_be_disabled_or_positive(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"clip_x0": value}))
    assert resolve_config(path)["clip_x0"] == value


@pytest.mark.parametrize("root", [[], [1, 2], "config", 5, None])
def test_json_configuration_root_must_be_an_object(tmp_path, root):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(root))
    with pytest.raises(ValueError, match="JSON object"):
        resolve_config(path)
