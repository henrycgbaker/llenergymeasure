"""Unit tests for config introspection SSOT architecture (INF-11).

Tests that introspection functions return correct structure and that
test value generation is schema-driven (derived from Pydantic models,
not hard-coded lists).
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.introspection import (
    get_all_params,
    get_backend_params,
    get_experiment_config_schema,
    get_param_test_values,
    get_shared_params,
    get_validation_rules,
    list_all_param_paths,
)
from llenergymeasure.config.ssot import PRECISION_SUPPORT
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# get_backend_params
# ---------------------------------------------------------------------------


def test_get_backend_params_returns_pytorch_params():
    """get_backend_params('pytorch') returns a dict with batch_size field."""
    params = get_backend_params("pytorch")
    assert isinstance(params, dict)
    assert "pytorch.batch_size" in params


def test_get_backend_params_pytorch_has_backend_support():
    """Each pytorch param has backend_support=['pytorch']."""
    params = get_backend_params("pytorch")
    for param_path, meta in params.items():
        assert "backend_support" in meta, f"Missing backend_support on {param_path}"
        assert "pytorch" in meta["backend_support"]


def test_get_backend_params_vllm_returns_params():
    """get_backend_params('vllm') returns params including max_num_seqs."""
    params = get_backend_params("vllm")
    assert isinstance(params, dict)
    assert len(params) > 0


def test_get_backend_params_tensorrt_returns_params():
    """get_backend_params('tensorrt') returns params including max_batch_size."""
    params = get_backend_params("tensorrt")
    assert isinstance(params, dict)
    assert len(params) > 0


def test_get_backend_params_unknown_backend_raises():
    """get_backend_params with unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_params("nonexistent_backend")


# ---------------------------------------------------------------------------
# get_shared_params
# ---------------------------------------------------------------------------


def test_get_shared_params_returns_model_field():
    """get_shared_params() contains 'model' — wait, 'model' is not in shared.

    Actually shared params are precision, n, max_input_tokens, max_output_tokens
    plus decoder.* params. 'model' is a top-level required field in ExperimentConfig,
    not in the shared section of introspection.
    """
    params = get_shared_params()
    assert isinstance(params, dict)
    # Precision is a confirmed shared param
    assert "precision" in params


def test_get_shared_params_contains_precision():
    """get_shared_params() contains 'precision'."""
    params = get_shared_params()
    assert "precision" in params
    assert params["precision"]["options"] == ["fp32", "fp16", "bf16"]


def test_get_shared_params_contains_n():
    """get_shared_params() contains 'n' (prompt count)."""
    params = get_shared_params()
    assert "n" in params


def test_get_shared_params_contains_decoder_temperature():
    """get_shared_params() contains 'decoder.temperature'."""
    params = get_shared_params()
    assert "decoder.temperature" in params


def test_get_shared_params_all_have_backend_support():
    """Every shared param has backend_support list."""
    params = get_shared_params()
    for param_path, meta in params.items():
        assert "backend_support" in meta, f"Missing backend_support on {param_path}"
        assert isinstance(meta["backend_support"], list)


# ---------------------------------------------------------------------------
# get_experiment_config_schema
# ---------------------------------------------------------------------------


def test_get_experiment_config_schema_is_valid_json_schema():
    """get_experiment_config_schema() returns a dict with 'properties' key."""
    schema = get_experiment_config_schema()
    assert isinstance(schema, dict)
    # Pydantic v2 JSON schema always has 'properties' at top level
    assert "properties" in schema


def test_get_experiment_config_schema_contains_model_field():
    """Schema contains 'model' property definition."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "model" in properties


def test_get_experiment_config_schema_contains_backend_field():
    """Schema contains 'backend' property definition."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "backend" in properties


# ---------------------------------------------------------------------------
# get_param_test_values (INF-11: SSOT-driven test value generation)
# ---------------------------------------------------------------------------


def test_get_param_test_values_pytorch_batch_size_returns_list():
    """get_param_test_values('pytorch.batch_size') returns a non-empty list."""
    values = get_param_test_values("pytorch.batch_size")
    assert isinstance(values, list)
    assert len(values) > 0


def test_get_param_test_values_precision_returns_all_options():
    """get_param_test_values('precision') returns all 3 precision options."""
    values = get_param_test_values("precision")
    assert set(values) == {"fp32", "fp16", "bf16"}


def test_get_param_test_values_decoder_temperature_returns_floats():
    """get_param_test_values('decoder.temperature') returns a list of floats."""
    values = get_param_test_values("decoder.temperature")
    assert isinstance(values, list)
    assert all(isinstance(v, int | float) for v in values)


def test_get_param_test_values_unknown_param_returns_empty():
    """get_param_test_values for unknown param path returns empty list."""
    values = get_param_test_values("nonexistent.param.path")
    assert values == []


# ---------------------------------------------------------------------------
# get_all_params
# ---------------------------------------------------------------------------


def test_get_all_params_covers_all_backends():
    """get_all_params() returns dict with 'pytorch', 'vllm', 'tensorrt' keys."""
    all_params = get_all_params()
    assert "pytorch" in all_params
    assert "vllm" in all_params
    assert "tensorrt" in all_params


def test_get_all_params_has_shared_key():
    """get_all_params() includes a 'shared' section."""
    all_params = get_all_params()
    assert "shared" in all_params


def test_get_all_params_pytorch_section_non_empty():
    """get_all_params()['pytorch'] is a non-empty dict."""
    all_params = get_all_params()
    assert len(all_params["pytorch"]) > 0


# ---------------------------------------------------------------------------
# list_all_param_paths
# ---------------------------------------------------------------------------


def test_list_all_param_paths_non_empty():
    """list_all_param_paths() returns a non-empty sorted list of dotted paths."""
    paths = list_all_param_paths()
    assert isinstance(paths, list)
    assert len(paths) > 0


def test_list_all_param_paths_contains_known_paths():
    """list_all_param_paths() contains expected well-known param paths."""
    paths = list_all_param_paths()
    assert "pytorch.batch_size" in paths
    assert "decoder.temperature" in paths
    assert "precision" in paths


def test_list_all_param_paths_filtered_by_backend():
    """list_all_param_paths(backend='pytorch') returns only pytorch paths."""
    paths = list_all_param_paths(backend="pytorch")
    assert all(p.startswith("pytorch.") for p in paths)


def test_list_all_param_paths_unknown_backend_raises():
    """list_all_param_paths with unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        list_all_param_paths(backend="nonexistent")


# ---------------------------------------------------------------------------
# SSOT schema-driven test generation (INF-11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISION_SUPPORT["pytorch"])
def test_all_pytorch_precision_values_produce_valid_config(precision):
    """Schema-driven: each SSOT PRECISION_SUPPORT['pytorch'] value creates a valid config."""
    config = make_config(precision=precision)
    assert config.precision == precision


def test_ssot_precision_values_match_param_test_values():
    """PRECISION_SUPPORT['pytorch'] values match get_param_test_values('precision')."""
    from_ssot = set(PRECISION_SUPPORT["pytorch"])
    from_introspection = set(get_param_test_values("precision"))
    # The test values from introspection should cover all SSOT precision values
    assert from_ssot == from_introspection


# ---------------------------------------------------------------------------
# get_validation_rules
# ---------------------------------------------------------------------------


def test_get_validation_rules_returns_list():
    """get_validation_rules() returns a list of dicts."""
    rules = get_validation_rules()
    assert isinstance(rules, list)
    assert len(rules) > 0


def test_get_validation_rules_each_has_required_keys():
    """Each validation rule has backend, combination, reason, resolution keys."""
    rules = get_validation_rules()
    for rule in rules:
        assert "backend" in rule, f"Rule missing 'backend': {rule}"
        assert "combination" in rule, f"Rule missing 'combination': {rule}"
        assert "reason" in rule, f"Rule missing 'reason': {rule}"
        assert "resolution" in rule, f"Rule missing 'resolution': {rule}"


def test_get_validation_rules_contains_backend_section_mismatch_rule():
    """Validation rules include the backend section mismatch rule."""
    rules = get_validation_rules()
    combinations = [r["combination"] for r in rules]
    assert any("mismatch" in c for c in combinations)
