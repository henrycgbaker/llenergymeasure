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
    get_display_label,
    get_experiment_config_schema,
    get_field_metadata,
    get_field_role,
    get_param_test_values,
    get_shared_params,
    get_swept_field_paths,
    get_validation_rules,
    list_all_param_paths,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import DTYPE_SUPPORT
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
    """get_backend_params('vllm') returns params including vllm.engine.max_num_seqs."""
    params = get_backend_params("vllm")
    assert isinstance(params, dict)
    assert "vllm.engine.max_num_seqs" in params


def test_get_backend_params_tensorrt_returns_params():
    """get_backend_params('tensorrt') returns params including nested sub-config paths."""
    params = get_backend_params("tensorrt")
    assert isinstance(params, dict)
    assert "tensorrt.max_batch_size" in params
    # Verify expanded nested sub-config params are registered
    assert "tensorrt.quant.quant_algo" in params
    assert "tensorrt.kv_cache.free_gpu_memory_fraction" in params
    assert "tensorrt.scheduler.capacity_scheduling_policy" in params
    assert "tensorrt.build_cache.max_records" in params
    assert "tensorrt.sampling.return_perf_metrics" in params
    assert len(params) >= 20


def test_get_backend_params_unknown_backend_raises():
    """get_backend_params with unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_params("nonexistent_backend")


# ---------------------------------------------------------------------------
# get_shared_params
# ---------------------------------------------------------------------------


def test_get_shared_params_returns_model_field():
    """get_shared_params() contains 'model' — wait, 'model' is not in shared.

    Actually shared params are dtype, n, max_input_tokens, max_output_tokens
    plus decoder.* params. 'model' is a top-level required field in ExperimentConfig,
    not in the shared section of introspection.
    """
    params = get_shared_params()
    assert isinstance(params, dict)
    # Precision is a confirmed shared param
    assert "dtype" in params


def test_get_shared_params_contains_dtype():
    """get_shared_params() contains 'dtype'."""
    params = get_shared_params()
    assert "dtype" in params
    assert params["dtype"]["options"] == ["float32", "float16", "bfloat16"]


def test_get_shared_params_contains_n():
    """get_shared_params() contains 'dataset.n_prompts' (prompt count)."""
    params = get_shared_params()
    assert "dataset.n_prompts" in params


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
    """get_param_test_values('pytorch.batch_size') returns a list containing 1."""
    values = get_param_test_values("pytorch.batch_size")
    assert isinstance(values, list)
    assert 1 in values


def test_get_param_test_values_dtype_returns_all_options():
    """get_param_test_values('dtype') returns all 3 dtype options."""
    values = get_param_test_values("dtype")
    assert set(values) == {"float32", "float16", "bfloat16"}


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


def test_get_all_params_pytorch_section_contains_batch_size():
    """get_all_params()['pytorch'] contains the batch_size param."""
    all_params = get_all_params()
    assert "pytorch.batch_size" in all_params["pytorch"]


# ---------------------------------------------------------------------------
# list_all_param_paths
# ---------------------------------------------------------------------------


def test_list_all_param_paths_contains_expected_paths():
    """list_all_param_paths() returns a sorted list containing known param paths."""
    paths = list_all_param_paths()
    assert isinstance(paths, list)
    assert "pytorch.batch_size" in paths
    assert "dtype" in paths


def test_list_all_param_paths_contains_known_paths():
    """list_all_param_paths() contains expected well-known param paths."""
    paths = list_all_param_paths()
    assert "pytorch.batch_size" in paths
    assert "decoder.temperature" in paths
    assert "dtype" in paths


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


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["pytorch"])
def test_all_pytorch_dtype_values_produce_valid_config(dt):
    """Schema-driven: each SSOT DTYPE_SUPPORT['pytorch'] value creates a valid config."""
    config = make_config(dtype=dt)
    assert config.dtype == dt


def test_ssot_dtype_values_match_param_test_values():
    """DTYPE_SUPPORT['pytorch'] values match get_param_test_values('dtype')."""
    from_ssot = set(DTYPE_SUPPORT["pytorch"])
    from_introspection = set(get_param_test_values("dtype"))
    # The test values from introspection should cover all SSOT dtype values
    assert from_ssot == from_introspection


# ---------------------------------------------------------------------------
# get_validation_rules
# ---------------------------------------------------------------------------


def test_get_validation_rules_returns_list():
    """get_validation_rules() returns a list containing the backend mismatch rule."""
    rules = get_validation_rules()
    assert isinstance(rules, list)
    combinations = [r["combination"] for r in rules]
    assert any("mismatch" in c for c in combinations)


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


# ---------------------------------------------------------------------------
# Field metadata helpers (display_label / role)
# ---------------------------------------------------------------------------


def test_get_display_label_from_metadata():
    """get_display_label() returns 'Model' for ExperimentConfig.model field."""
    fi = ExperimentConfig.model_fields["model"]
    label = get_display_label(fi, "model")
    assert label == "Model"


def test_get_display_label_fallback():
    """get_display_label() falls back to title-cased name for fields without metadata."""
    fi = ExperimentConfig.model_fields["random_seed"]
    # random_seed has no json_schema_extra; expect title-cased fallback
    label = get_display_label(fi, "random_seed")
    assert label == "Random Seed"


def test_get_field_role_workload():
    """get_field_role() returns 'workload' for model field."""
    fi = ExperimentConfig.model_fields["model"]
    assert get_field_role(fi) == "workload"


def test_get_field_role_experimental():
    """get_field_role() returns 'experimental' for dtype field."""
    fi = ExperimentConfig.model_fields["dtype"]
    assert get_field_role(fi) == "experimental"


def test_get_field_role_none_for_unannotated():
    """get_field_role() returns None for fields without role metadata."""
    fi = ExperimentConfig.model_fields["random_seed"]
    assert get_field_role(fi) is None


def test_get_field_metadata_returns_correct_dict():
    """get_field_metadata(ExperimentConfig) returns correct dict for key fields."""
    meta = get_field_metadata(ExperimentConfig)
    assert isinstance(meta, dict)
    assert meta["model"]["label"] == "Model"
    assert meta["model"]["role"] == "workload"
    assert meta["dtype"]["label"] == "Dtype"
    assert meta["dtype"]["role"] == "experimental"
    assert meta["energy_sampler"]["label"] == "Sampler"
    # experiment_name has no role metadata
    assert meta["experiment_name"]["role"] is None


# ---------------------------------------------------------------------------
# get_swept_field_paths
# ---------------------------------------------------------------------------


def test_get_swept_field_paths_single_experiment():
    """Single experiment yields an empty swept set."""
    exp = ExperimentConfig(model="gpt2")
    result = get_swept_field_paths([exp])
    assert result == set()


def test_get_swept_field_paths_dtype_swept():
    """Two experiments with different dtypes yield {'dtype'} in swept paths."""
    exp1 = ExperimentConfig(model="gpt2", dtype="float16")
    exp2 = ExperimentConfig(model="gpt2", dtype="bfloat16")
    result = get_swept_field_paths([exp1, exp2])
    assert "dtype" in result


def test_get_swept_field_paths_nested_field():
    """Two experiments with different n_prompts yield dataset and dataset.n_prompts in swept."""
    from llenergymeasure.config.models import DatasetConfig

    exp1 = ExperimentConfig(model="gpt2", dataset=DatasetConfig(n_prompts=10))
    exp2 = ExperimentConfig(model="gpt2", dataset=DatasetConfig(n_prompts=50))
    result = get_swept_field_paths([exp1, exp2])
    assert "dataset" in result
    assert "dataset.n_prompts" in result
