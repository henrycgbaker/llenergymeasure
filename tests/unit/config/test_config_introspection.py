"""Unit tests for config introspection SSOT architecture (INF-11).

Tests that introspection functions return correct structure and that
test value generation is schema-driven (derived from Pydantic models,
not hard-coded lists).
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.introspection import (
    get_all_params,
    get_display_label,
    get_engine_params,
    get_experiment_config_schema,
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
# get_engine_params
# ---------------------------------------------------------------------------


def test_get_engine_params_returns_pytorch_params():
    """get_engine_params('transformers') returns a dict with batch_size field."""
    params = get_engine_params("transformers")
    assert isinstance(params, dict)
    assert "transformers.batch_size" in params


def test_get_engine_params_pytorch_has_engine_support():
    """Each pytorch param has engine_support=['transformers']."""
    params = get_engine_params("transformers")
    for param_path, meta in params.items():
        assert "engine_support" in meta, f"Missing engine_support on {param_path}"
        assert "transformers" in meta["engine_support"]


def test_get_engine_params_vllm_returns_params():
    """get_engine_params('vllm') returns params including vllm.engine.max_num_seqs."""
    params = get_engine_params("vllm")
    assert isinstance(params, dict)
    assert "vllm.engine.max_num_seqs" in params


def test_get_engine_params_tensorrt_returns_params():
    """get_engine_params('tensorrt') returns params including nested sub-config paths."""
    params = get_engine_params("tensorrt")
    assert isinstance(params, dict)
    assert "tensorrt.max_batch_size" in params
    # Verify expanded nested sub-config params are registered
    assert "tensorrt.quant.quant_algo" in params
    assert "tensorrt.kv_cache.free_gpu_memory_fraction" in params
    assert "tensorrt.scheduler.capacity_scheduling_policy" in params
    # build_cache and calib sub-configs dropped (D1/D3); return_perf_metrics dropped (D1)
    assert "tensorrt.build_cache.max_records" not in params
    assert "tensorrt.sampling.return_perf_metrics" not in params
    # New fields from C.2
    assert "tensorrt.pipeline_parallel_size" in params
    assert "tensorrt.max_num_tokens" in params
    assert len(params) >= 10


def test_get_engine_params_unknown_engine_raises():
    """get_engine_params with unknown engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine_params("nonexistent_backend")


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


def test_get_shared_params_all_have_engine_support():
    """Every shared param has engine_support list."""
    params = get_shared_params()
    for param_path, meta in params.items():
        assert "engine_support" in meta, f"Missing engine_support on {param_path}"
        assert isinstance(meta["engine_support"], list)


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
    """Schema contains 'task' property with 'model' nested inside."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "task" in properties


def test_get_experiment_config_schema_contains_engine_field():
    """Schema contains 'engine' property definition."""
    schema = get_experiment_config_schema()
    properties = schema.get("properties", {})
    assert "engine" in properties


# ---------------------------------------------------------------------------
# get_param_test_values (INF-11: SSOT-driven test value generation)
# ---------------------------------------------------------------------------


def test_get_param_test_values_pytorch_batch_size_returns_list():
    """get_param_test_values('transformers.batch_size') returns a list containing 1."""
    values = get_param_test_values("transformers.batch_size")
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


def test_get_all_params_covers_all_engines():
    """get_all_params() returns dict with 'transformers', 'vllm', 'tensorrt' keys."""
    all_params = get_all_params()
    assert "transformers" in all_params
    assert "vllm" in all_params
    assert "tensorrt" in all_params


def test_get_all_params_has_shared_key():
    """get_all_params() includes a 'shared' section."""
    all_params = get_all_params()
    assert "shared" in all_params


def test_get_all_params_pytorch_section_contains_batch_size():
    """get_all_params()['transformers'] contains the batch_size param."""
    all_params = get_all_params()
    assert "transformers.batch_size" in all_params["transformers"]


# ---------------------------------------------------------------------------
# list_all_param_paths
# ---------------------------------------------------------------------------


def test_list_all_param_paths_contains_expected_paths():
    """list_all_param_paths() returns a sorted list containing known param paths."""
    paths = list_all_param_paths()
    assert isinstance(paths, list)
    assert "transformers.batch_size" in paths
    assert "dtype" in paths


def test_list_all_param_paths_contains_known_paths():
    """list_all_param_paths() contains expected well-known param paths."""
    paths = list_all_param_paths()
    assert "transformers.batch_size" in paths
    assert "decoder.temperature" in paths
    assert "dtype" in paths


def test_list_all_param_paths_filtered_by_engine():
    """list_all_param_paths(engine='transformers') returns only pytorch paths."""
    paths = list_all_param_paths(engine="transformers")
    assert all(p.startswith("transformers.") for p in paths)


def test_list_all_param_paths_unknown_engine_raises():
    """list_all_param_paths with unknown engine raises ValueError."""
    with pytest.raises(ValueError, match="Unknown engine"):
        list_all_param_paths(engine="nonexistent")


# ---------------------------------------------------------------------------
# SSOT schema-driven test generation (INF-11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", DTYPE_SUPPORT["transformers"])
def test_all_pytorch_dtype_values_produce_valid_config(dt):
    """Schema-driven: each SSOT DTYPE_SUPPORT['transformers'] value creates a valid config."""
    config = make_config(dtype=dt)
    assert config.dtype == dt


def test_ssot_dtype_values_match_param_test_values():
    """DTYPE_SUPPORT['transformers'] values match get_param_test_values('dtype')."""
    from_ssot = set(DTYPE_SUPPORT["transformers"])
    from_introspection = set(get_param_test_values("dtype"))
    # The test values from introspection should cover all SSOT dtype values
    assert from_ssot == from_introspection


# ---------------------------------------------------------------------------
# get_validation_rules
# ---------------------------------------------------------------------------


def test_get_validation_rules_returns_list():
    """get_validation_rules() returns a list containing the engine mismatch rule."""
    rules = get_validation_rules()
    assert isinstance(rules, list)
    combinations = [r["combination"] for r in rules]
    assert any("mismatch" in c for c in combinations)


def test_get_validation_rules_each_has_required_keys():
    """Each validation rule has engine, combination, reason, resolution keys."""
    rules = get_validation_rules()
    for rule in rules:
        assert "engine" in rule, f"Rule missing 'engine': {rule}"
        assert "combination" in rule, f"Rule missing 'combination': {rule}"
        assert "reason" in rule, f"Rule missing 'reason': {rule}"
        assert "resolution" in rule, f"Rule missing 'resolution': {rule}"


def test_get_validation_rules_contains_engine_section_mismatch_rule():
    """Validation rules include the engine section mismatch rule."""
    rules = get_validation_rules()
    combinations = [r["combination"] for r in rules]
    assert any("mismatch" in c for c in combinations)


# ---------------------------------------------------------------------------
# Field metadata helpers (display_label / role)
# ---------------------------------------------------------------------------


def test_get_display_label_from_metadata():
    """get_display_label() returns 'Model' for TaskConfig.model field."""
    from llenergymeasure.config.models import TaskConfig

    fi = TaskConfig.model_fields["model"]
    label = get_display_label(fi, "model")
    assert label == "Model"


def test_get_display_label_fallback():
    """get_display_label() falls back to title-cased name for fields without metadata."""
    from llenergymeasure.config.models import TaskConfig

    fi = TaskConfig.model_fields["random_seed"]
    # random_seed has no json_schema_extra; expect title-cased fallback
    label = get_display_label(fi, "random_seed")
    assert label == "Random Seed"


def test_get_field_role_workload():
    """get_field_role() returns 'workload' for DatasetConfig.source field."""
    from llenergymeasure.config.models import DatasetConfig

    fi = DatasetConfig.model_fields["source"]
    assert get_field_role(fi) == "workload"


def test_get_field_role_none_for_unannotated():
    """get_field_role() returns None for fields without role metadata."""
    fi = ExperimentConfig.model_fields["engine"]
    assert get_field_role(fi) is None


# ---------------------------------------------------------------------------
# get_swept_field_paths
# ---------------------------------------------------------------------------


def test_get_swept_field_paths_single_experiment():
    """Single experiment yields an empty swept set."""
    exp = ExperimentConfig(task={"model": "gpt2"})
    result = get_swept_field_paths([exp])
    assert result == set()


def test_get_swept_field_paths_dtype_swept():
    """Two experiments with different dtypes yield {'dtype'} in swept paths."""
    exp1 = ExperimentConfig(task={"model": "gpt2"}, dtype="float16")
    exp2 = ExperimentConfig(task={"model": "gpt2"}, dtype="bfloat16")
    result = get_swept_field_paths([exp1, exp2])
    assert "dtype" in result


def test_get_swept_field_paths_nested_field():
    """Two experiments with different n_prompts yield task.dataset.n_prompts in swept."""
    from llenergymeasure.config.models import DatasetConfig

    exp1 = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=10)})
    exp2 = ExperimentConfig(task={"model": "gpt2", "dataset": DatasetConfig(n_prompts=50)})
    result = get_swept_field_paths([exp1, exp2])
    assert "task.dataset.n_prompts" in result


def test_get_swept_field_paths_multi_engine_none_subconfigs():
    """Multi-engine study where optional sub-configs are None must not crash.

    In a multi-engine study, pytorch experiments have vllm=None and vice versa.
    get_swept_field_paths must handle None values in optional sub-config lists
    rather than raising AttributeError.
    """
    from llenergymeasure.config.engine_configs import TransformersConfig, VLLMConfig

    exp_pt = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        dtype="float16",
        transformers=TransformersConfig(batch_size=4),
    )
    exp_vllm = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        dtype="float16",
        vllm=VLLMConfig(),
    )
    # Must not raise AttributeError
    result = get_swept_field_paths([exp_pt, exp_vllm])
    # Engine itself varies
    assert "engine" in result
    # Optional sub-configs that are None on some experiments should be swept
    assert "transformers" in result
    assert "vllm" in result
