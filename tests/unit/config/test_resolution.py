"""Tests for config resolution log (semantic context + defaults in effect)."""

from __future__ import annotations

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.resolution import (
    SEMANTIC_RULES,
    SIGNIFICANT_DEFAULTS,
    _check_condition,
    _flatten_dict,
    _get_defaults_flat,
    _ValueEquals,
    _ValueGreaterThan,
    _ValueIsNotSet,
    _ValueIsSet,
    build_resolution_log,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config_dict() -> dict:
    """ExperimentConfig with only required fields (all defaults)."""
    cfg = ExperimentConfig(model="gpt2")
    return cfg.model_dump()


@pytest.fixture()
def config_with_convergence() -> dict:
    """Config with convergence detection enabled."""
    cfg = ExperimentConfig(
        model="gpt2",
        warmup={"convergence_detection": True, "max_prompts": 30},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_greedy() -> dict:
    """Config with greedy decoding."""
    cfg = ExperimentConfig(
        model="gpt2",
        decoder={"do_sample": False},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_4bit() -> dict:
    """Config with 4-bit quantisation."""
    cfg = ExperimentConfig(
        model="gpt2",
        pytorch={"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_8bit() -> dict:
    """Config with 8-bit quantisation."""
    cfg = ExperimentConfig(
        model="gpt2",
        pytorch={"load_in_8bit": True},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_torch_compile() -> dict:
    """Config with torch.compile enabled."""
    cfg = ExperimentConfig(
        model="gpt2",
        pytorch={"torch_compile": True, "torch_compile_mode": "reduce-overhead"},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_no_cache() -> dict:
    """Config with KV cache disabled."""
    cfg = ExperimentConfig(
        model="gpt2",
        pytorch={"use_cache": False},
    )
    return cfg.model_dump()


@pytest.fixture()
def config_beam_search() -> dict:
    """Config with beam search."""
    cfg = ExperimentConfig(
        model="gpt2",
        pytorch={"num_beams": 4, "early_stopping": True, "length_penalty": 0.8},
    )
    return cfg.model_dump()


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_schema_version(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        assert log["schema_version"] == "1.0"

    def test_has_all_sections(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        assert "overrides" in log
        assert "semantic_context" in log
        assert "defaults_in_effect" in log


# ---------------------------------------------------------------------------
# Overrides (backward-compatible with v1.0)
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_no_overrides_for_defaults(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        # model is required (no default), so it appears
        assert "model" in log["overrides"]
        # dtype has default bfloat16 -- should not appear
        assert "dtype" not in log["overrides"]

    def test_cli_override_source(self):
        cfg = ExperimentConfig(model="gpt2", dtype="float16")
        log = build_resolution_log(
            cfg.model_dump(),
            cli_overrides={"dtype": "float16"},
        )
        assert log["overrides"]["dtype"]["source"] == "cli_flag"

    def test_sweep_override_source(self):
        cfg = ExperimentConfig(model="meta-llama/Llama-2-7b")
        log = build_resolution_log(
            cfg.model_dump(),
            swept_fields={"model"},
        )
        assert log["overrides"]["model"]["source"] == "sweep"

    def test_yaml_override_source(self):
        cfg = ExperimentConfig(model="gpt2", dtype="float16")
        log = build_resolution_log(cfg.model_dump())
        assert log["overrides"]["dtype"]["source"] == "yaml"
        assert log["overrides"]["dtype"]["default"] == "bfloat16"
        assert log["overrides"]["dtype"]["effective"] == "float16"

    def test_cli_takes_priority_over_sweep(self):
        cfg = ExperimentConfig(model="gpt2", dtype="float16")
        log = build_resolution_log(
            cfg.model_dump(),
            cli_overrides={"dtype": "float16"},
            swept_fields={"dtype"},
        )
        assert log["overrides"]["dtype"]["source"] == "cli_flag"


# ---------------------------------------------------------------------------
# Semantic context -- warmup mode
# ---------------------------------------------------------------------------


class TestSemanticContextWarmup:
    def test_fixed_warmup_mode(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        ctx = log["semantic_context"]
        assert "warmup_mode" in ctx
        warmup = ctx["warmup_mode"]
        assert warmup["mode"] == "fixed"
        assert warmup["controlling_field"] == "warmup.convergence_detection"
        assert warmup["controlling_value"] is False
        assert "warmup.n_warmup" in warmup["active_fields"]
        assert "warmup.cv_threshold" in warmup["dormant_fields"]
        assert "warmup.max_prompts" in warmup["dormant_fields"]
        assert "warmup.window_size" in warmup["dormant_fields"]
        assert "warmup.min_prompts" in warmup["dormant_fields"]

    def test_convergence_warmup_mode(self, config_with_convergence):
        log = build_resolution_log(config_with_convergence)
        ctx = log["semantic_context"]
        warmup = ctx["warmup_mode"]
        assert warmup["mode"] == "convergence"
        assert warmup["controlling_value"] is True
        assert "warmup.cv_threshold" in warmup["active_fields"]
        assert "warmup.max_prompts" in warmup["active_fields"]
        assert warmup["dormant_fields"] == []


# ---------------------------------------------------------------------------
# Semantic context -- sampling mode
# ---------------------------------------------------------------------------


class TestSemanticContextSampling:
    def test_sampling_mode_default(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        ctx = log["semantic_context"]
        assert "sampling_mode" in ctx
        sampling = ctx["sampling_mode"]
        assert sampling["mode"] == "sampling"
        assert sampling["controlling_value"] is True

    def test_greedy_mode(self, config_greedy):
        log = build_resolution_log(config_greedy)
        ctx = log["semantic_context"]
        sampling = ctx["sampling_mode"]
        assert sampling["mode"] == "greedy"
        assert "decoder.temperature" in sampling["dormant_fields"]
        assert "decoder.top_k" in sampling["dormant_fields"]
        assert "decoder.top_p" in sampling["dormant_fields"]
        assert "decoder.min_p" in sampling["dormant_fields"]


# ---------------------------------------------------------------------------
# Semantic context -- quantisation
# ---------------------------------------------------------------------------


class TestSemanticContextQuantization:
    def test_4bit_mode(self, config_4bit):
        log = build_resolution_log(config_4bit)
        ctx = log["semantic_context"]
        assert "quantization_mode_4bit" in ctx
        quant = ctx["quantization_mode_4bit"]
        assert quant["mode"] == "4bit"
        assert "pytorch.bnb_4bit_compute_dtype" in quant["active_fields"]
        assert "pytorch.bnb_4bit_quant_type" in quant["active_fields"]
        assert "pytorch.load_in_8bit" in quant["dormant_fields"]

    def test_8bit_mode(self, config_8bit):
        log = build_resolution_log(config_8bit)
        ctx = log["semantic_context"]
        assert "quantization_mode_8bit" in ctx
        quant = ctx["quantization_mode_8bit"]
        assert quant["mode"] == "8bit"
        assert "pytorch.load_in_4bit" in quant["dormant_fields"]
        assert "pytorch.bnb_4bit_compute_dtype" in quant["dormant_fields"]

    def test_no_quantization_no_rule(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        ctx = log["semantic_context"]
        # Neither 4bit nor 8bit set, so no quant rules fire
        assert "quantization_mode_4bit" not in ctx
        assert "quantization_mode_8bit" not in ctx


# ---------------------------------------------------------------------------
# Semantic context -- torch.compile
# ---------------------------------------------------------------------------


class TestSemanticContextTorchCompile:
    def test_compile_enabled(self, config_torch_compile):
        log = build_resolution_log(config_torch_compile)
        ctx = log["semantic_context"]
        assert "torch_compile_mode" in ctx
        tc = ctx["torch_compile_mode"]
        assert tc["mode"] == "enabled"
        assert "pytorch.torch_compile_backend" in tc["active_fields"]
        assert "pytorch.torch_compile_mode" in tc["active_fields"]

    def test_compile_disabled(self):
        cfg = ExperimentConfig(model="gpt2", pytorch={"torch_compile": False})
        log = build_resolution_log(cfg.model_dump())
        ctx = log["semantic_context"]
        assert "torch_compile_mode" in ctx
        tc = ctx["torch_compile_mode"]
        assert tc["mode"] == "disabled"
        assert "pytorch.torch_compile_backend" in tc["dormant_fields"]

    def test_compile_not_set_no_rule(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        ctx = log["semantic_context"]
        # torch_compile is None by default (not True or False), so no rule fires
        assert "torch_compile_mode" not in ctx


# ---------------------------------------------------------------------------
# Semantic context -- KV cache
# ---------------------------------------------------------------------------


class TestSemanticContextKvCache:
    def test_cache_disabled(self, config_no_cache):
        log = build_resolution_log(config_no_cache)
        ctx = log["semantic_context"]
        assert "kv_cache_mode" in ctx
        kv = ctx["kv_cache_mode"]
        assert kv["mode"] == "disabled"
        assert "pytorch.cache_implementation" in kv["dormant_fields"]

    def test_cache_enabled(self):
        cfg = ExperimentConfig(model="gpt2", pytorch={"use_cache": True})
        log = build_resolution_log(cfg.model_dump())
        ctx = log["semantic_context"]
        assert "kv_cache_mode" in ctx
        kv = ctx["kv_cache_mode"]
        assert kv["mode"] == "enabled"
        assert "pytorch.cache_implementation" in kv["active_fields"]


# ---------------------------------------------------------------------------
# Semantic context -- beam search
# ---------------------------------------------------------------------------


class TestSemanticContextBeamSearch:
    def test_beam_search_active(self, config_beam_search):
        log = build_resolution_log(config_beam_search)
        ctx = log["semantic_context"]
        assert "beam_search_mode" in ctx
        beam = ctx["beam_search_mode"]
        assert beam["mode"] == "beam_search"
        assert "pytorch.early_stopping" in beam["active_fields"]
        assert "pytorch.length_penalty" in beam["active_fields"]
        assert "pytorch.no_repeat_ngram_size" in beam["active_fields"]

    def test_no_beam_search_no_rule(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        ctx = log["semantic_context"]
        assert "beam_search_mode" not in ctx

    def test_beam_search_single_beam_no_rule(self):
        cfg = ExperimentConfig(model="gpt2", pytorch={"num_beams": 1})
        log = build_resolution_log(cfg.model_dump())
        ctx = log["semantic_context"]
        # num_beams=1 is not > 1, so rule doesn't fire
        assert "beam_search_mode" not in ctx


# ---------------------------------------------------------------------------
# Defaults in effect
# ---------------------------------------------------------------------------


class TestDefaultsInEffect:
    def test_significant_defaults_present(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        defaults = log["defaults_in_effect"]
        assert "warmup.convergence_detection" in defaults
        assert defaults["warmup.convergence_detection"]["value"] is False
        assert "decoder.do_sample" in defaults
        assert defaults["decoder.do_sample"]["value"] is True

    def test_overridden_field_not_in_defaults(self, config_with_convergence):
        log = build_resolution_log(config_with_convergence)
        defaults = log["defaults_in_effect"]
        # convergence_detection was explicitly set to True (non-default), so excluded
        assert "warmup.convergence_detection" not in defaults

    def test_defaults_have_significance(self, default_config_dict):
        log = build_resolution_log(default_config_dict)
        for _field_path, entry in log["defaults_in_effect"].items():
            assert "value" in entry
            assert "significance" in entry
            assert len(entry["significance"]) > 0


# ---------------------------------------------------------------------------
# Condition matchers
# ---------------------------------------------------------------------------


class TestConditionMatchers:
    def test_value_equals_true(self):
        assert _check_condition(_ValueEquals(value=True), True) is True

    def test_value_equals_false(self):
        assert _check_condition(_ValueEquals(value=True), False) is False

    def test_value_equals_none(self):
        assert _check_condition(_ValueEquals(value=None), None) is True

    def test_value_greater_than(self):
        assert _check_condition(_ValueGreaterThan(threshold=1), 4) is True
        assert _check_condition(_ValueGreaterThan(threshold=1), 1) is False
        assert _check_condition(_ValueGreaterThan(threshold=1), None) is False

    def test_value_is_set(self):
        assert _check_condition(_ValueIsSet(), "something") is True
        assert _check_condition(_ValueIsSet(), None) is False

    def test_value_is_not_set(self):
        assert _check_condition(_ValueIsNotSet(), None) is True
        assert _check_condition(_ValueIsNotSet(), "something") is False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_flatten_dict_nested(self):
        d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = _flatten_dict(d)
        assert flat == {"a.b": 1, "a.c.d": 2, "e": 3}

    def test_flatten_dict_lists(self):
        d = {"a": [1, 2, 3]}
        flat = _flatten_dict(d)
        assert flat == {"a": [1, 2, 3]}

    def test_get_defaults_flat_returns_known_keys(self):
        defaults = _get_defaults_flat(ExperimentConfig)
        assert "backend" in defaults
        assert defaults["backend"] == "pytorch"
        assert "decoder.temperature" in defaults
        assert defaults["decoder.temperature"] == 1.0


# ---------------------------------------------------------------------------
# Rule registry integrity
# ---------------------------------------------------------------------------


class TestRuleRegistry:
    def test_all_rules_have_conditions(self):
        for rule in SEMANTIC_RULES:
            assert len(rule.conditions) > 0, f"Rule {rule.name} has no conditions"

    def test_all_rules_have_names(self):
        names = [r.name for r in SEMANTIC_RULES]
        assert len(names) == len(set(names)), "Duplicate rule names"

    def test_significant_defaults_are_all_strings(self):
        for key, desc in SIGNIFICANT_DEFAULTS.items():
            assert isinstance(key, str)
            assert isinstance(desc, str)
            assert "." in key or key in ExperimentConfig.model_fields


# ---------------------------------------------------------------------------
# Integration: combined scenarios
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_log_structure(self, default_config_dict):
        """Verify the full output structure matches the schema."""
        log = build_resolution_log(
            default_config_dict,
            cli_overrides={"model": "gpt2"},
            swept_fields=set(),
        )
        assert log["schema_version"] == "1.0"
        assert isinstance(log["overrides"], dict)
        assert isinstance(log["semantic_context"], dict)
        assert isinstance(log["defaults_in_effect"], dict)

    def test_complex_config(self):
        """Verify a complex config produces expected semantic context."""
        cfg = ExperimentConfig(
            model="meta-llama/Llama-2-7b",
            dtype="float16",
            decoder={"do_sample": False},
            warmup={"convergence_detection": True, "max_prompts": 50},
            pytorch={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "fp4",
                "torch_compile": True,
                "torch_compile_mode": "reduce-overhead",
            },
        )
        log = build_resolution_log(
            cfg.model_dump(),
            cli_overrides={"dtype": "float16"},
            swept_fields={"model"},
        )

        ctx = log["semantic_context"]
        # Warmup should be convergence mode
        assert ctx["warmup_mode"]["mode"] == "convergence"
        # Sampling should be greedy
        assert ctx["sampling_mode"]["mode"] == "greedy"
        # 4bit quant active
        assert ctx["quantization_mode_4bit"]["mode"] == "4bit"
        # torch.compile enabled
        assert ctx["torch_compile_mode"]["mode"] == "enabled"

        # Overrides should have correct sources
        assert log["overrides"]["dtype"]["source"] == "cli_flag"
        assert log["overrides"]["model"]["source"] == "sweep"

    def test_empty_semantic_context_when_no_backend_config(self):
        """Config without pytorch section has no pytorch-specific rules."""
        cfg = ExperimentConfig(model="gpt2")
        log = build_resolution_log(cfg.model_dump())
        ctx = log["semantic_context"]
        # Only warmup_mode and sampling_mode should fire (tier 1 configs)
        assert "warmup_mode" in ctx
        assert "sampling_mode" in ctx
        # No pytorch rules
        assert "quantization_mode_4bit" not in ctx
        assert "quantization_mode_8bit" not in ctx
        assert "torch_compile_mode" not in ctx
        assert "kv_cache_mode" not in ctx
        assert "beam_search_mode" not in ctx
