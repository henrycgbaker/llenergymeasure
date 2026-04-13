"""Tests for sweep group expansion in grid.py.

Covers: group detection (_is_group), mini-grid expansion within entries,
group x sweep crossing, engine scoping, fully-qualified key routing,
empty groups ({}), group + explicit experiments, collision detection,
combinatorial warnings, and hash stability.
"""

from __future__ import annotations

import logging

import pytest

from llenergymeasure.config.grid import (
    _apply_group_overlay,
    _expand_group,
    _expand_group_entry,
    _group_engine_scope,
    _is_group,
    _route_key_value,
    _validate_sweep_groups,
    expand_grid,
)
from llenergymeasure.utils.exceptions import ConfigError

# =============================================================================
# _is_group() detection
# =============================================================================


class TestIsGroup:
    def test_list_of_scalars_is_not_group(self):
        assert _is_group([1, 2, 3]) is False

    def test_list_of_strings_is_not_group(self):
        assert _is_group(["float16", "bfloat16"]) is False

    def test_list_of_dicts_is_group(self):
        assert _is_group([{"transformers.torch_compile": True}]) is True

    def test_list_with_empty_dict_is_group(self):
        assert _is_group([{}]) is True

    def test_empty_list_is_not_group(self):
        assert _is_group([]) is False

    def test_single_scalar_is_not_group(self):
        assert _is_group("float16") is False

    def test_none_is_not_group(self):
        assert _is_group(None) is False

    def test_mixed_list_raises_config_error(self):
        """Mixed dicts and scalars should raise ConfigError."""
        with pytest.raises(ConfigError, match="mixes dicts and scalars"):
            _is_group([{}, "scalar"])

    def test_list_of_booleans_is_not_group(self):
        assert _is_group([True, False]) is False


# =============================================================================
# _group_engine_scope()
# =============================================================================


class TestGroupBackendScope:
    def test_pytorch_scoped(self):
        assert _group_engine_scope("transformers.compilation") == "transformers"

    def test_vllm_scoped(self):
        assert _group_engine_scope("vllm.decoding") == "vllm"

    def test_tensorrt_scoped(self):
        assert _group_engine_scope("tensorrt.quant_config") == "tensorrt"

    def test_universal_no_dot(self):
        assert _group_engine_scope("compilation") is None

    def test_universal_non_engine_prefix(self):
        assert _group_engine_scope("decoder.sampling") is None


# =============================================================================
# _expand_group_entry() — mini-grid within entries
# =============================================================================


class TestExpandGroupEntry:
    def test_scalar_entry_passes_through(self):
        entry = {"transformers.torch_compile": True, "transformers.torch_compile_mode": "default"}
        result = _expand_group_entry(entry)
        assert result == [entry]

    def test_list_valued_field_expands(self):
        entry = {
            "transformers.load_in_4bit": True,
            "transformers.bnb_4bit_quant_type": ["nf4", "fp4"],
        }
        result = _expand_group_entry(entry)
        assert len(result) == 2
        assert result[0] == {
            "transformers.load_in_4bit": True,
            "transformers.bnb_4bit_quant_type": "nf4",
        }
        assert result[1] == {
            "transformers.load_in_4bit": True,
            "transformers.bnb_4bit_quant_type": "fp4",
        }

    def test_multiple_list_fields_cartesian(self):
        entry = {
            "tensorrt.quant.quant_algo": "INT8",
            "tensorrt.calib.calib_batches": [256, 512],
            "tensorrt.calib.calib_max_seq_length": [256, 512],
        }
        result = _expand_group_entry(entry)
        assert len(result) == 4  # 2 x 2

    def test_empty_dict_passes_through(self):
        result = _expand_group_entry({})
        assert result == [{}]

    def test_nested_list_treated_as_literal(self):
        """[[0, 1]] should not be expanded - it's a literal list value."""
        entry = {"transformers.some_param": [[0, 1]]}
        result = _expand_group_entry(entry)
        assert len(result) == 1
        assert result[0]["transformers.some_param"] == [[0, 1]]


# =============================================================================
# _expand_group() — union of variants
# =============================================================================


class TestExpandGroup:
    def test_simple_group(self):
        entries = [
            {"transformers.torch_compile": False},
            {"transformers.torch_compile": True, "transformers.torch_compile_mode": "default"},
        ]
        result = _expand_group(entries)
        assert len(result) == 2

    def test_group_with_mini_grid(self):
        entries = [
            {},
            {
                "transformers.load_in_4bit": True,
                "transformers.bnb_4bit_quant_type": ["nf4", "fp4"],
            },
        ]
        result = _expand_group(entries)
        assert len(result) == 3  # 1 baseline + 2 from mini-grid


# =============================================================================
# _apply_group_overlay() — key routing
# =============================================================================


class TestRouteKeyValue:
    def test_engine_scoped_key(self):
        config = {"engine": "transformers", "transformers": {"batch_size": 4}}
        result = _route_key_value(dict(config), "transformers.torch_compile", True)
        assert result["transformers"]["torch_compile"] is True
        assert result["transformers"]["batch_size"] == 4  # preserved

    def test_cross_section_key(self):
        config = {"engine": "transformers", "decoder": {"temperature": 1.0}}
        result = _route_key_value(dict(config), "decoder.do_sample", False)
        assert result["decoder"]["do_sample"] is False

    def test_simple_top_level_key(self):
        config = {"engine": "transformers"}
        result = _route_key_value(dict(config), "dtype", "float16")
        assert result["dtype"] == "float16"

    def test_deep_nested_key(self):
        """Multi-level dotted engine keys like vllm.engine.block_size."""
        config = {"engine": "vllm", "vllm": {}}
        result = _route_key_value(dict(config), "vllm.engine.block_size", 16)
        assert result["vllm"]["engine"]["block_size"] == 16


class TestApplyGroupOverlay:
    def test_engine_scoped_key(self):
        config = {"engine": "transformers", "transformers": {"batch_size": 4}}
        result = _apply_group_overlay(dict(config), {"transformers.torch_compile": True})
        assert result["transformers"]["torch_compile"] is True
        assert result["transformers"]["batch_size"] == 4  # preserved

    def test_cross_section_key(self):
        config = {"engine": "transformers", "decoder": {"temperature": 1.0}}
        result = _apply_group_overlay(
            dict(config),
            {"decoder.do_sample": False, "decoder.temperature": 0.0},
        )
        assert result["decoder"]["do_sample"] is False
        assert result["decoder"]["temperature"] == 0.0

    def test_simple_top_level_key(self):
        config = {"engine": "transformers"}
        result = _apply_group_overlay(dict(config), {"dtype": "float16"})
        assert result["dtype"] == "float16"

    def test_empty_overlay_no_change(self):
        config = {"engine": "transformers", "transformers": {"batch_size": 4}}
        result = _apply_group_overlay(dict(config), {})
        assert result == config

    def test_deep_nested_key(self):
        """Multi-level dotted engine keys like vllm.engine.block_size."""
        config = {"engine": "vllm", "vllm": {}}
        result = _apply_group_overlay(
            dict(config),
            {"vllm.engine.block_size": 16},
        )
        assert result["vllm"]["engine"]["block_size"] == 16


# =============================================================================
# _validate_sweep_groups() — collision detection
# =============================================================================


class TestValidateSweepGroups:
    def test_no_collision_passes(self):
        groups = {"transformers.compilation": [{}]}
        axis_keys = {"transformers.batch_size"}
        _validate_sweep_groups(groups, axis_keys)  # should not raise

    def test_collision_raises(self):
        groups = {"transformers.batch_size": [{}]}
        axis_keys = {"transformers.batch_size"}
        with pytest.raises(ConfigError, match="collide"):
            _validate_sweep_groups(groups, axis_keys)


# =============================================================================
# expand_grid() — integration tests with groups
# =============================================================================


class TestExpandGridSweepGroups:
    def test_single_group_produces_union(self):
        """A group with 3 entries crossed with 2 dtype = 6 experiments."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "max-autotune",
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 6  # 2 dtype x 3 compilation variants
        # Check that compile variants are present
        compile_values = [c.transformers.torch_compile for c in valid]
        assert compile_values.count(False) == 2
        assert compile_values.count(True) == 4

    def test_empty_dict_baseline_variant(self):
        """An empty dict {} in a group means 'no override' (baseline)."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "transformers.quantization": [
                    {},  # baseline: no quantisation
                    {"transformers.load_in_8bit": True},
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2
        # One has load_in_8bit, one doesn't
        quant_flags = [
            c.transformers.load_in_8bit if c.transformers is not None else None for c in valid
        ]
        assert True in quant_flags
        assert quant_flags.count(True) == 1

    def test_two_groups_crossed(self):
        """Two groups are crossed with each other (not unioned)."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
                "transformers.quantization": [
                    {},
                    {"transformers.load_in_8bit": True},
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 4  # 2 compilation x 2 quantisation

    def test_groups_crossed_with_axes(self):
        """Groups are crossed with independent axes."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 4  # 2 dtype x 2 compilation

    def test_cross_section_group_overlay(self):
        """Group entries can override non-engine fields like decoder settings."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "transformers.decoding": [
                    {},  # baseline: use shared decoder settings
                    {
                        "decoder.do_sample": False,
                        "decoder.temperature": 0.0,
                        "transformers.num_beams": 4,
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2
        beam_config = next(
            c for c in valid if c.transformers is not None and c.transformers.num_beams is not None
        )
        assert beam_config.decoder.do_sample is False
        assert beam_config.decoder.temperature == 0.0
        assert beam_config.transformers.num_beams == 4

    def test_group_plus_explicit_experiments(self):
        """Groups and explicit experiments coexist."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
            },
            "experiments": [
                {"model": "gpt2", "engine": "transformers", "dtype": "float32"},
            ],
        }
        valid, _skipped = expand_grid(raw)
        # 2 dtype x 2 compilation = 4 from sweep, + 1 explicit = 5
        assert len(valid) == 5

    def test_engine_scoped_group(self):
        """Groups scoped to a engine only apply to that engine's experiments."""
        raw = {
            "model": "gpt2",
            "engine": ["transformers", "vllm"],
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        pytorch_configs = [c for c in valid if c.engine == "transformers"]
        vllm_configs = [c for c in valid if c.engine == "vllm"]
        # PyTorch: 2 dtype x 2 compilation = 4
        assert len(pytorch_configs) == 4
        # vLLM: 2 dtype x no groups = 2
        assert len(vllm_configs) == 2

    def test_mini_grid_within_group(self):
        """List-valued fields within a group entry expand as mini-grid."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "transformers.quantization": [
                    {},
                    {
                        "transformers.load_in_4bit": True,
                        "transformers.bnb_4bit_quant_type": ["nf4", "fp4"],
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        # 1 baseline + 2 from mini-grid = 3
        assert len(valid) == 3
        quant_types = [
            c.transformers.bnb_4bit_quant_type
            for c in valid
            if c.transformers is not None and c.transformers.bnb_4bit_quant_type is not None
        ]
        assert set(quant_types) == {"nf4", "fp4"}

    def test_only_groups_no_axes(self):
        """Study with only groups and no independent axes."""
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        assert len(valid) == 2


class TestExpandGridSweepGroupsMultiBackend:
    def test_groups_scoped_to_different_engines(self):
        """Backend-scoped groups only apply to their respective engine."""
        raw = {
            "model": "gpt2",
            "engine": ["transformers", "vllm"],
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
                "vllm.decoding": [
                    {},
                    {"vllm.sampling.presence_penalty": 0.6},
                ],
            },
        }
        valid, _skipped = expand_grid(raw)
        pytorch_configs = [c for c in valid if c.engine == "transformers"]
        vllm_configs = [c for c in valid if c.engine == "vllm"]
        # PyTorch: 2 dtype x 2 compilation = 4
        assert len(pytorch_configs) == 4
        # vLLM: 2 dtype x 2 decoding = 4
        assert len(vllm_configs) == 4


# =============================================================================
# Combinatorial explosion warnings
# =============================================================================


class TestCombinatorialWarnings:
    def test_large_study_info_log(self, caplog):
        """Studies with >100 valid experiments log an info message."""
        # 3 dtype x 5 batch x 3 attn x 3 compile = 135 raw combos, minus 15
        # invalid (flash_attention_2 + float32) = 120 valid experiments
        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "dtype": ["float32", "float16", "bfloat16"],
                "transformers.batch_size": [1, 4, 8, 16, 32],
                "transformers.attn_implementation": ["sdpa", "flash_attention_2", "eager"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "max-autotune",
                    },
                ],
            },
        }
        with caplog.at_level(logging.INFO, logger="llenergymeasure.config.grid"):
            valid, _ = expand_grid(raw)
        assert len(valid) == 120
        assert any("Large study" in r.message for r in caplog.records)


# =============================================================================
# Hash stability with groups
# =============================================================================


class TestHashStabilityWithGroups:
    def test_groups_produce_deterministic_hash(self):
        """Same group config produces the same design hash."""
        from llenergymeasure.config.grid import compute_study_design_hash

        raw = {
            "model": "gpt2",
            "engine": "transformers",
            "sweep": {
                "dtype": ["float16", "bfloat16"],
                "transformers.compilation": [
                    {"transformers.torch_compile": False},
                    {
                        "transformers.torch_compile": True,
                        "transformers.torch_compile_mode": "default",
                    },
                ],
            },
        }
        valid1, _ = expand_grid(raw)
        valid2, _ = expand_grid(raw)
        h1 = compute_study_design_hash(valid1)
        h2 = compute_study_design_hash(valid2)
        assert h1 == h2
        assert len(h1) == 16
