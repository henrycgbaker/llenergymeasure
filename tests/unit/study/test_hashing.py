"""Tests for canonical serialisation + H1/H3 hashing.

Normalisation rules come from ``sweep-dedup.md`` §9.Q3. Each test pins one
rule from the table so a rule change surfaces as a targeted failure rather
than a diffuse hash-mismatch.
"""

from __future__ import annotations

import math

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.hashing import (
    ConfigHashView,
    build_h1_view,
    build_h3_view,
    canonical_serialise,
    hash_config,
)


def _mk_config(**overrides):
    base = {"task": {"model": "gpt2"}, "engine": "transformers"}
    base.update(overrides)
    return ExperimentConfig(**base)


class TestCanonicalSerialise:
    def test_none_and_missing_differ(self):
        a = {"seed": None}
        b: dict = {}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_int_and_float_differ(self):
        a = {"top_k": 0}
        b = {"top_k": 0.0}
        # Type difference preserved: int 0 and float 0.0 serialise differently
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_bool_and_int_differ(self):
        a = {"do_sample": True}
        b = {"do_sample": 1}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_tuple_normalises_to_list(self):
        a = {"stop": ("a", "b")}
        b = {"stop": ["a", "b"]}
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_dict_keys_sorted(self):
        a = {"a": 1, "b": 2}
        b = {"b": 2, "a": 1}
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_nan_is_stable_string(self):
        out = canonical_serialise({"x": math.nan})
        assert b'"NaN"' in out

    def test_infinity_is_stable(self):
        pos = canonical_serialise({"x": math.inf})
        neg = canonical_serialise({"x": -math.inf})
        assert b"Infinity" in pos and b"-Infinity" in neg
        assert pos != neg

    def test_float_rounding_stability(self):
        # Jitter in the 13th+ significant digit should collapse to the same hash.
        a = {"temp": 0.123456789012345}
        b = {"temp": 0.1234567890123459}  # 13+ digits differ
        assert canonical_serialise(a) == canonical_serialise(b)

    def test_float_distinct_values_stay_distinct(self):
        # Two values that differ at the 11th digit must not collapse.
        a = {"temp": 0.12345678901}
        b = {"temp": 0.12345678902}
        assert canonical_serialise(a) != canonical_serialise(b)

    def test_nested_dict_recursive_normalisation(self):
        a = {"outer": {"inner": (1.0, 2.0)}}
        b = {"outer": {"inner": [1.0, 2.0]}}
        assert canonical_serialise(a) == canonical_serialise(b)


class TestHashConfig:
    def test_returns_hex_digest(self):
        view = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        h = hash_config(view)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # parses as hex

    def test_identical_views_same_hash(self):
        v1 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        v2 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        assert hash_config(v1) == hash_config(v2)

    def test_different_engine_different_hash(self):
        v1 = ConfigHashView(engine="transformers", task={"model": "gpt2"})
        v2 = ConfigHashView(engine="vllm", task={"model": "gpt2"})
        assert hash_config(v1) != hash_config(v2)


class TestBuildH1View:
    def test_extracts_engine_and_task(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": False}})
        view = build_h1_view(cfg)
        assert view.engine == "transformers"
        assert view.task["model"] == "gpt2"

    def test_sampling_lifted_into_sampling_bucket(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": True, "temperature": 0.7}})
        view = build_h1_view(cfg)
        assert view.effective_sampling_params["do_sample"] is True
        assert view.effective_sampling_params["temperature"] == 0.7
        assert "sampling" not in view.effective_engine_params

    def test_passthrough_kwargs_propagated(self):
        cfg = _mk_config(passthrough_kwargs={"my_key": "my_val"})
        view = build_h1_view(cfg)
        assert view.passthrough_kwargs == {"my_key": "my_val"}


class TestBuildH3View:
    def test_carries_inputs_through(self):
        view = build_h3_view(
            engine="vllm",
            task={"model": "gpt2"},
            effective_engine_params={"dtype": "float16"},
            effective_sampling_params={"temperature": 1.0},
        )
        assert view.engine == "vllm"
        assert view.effective_engine_params["dtype"] == "float16"
        assert view.effective_sampling_params["temperature"] == 1.0

    def test_h1_and_h3_match_on_same_inputs(self):
        # Symmetry: both views hashed through the same pipe produce the same hash
        # when the underlying fields match. This is what makes the H3-collision
        # invariant meaningful.
        task = {"model": "gpt2"}
        engine_params = {"dtype": "float16"}
        sampling_params = {"temperature": 1.0}

        h1_view = ConfigHashView(
            engine="vllm",
            task=task,
            effective_engine_params=engine_params,
            effective_sampling_params=sampling_params,
        )
        h3_view = build_h3_view(
            engine="vllm",
            task=task,
            effective_engine_params=engine_params,
            effective_sampling_params=sampling_params,
        )
        assert hash_config(h1_view) == hash_config(h3_view)


class TestHashStability:
    @pytest.mark.parametrize("_", range(5))
    def test_hash_stable_across_repeat_calls(self, _):
        cfg = _mk_config(
            transformers={"sampling": {"do_sample": False, "temperature": 1.0}},
        )
        h1 = hash_config(build_h1_view(cfg))
        h2 = hash_config(build_h1_view(cfg))
        assert h1 == h2
