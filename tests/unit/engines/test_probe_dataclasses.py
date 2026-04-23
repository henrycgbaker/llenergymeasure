"""Tests for the :class:`ConfigProbe` dataclass, dormancy helpers, and the
sampling-params builders retained after per-engine ``probe_config()`` was
deprecated in phase 50.2c.

Scope:
  A. :class:`ConfigProbe` / :class:`DormantField` / :func:`compute_dormant_fields`
     shape and invariants.
  E. Refactoring verification: ``_build_sampling_params`` still constructs the
     engine-native SamplingParams object, and ``_build_sampling_kwargs``
     returns the same dict it would pass to the class.

End-to-end probe-adapter coverage lives in
``tests/unit/engines/test_probe_adapter.py``. Generic-validator coverage
lives in ``tests/unit/config/test_generic_validator.py``.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import patch

from llenergymeasure.config.engine_configs import (
    TensorRTConfig,
    TensorRTSamplingConfig,
    VLLMConfig,
    VLLMSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines._helpers import compute_dormant_fields
from llenergymeasure.engines.protocol import ConfigProbe, DormantField
from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.vllm import VLLMEngine

# =============================================================================
# A. Dataclass and helper tests
# =============================================================================


def test_config_probe_is_valid_when_no_errors():
    probe = ConfigProbe(effective_engine_params={}, effective_sampling_params={}, dormant_fields={})
    assert probe.is_valid is True


def test_config_probe_invalid_when_errors_present():
    probe = ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields={},
        errors=["boom"],
    )
    assert probe.is_valid is False


def test_config_probe_warnings_do_not_invalidate():
    probe = ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields={},
        warnings=["heads up"],
    )
    assert probe.is_valid is True


def test_dormant_field_defaults():
    f = DormantField(declared_value=0.9, effective_value=None)
    assert f.declared_value == 0.9
    assert f.effective_value is None
    assert f.reason is None


def test_dormant_field_with_reason():
    f = DormantField(declared_value=0.9, effective_value=0.0, reason="greedy")
    assert f.reason == "greedy"


def test_compute_dormant_fields_returns_empty_when_matching():
    assert compute_dormant_fields({"a": 1}, {"a": 1}) == {}


def test_compute_dormant_fields_detects_stripped_field():
    dormant = compute_dormant_fields({"a": 1, "b": 2}, {"a": 1})
    assert set(dormant.keys()) == {"b"}
    assert dormant["b"].declared_value == 2
    assert dormant["b"].effective_value is None


def test_compute_dormant_fields_detects_overridden_field():
    dormant = compute_dormant_fields({"top_k": 0}, {"top_k": -1})
    assert "top_k" in dormant
    assert dormant["top_k"].declared_value == 0
    assert dormant["top_k"].effective_value == -1


def test_compute_dormant_fields_prefix_applied():
    dormant = compute_dormant_fields({"t": 0.9}, {}, prefix="vllm.sampling.")
    assert "vllm.sampling.t" in dormant


def test_compute_dormant_fields_reason_fn_called():
    def reason_fn(key: str, declared: Any, effective: Any | None) -> str | None:
        return f"{key}:{declared}->{effective}"

    dormant = compute_dormant_fields({"t": 0.9}, {}, reason_fn=reason_fn)
    assert dormant["t"].reason == "t:0.9->None"


def test_compute_dormant_fields_ignores_effective_only_keys():
    dormant = compute_dormant_fields({"a": 1}, {"a": 1, "extra": 99})
    assert dormant == {}


# =============================================================================
# E. Refactoring verification — sampling-params builders still functional
# =============================================================================


def test_vllm_build_sampling_params_still_returns_object():
    """VLLMEngine._build_sampling_params constructs SamplingParams from the dict."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(sampling=VLLMSamplingConfig(temperature=0.9, top_p=0.95)),
    )

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = VLLMEngine._build_sampling_params(cfg, _SP)
    assert isinstance(result, _SP)
    assert result.kwargs["temperature"] == 0.9
    assert result.kwargs["top_p"] == 0.95


def test_vllm_build_sampling_kwargs_matches_build_sampling_params_inputs():
    """The kwargs dict is exactly what ``_build_sampling_params`` passes to the class."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(
            sampling=VLLMSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    kwargs = VLLMEngine._build_sampling_kwargs(cfg)

    captured: dict[str, Any] = {}

    class _SP:
        def __init__(self, **kw):
            captured.update(kw)

    VLLMEngine._build_sampling_params(cfg, _SP)
    assert captured == kwargs


def test_tensorrt_build_sampling_params_still_constructs_from_module():
    """TensorRTEngine._build_sampling_params wires kwargs into tensorrt_llm.SamplingParams."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(
            sampling=TensorRTSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    engine = TensorRTEngine()

    captured: dict[str, Any] = {}

    class _FakeSP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_trt = types.ModuleType("tensorrt_llm")
    fake_trt.SamplingParams = _FakeSP  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"tensorrt_llm": fake_trt}):
        engine._build_sampling_params(cfg)

    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.9
    assert captured["top_k"] == 50
    assert "random_seed" in captured


def test_tensorrt_build_sampling_kwargs_matches_params_construction():
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(
            sampling=TensorRTSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    engine = TensorRTEngine()
    kwargs = engine._build_sampling_kwargs(cfg)

    captured: dict[str, Any] = {}

    class _FakeSP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_trt = types.ModuleType("tensorrt_llm")
    fake_trt.SamplingParams = _FakeSP  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"tensorrt_llm": fake_trt}):
        engine._build_sampling_params(cfg)

    assert captured == kwargs
