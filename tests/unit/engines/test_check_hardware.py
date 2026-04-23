"""Unit tests for ``EnginePlugin.check_hardware`` across all three engines.

The ``check_hardware`` seam is the host-GPU-dependent counterpart to the
vendored-rules validator that runs at ``ExperimentConfig`` construction time.
Tests here cover:

- Static-method contract (``check_hardware`` callable without an instance).
- Transformers / vLLM return ``[]`` (behavioural stubs; rules move to the
  vendored corpus when their respective walkers ship).
- TensorRT's SM floor, FP8 gate, FP8 KV-cache gate, and multi-error collection.
- Structural property: ``check_hardware`` and ``_build_llm_kwargs`` are
  independent code paths, so a T0 kwargs-build failure can no longer
  short-circuit hardware compat (the bug the new seam exists to preclude).
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.engine_configs import TensorRTConfig, TensorRTQuantConfig
from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Static-method contract (all three engines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine_cls",
    [TransformersEngine, VLLMEngine, TensorRTEngine],
    ids=["transformers", "vllm", "tensorrt"],
)
def test_check_hardware_is_static(engine_cls, monkeypatch):
    """``check_hardware`` is callable on the class (no instance required)."""
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: (8, 0),
    )
    engine_name = {
        TransformersEngine: "transformers",
        VLLMEngine: "vllm",
        TensorRTEngine: "tensorrt",
    }[engine_cls]
    config = make_config(model="test-model", engine=engine_name)
    result = engine_cls.check_hardware(config)
    assert isinstance(result, list)
    assert all(isinstance(e, str) for e in result)


# ---------------------------------------------------------------------------
# Transformers: no host-hardware rules at MVP
# ---------------------------------------------------------------------------


class TestTransformersCheckHardware:
    def test_returns_empty_on_any_sm(self, monkeypatch):
        """Transformers has no host-hardware rules; always returns ``[]``."""
        for sm in [(7, 0), (8, 0), (8, 9), (9, 0), None]:
            monkeypatch.setattr(
                "llenergymeasure.device.gpu_info.get_compute_capability",
                lambda gpu_index=0, _sm=sm: _sm,
            )
            config = make_config(model="test-model", engine="transformers")
            assert TransformersEngine.check_hardware(config) == []


# ---------------------------------------------------------------------------
# vLLM: no host-hardware rules at MVP
# ---------------------------------------------------------------------------


class TestVLLMCheckHardware:
    def test_returns_empty_on_any_sm(self, monkeypatch):
        """vLLM has no host-hardware rules at MVP; always returns ``[]``."""
        for sm in [(7, 0), (8, 0), (8, 9), (9, 0), None]:
            monkeypatch.setattr(
                "llenergymeasure.device.gpu_info.get_compute_capability",
                lambda gpu_index=0, _sm=sm: _sm,
            )
            config = make_config(model="test-model", engine="vllm")
            assert VLLMEngine.check_hardware(config) == []


# ---------------------------------------------------------------------------
# TensorRT: SM floor, FP8 gates, multi-error collection
# ---------------------------------------------------------------------------


_TRT_DEFAULTS = {"model": "test-model", "engine": "tensorrt"}


def _patch_sm(monkeypatch, sm: tuple[int, int] | None) -> None:
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: sm,
    )


class TestTensorRTCheckHardware:
    def test_sm_none_returns_empty(self, monkeypatch):
        """SM detection returns None (no GPU visible) → no errors."""
        _patch_sm(monkeypatch, None)
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_sm_below_floor_errors(self, monkeypatch):
        """SM 7.0 (V100) fails the 7.5 floor."""
        _patch_sm(monkeypatch, (7, 0))
        config = make_config(**_TRT_DEFAULTS)
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]

    def test_sm_at_floor_passes(self, monkeypatch):
        """SM 7.5 (Turing T4) passes exactly at the floor."""
        _patch_sm(monkeypatch, (7, 5))
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_on_a100_errors(self, monkeypatch):
        """FP8 quant on SM 8.0 (A100) is blocked."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(quant_algo="FP8")),
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "SM >= 8.9" in errors[0]

    def test_fp8_on_ada_passes(self, monkeypatch):
        """FP8 quant on SM 8.9 (Ada Lovelace) passes."""
        _patch_sm(monkeypatch, (8, 9))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(quant_algo="FP8")),
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_kv_cache_on_a100_errors(self, monkeypatch):
        """FP8 KV-cache quant on SM 8.0 is blocked."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(kv_cache_quant_algo="FP8")),
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "KV cache" in errors[0]

    def test_both_fp8_errors_collected(self, monkeypatch):
        """FP8 weight quant AND FP8 KV cache on SM 8.0 produces 2 errors."""
        _patch_sm(monkeypatch, (8, 0))
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                quant_config=TensorRTQuantConfig(quant_algo="FP8", kv_cache_quant_algo="FP8")
            ),
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Short-circuit regression: pre-check_hardware, TensorRT's hardware check was
# only reachable downstream of ``_build_llm_kwargs``, so a kwargs-build failure
# silently skipped hardware compat. The fix is structural: ``check_hardware``
# is now a separate code path. This test exercises the same fixture against
# both: ``_build_llm_kwargs`` raises, yet ``check_hardware`` still returns the
# SM error.
# ---------------------------------------------------------------------------


class TestShortCircuitRegression:
    def test_kwargs_build_and_hardware_check_are_independent(self, monkeypatch, tmp_path):
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        _patch_sm(monkeypatch, (7, 0))  # below the 7.5 floor

        # engine_path pointing at a non-existent directory makes
        # _build_llm_kwargs raise ConfigError.
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(engine_path=tmp_path / "does-not-exist"),
        )

        engine = TensorRTEngine()

        with pytest.raises(ConfigError):
            engine._build_llm_kwargs(config)

        errors = TensorRTEngine.check_hardware(config)
        assert any("SM >= 7.5" in e for e in errors), (
            f"expected SM-floor error from check_hardware; got {errors!r}"
        )
