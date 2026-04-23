"""Unit tests for ``EnginePlugin.check_hardware`` across all three engines.

The ``check_hardware`` seam is the host-GPU-dependent counterpart to the
vendored-rules validator that runs at ``ExperimentConfig`` construction time.
These tests cover:

- Static-method contract (``check_hardware`` callable without an instance).
- Transformers / vLLM return ``[]`` (behavioural stubs; rules move to the
  vendored corpus when their respective walkers ship).
- TensorRT's SM floor, FP8 gate, and FP8 KV-cache gate fire on the right SM.
- Structural property: TensorRT's ``check_hardware`` is independent of
  engine-kwargs construction, so a failure in ``_build_llm_kwargs`` cannot
  short-circuit it (closes the pre-50.2 ``probe_config`` T0-failure bug at
  the new seam).
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
    # Calls without instantiating
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
# TensorRT: SM floor + FP8 gates (same logic previously on validate_config,
# now on the dedicated check_hardware seam)
# ---------------------------------------------------------------------------


_TRT_DEFAULTS = {"model": "test-model", "engine": "tensorrt"}


class TestTensorRTCheckHardware:
    def test_sm_none_returns_empty(self, monkeypatch):
        """SM detection returns None (no GPU visible) → no errors."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: None,
        )
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_sm_below_floor_errors(self, monkeypatch):
        """SM 7.0 (V100) fails the 7.5 floor."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 0),
        )
        config = make_config(**_TRT_DEFAULTS)
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]

    def test_sm_at_floor_passes(self, monkeypatch):
        """SM 7.5 (Turing T4) passes exactly at the floor."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 5),
        )
        config = make_config(**_TRT_DEFAULTS)
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_on_a100_errors(self, monkeypatch):
        """FP8 quant on SM 8.0 (A100) is blocked."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
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
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 9),
        )
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(quant_algo="FP8")),
        )
        assert TensorRTEngine.check_hardware(config) == []

    def test_fp8_kv_cache_on_a100_errors(self, monkeypatch):
        """FP8 KV-cache quant on SM 8.0 is blocked."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(kv_cache_quant_algo="FP8")),
        )
        errors = TensorRTEngine.check_hardware(config)
        assert len(errors) == 1
        assert "KV cache" in errors[0]


# ---------------------------------------------------------------------------
# Structural property: check_hardware is independent of engine-kwargs build.
#
# Pre-check_hardware, TensorRT's probe_config returned early on a T0
# ``_build_llm_kwargs`` failure and therefore skipped the T5 hardware check.
# The new seam is a separate code path — the hardware check must run even
# when engine-kwargs construction would fail.
# ---------------------------------------------------------------------------


class TestTensorRTCheckHardwareIndependentOfKwargs:
    def test_hardware_check_runs_when_engine_path_invalid(self, monkeypatch, tmp_path):
        """A config whose engine_path would make ``_build_llm_kwargs`` raise must
        still surface hardware errors from ``check_hardware``.
        """
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 0),  # below the 7.5 floor
        )
        # Point engine_path at a non-existent directory — _build_llm_kwargs
        # raises ConfigError on this. check_hardware should not even look.
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(engine_path=tmp_path / "does-not-exist"),
        )

        errors = TensorRTEngine.check_hardware(config)

        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]
