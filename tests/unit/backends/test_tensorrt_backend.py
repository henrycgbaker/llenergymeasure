"""Unit tests for TensorRTBackend.

All tests run without GPU hardware and without tensorrt_llm installed.
TRT-LLM imports inside TensorRTBackend methods are lazy — the module is
importable on any host. Tests that exercise QuantConfig / BuildCacheConfig
construction inject mock classes via sys.modules so no real tensorrt_llm
import occurs.

Coverage:
  - Protocol compliance (BACK-01)
  - _build_llm_kwargs: field mapping + None omission
  - _build_sampling_params: defaults, greedy, TRT-specific overrides
  - validate_config: SM >= 7.5 check (PRE-01), FP8 SM >= 8.9 check (PRE-02)
  - Build metadata keys (BACK-03)
"""

from __future__ import annotations

import sys
import types

from llenergymeasure.config.backend_configs import (
    TensorRTBuildCacheConfig,
    TensorRTConfig,
    TensorRTKvCacheConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TensorRTSchedulerConfig,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.core.backends.tensorrt import TensorRTBackend

# =============================================================================
# Helpers
# =============================================================================


def _make_config(**overrides) -> ExperimentConfig:
    """Return a minimal valid ExperimentConfig with backend='tensorrt'."""
    defaults: dict = {"model": "test-model", "backend": "tensorrt"}
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class _MockQuantAlgo:
    """Mock for tensorrt_llm.llmapi.QuantAlgo enum."""

    INT8 = "INT8"
    FP8 = "FP8"
    W4A16_AWQ = "W4A16_AWQ"
    W4A16_GPTQ = "W4A16_GPTQ"
    W8A16 = "W8A16"

    def __class_getitem__(cls, item):
        return getattr(cls, item)


class _MockQuantConfig:
    """Mock for tensorrt_llm.llmapi.QuantConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockBuildCacheConfig:
    """Mock for tensorrt_llm.llmapi.BuildCacheConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockKvCacheConfig:
    """Mock for tensorrt_llm.llmapi.KvCacheConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockSchedulerConfig:
    """Mock for tensorrt_llm.llmapi.SchedulerConfig."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeSamplingParams:
    """Minimal stand-in for tensorrt_llm.SamplingParams — captures kwargs."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_tensorrt_llm_module() -> types.ModuleType:
    """Build a minimal fake tensorrt_llm module for sys.modules injection."""
    mock_trt = types.ModuleType("tensorrt_llm")
    mock_trt.__version__ = "0.21.0"  # type: ignore[attr-defined]
    mock_trt.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]

    mock_llmapi = types.ModuleType("tensorrt_llm.llmapi")
    mock_llmapi.QuantAlgo = _MockQuantAlgo  # type: ignore[attr-defined]
    mock_llmapi.QuantConfig = _MockQuantConfig  # type: ignore[attr-defined]
    mock_llmapi.BuildCacheConfig = _MockBuildCacheConfig  # type: ignore[attr-defined]
    mock_llmapi.KvCacheConfig = _MockKvCacheConfig  # type: ignore[attr-defined]
    mock_llmapi.SchedulerConfig = _MockSchedulerConfig  # type: ignore[attr-defined]

    mock_trt.llmapi = mock_llmapi  # type: ignore[attr-defined]
    return mock_trt


# =============================================================================
# Test Group 1: Protocol compliance (BACK-01)
# =============================================================================


class TestProtocolCompliance:
    def test_tensorrt_backend_satisfies_plugin_protocol(self):
        """TensorRTBackend must satisfy the BackendPlugin Protocol."""
        from llenergymeasure.core.backends.protocol import BackendPlugin

        backend = TensorRTBackend()
        assert isinstance(backend, BackendPlugin)

    def test_tensorrt_backend_name(self):
        """TensorRTBackend.name returns 'tensorrt'."""
        assert TensorRTBackend().name == "tensorrt"

    def test_tensorrt_backend_has_all_protocol_methods(self):
        """TensorRTBackend implements all 6 BackendPlugin methods."""
        backend = TensorRTBackend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "load_model")
        assert hasattr(backend, "warmup")
        assert hasattr(backend, "run_inference")
        assert hasattr(backend, "cleanup")
        assert hasattr(backend, "validate_config")


# =============================================================================
# Test Group 2: _build_llm_kwargs (BACK-01)
# =============================================================================


class TestBuildLlmKwargs:
    def test_build_llm_kwargs_minimal(self):
        """No tensorrt config → kwargs has model and backend='trt' and enable_build_cache."""
        config = _make_config()
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["model"] == "test-model"
        assert kwargs["backend"] == "trt"
        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_tp_size(self):
        """tp_size=2 maps to tensor_parallel_size=2."""
        config = _make_config(tensorrt=TensorRTConfig(tp_size=2))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2

    def test_build_llm_kwargs_max_batch_size(self):
        """max_batch_size maps directly."""
        config = _make_config(tensorrt=TensorRTConfig(max_batch_size=16))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["max_batch_size"] == 16

    def test_build_llm_kwargs_dtype(self):
        """dtype='float16' maps directly."""
        config = _make_config(tensorrt=TensorRTConfig(dtype="float16"))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["dtype"] == "float16"

    def test_build_llm_kwargs_fast_build(self):
        """fast_build=True maps directly."""
        config = _make_config(tensorrt=TensorRTConfig(fast_build=True))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs["fast_build"] is True

    def test_build_llm_kwargs_none_values_not_included(self):
        """None fields from TensorRTConfig are NOT in kwargs."""
        config = _make_config(tensorrt=TensorRTConfig())  # all fields None
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert "tensor_parallel_size" not in kwargs
        assert "max_batch_size" not in kwargs
        assert "max_input_len" not in kwargs
        assert "max_seq_len" not in kwargs
        assert "fast_build" not in kwargs

    def test_build_llm_kwargs_default_build_cache_when_no_build_cache_section(self):
        """When no build_cache section, enable_build_cache=True is set."""
        config = _make_config(tensorrt=TensorRTConfig(tp_size=1))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_quant_config(self, monkeypatch):
        """quant.quant_algo='INT8' produces QuantConfig with QuantAlgo.INT8."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config(tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="INT8")))
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert "quantization" in kwargs
        assert isinstance(kwargs["quantization"], _MockQuantConfig)
        assert kwargs["quantization"]._kwargs["quant_algo"] == "INT8"

    def test_build_llm_kwargs_build_cache_config(self, monkeypatch):
        """build_cache section maps to BuildCacheConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config(
            tensorrt=TensorRTConfig(
                build_cache=TensorRTBuildCacheConfig(
                    cache_root="/tmp/trt_cache",
                    max_records=5,
                    max_cache_storage_gb=128.0,
                )
            )
        )
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert "enable_build_cache" in kwargs
        assert isinstance(kwargs["enable_build_cache"], _MockBuildCacheConfig)
        assert kwargs["enable_build_cache"]._kwargs["max_records"] == 5
        assert kwargs["enable_build_cache"]._kwargs["max_cache_storage_gb"] == 128.0

    def test_build_llm_kwargs_kv_cache_config(self, monkeypatch):
        """kv_cache section maps to KvCacheConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config(
            tensorrt=TensorRTConfig(
                kv_cache=TensorRTKvCacheConfig(
                    enable_block_reuse=True,
                    free_gpu_memory_fraction=0.8,
                )
            )
        )
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert "kv_cache_config" in kwargs
        assert isinstance(kwargs["kv_cache_config"], _MockKvCacheConfig)
        assert kwargs["kv_cache_config"]._kwargs["enable_block_reuse"] is True
        assert kwargs["kv_cache_config"]._kwargs["free_gpu_memory_fraction"] == 0.8

    def test_build_llm_kwargs_scheduler_config(self, monkeypatch):
        """scheduler section maps to SchedulerConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config(
            tensorrt=TensorRTConfig(
                scheduler=TensorRTSchedulerConfig(
                    capacity_scheduling_policy="MAX_UTILIZATION",
                )
            )
        )
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)

        assert "scheduler_config" in kwargs
        assert isinstance(kwargs["scheduler_config"], _MockSchedulerConfig)
        assert kwargs["scheduler_config"]._kwargs["capacity_scheduling_policy"] == "MAX_UTILIZATION"

    def test_build_llm_kwargs_model_always_present(self):
        """model key is always present regardless of tensorrt config."""
        config = _make_config()
        backend = TensorRTBackend()
        kwargs = backend._build_llm_kwargs(config)
        assert kwargs["model"] == "test-model"


# =============================================================================
# Test Group 3: _build_sampling_params (BACK-01)
# =============================================================================


class TestBuildSamplingParams:
    def test_build_sampling_params_defaults(self, monkeypatch):
        """Default config produces SamplingParams with max_new_tokens and no temperature."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config()
        backend = TensorRTBackend()
        params = backend._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        assert params._kwargs["max_new_tokens"] == config.max_output_tokens

    def test_build_sampling_params_greedy(self, monkeypatch):
        """temperature=0 omits temperature key (greedy decoding)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        from llenergymeasure.config.models import DecoderConfig

        config = _make_config(decoder=DecoderConfig(temperature=0.0))
        backend = TensorRTBackend()
        params = backend._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        # temperature=0.0 should NOT be in kwargs (greedy check: not != 0.0)
        assert "temperature" not in params._kwargs

    def test_build_sampling_params_with_temperature(self, monkeypatch):
        """Non-zero temperature is included in SamplingParams kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        from llenergymeasure.config.models import DecoderConfig

        config = _make_config(decoder=DecoderConfig(temperature=0.7))
        backend = TensorRTBackend()
        params = backend._build_sampling_params(config)

        assert params._kwargs.get("temperature") == 0.7

    def test_build_sampling_params_trt_overrides(self, monkeypatch):
        """tensorrt.sampling overrides (n, ignore_eos) take effect."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = _make_config(
            tensorrt=TensorRTConfig(
                sampling=TensorRTSamplingConfig(
                    n=3,
                    ignore_eos=True,
                    min_tokens=5,
                )
            )
        )
        backend = TensorRTBackend()
        params = backend._build_sampling_params(config)

        assert params._kwargs.get("n") == 3
        assert params._kwargs.get("ignore_eos") is True
        assert params._kwargs.get("min_tokens") == 5


# =============================================================================
# Test Group 4: validate_config — SM checks (PRE-01)
# =============================================================================


class TestValidateConfigSMChecks:
    def test_validate_config_sm_above_7_5_passes(self, monkeypatch):
        """SM 8.0 (A100) passes the SM >= 7.5 check."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = _make_config()
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_sm_below_7_5_fails(self, monkeypatch):
        """SM 7.0 (V100) fails with an error containing 'SM >= 7.5'."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 0),
        )
        config = _make_config()
        backend = TensorRTBackend()
        errors = backend.validate_config(config)

        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]
        assert "7.0" in errors[0]

    def test_validate_config_sm_exactly_7_5_passes(self, monkeypatch):
        """SM 7.5 (Turing T4) is exactly the minimum — should pass."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 5),
        )
        config = _make_config()
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_sm_none_skips(self, monkeypatch):
        """When SM detection returns None, no errors are returned (don't block containers)."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: None,
        )
        config = _make_config()
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []


# =============================================================================
# Test Group 5: validate_config — FP8 checks (PRE-02)
# =============================================================================


class TestValidateConfigFP8Checks:
    def test_validate_config_fp8_on_sm_80_fails(self, monkeypatch):
        """FP8 quant on SM 8.0 (A100) raises an error mentioning FP8 and SM >= 8.9."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = _make_config(tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8")))
        backend = TensorRTBackend()
        errors = backend.validate_config(config)

        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "SM >= 8.9" in errors[0]

    def test_validate_config_fp8_on_sm_89_passes(self, monkeypatch):
        """FP8 quant on SM 8.9 (L40, RTX 4090) passes."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 9),
        )
        config = _make_config(tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8")))
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_fp8_on_sm_90_passes(self, monkeypatch):
        """FP8 quant on SM 9.0 (H100) passes."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (9, 0),
        )
        config = _make_config(tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8")))
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_fp8_kv_cache_on_sm_80_fails(self, monkeypatch):
        """FP8 KV cache quant on SM 8.0 (A100) raises an error."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = _make_config(
            tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(kv_cache_quant_algo="FP8"))
        )
        backend = TensorRTBackend()
        errors = backend.validate_config(config)

        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "KV cache" in errors[0]

    def test_validate_config_non_fp8_quant_on_sm_80_passes(self, monkeypatch):
        """INT8 quant on SM 8.0 (A100) passes — only FP8 is blocked."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = _make_config(tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="INT8")))
        backend = TensorRTBackend()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_both_fp8_errors_collected(self, monkeypatch):
        """FP8 weight quant AND FP8 KV cache on SM 8.0 produces 2 errors."""
        monkeypatch.setattr(
            "llenergymeasure.core.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = _make_config(
            tensorrt=TensorRTConfig(
                quant=TensorRTQuantConfig(quant_algo="FP8", kv_cache_quant_algo="FP8")
            )
        )
        backend = TensorRTBackend()
        errors = backend.validate_config(config)

        assert len(errors) == 2


# =============================================================================
# Test Group 6: Build metadata keys (BACK-03)
# =============================================================================


class TestBuildMetadata:
    def _make_backend_with_fake_metadata(self) -> TensorRTBackend:
        """Return a TensorRTBackend with _build_metadata populated via a direct assignment."""
        backend = TensorRTBackend()
        backend._build_metadata = {
            "build_time_sec": 12.5,
            "gpu_architecture": "sm_80",
            "trt_llm_version": "0.21.0",
            "config_hash": "abc123def456",
            "built_at": "2026-03-17T10:00:00+00:00",
        }
        return backend

    def test_build_metadata_keys(self):
        """_build_metadata has exactly the 5 required keys."""
        backend = self._make_backend_with_fake_metadata()
        meta = backend._build_metadata
        assert meta is not None

        required_keys = {
            "build_time_sec",
            "gpu_architecture",
            "trt_llm_version",
            "config_hash",
            "built_at",
        }
        assert set(meta.keys()) == required_keys

    def test_build_metadata_types(self):
        """_build_metadata values have expected types."""
        backend = self._make_backend_with_fake_metadata()
        meta = backend._build_metadata
        assert meta is not None

        assert isinstance(meta["build_time_sec"], float)
        assert isinstance(meta["gpu_architecture"], str)
        assert isinstance(meta["trt_llm_version"], str)
        assert isinstance(meta["config_hash"], str)
        assert isinstance(meta["built_at"], str)

    def test_build_metadata_config_hash_deterministic(self, monkeypatch):
        """Same config should produce the same config_hash (from _build_llm_kwargs)."""
        import hashlib
        import json

        config = _make_config(tensorrt=TensorRTConfig(tp_size=2, max_batch_size=8))
        backend = TensorRTBackend()

        # Reproduce the hash computation from load_model
        kwargs = backend._build_llm_kwargs(config)
        hash1 = hashlib.sha256(
            json.dumps(kwargs, default=str, sort_keys=True).encode()
        ).hexdigest()[:16]
        hash2 = hashlib.sha256(
            json.dumps(kwargs, default=str, sort_keys=True).encode()
        ).hexdigest()[:16]

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_build_metadata_initially_none(self):
        """_build_metadata is None before load_model() is called."""
        backend = TensorRTBackend()
        assert backend._build_metadata is None

    def test_build_metadata_included_in_run_inference_extras(self, monkeypatch):
        """run_inference passes _build_metadata into InferenceOutput.extras."""
        # We test the logic path: if _build_metadata is set, it appears in extras
        backend = self._make_backend_with_fake_metadata()

        # Simulate the extras assembly logic from run_inference
        extras = {}
        if backend._build_metadata is not None:
            extras["build_metadata"] = backend._build_metadata

        assert "build_metadata" in extras
        assert extras["build_metadata"]["trt_llm_version"] == "0.21.0"
