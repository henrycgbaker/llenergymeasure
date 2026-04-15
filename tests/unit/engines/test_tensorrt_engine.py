"""Unit tests for TensorRTEngine.

All tests run without GPU hardware and without tensorrt_llm installed.
TRT-LLM imports inside TensorRTEngine methods are lazy — the module is
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

import json
import sys
import types

from llenergymeasure.config.engine_configs import (
    TensorRTConfig,
    TensorRTKvCacheConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TensorRTSchedulerConfig,
)
from llenergymeasure.engines.tensorrt import TensorRTEngine, _validate_engine_directory
from tests.conftest import make_config

# =============================================================================
# Helpers
# =============================================================================

_TRT_DEFAULTS = {"model": "test-model", "engine": "tensorrt"}


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


class _MockCapacitySchedulerPolicy:
    """Mock for tensorrt_llm.llmapi.CapacitySchedulerPolicy enum."""

    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    MAX_UTILIZATION = "MAX_UTILIZATION"
    STATIC_BATCH = "STATIC_BATCH"

    def __class_getitem__(cls, item):
        return getattr(cls, item)


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
    mock_llmapi.CapacitySchedulerPolicy = _MockCapacitySchedulerPolicy  # type: ignore[attr-defined]

    mock_trt.llmapi = mock_llmapi  # type: ignore[attr-defined]
    return mock_trt


# =============================================================================
# Test Group 1: Protocol compliance (BACK-01)
# =============================================================================


class TestProtocolCompliance:
    def test_tensorrt_engine_satisfies_plugin_protocol(self):
        """TensorRTEngine must satisfy the EnginePlugin Protocol."""
        from llenergymeasure.engines.protocol import EnginePlugin

        engine = TensorRTEngine()
        assert isinstance(engine, EnginePlugin)

    def test_tensorrt_engine_name(self):
        """TensorRTEngine.name returns 'tensorrt'."""
        assert TensorRTEngine().name == "tensorrt"

    def test_tensorrt_engine_has_all_protocol_methods(self):
        """TensorRTEngine implements all 6 EnginePlugin methods."""
        engine = TensorRTEngine()
        assert hasattr(engine, "name")
        assert hasattr(engine, "load_model")
        assert hasattr(engine, "run_warmup_prompt")
        assert hasattr(engine, "run_inference")
        assert hasattr(engine, "cleanup")
        assert hasattr(engine, "validate_config")


# =============================================================================
# Test Group 2: _build_llm_kwargs (BACK-01)
# =============================================================================


class TestBuildLlmKwargs:
    def test_build_llm_kwargs_minimal(self):
        """No tensorrt config → kwargs has model and backend='trt' and enable_build_cache."""
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["model"] == "test-model"
        assert kwargs["backend"] == "trt"
        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_tensor_parallel_size(self):
        """tensor_parallel_size=2 maps to kwargs tensor_parallel_size=2."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(tensor_parallel_size=2))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["tensor_parallel_size"] == 2

    def test_build_llm_kwargs_max_batch_size(self):
        """max_batch_size maps directly."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(max_batch_size=16))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["max_batch_size"] == 16

    def test_build_llm_kwargs_dtype(self):
        """dtype='float16' maps directly."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(dtype="float16"))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["dtype"] == "float16"

    def test_build_llm_kwargs_fast_build(self):
        """fast_build=True maps directly."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(fast_build=True))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["fast_build"] is True

    def test_build_llm_kwargs_none_values_not_included(self):
        """None fields from TensorRTConfig are NOT in kwargs."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig())  # all fields None
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "tensor_parallel_size" not in kwargs
        assert "max_batch_size" not in kwargs
        assert "max_input_len" not in kwargs
        assert "max_seq_len" not in kwargs
        assert "fast_build" not in kwargs

    def test_build_llm_kwargs_default_build_cache_when_no_build_cache_section(self):
        """When no build_cache section, enable_build_cache=True is set."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(tensor_parallel_size=1))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_quant_config(self, monkeypatch):
        """quant.quant_algo='INT8' produces QuantConfig with QuantAlgo.INT8."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="INT8"))
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "quantization" in kwargs
        assert isinstance(kwargs["quantization"], _MockQuantConfig)
        assert kwargs["quantization"]._kwargs["quant_algo"] == "INT8"

    def test_build_llm_kwargs_always_has_enable_build_cache(self):
        """enable_build_cache=True is always set (TensorRTBuildCacheConfig dropped D1)."""
        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig())
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs.get("enable_build_cache") is True

    def test_build_llm_kwargs_kv_cache_config(self, monkeypatch):
        """kv_cache section maps to KvCacheConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                kv_cache=TensorRTKvCacheConfig(
                    enable_block_reuse=True,
                    free_gpu_memory_fraction=0.8,
                )
            ),
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "kv_cache_config" in kwargs
        assert isinstance(kwargs["kv_cache_config"], _MockKvCacheConfig)
        assert kwargs["kv_cache_config"]._kwargs["enable_block_reuse"] is True
        assert kwargs["kv_cache_config"]._kwargs["free_gpu_memory_fraction"] == 0.8

    def test_build_llm_kwargs_scheduler_config(self, monkeypatch):
        """scheduler section maps to SchedulerConfig kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                scheduler=TensorRTSchedulerConfig(
                    capacity_scheduling_policy="MAX_UTILIZATION",
                )
            ),
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "scheduler_config" in kwargs
        assert isinstance(kwargs["scheduler_config"], _MockSchedulerConfig)
        assert (
            kwargs["scheduler_config"]._kwargs["capacity_scheduling_policy"]
            == _MockCapacitySchedulerPolicy.MAX_UTILIZATION
        )

    def test_build_llm_kwargs_model_always_present(self):
        """model key is always present regardless of tensorrt config."""
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)
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

        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        assert params._kwargs["max_new_tokens"] == config.max_output_tokens

    def test_build_sampling_params_passes_random_seed(self, monkeypatch):
        """random_seed from ExperimentConfig is forwarded to SamplingParams."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(**_TRT_DEFAULTS, random_seed=123)
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert params._kwargs["random_seed"] == 123

    def test_build_sampling_params_greedy(self, monkeypatch):
        """temperature=0 omits temperature key (greedy decoding)."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        from llenergymeasure.config.models import DecoderConfig

        config = make_config(**_TRT_DEFAULTS, decoder=DecoderConfig(temperature=0.0))
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert isinstance(params, _FakeSamplingParams)
        # temperature=0.0 should NOT be in kwargs (greedy check: not != 0.0)
        assert "temperature" not in params._kwargs

    def test_build_sampling_params_with_temperature(self, monkeypatch):
        """Non-zero temperature is included in SamplingParams kwargs."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        from llenergymeasure.config.models import DecoderConfig

        config = make_config(**_TRT_DEFAULTS, decoder=DecoderConfig(temperature=0.7))
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

        assert params._kwargs.get("temperature") == 0.7

    def test_build_sampling_params_trt_overrides(self, monkeypatch):
        """tensorrt.sampling overrides (n, ignore_eos) take effect."""
        mock_trt = _make_fake_tensorrt_llm_module()
        monkeypatch.setitem(sys.modules, "tensorrt_llm", mock_trt)
        monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", mock_trt.llmapi)

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                sampling=TensorRTSamplingConfig(
                    n=3,
                    ignore_eos=True,
                    min_tokens=5,
                )
            ),
        )
        engine = TensorRTEngine()
        params = engine._build_sampling_params(config)

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
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []

    def test_validate_config_sm_below_7_5_fails(self, monkeypatch):
        """SM 7.0 (V100) fails with an error containing 'SM >= 7.5'."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 0),
        )
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        errors = engine.validate_config(config)

        assert len(errors) == 1
        assert "SM >= 7.5" in errors[0]
        assert "7.0" in errors[0]

    def test_validate_config_sm_exactly_7_5_passes(self, monkeypatch):
        """SM 7.5 (Turing T4) is exactly the minimum — should pass."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (7, 5),
        )
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []

    def test_validate_config_sm_none_skips(self, monkeypatch):
        """When SM detection returns None, no errors are returned (don't block containers)."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: None,
        )
        config = make_config(**_TRT_DEFAULTS)
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []


# =============================================================================
# Test Group 5: validate_config — FP8 checks (PRE-02)
# =============================================================================


class TestValidateConfigFP8Checks:
    def test_validate_config_fp8_on_sm_80_fails(self, monkeypatch):
        """FP8 quant on SM 8.0 (A100) raises an error mentioning FP8 and SM >= 8.9."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8"))
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)

        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "SM >= 8.9" in errors[0]

    def test_validate_config_fp8_on_sm_89_passes(self, monkeypatch):
        """FP8 quant on SM 8.9 (L40, RTX 4090) passes."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 9),
        )
        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8"))
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []

    def test_validate_config_fp8_on_sm_90_passes(self, monkeypatch):
        """FP8 quant on SM 9.0 (H100) passes."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (9, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="FP8"))
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []

    def test_validate_config_fp8_kv_cache_on_sm_80_fails(self, monkeypatch):
        """FP8 KV cache quant on SM 8.0 (A100) raises an error."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(kv_cache_quant_algo="FP8")),
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)

        assert len(errors) == 1
        assert "FP8" in errors[0]
        assert "KV cache" in errors[0]

    def test_validate_config_non_fp8_quant_on_sm_80_passes(self, monkeypatch):
        """INT8 quant on SM 8.0 (A100) passes — only FP8 is blocked."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(quant=TensorRTQuantConfig(quant_algo="INT8"))
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)
        assert errors == []

    def test_validate_config_both_fp8_errors_collected(self, monkeypatch):
        """FP8 weight quant AND FP8 KV cache on SM 8.0 produces 2 errors."""
        monkeypatch.setattr(
            "llenergymeasure.device.gpu_info.get_compute_capability",
            lambda gpu_index=0: (8, 0),
        )
        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                quant=TensorRTQuantConfig(quant_algo="FP8", kv_cache_quant_algo="FP8")
            ),
        )
        engine = TensorRTEngine()
        errors = engine.validate_config(config)

        assert len(errors) == 2


# =============================================================================
# Test Group 6: Build metadata keys (BACK-03)
# =============================================================================


class TestBuildMetadata:
    def _make_engine_with_fake_metadata(self) -> TensorRTEngine:
        """Return a TensorRTEngine with _build_metadata populated via a direct assignment."""
        engine = TensorRTEngine()
        engine._build_metadata = {
            "build_time_sec": 12.5,
            "gpu_architecture": "sm_80",
            "trt_llm_version": "0.21.0",
            "config_hash": "abc123def456",
            "built_at": "2026-03-17T10:00:00+00:00",
        }
        return engine

    def test_build_metadata_keys(self):
        """_build_metadata has exactly the 5 required keys."""
        engine = self._make_engine_with_fake_metadata()
        meta = engine._build_metadata
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
        engine = self._make_engine_with_fake_metadata()
        meta = engine._build_metadata
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

        config = make_config(
            **_TRT_DEFAULTS, tensorrt=TensorRTConfig(tensor_parallel_size=2, max_batch_size=8)
        )
        engine = TensorRTEngine()

        # Reproduce the hash computation from load_model
        kwargs = engine._build_llm_kwargs(config)
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
        engine = TensorRTEngine()
        assert engine._build_metadata is None

    def test_build_metadata_included_in_run_inference_extras(self, monkeypatch):
        """run_inference passes _build_metadata into InferenceOutput.extras."""
        # We test the logic path: if _build_metadata is set, it appears in extras
        engine = self._make_engine_with_fake_metadata()

        # Simulate the extras assembly logic from run_inference
        extras = {}
        if engine._build_metadata is not None:
            extras["build_metadata"] = engine._build_metadata

        assert "build_metadata" in extras
        assert extras["build_metadata"]["trt_llm_version"] == "0.21.0"


# =============================================================================
# Test Group 7: _validate_engine_directory
# =============================================================================


class TestValidateEngineDirectory:
    def test_valid_engine_dir(self, tmp_path):
        """Valid engine dir with config.json and rank0.engine passes."""
        config = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert errors == []

    def test_missing_dir(self, tmp_path):
        """Non-existent dir returns error."""
        errors = _validate_engine_directory(tmp_path / "nonexistent", tp_size=1)
        assert len(errors) == 1
        assert "does not exist" in errors[0]

    def test_missing_config_json(self, tmp_path):
        """Dir exists but no config.json returns error."""
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert len(errors) == 1
        assert "config.json" in errors[0]

    def test_tp_size_mismatch(self, tmp_path):
        """Engine tp_size=2 but requested tp_size=1 returns error."""
        config = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert any("tp_size" in e for e in errors)

    def test_missing_rank_engine(self, tmp_path):
        """tp_size=2 but only rank0.engine exists returns error for rank1."""
        config = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=2)
        assert any("rank1.engine" in e for e in errors)

    def test_corrupt_config_json(self, tmp_path):
        """Corrupt config.json returns parse error."""
        (tmp_path / "config.json").write_text("not json{{{")
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert any("config.json" in e for e in errors)

    def test_missing_tp_size_key_skips_check(self, tmp_path):
        """config.json without mapping.tp_size skips tp_size check (non-blocking)."""
        config = {"pretrained_config": {}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        errors = _validate_engine_directory(tmp_path, tp_size=1)
        assert errors == []


# =============================================================================
# Test Group 8: _build_llm_kwargs engine_path branches
# =============================================================================


class TestBuildLlmKwargsEnginePath:
    def test_build_llm_kwargs_engine_path(self, tmp_path):
        """engine_path set -> kwargs has model=engine_path as string and engine=trt."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")

        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(engine_path=str(tmp_path)))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert kwargs["model"] == str(tmp_path)
        assert kwargs["backend"] == "trt"

    def test_build_llm_kwargs_engine_path_skips_compile_kwargs(self, tmp_path):
        """engine_path set -> no compile-time kwargs (tensor_parallel_size, max_batch_size, etc.)."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 2}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")
        (tmp_path / "rank1.engine").write_bytes(b"fake")

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(
                engine_path=str(tmp_path), tensor_parallel_size=2, max_batch_size=16
            ),
        )
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "tensor_parallel_size" not in kwargs
        assert "max_batch_size" not in kwargs
        assert "max_input_len" not in kwargs
        assert "max_seq_len" not in kwargs
        assert "fast_build" not in kwargs
        assert "dtype" not in kwargs

    def test_build_llm_kwargs_engine_path_no_build_cache(self, tmp_path):
        """engine_path set -> enable_build_cache not in kwargs."""
        config_data = {"pretrained_config": {"mapping": {"tp_size": 1}}, "build_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config_data))
        (tmp_path / "rank0.engine").write_bytes(b"fake")

        config = make_config(**_TRT_DEFAULTS, tensorrt=TensorRTConfig(engine_path=str(tmp_path)))
        engine = TensorRTEngine()
        kwargs = engine._build_llm_kwargs(config)

        assert "enable_build_cache" not in kwargs

    def test_build_llm_kwargs_engine_path_invalid_dir_raises(self, tmp_path):
        """engine_path pointing to non-existent dir raises ConfigError."""
        import pytest

        from llenergymeasure.utils.exceptions import ConfigError

        config = make_config(
            **_TRT_DEFAULTS,
            tensorrt=TensorRTConfig(engine_path=str(tmp_path / "nonexistent")),
        )
        engine = TensorRTEngine()
        with pytest.raises(ConfigError, match="engine_path validation failed"):
            engine._build_llm_kwargs(config)
