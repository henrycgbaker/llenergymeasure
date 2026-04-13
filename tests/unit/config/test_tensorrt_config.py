"""Unit tests for expanded TensorRT-LLM config validation.

Tests cover all 7 config requirements (CFG-01 through CFG-07):
- CFG-01: Compile-time params (tp_size, max_batch_size, max_input_len, max_seq_len, dtype, fast_build)
- CFG-02: Quantisation (QuantAlgo Literal type, kv_cache_quant_algo)
- CFG-03: Calibration (calib_batches, calib_max_seq_length)
- CFG-04: KV cache (enable_block_reuse, free_gpu_memory_fraction, max_tokens, host_cache_size)
- CFG-05: Scheduler (capacity_scheduling_policy Literal)
- CFG-06: Build cache (cache_root, max_records, max_cache_storage_gb)
- CFG-07: Sampling (min_tokens, n, ignore_eos, return_perf_metrics)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.engine_configs import (
    TensorRTBuildCacheConfig,
    TensorRTCalibConfig,
    TensorRTConfig,
    TensorRTKvCacheConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TensorRTSchedulerConfig,
)
from llenergymeasure.config.models import ExperimentConfig

# ---------------------------------------------------------------------------
# CFG-01: Compile-time params
# ---------------------------------------------------------------------------


class TestCompileTimeParams:
    """Tests for TensorRT compile-time parameters."""

    def test_tensorrt_compile_params_accepted(self):
        """All compile-time params validate when set together."""
        config = TensorRTConfig(
            max_batch_size=8,
            max_input_len=1024,
            max_seq_len=2048,
            tp_size=2,
            dtype="float16",
            fast_build=True,
        )
        assert config.max_batch_size == 8
        assert config.max_input_len == 1024
        assert config.max_seq_len == 2048
        assert config.tp_size == 2
        assert config.dtype == "float16"
        assert config.fast_build is True

    def test_tensorrt_dtype_literal_validation(self):
        """dtype only accepts 'float16' or 'bfloat16', rejects 'float32'."""
        with pytest.raises(ValidationError):
            TensorRTConfig(dtype="float32")

    def test_tensorrt_dtype_bfloat16_accepted(self):
        """dtype='bfloat16' is valid."""
        config = TensorRTConfig(dtype="bfloat16")
        assert config.dtype == "bfloat16"

    def test_tensorrt_tp_size_ge_1(self):
        """tp_size=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTConfig(tp_size=0)

    def test_tensorrt_max_batch_size_ge_1(self):
        """max_batch_size=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTConfig(max_batch_size=0)

    def test_tensorrt_max_input_len_ge_1(self):
        """max_input_len=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTConfig(max_input_len=0)

    def test_tensorrt_max_seq_len_ge_1(self):
        """max_seq_len=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTConfig(max_seq_len=0)


# ---------------------------------------------------------------------------
# CFG-02: Quantisation
# ---------------------------------------------------------------------------


_ALL_QUANT_ALGOS = [
    "INT8",
    "W4A16_AWQ",
    "W4A16_GPTQ",
    "W8A16",
    "W8A16_GPTQ",
    "W4A8_AWQ",
    "FP8",
    "NO_QUANT",
]


class TestQuantisation:
    """Tests for TensorRT quantisation config."""

    @pytest.mark.parametrize("algo", _ALL_QUANT_ALGOS)
    def test_valid_quant_algo_accepted(self, algo: str):
        """All 8 required QuantAlgo values are accepted."""
        config = TensorRTQuantConfig(quant_algo=algo)
        assert config.quant_algo == algo

    def test_invalid_quant_algo_rejected(self):
        """Misspelled value like 'fp8' (lowercase) raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTQuantConfig(quant_algo="fp8")

    @pytest.mark.parametrize("algo", ["FP8", "INT8"])
    def test_kv_cache_quant_algo_accepted(self, algo: str):
        """FP8 and INT8 accepted for kv_cache_quant_algo."""
        config = TensorRTQuantConfig(kv_cache_quant_algo=algo)
        assert config.kv_cache_quant_algo == algo

    def test_invalid_kv_cache_quant_algo_rejected(self):
        """Invalid kv_cache_quant_algo value raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTQuantConfig(kv_cache_quant_algo="INVALID")


# ---------------------------------------------------------------------------
# CFG-03: Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    """Tests for TensorRT calibration config."""

    def test_calib_config_accepted(self):
        """Calibration section with valid values validates."""
        config = TensorRTCalibConfig(
            calib_batches=64,
            calib_max_seq_length=256,
            calib_dataset="cnn_dailymail",
        )
        assert config.calib_batches == 64
        assert config.calib_max_seq_length == 256
        assert config.calib_dataset == "cnn_dailymail"

    def test_calib_batches_ge_1(self):
        """calib_batches=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTCalibConfig(calib_batches=0)


# ---------------------------------------------------------------------------
# CFG-04: KV Cache
# ---------------------------------------------------------------------------


class TestKvCache:
    """Tests for TensorRT KV cache config."""

    def test_kv_cache_config_accepted(self):
        """KV cache section with valid values validates."""
        config = TensorRTKvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.85,
            max_tokens=4096,
            host_cache_size=1073741824,
        )
        assert config.enable_block_reuse is True
        assert config.free_gpu_memory_fraction == 0.85
        assert config.max_tokens == 4096
        assert config.host_cache_size == 1073741824

    def test_kv_cache_free_gpu_memory_fraction_range(self):
        """free_gpu_memory_fraction must be 0.0-1.0."""
        # Valid boundary
        config = TensorRTKvCacheConfig(free_gpu_memory_fraction=0.0)
        assert config.free_gpu_memory_fraction == 0.0
        config = TensorRTKvCacheConfig(free_gpu_memory_fraction=1.0)
        assert config.free_gpu_memory_fraction == 1.0

        # Invalid: above 1.0
        with pytest.raises(ValidationError):
            TensorRTKvCacheConfig(free_gpu_memory_fraction=1.5)

        # Invalid: below 0.0
        with pytest.raises(ValidationError):
            TensorRTKvCacheConfig(free_gpu_memory_fraction=-0.1)


# ---------------------------------------------------------------------------
# CFG-05: Scheduler
# ---------------------------------------------------------------------------


_VALID_SCHEDULER_POLICIES = [
    "GUARANTEED_NO_EVICT",
    "MAX_UTILIZATION",
    "STATIC_BATCH",
]


class TestScheduler:
    """Tests for TensorRT scheduler config."""

    def test_scheduler_config_accepted(self):
        """Scheduler section with valid policy validates."""
        config = TensorRTSchedulerConfig(
            capacity_scheduling_policy="GUARANTEED_NO_EVICT",
        )
        assert config.capacity_scheduling_policy == "GUARANTEED_NO_EVICT"

    @pytest.mark.parametrize("policy", _VALID_SCHEDULER_POLICIES)
    def test_valid_scheduler_policies(self, policy: str):
        """All valid scheduler policies are accepted."""
        config = TensorRTSchedulerConfig(capacity_scheduling_policy=policy)
        assert config.capacity_scheduling_policy == policy

    def test_invalid_scheduler_policy_rejected(self):
        """Invalid policy raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTSchedulerConfig(capacity_scheduling_policy="INVALID_POLICY")


# ---------------------------------------------------------------------------
# CFG-06: Build Cache
# ---------------------------------------------------------------------------


class TestBuildCache:
    """Tests for TensorRT build cache config."""

    def test_build_cache_config_accepted(self):
        """Build cache section with valid values validates."""
        config = TensorRTBuildCacheConfig(
            cache_root="/tmp/trt_cache",
            max_records=20,
            max_cache_storage_gb=512.0,
        )
        assert config.cache_root == "/tmp/trt_cache"
        assert config.max_records == 20
        assert config.max_cache_storage_gb == 512.0

    def test_build_cache_max_records_ge_1(self):
        """max_records=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTBuildCacheConfig(max_records=0)


# ---------------------------------------------------------------------------
# CFG-07: Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    """Tests for TensorRT sampling config."""

    def test_sampling_config_accepted(self):
        """Sampling section with valid values validates."""
        config = TensorRTSamplingConfig(
            min_tokens=10,
            n=4,
            ignore_eos=True,
            return_perf_metrics=True,
        )
        assert config.min_tokens == 10
        assert config.n == 4
        assert config.ignore_eos is True
        assert config.return_perf_metrics is True

    def test_sampling_n_ge_1(self):
        """n=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            TensorRTSamplingConfig(n=0)


# ---------------------------------------------------------------------------
# Integration with ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfigIntegration:
    """Tests for TensorRTConfig integration with ExperimentConfig."""

    def test_experiment_config_with_full_tensorrt(self):
        """ExperimentConfig with engine='tensorrt' and full tensorrt section validates."""
        config = ExperimentConfig(
            model="gpt2",
            engine="tensorrt",
            tensorrt={
                "tp_size": 2,
                "max_batch_size": 8,
                "max_input_len": 1024,
                "max_seq_len": 2048,
                "dtype": "bfloat16",
                "fast_build": True,
                "quant": {"quant_algo": "W4A16_AWQ"},
                "kv_cache": {
                    "enable_block_reuse": True,
                    "free_gpu_memory_fraction": 0.9,
                },
                "scheduler": {
                    "capacity_scheduling_policy": "MAX_UTILIZATION",
                },
                "calib": {
                    "calib_batches": 128,
                    "calib_max_seq_length": 512,
                },
                "build_cache": {
                    "max_cache_storage_gb": 256,
                    "max_records": 10,
                },
                "sampling": {
                    "min_tokens": 5,
                    "n": 1,
                    "return_perf_metrics": True,
                },
            },
        )
        assert config.engine == "tensorrt"
        assert config.tensorrt is not None
        assert config.tensorrt.tp_size == 2
        assert config.tensorrt.quant is not None
        assert config.tensorrt.quant.quant_algo == "W4A16_AWQ"
        assert config.tensorrt.kv_cache is not None
        assert config.tensorrt.kv_cache.enable_block_reuse is True
        assert config.tensorrt.scheduler is not None
        assert config.tensorrt.scheduler.capacity_scheduling_policy == "MAX_UTILIZATION"
        assert config.tensorrt.sampling is not None
        assert config.tensorrt.sampling.return_perf_metrics is True

    def test_tensorrt_extra_allow_forwards_unknown(self):
        """Extra fields on TensorRTConfig and sub-configs are accepted (not rejected)."""
        config = TensorRTConfig(
            tp_size=1,
            custom_future_field="value",
        )
        # Should not raise - extra="allow"
        assert config.tp_size == 1

        quant = TensorRTQuantConfig(
            quant_algo="INT8",
            custom_quant_field=42,
        )
        assert quant.quant_algo == "INT8"

    def test_tensorrt_none_defaults(self):
        """All fields default to None when not specified."""
        config = TensorRTConfig()
        assert config.max_batch_size is None
        assert config.tp_size is None
        assert config.max_input_len is None
        assert config.max_seq_len is None
        assert config.dtype is None
        assert config.fast_build is None
        assert config.backend is None
        assert config.engine_path is None
        assert config.quant is None
        assert config.kv_cache is None
        assert config.scheduler is None
        assert config.calib is None
        assert config.build_cache is None
        assert config.sampling is None
