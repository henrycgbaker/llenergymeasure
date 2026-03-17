"""Unit tests for TensorRT sweep expansion (CFG-08).

Tests that backend is a sweepable axis producing distinct experiment configs,
and that dotted nested sweep keys expand correctly into TensorRT sub-configs.
"""

from __future__ import annotations

from llenergymeasure.config.grid import expand_grid


class TestBackendSweepAxis:
    """Tests for backend as a sweepable axis."""

    def test_backend_sweep_axis_produces_distinct_configs(self):
        """sweep over backend: [pytorch, tensorrt] produces 2 ExperimentConfig objects."""
        raw_study = {
            "model": "gpt2",
            "sweep": {
                "backend": ["pytorch", "tensorrt"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        assert {c.backend for c in valid} == {"pytorch", "tensorrt"}

    def test_backend_sweep_with_trt_scoped_params(self):
        """backend: [pytorch, tensorrt] with tensorrt.tp_size: [1, 2] produces 3 configs."""
        raw_study = {
            "model": "gpt2",
            "backend": ["pytorch", "tensorrt"],
            "sweep": {
                "tensorrt.tp_size": [1, 2],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        # pytorch: 1 config (no scoped dims), tensorrt: 2 configs (tp_size=[1,2])
        assert len(valid) == 3
        pytorch_configs = [c for c in valid if c.backend == "pytorch"]
        tensorrt_configs = [c for c in valid if c.backend == "tensorrt"]
        assert len(pytorch_configs) == 1
        assert len(tensorrt_configs) == 2
        tp_sizes = {c.tensorrt.tp_size for c in tensorrt_configs}
        assert tp_sizes == {1, 2}


class TestDottedNestedSweep:
    """Tests for dotted sweep keys expanding into nested TensorRT config."""

    def test_dotted_quant_algo_sweep(self):
        """tensorrt.quant.quant_algo: [INT8, FP8, W4A16_AWQ] produces 3 configs."""
        raw_study = {
            "model": "gpt2",
            "backend": "tensorrt",
            "sweep": {
                "tensorrt.quant.quant_algo": ["INT8", "FP8", "W4A16_AWQ"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 3
        algos = [c.tensorrt.quant.quant_algo for c in valid]
        assert set(algos) == {"INT8", "FP8", "W4A16_AWQ"}

    def test_dotted_nested_build_cache_sweep(self):
        """tensorrt.build_cache.max_cache_storage_gb: [128, 256] produces 2 configs."""
        raw_study = {
            "model": "gpt2",
            "backend": "tensorrt",
            "sweep": {
                "tensorrt.build_cache.max_cache_storage_gb": [128, 256],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        storage_values = {c.tensorrt.build_cache.max_cache_storage_gb for c in valid}
        assert storage_values == {128, 256}

    def test_full_tensorrt_study_yaml_parses(self):
        """Comprehensive study YAML with all sub-sections and quant sweep round-trips."""
        raw_study = {
            "model": "gpt2",
            "backend": "tensorrt",
            "tensorrt": {
                "tp_size": 2,
                "max_batch_size": 8,
                "max_input_len": 1024,
                "max_seq_len": 2048,
                "dtype": "bfloat16",
                "kv_cache": {
                    "enable_block_reuse": True,
                    "free_gpu_memory_fraction": 0.9,
                },
                "scheduler": {
                    "capacity_scheduling_policy": "MAX_UTILIZATION",
                },
                "calib": {
                    "calib_batches": 128,
                },
                "build_cache": {
                    "max_cache_storage_gb": 256,
                },
                "sampling": {
                    "return_perf_metrics": True,
                },
            },
            "sweep": {
                "tensorrt.quant.quant_algo": ["INT8", "W4A16_AWQ"],
            },
        }
        valid, _skipped = expand_grid(raw_study)
        assert len(valid) == 2
        for config in valid:
            assert config.backend == "tensorrt"
            assert config.tensorrt is not None
            assert config.tensorrt.tp_size == 2
            assert config.tensorrt.max_batch_size == 8
            assert config.tensorrt.dtype == "bfloat16"
            assert config.tensorrt.kv_cache is not None
            assert config.tensorrt.kv_cache.enable_block_reuse is True
            assert config.tensorrt.scheduler is not None
            assert config.tensorrt.sampling is not None
            assert config.tensorrt.sampling.return_perf_metrics is True
        algos = {c.tensorrt.quant.quant_algo for c in valid}
        assert algos == {"INT8", "W4A16_AWQ"}

    def test_invalid_quant_algo_in_sweep_is_skipped(self):
        """Sweep with invalid quant algo produces 1 valid + 1 skipped."""
        raw_study = {
            "model": "gpt2",
            "backend": "tensorrt",
            "sweep": {
                "tensorrt.quant.quant_algo": ["INT8", "INVALID_VALUE"],
            },
        }
        valid, skipped = expand_grid(raw_study)
        assert len(valid) == 1
        assert valid[0].tensorrt.quant.quant_algo == "INT8"
        assert len(skipped) == 1
        assert "INVALID_VALUE" in skipped[0].reason
