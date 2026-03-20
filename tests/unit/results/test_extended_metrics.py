"""Unit tests for results/extended_metrics.py — compute and aggregate functions."""

from __future__ import annotations

import pytest

from llenergymeasure.domain.metrics import (
    BatchEfficiencyMetrics,
    ExtendedEfficiencyMetrics,
    KVCacheEfficiencyMetrics,
    MemoryEfficiencyMetrics,
)
from llenergymeasure.results.extended_metrics import (
    _compute_batch_metrics,
    _compute_gpu_utilisation_metrics,
    _compute_kv_cache_metrics,
    _compute_memory_metrics,
    _compute_request_latency_metrics,
    aggregate_extended_metrics,
    compute_extended_metrics,
)

# ---------------------------------------------------------------------------
# TestComputeExtendedMetrics
# ---------------------------------------------------------------------------


class TestComputeExtendedMetrics:
    """Top-level compute_extended_metrics() orchestrator."""

    def test_zero_output_tokens_returns_defaults(self):
        result = compute_extended_metrics(
            output_tokens=0, total_energy_j=10.0, tokens_per_second=50.0
        )
        assert result.tpot_ms is None
        assert result.token_efficiency_index is None

    def test_tpot_from_itl_mean(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=50.0, itl_mean_ms=15.0
        )
        assert result.tpot_ms == pytest.approx(15.0)

    def test_tpot_none_when_no_itl(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=50.0
        )
        assert result.tpot_ms is None

    def test_tei_normal(self):
        result = compute_extended_metrics(
            output_tokens=200, total_energy_j=10.0, tokens_per_second=100.0, precision_factor=0.5
        )
        # TEI = tps * (output_tokens / energy) * precision_factor
        # = 100 * (200 / 10) * 0.5 = 100 * 20 * 0.5 = 1000.0
        assert result.token_efficiency_index == pytest.approx(1000.0)

    def test_tei_none_zero_throughput(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=0.0
        )
        assert result.token_efficiency_index is None

    def test_tei_none_zero_energy(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=0.0, tokens_per_second=50.0
        )
        assert result.token_efficiency_index is None

    def test_all_optional_none_graceful(self):
        result = compute_extended_metrics(
            output_tokens=100,
            total_energy_j=10.0,
            tokens_per_second=50.0,
            itl_mean_ms=None,
            per_request_latencies_ms=None,
            gpu_utilisation_samples=None,
            memory_stats=None,
            batch_stats=None,
            kv_cache_stats=None,
        )
        assert isinstance(result, ExtendedEfficiencyMetrics)
        assert result.memory.peak_memory_mb == 0.0
        assert result.gpu_utilisation.sm_utilisation_mean is None


# ---------------------------------------------------------------------------
# TestComputeMemoryMetrics
# ---------------------------------------------------------------------------


class TestComputeMemoryMetrics:
    """_compute_memory_metrics() from memory_stats dict."""

    def test_none_memory_stats(self):
        mem = _compute_memory_metrics(100, None)
        assert mem.peak_memory_mb == 0.0
        assert mem.tokens_per_gb_vram is None

    def test_tokens_per_gb_vram(self):
        mem = _compute_memory_metrics(
            1024,
            {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 8000.0},
        )
        # 1024 tokens / (1024 MB / 1024) = 1024 / 1.0 = 1024.0
        assert mem.tokens_per_gb_vram == pytest.approx(1024.0)

    def test_model_memory_utilisation(self):
        mem = _compute_memory_metrics(
            100,
            {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 8192.0},
        )
        assert mem.model_memory_utilisation == pytest.approx(8192.0 / 40960.0)

    def test_kv_cache_memory_ratio(self):
        mem = _compute_memory_metrics(
            100,
            {"peak_mb": 2048.0, "total_vram_mb": 40960.0, "model_mb": 1000.0, "kv_cache_mb": 512.0},
        )
        assert mem.kv_cache_memory_ratio == pytest.approx(512.0 / 2048.0)

    def test_zero_peak_tokens_per_gb_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 0.0, "total_vram_mb": 40960.0, "model_mb": 0.0}
        )
        assert mem.tokens_per_gb_vram is None

    def test_zero_total_vram_utilisation_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 1024.0, "total_vram_mb": 0.0, "model_mb": 1000.0}
        )
        assert mem.model_memory_utilisation is None

    def test_kv_cache_mb_none_ratio_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 1000.0}
        )
        assert mem.kv_cache_memory_ratio is None


# ---------------------------------------------------------------------------
# TestComputeGPUUtilisationMetrics
# ---------------------------------------------------------------------------


class TestComputeGPUUtilisationMetrics:
    """_compute_gpu_utilisation_metrics() from sample list."""

    def test_none_input(self):
        gpu = _compute_gpu_utilisation_metrics(None)
        assert gpu.sm_utilisation_mean is None
        assert gpu.sm_utilisation_samples == 0

    def test_empty_list(self):
        gpu = _compute_gpu_utilisation_metrics([])
        assert gpu.sm_utilisation_mean is None

    def test_normal_samples(self):
        gpu = _compute_gpu_utilisation_metrics([50.0, 60.0, 70.0, 80.0])
        assert gpu.sm_utilisation_mean == pytest.approx(65.0)
        assert gpu.sm_utilisation_samples == 4

    def test_single_sample(self):
        gpu = _compute_gpu_utilisation_metrics([42.0])
        assert gpu.sm_utilisation_mean == pytest.approx(42.0)
        assert gpu.sm_utilisation_samples == 1


# ---------------------------------------------------------------------------
# TestComputeBatchMetrics
# ---------------------------------------------------------------------------


class TestComputeBatchMetrics:
    """_compute_batch_metrics() from batch_stats dict."""

    def test_none_returns_empty(self):
        batch = _compute_batch_metrics(None)
        assert batch.effective_batch_size is None
        assert batch.batch_utilisation is None

    def test_batch_utilisation(self):
        batch = _compute_batch_metrics(
            {"effective_batch_size": 8.0, "configured_batch_size": 16, "padding_overhead": 0.1}
        )
        assert batch.batch_utilisation == pytest.approx(0.5)
        assert batch.padding_overhead == pytest.approx(0.1)

    def test_zero_configured_utilisation_none(self):
        batch = _compute_batch_metrics({"effective_batch_size": 8.0, "configured_batch_size": 0})
        assert batch.batch_utilisation is None

    def test_num_batches_cast_to_int(self):
        batch = _compute_batch_metrics({"num_batches": 5.0})
        assert batch.num_batches == 5
        assert isinstance(batch.num_batches, int)

    def test_padding_overhead_passthrough(self):
        batch = _compute_batch_metrics({"padding_overhead": 0.25})
        assert batch.padding_overhead == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# TestComputeKVCacheMetrics
# ---------------------------------------------------------------------------


class TestComputeKVCacheMetrics:
    """_compute_kv_cache_metrics() from kv_cache_stats dict."""

    def test_none_returns_empty(self):
        kv = _compute_kv_cache_metrics(None)
        assert kv.kv_cache_hit_rate is None
        assert kv.kv_cache_blocks_used is None

    def test_normal_case(self):
        kv = _compute_kv_cache_metrics({"hit_rate": 0.85, "blocks_used": 128, "blocks_total": 256})
        assert kv.kv_cache_hit_rate == pytest.approx(0.85)
        assert kv.kv_cache_blocks_used == 128
        assert kv.kv_cache_blocks_total == 256


# ---------------------------------------------------------------------------
# TestComputeRequestLatencyMetrics
# ---------------------------------------------------------------------------


class TestComputeRequestLatencyMetrics:
    """_compute_request_latency_metrics() from latency list."""

    def test_none_returns_empty(self):
        req = _compute_request_latency_metrics(None)
        assert req.e2e_latency_mean_ms is None
        assert req.e2e_latency_samples == 0

    def test_empty_list_returns_empty(self):
        req = _compute_request_latency_metrics([])
        assert req.e2e_latency_mean_ms is None

    def test_normal_latencies(self):
        req = _compute_request_latency_metrics([100.0, 200.0, 150.0, 180.0, 120.0])
        assert req.e2e_latency_mean_ms == pytest.approx(150.0)
        assert req.e2e_latency_median_ms == pytest.approx(150.0)
        assert req.e2e_latency_samples == 5

    def test_single_element(self):
        req = _compute_request_latency_metrics([42.0])
        assert req.e2e_latency_mean_ms == pytest.approx(42.0)
        assert req.e2e_latency_median_ms == pytest.approx(42.0)
        assert req.e2e_latency_p95_ms == pytest.approx(42.0)
        assert req.e2e_latency_p99_ms == pytest.approx(42.0)

    def test_ordering_p95_lte_p99(self):
        req = _compute_request_latency_metrics([10.0, 20.0, 30.0, 100.0, 200.0, 500.0])
        assert req.e2e_latency_p95_ms <= req.e2e_latency_p99_ms


# ---------------------------------------------------------------------------
# TestAggregateExtendedMetrics
# ---------------------------------------------------------------------------


class TestAggregateExtendedMetrics:
    """aggregate_extended_metrics() from multiple process metrics."""

    def test_empty_input(self):
        result = aggregate_extended_metrics(
            raw_extended_metrics=[],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=0,
            aggregated_energy_j=0.0,
            aggregated_tokens_per_sec=0.0,
            itl_mean_ms=None,
        )
        assert isinstance(result, ExtendedEfficiencyMetrics)
        assert result.tpot_ms is None

    def test_memory_max_peak(self):
        m1 = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                peak_memory_mb=1024.0, total_vram_mb=40960.0, model_memory_mb=8000.0
            )
        )
        m2 = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                peak_memory_mb=2048.0, total_vram_mb=40960.0, model_memory_mb=8000.0
            )
        )
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1, m2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.memory.peak_memory_mb == pytest.approx(2048.0)

    def test_memory_sum_vram_and_model(self):
        m1 = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                peak_memory_mb=1024.0, total_vram_mb=20000.0, model_memory_mb=4000.0
            )
        )
        m2 = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(
                peak_memory_mb=1024.0, total_vram_mb=20000.0, model_memory_mb=4000.0
            )
        )
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1, m2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.memory.total_vram_mb == pytest.approx(40000.0)
        assert result.memory.model_memory_mb == pytest.approx(8000.0)

    def test_kv_cache_from_first_non_null(self):
        m1 = ExtendedEfficiencyMetrics(memory=MemoryEfficiencyMetrics(peak_memory_mb=1024.0))
        m2 = ExtendedEfficiencyMetrics(
            memory=MemoryEfficiencyMetrics(peak_memory_mb=1024.0, kv_cache_mb=256.0)
        )
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1, m2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        # kv_cache_mb should come from m2 (first non-null)
        assert result.memory.kv_cache_mb == pytest.approx(256.0)

    def test_batch_mean_effective_size(self):
        m1 = ExtendedEfficiencyMetrics(
            batch=BatchEfficiencyMetrics(
                effective_batch_size=8.0, padding_overhead=0.1, num_batches=10
            )
        )
        m2 = ExtendedEfficiencyMetrics(
            batch=BatchEfficiencyMetrics(
                effective_batch_size=12.0, padding_overhead=0.2, num_batches=10
            )
        )
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1, m2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.batch.effective_batch_size == pytest.approx(10.0)
        assert result.batch.padding_overhead == pytest.approx(0.15)
        assert result.batch.num_batches == 20

    def test_kv_cache_mean_hit_rate_sum_blocks(self):
        m1 = ExtendedEfficiencyMetrics(
            kv_cache=KVCacheEfficiencyMetrics(
                kv_cache_hit_rate=0.8, kv_cache_blocks_used=100, kv_cache_blocks_total=200
            )
        )
        m2 = ExtendedEfficiencyMetrics(
            kv_cache=KVCacheEfficiencyMetrics(
                kv_cache_hit_rate=0.6, kv_cache_blocks_used=50, kv_cache_blocks_total=200
            )
        )
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1, m2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.kv_cache.kv_cache_hit_rate == pytest.approx(0.7)
        assert result.kv_cache.kv_cache_blocks_used == 150
        assert result.kv_cache.kv_cache_blocks_total == 400

    def test_recomputes_tei_from_aggregated(self):
        result = aggregate_extended_metrics(
            raw_extended_metrics=[ExtendedEfficiencyMetrics()],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=200,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=100.0,
            itl_mean_ms=None,
            precision_factor=1.0,
        )
        # TEI = 100 * (200/10) * 1.0 = 2000.0
        assert result.token_efficiency_index == pytest.approx(2000.0)

    def test_concatenated_latencies_single_percentile(self):
        result = aggregate_extended_metrics(
            raw_extended_metrics=[ExtendedEfficiencyMetrics()],
            all_request_latencies=[10.0, 20.0, 30.0, 40.0, 50.0],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.request_latency.e2e_latency_samples == 5
        assert result.request_latency.e2e_latency_mean_ms == pytest.approx(30.0)

    def test_concatenated_gpu_samples_single_mean(self):
        result = aggregate_extended_metrics(
            raw_extended_metrics=[ExtendedEfficiencyMetrics()],
            all_request_latencies=[],
            all_gpu_samples=[50.0, 60.0, 70.0, 80.0],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.gpu_utilisation.sm_utilisation_mean == pytest.approx(65.0)
        assert result.gpu_utilisation.sm_utilisation_samples == 4

    def test_tpot_from_itl_mean(self):
        result = aggregate_extended_metrics(
            raw_extended_metrics=[ExtendedEfficiencyMetrics()],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=8.5,
        )
        assert result.tpot_ms == pytest.approx(8.5)

    def test_no_peak_memory_no_aggregation(self):
        """Processes with peak_memory_mb=0 should not contribute to memory aggregation."""
        m1 = ExtendedEfficiencyMetrics(memory=MemoryEfficiencyMetrics(peak_memory_mb=0.0))
        result = aggregate_extended_metrics(
            raw_extended_metrics=[m1],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=100,
            aggregated_energy_j=10.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )
        assert result.memory.peak_memory_mb == 0.0
