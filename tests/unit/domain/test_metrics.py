"""Unit tests for domain/metrics.py — precision, normalised, energy, inference, latency models."""

from __future__ import annotations

import pytest

from llenergymeasure.domain.experiment import Timestamps
from llenergymeasure.domain.metrics import (
    CombinedMetrics,
    EnergyMetrics,
    FlopsResult,
    LatencyMeasurementMode,
    LatencyMeasurements,
    NormalisedMetrics,
    PrecisionMetadata,
    collect_itl_measurements,
)
from tests.conftest import make_compute_metrics, make_energy_metrics, make_inference_metrics

# ---------------------------------------------------------------------------
# TestPrecisionMetadata
# ---------------------------------------------------------------------------


class TestPrecisionMetadata:
    """Precision factor mapping for all compute types."""

    @pytest.mark.parametrize(
        ("compute", "expected"),
        [
            ("fp32", 1.0),
            ("fp16", 1.0),
            ("bf16", 1.0),
            ("tf32", 1.0),
            ("fp8", 0.5),
            ("int8", 0.5),
        ],
    )
    def test_precision_factor_known(self, compute: str, expected: float):
        pm = PrecisionMetadata(compute=compute)
        assert pm.precision_factor == expected

    def test_precision_factor_mixed(self):
        pm = PrecisionMetadata(weights="mixed", compute="fp16")
        # compute controls precision_factor, not weights
        assert pm.precision_factor == 1.0

    def test_precision_factor_mixed_compute(self):
        """mixed as a weights value doesn't affect precision_factor directly.

        But PrecisionMetadata's compute field doesn't accept 'mixed' — only
        weights does. Test through the property with known fallback.
        """
        pm = PrecisionMetadata(weights="mixed")
        # Default compute is fp16
        assert pm.precision_factor == 1.0

    def test_precision_factor_default(self):
        pm = PrecisionMetadata()
        # Default compute is fp16 -> 1.0
        assert pm.precision_factor == 1.0


# ---------------------------------------------------------------------------
# TestNormalisedMetrics
# ---------------------------------------------------------------------------


class TestNormalisedMetrics:
    """from_metrics() classmethod with division-by-zero guards."""

    def test_normal_case(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=1e15,
        )
        assert result.tokens_per_joule == pytest.approx(10.0)
        assert result.tokens_per_second_per_watt == pytest.approx(0.4)
        assert result.effective_flops == pytest.approx(1e15)

    def test_zero_energy(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=0.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=1e12,
        )
        assert result.tokens_per_joule == 0.0

    def test_zero_time(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=0.0,
            theoretical_flops=1e12,
        )
        assert result.tokens_per_second_per_watt == 0.0

    def test_zero_power(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=0.0,
            inference_time_sec=5.0,
            theoretical_flops=1e12,
        )
        assert result.tokens_per_second_per_watt == 0.0

    def test_zero_flops(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=0.0,
        )
        assert result.tokens_per_effective_pflop == 0.0
        assert result.effective_flops == 0.0

    def test_precision_factor_application(self):
        prec = PrecisionMetadata(compute="int8")  # factor = 0.5
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=1e15,
            precision=prec,
        )
        assert result.effective_flops == pytest.approx(0.5e15)
        assert result.precision is prec

    def test_none_precision_fallback(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=1e15,
            precision=None,
        )
        # None precision -> factor=1.0
        assert result.effective_flops == pytest.approx(1e15)

    def test_tokens_per_effective_pflop(self):
        result = NormalisedMetrics.from_metrics(
            total_output_tokens=100,
            total_energy_j=10.0,
            mean_power_w=50.0,
            inference_time_sec=5.0,
            theoretical_flops=1e15,
        )
        # 100 / (1e15 / 1e15) = 100.0
        assert result.tokens_per_effective_pflop == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestFlopsResult
# ---------------------------------------------------------------------------


class TestFlopsResult:
    """is_valid property based on value."""

    def test_positive_is_valid(self):
        fr = FlopsResult(value=1e9, method="calflops", confidence="high", precision="fp16")
        assert fr.is_valid is True

    def test_zero_not_valid(self):
        fr = FlopsResult(value=0.0, method="calflops", confidence="low", precision="fp16")
        assert fr.is_valid is False

    def test_negative_not_valid(self):
        fr = FlopsResult(
            value=-1.0, method="parameter_estimate", confidence="low", precision="fp16"
        )
        assert fr.is_valid is False


# ---------------------------------------------------------------------------
# TestEnergyMetrics
# ---------------------------------------------------------------------------


class TestEnergyMetrics:
    """placeholder() and total_power_w property."""

    def test_placeholder_defaults(self):
        em = EnergyMetrics.placeholder()
        assert em.total_energy_j == 0.0
        assert em.gpu_energy_j == 0.0
        assert em.cpu_energy_j == 0.0
        assert em.duration_sec == 0.0

    def test_placeholder_custom_duration(self):
        em = EnergyMetrics.placeholder(duration_sec=7.5)
        assert em.duration_sec == 7.5
        assert em.total_energy_j == 0.0

    def test_total_power_w(self):
        em = make_energy_metrics(gpu_power_w=100.0, cpu_power_w=25.0)
        assert em.total_power_w == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# TestInferenceMetrics
# ---------------------------------------------------------------------------


class TestInferenceMetrics:
    """throughput alias."""

    def test_throughput_alias(self):
        im = make_inference_metrics(tokens_per_second=42.5)
        assert im.throughput == im.tokens_per_second == 42.5


# ---------------------------------------------------------------------------
# TestCombinedMetrics
# ---------------------------------------------------------------------------


class TestCombinedMetrics:
    """efficiency_tokens_per_joule and efficiency_flops_per_watt properties."""

    def _make(self, **energy_overrides) -> CombinedMetrics:
        return CombinedMetrics(
            inference=make_inference_metrics(),
            energy=make_energy_metrics(**energy_overrides),
            compute=make_compute_metrics(flops_per_second=1e10),
        )

    def test_efficiency_tokens_per_joule(self):
        cm = self._make(total_energy_j=10.0)
        # 500 tokens / 10 J = 50.0
        assert cm.efficiency_tokens_per_joule == pytest.approx(50.0)

    def test_efficiency_tokens_per_joule_zero_energy(self):
        cm = self._make(total_energy_j=0.0)
        assert cm.efficiency_tokens_per_joule == 0.0

    def test_efficiency_flops_per_watt(self):
        cm = self._make(gpu_power_w=100.0, cpu_power_w=0.0)
        # 1e10 / 100 = 1e8
        assert cm.efficiency_flops_per_watt == pytest.approx(1e8)

    def test_efficiency_flops_per_watt_zero_power(self):
        cm = self._make(gpu_power_w=0.0, cpu_power_w=0.0)
        assert cm.efficiency_flops_per_watt == 0.0


# ---------------------------------------------------------------------------
# TestTimestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    """from_times() classmethod."""

    def test_positive_duration(self):
        from datetime import datetime, timezone

        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
        ts = Timestamps.from_times(start, end)
        assert ts.duration_sec == pytest.approx(10.0)
        assert ts.start == start
        assert ts.end == end

    def test_zero_duration(self):
        from datetime import datetime, timezone

        t = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = Timestamps.from_times(t, t)
        assert ts.duration_sec == 0.0

    def test_negative_duration(self):
        from datetime import datetime, timezone

        start = datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts = Timestamps.from_times(start, end)
        assert ts.duration_sec < 0


# ---------------------------------------------------------------------------
# TestCollectItlMeasurements
# ---------------------------------------------------------------------------


class TestCollectItlMeasurements:
    """collect_itl_measurements() from per-request token timestamps."""

    def test_normal_multi_request(self):
        timestamps = [
            [0.0, 10.0, 20.0, 30.0, 40.0],  # 4 intervals, trimmed: [10, 10]
            [0.0, 15.0, 25.0, 35.0, 45.0],  # 4 intervals, trimmed: [10, 10]
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 8  # 4 + 4
        assert len(trimmed) == 4  # 2 + 2
        assert excluded == 4  # 2 per request

    def test_trimming_removes_first_last(self):
        timestamps = [[0.0, 100.0, 20.0, 30.0, 5.0]]  # 4 intervals
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        # full = [100, -80, 10, -25]
        assert len(full) == 4
        # trimmed = [-80, 10] (indices 1:-1)
        assert len(trimmed) == 2
        assert excluded == 2

    def test_short_request_two_tokens(self):
        """Two tokens = 1 interval; too short to trim -> excluded."""
        timestamps = [[0.0, 10.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 1
        assert len(trimmed) == 0
        assert excluded == 1

    def test_short_request_three_tokens(self):
        """Three tokens = 2 intervals; still too short for 3-interval trim."""
        timestamps = [[0.0, 10.0, 20.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 2
        assert len(trimmed) == 0
        assert excluded == 2

    def test_single_token_skipped(self):
        """Single token has no intervals."""
        timestamps = [[5.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert full == []
        assert trimmed == []
        assert excluded == 0

    def test_empty_input(self):
        full, trimmed, excluded = collect_itl_measurements([])
        assert full == []
        assert trimmed == []
        assert excluded == 0

    def test_mixed_lengths(self):
        timestamps = [
            [0.0, 10.0, 20.0, 30.0, 40.0],  # 4 intervals, trimmed: 2
            [0.0, 5.0],  # 1 interval, not trimmable
            [0.0],  # 0 intervals, skipped
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 5  # 4 + 1
        assert len(trimmed) == 2
        assert excluded == 3  # 2 from first + 1 from second

    def test_excluded_count_accuracy(self):
        """Excluded count must match intervals removed from full to get trimmed."""
        timestamps = [
            [0.0, 10.0, 20.0, 30.0],  # 3 intervals -> trim first/last -> 1 trimmed, 2 excluded
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 3
        assert len(trimmed) == 1
        assert excluded == 2

    def test_four_token_request(self):
        """Four tokens = 3 intervals -> exactly trimmable."""
        timestamps = [[0.0, 10.0, 20.0, 30.0]]
        full, trimmed, _excluded = collect_itl_measurements(timestamps)
        assert len(full) == 3
        assert len(trimmed) == 1  # middle interval only
        assert trimmed[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestLatencyMeasurements
# ---------------------------------------------------------------------------


class TestLatencyMeasurements:
    """LatencyMeasurements dataclass legacy alias and defaults."""

    def test_measurement_method_legacy_alias(self):
        lm = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[5.0],
            request_count=1,
            total_output_tokens=10,
            excluded_tokens=0,
            streaming_mode=True,
            warmup_requests_excluded=0,
            measurement_mode=LatencyMeasurementMode.PER_REQUEST_BATCH,
        )
        assert lm.measurement_method == "per_request_batch"

    def test_default_measurement_mode(self):
        lm = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[5.0],
            request_count=1,
            total_output_tokens=10,
            excluded_tokens=0,
            streaming_mode=True,
            warmup_requests_excluded=0,
        )
        assert lm.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        assert lm.measurement_method == "true_streaming"
