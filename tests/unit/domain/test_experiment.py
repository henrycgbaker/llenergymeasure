"""Unit tests for domain/experiment.py — RawProcessResult, hashing, StudySummary, edge cases."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    StudySummary,
    compute_measurement_config_hash,
)
from llenergymeasure.domain.metrics import ExtendedEfficiencyMetrics
from tests.conftest import (
    make_config,
    make_raw_process_result,
    make_result,
)

# ---------------------------------------------------------------------------
# TestRawProcessResult
# ---------------------------------------------------------------------------


class TestRawProcessResult:
    """RawProcessResult construction, frozen enforcement, defaults."""

    def test_frozen_model_enforcement(self):
        rpr = make_raw_process_result()
        with pytest.raises(ValidationError):
            rpr.backend = "vllm"

    def test_minimal_construction(self):
        """Builds with only required fields — no extra kwargs needed."""
        rpr = make_raw_process_result()
        assert rpr.experiment_id == "test-001"
        assert rpr.process_index == 0
        assert rpr.gpu_id == 0

    def test_default_extended_metrics(self):
        rpr = make_raw_process_result()
        assert isinstance(rpr.extended_metrics, ExtendedEfficiencyMetrics)
        # Default should have null tpot
        assert rpr.extended_metrics.tpot_ms is None

    def test_default_per_request_latencies(self):
        rpr = make_raw_process_result()
        assert rpr.per_request_latencies_ms == []

    def test_default_gpu_utilisation_samples(self):
        rpr = make_raw_process_result()
        assert rpr.gpu_utilisation_samples == []

    def test_optional_fields_default_none(self):
        rpr = make_raw_process_result()
        assert rpr.energy_breakdown is None
        assert rpr.thermal_throttle is None
        assert rpr.warmup_result is None


# ---------------------------------------------------------------------------
# TestAggregationMetadata
# ---------------------------------------------------------------------------


class TestAggregationMetadata:
    """AggregationMetadata defaults."""

    def test_defaults(self):
        am = AggregationMetadata(num_processes=2)
        assert am.method == "sum_energy_avg_throughput"
        assert am.warnings == []
        assert am.temporal_overlap_verified is False
        assert am.gpu_attribution_verified is False

    def test_with_warnings(self):
        am = AggregationMetadata(num_processes=1, warnings=["something off"])
        assert len(am.warnings) == 1

    def test_verification_flags(self):
        am = AggregationMetadata(
            num_processes=4,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
        )
        assert am.temporal_overlap_verified is True
        assert am.gpu_attribution_verified is True


# ---------------------------------------------------------------------------
# TestStudySummary
# ---------------------------------------------------------------------------


class TestStudySummary:
    """StudySummary optional fields and defaults."""

    def test_unique_configurations_default_none(self):
        ss = StudySummary(total_experiments=5)
        assert ss.unique_configurations is None

    def test_warnings_default_empty(self):
        ss = StudySummary(total_experiments=3)
        assert ss.warnings == []


# ---------------------------------------------------------------------------
# TestExperimentResultEdgeCases
# ---------------------------------------------------------------------------


class TestExperimentResultEdgeCases:
    """Properties and edge cases for ExperimentResult."""

    def test_tokens_per_joule_zero_energy(self):
        r = make_result(total_energy_j=0.0)
        assert r.tokens_per_joule == 0.0

    def test_tokens_per_joule_small_energy(self):
        r = make_result(total_energy_j=1e-10, total_tokens=1000)
        # Should produce a large number without overflow
        assert r.tokens_per_joule == pytest.approx(1000 / 1e-10)
        assert r.tokens_per_joule > 0

    def test_duration_sec_subsecond(self):
        start = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
        r = make_result(start_time=start, end_time=end)
        assert r.duration_sec == pytest.approx(0.5)

    def test_duration_sec_zero_when_same_time(self):
        t = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        r = make_result(start_time=t, end_time=t)
        assert r.duration_sec == 0.0

    def test_frozen_and_extra_forbid(self):
        r = make_result()
        with pytest.raises(ValidationError):
            r.total_tokens = 9999


# ---------------------------------------------------------------------------
# TestMeasurementConfigHash
# ---------------------------------------------------------------------------


class TestMeasurementConfigHash:
    """compute_measurement_config_hash() determinism and shape."""

    def test_hash_length(self):
        config = make_config()
        h = compute_measurement_config_hash(config)
        assert len(h) == 16

    def test_deterministic(self):
        config = make_config()
        h1 = compute_measurement_config_hash(config)
        h2 = compute_measurement_config_hash(config)
        assert h1 == h2

    def test_different_backends_different_hash(self):
        h1 = compute_measurement_config_hash(make_config(backend="pytorch"))
        h2 = compute_measurement_config_hash(make_config(backend="vllm"))
        assert h1 != h2

    def test_hash_is_string(self):
        h = compute_measurement_config_hash(make_config())
        assert isinstance(h, str)
