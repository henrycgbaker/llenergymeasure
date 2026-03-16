"""Unit tests for v2.0 aggregation: aggregate_results() -> ExperimentResult."""

from __future__ import annotations

from datetime import datetime

import pytest

from llenergymeasure.domain.experiment import (
    ExperimentResult,
    RawProcessResult,
    Timestamps,
)
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyMetrics,
    InferenceMetrics,
)
from llenergymeasure.results.aggregation import aggregate_results, validate_process_completeness

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_raw_result():
    """Factory fixture for constructing valid RawProcessResult objects."""

    def _make(process_index: int = 0, gpu_id: int = 0, **overrides) -> RawProcessResult:
        defaults: dict = dict(
            experiment_id="test-001",
            process_index=process_index,
            gpu_id=gpu_id,
            config_name="test",
            model_name="gpt2",
            timestamps=Timestamps.from_times(
                datetime(2026, 2, 26, 14, 0, 0),
                datetime(2026, 2, 26, 14, 0, 10),
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=500,
                input_tokens=100,
                output_tokens=400,
                inference_time_sec=10.0,
                tokens_per_second=50.0,
                latency_per_token_ms=2.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=25.0,
                duration_sec=10.0,
                energy_per_token_j=0.05,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=5e11,
            ),
            per_request_latencies_ms=[100.0, 110.0, 95.0, 105.0, 90.0],
        )
        defaults.update(overrides)
        return RawProcessResult(**defaults)

    return _make


# ---------------------------------------------------------------------------
# Task 1: single-process aggregation produces correct ExperimentResult
# ---------------------------------------------------------------------------


def test_aggregate_single_process(make_raw_result):
    """Single RawProcessResult produces ExperimentResult with matching metrics."""
    raw = make_raw_result()
    result = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )
    assert result.total_tokens == 500
    assert result.total_energy_j == 25.0
    assert result.total_flops == 5e11
    assert result.total_inference_time_sec == 10.0
    assert result.avg_tokens_per_second == 50.0


def test_aggregate_returns_experiment_result(make_raw_result):
    """Return type is ExperimentResult, not some other class."""
    raw = make_raw_result()
    result = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )
    assert type(result).__name__ == "ExperimentResult"
    assert isinstance(result, ExperimentResult)


def test_aggregate_schema_version(make_raw_result):
    """Aggregated result carries schema_version == '2.0'."""
    raw = make_raw_result()
    result = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )
    assert result.schema_version == "2.0"


def test_aggregate_measurement_config_hash(make_raw_result):
    """measurement_config_hash passed through to result unchanged."""
    raw = make_raw_result()
    result = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="deadbeef12345678",
    )
    assert result.measurement_config_hash == "deadbeef12345678"


def test_aggregate_measurement_methodology(make_raw_result):
    """measurement_methodology passed through to result unchanged."""
    raw = make_raw_result()

    result_total = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
        measurement_methodology="total",
    )
    assert result_total.measurement_methodology == "total"

    result_ss = aggregate_results(
        raw_results=[raw],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
        measurement_methodology="steady_state",
    )
    assert result_ss.measurement_methodology == "steady_state"


# ---------------------------------------------------------------------------
# Multi-process aggregation
# ---------------------------------------------------------------------------


def test_aggregate_energy_sum(make_raw_result):
    """Two processes: energy is summed (25 + 30 = 55)."""
    raw0 = make_raw_result(
        process_index=0,
        gpu_id=0,
        energy_metrics=EnergyMetrics(total_energy_j=25.0, duration_sec=10.0),
    )
    raw1 = make_raw_result(
        process_index=1,
        gpu_id=1,
        energy_metrics=EnergyMetrics(total_energy_j=30.0, duration_sec=10.0),
    )
    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )
    assert result.total_energy_j == pytest.approx(55.0)


def test_aggregate_tokens_sum(make_raw_result):
    """Two processes: tokens are summed."""
    raw0 = make_raw_result(
        process_index=0,
        gpu_id=0,
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=100,
            output_tokens=400,
            inference_time_sec=10.0,
            tokens_per_second=50.0,
            latency_per_token_ms=2.0,
        ),
    )
    raw1 = make_raw_result(
        process_index=1,
        gpu_id=1,
        inference_metrics=InferenceMetrics(
            total_tokens=600,
            input_tokens=120,
            output_tokens=480,
            inference_time_sec=12.0,
            tokens_per_second=50.0,
            latency_per_token_ms=2.0,
        ),
    )
    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )
    assert result.total_tokens == 1100


def test_aggregate_late_aggregation_latencies(make_raw_result):
    """Two processes with different latency lists: all latencies concatenated, not averaged."""
    latencies_p0 = [100.0, 110.0, 95.0]
    latencies_p1 = [200.0, 210.0, 195.0]

    raw0 = make_raw_result(process_index=0, gpu_id=0, per_request_latencies_ms=latencies_p0)
    raw1 = make_raw_result(process_index=1, gpu_id=1, per_request_latencies_ms=latencies_p1)

    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )

    # The extended_metrics should reflect the combined latency distribution
    # (late aggregation means raw latencies were concatenated before computing stats)
    # We verify this indirectly: the result includes process_results with all raw latencies
    all_latencies_in_process_results = []
    for proc in result.process_results:
        all_latencies_in_process_results.extend(proc.per_request_latencies_ms)

    expected_all = latencies_p0 + latencies_p1
    assert sorted(all_latencies_in_process_results) == sorted(expected_all)


def test_aggregate_process_results_embedded(make_raw_result):
    """process_results list contains the original RawProcessResult objects."""
    raw0 = make_raw_result(process_index=0, gpu_id=0)
    raw1 = make_raw_result(process_index=1, gpu_id=1)

    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )

    assert len(result.process_results) == 2
    indices = {p.process_index for p in result.process_results}
    assert indices == {0, 1}


def test_aggregate_metadata_num_processes(make_raw_result):
    """aggregation.num_processes matches input count."""
    raw0 = make_raw_result(process_index=0, gpu_id=0)
    raw1 = make_raw_result(process_index=1, gpu_id=1)
    raw2 = make_raw_result(process_index=2, gpu_id=2)

    result = aggregate_results(
        raw_results=[raw0, raw1, raw2],
        experiment_id="test-001",
        measurement_config_hash="abc123def456abcd",
    )

    assert result.aggregation is not None
    assert result.aggregation.num_processes == 3


# ---------------------------------------------------------------------------
# validate_process_completeness
# ---------------------------------------------------------------------------


def test_validate_process_completeness_complete(make_raw_result, tmp_path):
    """2 processes with indices 0,1, both with markers -> is_complete=True."""
    experiment_id = "exp-complete"
    raw_dir = tmp_path / "raw" / experiment_id
    raw_dir.mkdir(parents=True)

    # Create completion markers
    (raw_dir / ".completed_0").touch()
    (raw_dir / ".completed_1").touch()

    raw0 = make_raw_result(process_index=0, experiment_id=experiment_id)
    raw1 = make_raw_result(process_index=1, experiment_id=experiment_id)

    report = validate_process_completeness(
        experiment_id=experiment_id,
        raw_results=[raw0, raw1],
        expected_processes=2,
        results_dir=tmp_path,
    )

    assert report.is_complete is True
    assert report.missing_indices == []
    assert report.duplicate_indices == []
    assert report.error_message is None


def test_validate_process_completeness_missing(make_raw_result, tmp_path):
    """1 of 2 processes missing -> is_complete=False."""
    experiment_id = "exp-missing"
    raw_dir = tmp_path / "raw" / experiment_id
    raw_dir.mkdir(parents=True)

    # Only marker for process 0
    (raw_dir / ".completed_0").touch()

    raw0 = make_raw_result(process_index=0, experiment_id=experiment_id)
    # Only one process result, expected two

    report = validate_process_completeness(
        experiment_id=experiment_id,
        raw_results=[raw0],
        expected_processes=2,
        results_dir=tmp_path,
    )

    assert report.is_complete is False
    assert 1 in report.missing_indices
    assert report.error_message is not None


def test_validate_process_completeness_duplicate(make_raw_result, tmp_path):
    """Two results with process_index=0 -> duplicate detected."""
    experiment_id = "exp-dup"
    raw_dir = tmp_path / "raw" / experiment_id
    raw_dir.mkdir(parents=True)

    # Both markers present
    (raw_dir / ".completed_0").touch()
    (raw_dir / ".completed_1").touch()

    # Both have process_index=0 (duplicate)
    raw0a = make_raw_result(process_index=0, gpu_id=0, experiment_id=experiment_id)
    raw0b = make_raw_result(process_index=0, gpu_id=1, experiment_id=experiment_id)

    report = validate_process_completeness(
        experiment_id=experiment_id,
        raw_results=[raw0a, raw0b],
        expected_processes=2,
        results_dir=tmp_path,
    )

    assert report.is_complete is False
    assert 0 in report.duplicate_indices
    assert report.error_message is not None


# ---------------------------------------------------------------------------
# C2: Wall-clock inference time (not sum of per-process times)
# ---------------------------------------------------------------------------


def test_aggregate_uses_wall_clock_not_sum(make_raw_result):
    """Aggregated inference time is wall-clock duration, not sum of per-process times.

    Two processes each have inference_time_sec=10.0 (sum=20.0).
    But they ran concurrently: start=14:00:00, end=14:00:12 -> wall-clock=12s.
    The aggregated result must report 12.0, not 20.0.
    """
    raw0 = make_raw_result(
        process_index=0,
        gpu_id=0,
        timestamps=Timestamps.from_times(
            datetime(2026, 2, 26, 14, 0, 0),
            datetime(2026, 2, 26, 14, 0, 12),
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=100,
            output_tokens=400,
            inference_time_sec=10.0,
            tokens_per_second=50.0,
            latency_per_token_ms=2.0,
        ),
    )
    raw1 = make_raw_result(
        process_index=1,
        gpu_id=1,
        timestamps=Timestamps.from_times(
            datetime(2026, 2, 26, 14, 0, 1),
            datetime(2026, 2, 26, 14, 0, 12),
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=100,
            output_tokens=400,
            inference_time_sec=10.0,
            tokens_per_second=50.0,
            latency_per_token_ms=2.0,
        ),
    )
    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-wall-clock",
        measurement_config_hash="abc123def456abcd",
    )

    # Wall-clock is max(end) - min(start) = 14:00:12 - 14:00:00 = 12 seconds
    assert result.total_inference_time_sec == pytest.approx(12.0)
    # Definitely NOT the sum of per-process times (20.0)
    assert result.total_inference_time_sec != pytest.approx(20.0)


# ---------------------------------------------------------------------------
# C3: FLOPs-derived fields populated in aggregated results
# ---------------------------------------------------------------------------


def test_aggregate_populates_flops_derived_fields(make_raw_result):
    """Aggregated result has non-None flops_per_output_token, flops_per_input_token,
    flops_per_second when FLOPs and token counts are available.

    Process 0: flops=1e12, input_tokens=100, output_tokens=400, duration 10s
    Process 1: flops=2e12, input_tokens=200, output_tokens=800, duration 10s
    Wall-clock span: 10 seconds (concurrent).

    Expected:
      total_flops = 3e12
      total_output_tokens = 1200
      total_input_tokens = 300
      flops_per_output_token = 3e12 / 1200 = 2.5e9
      flops_per_input_token = 3e12 / 300 = 1e10
      flops_per_second = 3e12 / 10 = 3e11
    """
    from llenergymeasure.domain.metrics import ComputeMetrics

    raw0 = make_raw_result(
        process_index=0,
        gpu_id=0,
        timestamps=Timestamps.from_times(
            datetime(2026, 2, 26, 14, 0, 0),
            datetime(2026, 2, 26, 14, 0, 10),
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=500,
            input_tokens=100,
            output_tokens=400,
            inference_time_sec=10.0,
            tokens_per_second=50.0,
            latency_per_token_ms=2.0,
        ),
        compute_metrics=ComputeMetrics(flops_total=1e12),
    )
    raw1 = make_raw_result(
        process_index=1,
        gpu_id=1,
        timestamps=Timestamps.from_times(
            datetime(2026, 2, 26, 14, 0, 0),
            datetime(2026, 2, 26, 14, 0, 10),
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=1000,
            input_tokens=200,
            output_tokens=800,
            inference_time_sec=10.0,
            tokens_per_second=100.0,
            latency_per_token_ms=1.0,
        ),
        compute_metrics=ComputeMetrics(flops_total=2e12),
    )

    result = aggregate_results(
        raw_results=[raw0, raw1],
        experiment_id="test-flops-derived",
        measurement_config_hash="abc123def456abcd",
    )

    assert result.flops_per_output_token is not None
    assert result.flops_per_input_token is not None
    assert result.flops_per_second is not None

    assert result.flops_per_output_token == pytest.approx(3e12 / 1200)
    assert result.flops_per_input_token == pytest.approx(3e12 / 300)
    assert result.flops_per_second == pytest.approx(3e12 / 10)
