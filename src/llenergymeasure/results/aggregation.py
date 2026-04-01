"""Aggregation logic for combining raw process results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from llenergymeasure._version import __version__
from llenergymeasure.domain.environment import EnvironmentSnapshot
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    RawProcessResult,
    mj_per_token,
)
from llenergymeasure.domain.metrics import (
    EnergyBreakdown,
    ExtendedEfficiencyMetrics,
    LatencyMeasurements,
    LatencyStatistics,
    MultiGPUMetrics,
    ThermalThrottleInfo,
    WarmupResult,
)
from llenergymeasure.utils.constants import COMPLETION_MARKER_PREFIX
from llenergymeasure.utils.exceptions import AggregationError

logger = logging.getLogger(__name__)


class CompletenessReport(BaseModel):
    """Report on process completion status."""

    expected_processes: int = Field(..., description="Expected number of processes")
    found_processes: int = Field(..., description="Number of results found")
    missing_indices: list[int] = Field(default_factory=list, description="Missing process indices")
    duplicate_indices: list[int] = Field(
        default_factory=list, description="Duplicate process indices"
    )
    marker_status: dict[int, bool] = Field(
        default_factory=dict, description="Process index -> has completion marker"
    )
    is_complete: bool = Field(default=False, description="Whether all processes completed")
    error_message: str | None = Field(default=None, description="Error description if incomplete")


def validate_process_completeness(
    experiment_id: str,
    raw_results: list[RawProcessResult],
    expected_processes: int,
    results_dir: Path,
) -> CompletenessReport:
    """Validate all expected processes completed successfully.

    Checks:
    1. Count: len(results) == expected_processes
    2. Index contiguity: indices are 0, 1, 2, ..., N-1 with no gaps
    3. No duplicates: each index appears exactly once
    4. Markers exist: each process has a completion marker

    Args:
        experiment_id: Experiment identifier.
        raw_results: List of raw process results.
        expected_processes: Number of processes expected.
        results_dir: Base results directory.

    Returns:
        CompletenessReport with validation results.
    """
    found_indices = [r.process_index for r in raw_results]
    expected_indices = set(range(expected_processes))
    found_set = set(found_indices)

    # Check for missing indices
    missing = sorted(expected_indices - found_set)

    # Check for duplicates
    duplicates = sorted({i for i in found_indices if found_indices.count(i) > 1})

    # Check markers
    marker_status: dict[int, bool] = {}
    raw_dir = results_dir / "raw" / experiment_id
    for i in range(expected_processes):
        marker_path = raw_dir / f"{COMPLETION_MARKER_PREFIX}{i}"
        marker_status[i] = marker_path.exists()

    # Determine completeness
    is_complete = (
        len(raw_results) == expected_processes
        and not missing
        and not duplicates
        and all(marker_status.values())
    )

    error_message = None
    if not is_complete:
        issues = []
        if len(raw_results) != expected_processes:
            issues.append(f"Expected {expected_processes} results, found {len(raw_results)}")
        if missing:
            issues.append(f"Missing process indices: {missing}")
        if duplicates:
            issues.append(f"Duplicate process indices: {duplicates}")
        missing_markers = [i for i, has in marker_status.items() if not has]
        if missing_markers:
            issues.append(f"Missing completion markers for processes: {missing_markers}")
        error_message = "; ".join(issues)

    return CompletenessReport(
        expected_processes=expected_processes,
        found_processes=len(raw_results),
        missing_indices=missing,
        duplicate_indices=duplicates,
        marker_status=marker_status,
        is_complete=is_complete,
        error_message=error_message,
    )


@dataclass
class AggregationContext:
    """All parameters needed by aggregate_results(), grouped logically.

    Required fields must be provided; everything else has sensible defaults
    matching the previous keyword-argument interface.
    """

    # -- Required --------------------------------------------------------
    experiment_id: str
    measurement_config_hash: str

    # -- Measurement context ---------------------------------------------
    measurement_methodology: Literal["total", "steady_state", "windowed"] = "total"
    steady_state_window: tuple[float, float] | None = None
    baseline_power_w: float | None = None
    energy_adjusted_j: float | None = None
    energy_per_device_j: list[float] | None = None
    energy_breakdown: EnergyBreakdown | None = None
    multi_gpu: MultiGPUMetrics | None = None

    # -- Environment -----------------------------------------------------
    environment_snapshot: EnvironmentSnapshot | None = None
    thermal_throttle: ThermalThrottleInfo | None = None

    # -- Warmup ----------------------------------------------------------
    warmup_excluded_samples: int | None = None
    warmup_result: WarmupResult | None = None

    # -- Verification ----------------------------------------------------
    verify_temporal_overlap: bool = True
    verify_gpu_attribution: bool = True
    expected_processes: int | None = None

    # -- Output ----------------------------------------------------------
    results_dir: Path | None = None
    strict: bool = True
    allow_mixed_backends: bool = False

    # -- Timeseries + config ---------------------------------------------
    timeseries: str | None = None
    effective_config: dict[str, Any] | None = None
    measurement_warnings: list[str] | None = None


# NOTE: Not yet wired into the harness. The harness currently builds
# ExperimentResult directly via _build_result() for single-process runs.
# This function is designed for future data-parallel multi-process
# experiments where multiple workers each produce a RawProcessResult.
def aggregate_results(
    raw_results: list[RawProcessResult],
    ctx: AggregationContext,
) -> ExperimentResult:
    """Aggregate raw per-process results into a single ExperimentResult (v2.0).

    Aggregation strategy:
    - Energy: Sum across processes (each GPU's energy is additive)
    - Tokens: Sum across processes (each process handles different data)
    - Throughput: Average (tokens_per_second / num_processes gives per-GPU rate)
    - FLOPs: Sum across processes
    - Time: Use wall-clock range (earliest start to latest end)
    - Latencies: Concatenated (late aggregation — no average of averages)

    Args:
        raw_results: List of raw results from each process.
        ctx: AggregationContext with all configuration parameters.

    Returns:
        ExperimentResult (v2.0) combining all process data.

    Raises:
        AggregationError: If raw_results is empty, strict and incomplete,
            or mixed backends without allow_mixed_backends=True.
    """
    if not raw_results:
        raise AggregationError("Cannot aggregate empty results list")

    warnings: list[str] = list(ctx.measurement_warnings or [])
    num_processes = len(raw_results)

    # Backend consistency validation
    backends = {r.backend for r in raw_results}
    if len(backends) > 1:
        backend_list = ", ".join(sorted(backends))
        msg = (
            f"Mixed backends detected: {backend_list}. "
            "Aggregating results from different backends produces statistically invalid comparisons."
        )
        if not ctx.allow_mixed_backends:
            raise AggregationError(
                f"{msg} Use --allow-mixed-backends to override (not recommended)."
            )
        warnings.append(msg)
        logger.warning("Proceeding with mixed-backend aggregation: %s", backend_list)

    # Completeness validation
    if ctx.expected_processes is not None and ctx.results_dir is not None:
        report = validate_process_completeness(
            ctx.experiment_id, raw_results, ctx.expected_processes, ctx.results_dir
        )
        if not report.is_complete:
            if ctx.strict:
                raise AggregationError(f"Incomplete experiment results: {report.error_message}")
            else:
                warnings.append(f"Incomplete results: {report.error_message}")
                logger.warning("Proceeding with incomplete results: %s", report.error_message)

    # Verify temporal overlap
    temporal_overlap_ok = False
    if ctx.verify_temporal_overlap and num_processes > 1:
        temporal_overlap_ok = _check_temporal_overlap(raw_results)
        if not temporal_overlap_ok:
            warnings.append("Processes did not run concurrently - aggregation may be inaccurate")
            logger.warning("Temporal overlap verification failed")

    # Verify GPU attribution
    gpu_attribution_ok = False
    if ctx.verify_gpu_attribution:
        gpu_attribution_ok = _check_gpu_attribution(raw_results)
        if not gpu_attribution_ok:
            warnings.append("Duplicate GPU IDs detected - energy may be double-counted")
            logger.warning("GPU attribution verification failed")

    # Check for MIG instances and warn about energy measurement
    mig_instances = [r for r in raw_results if r.gpu_is_mig]
    if mig_instances:
        mig_profiles = {r.gpu_mig_profile for r in mig_instances if r.gpu_mig_profile}
        profile_str = ", ".join(sorted(mig_profiles)) if mig_profiles else "unknown"
        warnings.append(
            f"Experiment ran on {len(mig_instances)} MIG instance(s) ({profile_str}). "
            "Energy measurements reflect parent GPU total, not per-instance consumption."
        )
        logger.info("MIG instances detected: %d with profiles: %s", len(mig_instances), profile_str)

    # Aggregate core metrics
    total_tokens = sum(r.inference_metrics.total_tokens for r in raw_results)
    total_energy_j = sum(r.energy_metrics.total_energy_j for r in raw_results)
    total_flops = sum(r.compute_metrics.flops_total for r in raw_results)
    total_input_tokens = sum(r.inference_metrics.input_tokens for r in raw_results)
    total_output_tokens = sum(r.inference_metrics.output_tokens for r in raw_results)

    # Calculate averages
    avg_tokens_per_second = (
        sum(r.inference_metrics.tokens_per_second for r in raw_results) / num_processes
        if num_processes > 0
        else 0.0
    )
    avg_energy_per_token = total_energy_j / total_tokens if total_tokens > 0 else 0.0

    # Find time range (wall-clock)
    start_time = min(r.timestamps.start for r in raw_results)
    end_time = max(r.timestamps.end for r in raw_results)
    wall_clock_duration = (end_time - start_time).total_seconds()

    # Build aggregation metadata
    metadata = AggregationMetadata(
        method="sum_energy_avg_throughput",
        num_processes=num_processes,
        temporal_overlap_verified=temporal_overlap_ok,
        gpu_attribution_verified=gpu_attribution_ok,
        warnings=warnings,
    )

    # Aggregate latency measurements if present (streaming mode)
    latency_stats: LatencyStatistics | None = None
    latency_measurements_list: list[LatencyMeasurements] = []
    for r in raw_results:
        lm = r.inference_metrics.latency_measurements
        if lm is not None:
            if isinstance(lm, LatencyMeasurements):
                latency_measurements_list.append(lm)
            elif isinstance(lm, dict) and "ttft_ms" in lm:
                latency_measurements_list.append(LatencyMeasurements(**lm))

    if latency_measurements_list:
        latency_stats = aggregate_latency_measurements(latency_measurements_list)
        if latency_stats:
            logger.info(
                "Latency stats: TTFT mean=%.1fms",
                latency_stats.ttft_mean_ms,
            )

    logger.info(
        "Aggregated %d processes: tokens=%d, energy=%.2fJ, throughput=%.2f tok/s",
        num_processes,
        total_tokens,
        total_energy_j,
        avg_tokens_per_second,
    )

    # Aggregate extended efficiency metrics (late aggregation)
    extended_metrics = _aggregate_extended_metrics_from_results(
        raw_results=raw_results,
        total_energy_j=total_energy_j,
        avg_tokens_per_second=avg_tokens_per_second,
        total_output_tokens=sum(r.inference_metrics.output_tokens for r in raw_results),
        latency_stats=latency_stats,
    )

    # Resolve effective_config and backend from first result if not provided
    resolved_effective_config: dict[str, Any] = ctx.effective_config or (
        raw_results[0].effective_config if raw_results else {}
    )
    backend = raw_results[0].backend if raw_results else "pytorch"
    backend_version: str | None = raw_results[0].backend_version if raw_results else None

    # Energy breakdown: sum raw_j and adjusted_j across processes (if not provided)
    energy_breakdown = ctx.energy_breakdown
    if energy_breakdown is None:
        breakdowns = [p.energy_breakdown for p in raw_results if p.energy_breakdown]
        if breakdowns:
            total_raw = sum(b.raw_j for b in breakdowns)
            adjusted_values = [b.adjusted_j for b in breakdowns if b.adjusted_j is not None]
            total_adjusted = sum(adjusted_values) if adjusted_values else None
            first = breakdowns[0]
            energy_breakdown = EnergyBreakdown(
                raw_j=total_raw,
                adjusted_j=total_adjusted,
                baseline_power_w=first.baseline_power_w,
                baseline_method=first.baseline_method,
                baseline_timestamp=first.baseline_timestamp,
                baseline_cache_age_sec=first.baseline_cache_age_sec,
            )

    # Thermal throttle: merge across processes (if not provided)
    thermal_throttle = ctx.thermal_throttle
    if thermal_throttle is None:
        throttles = [p.thermal_throttle for p in raw_results if p.thermal_throttle]
        if throttles:
            max_temps = [t.max_temperature_c for t in throttles if t.max_temperature_c is not None]
            thermal_throttle = ThermalThrottleInfo(
                detected=any(t.detected for t in throttles),
                thermal=any(t.thermal for t in throttles),
                power=any(t.power for t in throttles),
                sw_thermal=any(t.sw_thermal for t in throttles),
                hw_thermal=any(t.hw_thermal for t in throttles),
                hw_power=any(t.hw_power for t in throttles),
                throttle_duration_sec=max(t.throttle_duration_sec for t in throttles),
                max_temperature_c=max(max_temps) if max_temps else None,
            )

    # Warmup result: take from first process if not provided
    warmup_result = ctx.warmup_result
    if warmup_result is None:
        warmup_result = next((p.warmup_result for p in raw_results if p.warmup_result), None)

    # Compute FLOPs-derived fields (C3)
    flops_per_output_token = total_flops / total_output_tokens if total_output_tokens > 0 else None
    flops_per_input_token = total_flops / total_input_tokens if total_input_tokens > 0 else None
    flops_per_second = total_flops / wall_clock_duration if wall_clock_duration > 0 else None

    # mJ/tok derived fields
    _mj_total = mj_per_token(total_energy_j, total_tokens)
    _energy_adj = energy_breakdown.adjusted_j if energy_breakdown else None
    _mj_adjusted = mj_per_token(_energy_adj, total_tokens) if _energy_adj is not None else None

    return ExperimentResult(
        schema_version="2.0",
        experiment_id=ctx.experiment_id,
        measurement_config_hash=ctx.measurement_config_hash,
        llenergymeasure_version=__version__,
        backend=backend,
        backend_version=backend_version,
        measurement_methodology=ctx.measurement_methodology,
        steady_state_window=ctx.steady_state_window,
        total_tokens=total_tokens,
        total_energy_j=total_energy_j,
        total_inference_time_sec=wall_clock_duration,
        avg_tokens_per_second=avg_tokens_per_second,
        avg_energy_per_token_j=avg_energy_per_token,
        total_flops=total_flops,
        flops_per_output_token=flops_per_output_token,
        flops_per_input_token=flops_per_input_token,
        flops_per_second=flops_per_second,
        mj_per_tok_total=_mj_total,
        mj_per_tok_adjusted=_mj_adjusted,
        baseline_power_w=ctx.baseline_power_w,
        energy_adjusted_j=ctx.energy_adjusted_j,
        energy_per_device_j=ctx.energy_per_device_j,
        energy_breakdown=energy_breakdown,
        multi_gpu=ctx.multi_gpu,
        environment_snapshot=ctx.environment_snapshot,
        measurement_warnings=warnings,
        warmup_excluded_samples=ctx.warmup_excluded_samples,
        timeseries=ctx.timeseries,
        start_time=start_time,
        end_time=end_time,
        effective_config=resolved_effective_config,
        process_results=raw_results,
        aggregation=metadata,
        thermal_throttle=thermal_throttle,
        warmup_result=warmup_result,
        latency_stats=latency_stats,
        extended_metrics=extended_metrics,
    )


def _check_temporal_overlap(results: list[RawProcessResult]) -> bool:
    """Check if process execution times overlap.

    For valid distributed execution, all processes should run concurrently.
    Returns True if there's significant overlap between all processes.
    """
    if len(results) < 2:
        return True

    # Find the intersection of all time ranges
    max_start = max(r.timestamps.start for r in results)
    min_end = min(r.timestamps.end for r in results)

    # Check if there's positive overlap
    if max_start >= min_end:
        return False

    # Check that overlap is at least 50% of the shortest process duration
    overlap_duration = (min_end - max_start).total_seconds()
    min_process_duration = min(r.timestamps.duration_sec for r in results)

    return overlap_duration >= (min_process_duration * 0.5)


def _check_gpu_attribution(results: list[RawProcessResult]) -> bool:
    """Check that GPU IDs are unique across processes.

    Duplicate GPU IDs could indicate double-counting of energy.
    """
    gpu_ids = [r.gpu_id for r in results]
    return len(gpu_ids) == len(set(gpu_ids))


def _aggregate_extended_metrics_from_results(
    raw_results: list[RawProcessResult],
    total_energy_j: float,
    avg_tokens_per_second: float,
    total_output_tokens: int,
    latency_stats: LatencyStatistics | None,
) -> ExtendedEfficiencyMetrics | None:
    """Aggregate extended efficiency metrics from per-process results.

    Collects raw samples from all processes and computes aggregated statistics.
    Uses late aggregation pattern: raw samples stored per-process, stats computed here.

    Args:
        raw_results: List of raw results from each process.
        total_energy_j: Total energy consumption across all processes.
        avg_tokens_per_second: Average throughput.
        total_output_tokens: Total output tokens generated.
        latency_stats: Aggregated latency statistics (for ITL/TPOT).

    Returns:
        Aggregated ExtendedEfficiencyMetrics, or None if aggregation fails.
    """
    from llenergymeasure.results.extended_metrics import aggregate_extended_metrics

    # Collect raw data from all processes for late aggregation
    all_request_latencies: list[float] = []
    all_gpu_samples: list[float] = []
    raw_extended_metrics: list[ExtendedEfficiencyMetrics] = []

    for r in raw_results:
        # Collect per-request latencies (late aggregation — concatenate, not average)
        if r.per_request_latencies_ms:
            all_request_latencies.extend(r.per_request_latencies_ms)

        # Collect GPU utilisation samples
        if r.gpu_utilisation_samples:
            all_gpu_samples.extend(r.gpu_utilisation_samples)

        # Collect per-process extended metrics
        raw_extended_metrics.append(r.extended_metrics)

    # Get ITL mean for TPOT (from aggregated latency stats)
    itl_mean_ms: float | None = None
    if latency_stats and latency_stats.itl_mean_ms is not None:
        itl_mean_ms = latency_stats.itl_mean_ms

    try:
        extended_metrics = aggregate_extended_metrics(
            raw_extended_metrics=raw_extended_metrics,
            all_request_latencies=all_request_latencies,
            all_gpu_samples=all_gpu_samples,
            aggregated_output_tokens=total_output_tokens,
            aggregated_energy_j=total_energy_j,
            aggregated_tokens_per_sec=avg_tokens_per_second,
            itl_mean_ms=itl_mean_ms,
            precision_factor=1.0,
        )
        logger.debug(
            "Aggregated extended metrics: TPOT=%s, TEI=%s",
            extended_metrics.tpot_ms,
            extended_metrics.token_efficiency_index,
        )
    except Exception as e:
        logger.warning("Extended metrics aggregation failed (non-fatal): %s", e)
        extended_metrics = ExtendedEfficiencyMetrics()

    return extended_metrics


def aggregate_latency_measurements(
    measurements: list[LatencyMeasurements],
) -> LatencyStatistics | None:
    """Aggregate raw latency measurements from multiple processes.

    Concatenates raw samples from all processes, then computes statistics.
    This is the correct way to aggregate percentiles (not mean of percentiles).

    Args:
        measurements: List of LatencyMeasurements from each process.

    Returns:
        LatencyStatistics with computed percentiles, or None if no data.
    """
    if not measurements:
        return None

    # Concatenate all raw samples
    all_ttft: list[float] = []
    all_itl_full: list[float] = []
    all_itl_trimmed: list[float] = []

    for m in measurements:
        all_ttft.extend(m.ttft_ms)
        all_itl_full.extend(m.itl_full_ms)
        all_itl_trimmed.extend(m.itl_trimmed_ms)

    if not all_ttft:
        logger.warning("No TTFT samples to aggregate")
        return None

    # Compute TTFT statistics
    ttft_arr = np.array(all_ttft)

    # Compute ITL statistics (trimmed - primary metric)
    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_samples = 0

    if all_itl_trimmed:
        itl_arr = np.array(all_itl_trimmed)
        itl_mean_ms = float(np.mean(itl_arr))
        itl_median_ms = float(np.median(itl_arr))
        itl_p95_ms = float(np.percentile(itl_arr, 95))
        itl_p99_ms = float(np.percentile(itl_arr, 99))
        itl_samples = len(all_itl_trimmed)

    # Compute ITL full statistics (for comparison)
    itl_full_mean_ms: float | None = None
    itl_full_p99_ms: float | None = None

    if all_itl_full:
        itl_full_arr = np.array(all_itl_full)
        itl_full_mean_ms = float(np.mean(itl_full_arr))
        itl_full_p99_ms = float(np.percentile(itl_full_arr, 99))

    logger.info(
        "Aggregated latency stats: TTFT samples=%d, ITL samples=%d (trimmed)",
        len(all_ttft),
        itl_samples,
    )

    return LatencyStatistics(
        ttft_mean_ms=float(np.mean(ttft_arr)),
        ttft_median_ms=float(np.median(ttft_arr)),
        ttft_p95_ms=float(np.percentile(ttft_arr, 95)),
        ttft_p99_ms=float(np.percentile(ttft_arr, 99)),
        ttft_min_ms=float(np.min(ttft_arr)),
        ttft_max_ms=float(np.max(ttft_arr)),
        ttft_samples=len(all_ttft),
        itl_mean_ms=itl_mean_ms,
        itl_median_ms=itl_median_ms,
        itl_p95_ms=itl_p95_ms,
        itl_p99_ms=itl_p99_ms,
        itl_samples=itl_samples,
        itl_full_mean_ms=itl_full_mean_ms,
        itl_full_p99_ms=itl_full_p99_ms,
    )
