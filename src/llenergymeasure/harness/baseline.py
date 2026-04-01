"""Baseline power measurement with session-level caching.

Measures idle GPU power draw to enable baseline-adjusted energy measurements.
The baseline represents idle power consumption; subtracting it from total energy
isolates the energy attributable to inference work.

For multi-GPU setups, baseline power is summed across all specified GPUs.

Gracefully degrades when NVML is unavailable — returns None instead of crashing.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from llenergymeasure.device.gpu_info import nvml_context
from llenergymeasure.domain.metrics import EnergyBreakdown

logger = logging.getLogger(__name__)


@dataclass
class BaselineCache:
    """Cached baseline power measurement."""

    power_w: float
    timestamp: float
    gpu_indices: list[int]
    sample_count: int
    duration_sec: float
    from_cache: bool = False


# Module-level cache keyed by sorted tuple of gpu_indices
_baseline_cache: dict[tuple[int, ...], BaselineCache] = {}


def measure_baseline_power(
    gpu_indices: list[int] | None = None,
    duration_sec: float = 30.0,
    sample_interval_ms: int = 100,
    cache_ttl_sec: float = 3600.0,
    device_index: int | None = None,
) -> BaselineCache | None:
    """Measure idle GPU baseline power with session caching.

    Samples GPU power draw at the specified interval for the given duration,
    then returns the mean summed power across all specified GPUs. Results are
    cached per GPU set with a configurable TTL.

    Args:
        gpu_indices: CUDA device indices to measure. Defaults to [0] when None.
        duration_sec: How long to sample idle power.
        sample_interval_ms: Interval between power samples in milliseconds.
        cache_ttl_sec: How long cached results remain valid in seconds.
        device_index: Deprecated. Use gpu_indices instead. If provided and
            gpu_indices is None, treated as gpu_indices=[device_index].

    Returns:
        BaselineCache with measured power (summed across GPUs), or None if
        measurement failed.
    """
    # Resolve gpu_indices
    if gpu_indices is not None:
        resolved_indices = gpu_indices
    elif device_index is not None:
        logger.warning(
            "measure_baseline_power: device_index is deprecated, use gpu_indices instead"
        )
        resolved_indices = [device_index]
    else:
        resolved_indices = [0]

    cache_key = tuple(sorted(resolved_indices))

    # Check cache first
    cached = _baseline_cache.get(cache_key)
    if cached is not None and (time.time() - cached.timestamp) < cache_ttl_sec:
        logger.debug(
            "Using cached baseline: %.1fW (age: %.0fs)",
            cached.power_w,
            time.time() - cached.timestamp,
        )
        return replace(cached, from_cache=True)

    # Measure fresh baseline
    if importlib.util.find_spec("pynvml") is None:
        logger.warning("Baseline: pynvml not available, cannot measure baseline power")
        return None

    import pynvml

    interval = sample_interval_ms / 1000.0
    start = time.monotonic()

    # Per-GPU sample lists
    per_gpu_samples: dict[int, list[float]] = {idx: [] for idx in resolved_indices}

    with nvml_context():
        # Get handles for all GPUs
        handles: dict[int, object] = {}
        for gpu_idx in resolved_indices:
            try:
                handles[gpu_idx] = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            except Exception as e:
                logger.warning("Baseline: failed to get handle for GPU %d: %s", gpu_idx, e)

        if not handles:
            logger.warning("Baseline: no GPU handles obtained")
            return None

        try:
            while (time.monotonic() - start) < duration_sec:
                for gpu_idx, handle in handles.items():
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                        per_gpu_samples[gpu_idx].append(power_mw / 1000.0)
                    except pynvml.NVMLError:
                        logger.debug("Power read failed for GPU %d", gpu_idx, exc_info=True)
                time.sleep(interval)
        except Exception as e:
            logger.warning("Baseline: sampling failed: %s", e)

    # Validate sample count against expected rate
    actual_duration = time.monotonic() - start
    expected_samples = int(actual_duration / interval) * len(resolved_indices)
    total_collected = sum(len(s) for s in per_gpu_samples.values())
    if expected_samples > 0 and total_collected < int(expected_samples * 0.8):
        logger.warning(
            "Baseline measurement got %d samples, expected ~%d. Results may be unreliable.",
            total_collected,
            expected_samples,
        )

    # Compute mean power per GPU, then sum across GPUs
    per_gpu_means: list[float] = []
    total_samples = 0
    for gpu_idx in resolved_indices:
        samples = per_gpu_samples.get(gpu_idx, [])
        if samples:
            per_gpu_means.append(sum(samples) / len(samples))
            total_samples += len(samples)

    if not per_gpu_means:
        logger.warning("Baseline: no power samples collected")
        return None

    mean_power = sum(per_gpu_means)

    result = BaselineCache(
        power_w=mean_power,
        timestamp=time.time(),
        gpu_indices=list(resolved_indices),
        sample_count=total_samples,
        duration_sec=actual_duration,
    )

    _baseline_cache[cache_key] = result

    logger.debug(
        "Baseline power measured: %.1fW (%d total samples over %.1fs, %d GPUs)",
        mean_power,
        total_samples,
        actual_duration,
        len(resolved_indices),
    )

    return result


def invalidate_baseline_cache(
    gpu_indices: list[int] | tuple[int, ...] | None = None,
) -> None:
    """Invalidate cached baseline measurements.

    Args:
        gpu_indices: Specific GPU set to invalidate (as list or tuple), or None
            to clear all cached baselines.
    """
    if gpu_indices is not None:
        cache_key = tuple(sorted(gpu_indices))
        _baseline_cache.pop(cache_key, None)
        logger.debug("Baseline cache invalidated for GPUs %s", list(gpu_indices))
    else:
        _baseline_cache.clear()
        logger.debug("Baseline cache cleared (all devices)")


def adjust_energy_for_baseline(
    total_energy_j: float,
    baseline_power_w: float,
    duration_sec: float,
) -> float:
    """Subtract baseline energy from total to isolate inference energy.

    Args:
        total_energy_j: Total measured energy in Joules.
        baseline_power_w: Baseline idle power in Watts.
        duration_sec: Duration of the measurement in seconds.

    Returns:
        Adjusted energy in Joules, floored at 0.0 (negative is physically
        meaningless and indicates measurement noise).
    """
    baseline_energy_j = baseline_power_w * duration_sec
    adjusted = total_energy_j - baseline_energy_j
    return max(0.0, adjusted)


def create_energy_breakdown(
    total_energy_j: float,
    baseline: BaselineCache | None,
    duration_sec: float,
) -> EnergyBreakdown:
    """Create detailed energy breakdown with baseline adjustment.

    Args:
        total_energy_j: Total measured energy in Joules.
        baseline: Cached baseline measurement, or None if unavailable.
        duration_sec: Duration of the inference measurement in seconds.

    Returns:
        EnergyBreakdown with raw and baseline-adjusted values.
    """
    if baseline is not None:
        adjusted_j = adjust_energy_for_baseline(total_energy_j, baseline.power_w, duration_sec)
        cache_age = time.time() - baseline.timestamp

        return EnergyBreakdown(
            raw_j=total_energy_j,
            adjusted_j=adjusted_j,
            baseline_power_w=baseline.power_w,
            baseline_method="cached" if baseline.from_cache else "fresh",
            baseline_timestamp=datetime.fromtimestamp(baseline.timestamp, tz=timezone.utc),
            baseline_cache_age_sec=cache_age,
        )

    return EnergyBreakdown(
        raw_j=total_energy_j,
        adjusted_j=None,
        baseline_power_w=None,
        baseline_method="unavailable",
        baseline_timestamp=None,
        baseline_cache_age_sec=None,
    )
