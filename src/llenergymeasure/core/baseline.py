"""Baseline power measurement with session-level caching.

Measures idle GPU power draw to enable baseline-adjusted energy measurements.
The baseline represents idle power consumption; subtracting it from total energy
isolates the energy attributable to inference work.

Gracefully degrades when NVML is unavailable — returns None instead of crashing.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from llenergymeasure.core.gpu_info import nvml_context
from llenergymeasure.domain.metrics import EnergyBreakdown

logger = logging.getLogger(__name__)


@dataclass
class BaselineCache:
    """Cached baseline power measurement."""

    power_w: float
    timestamp: float
    device_index: int
    sample_count: int
    duration_sec: float


# Module-level cache keyed by device index
_baseline_cache: dict[int, BaselineCache] = {}


def measure_baseline_power(
    device_index: int = 0,
    duration_sec: float = 30.0,
    sample_interval_ms: int = 100,
    cache_ttl_sec: float = 3600.0,
) -> BaselineCache | None:
    """Measure idle GPU baseline power with session caching.

    Samples GPU power draw at the specified interval for the given duration,
    then returns the mean power. Results are cached per device index with
    a configurable TTL.

    Args:
        device_index: CUDA device index to measure.
        duration_sec: How long to sample idle power.
        sample_interval_ms: Interval between power samples in milliseconds.
        cache_ttl_sec: How long cached results remain valid in seconds.

    Returns:
        BaselineCache with measured power, or None if measurement failed.
    """
    # Check cache first
    cached = _baseline_cache.get(device_index)
    if cached is not None and (time.time() - cached.timestamp) < cache_ttl_sec:
        logger.info(
            "Using cached baseline: %.1fW (age: %.0fs)",
            cached.power_w,
            time.time() - cached.timestamp,
        )
        return cached

    # Measure fresh baseline
    if importlib.util.find_spec("pynvml") is None:
        logger.warning("Baseline: pynvml not available, cannot measure baseline power")
        return None

    import pynvml

    samples: list[float] = []
    interval = sample_interval_ms / 1000.0
    start = time.monotonic()

    with nvml_context():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            logger.warning("Baseline: failed to get device handle: %s", e)
            return None

        try:
            while (time.monotonic() - start) < duration_sec:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    samples.append(power_mw / 1000.0)
                except pynvml.NVMLError:
                    pass
                time.sleep(interval)
        except Exception as e:
            logger.warning("Baseline: sampling failed: %s", e)

    if not samples:
        logger.warning("Baseline: no power samples collected")
        return None

    mean_power = sum(samples) / len(samples)
    actual_duration = time.monotonic() - start

    result = BaselineCache(
        power_w=mean_power,
        timestamp=time.time(),
        device_index=device_index,
        sample_count=len(samples),
        duration_sec=actual_duration,
    )

    _baseline_cache[device_index] = result

    logger.info(
        "Baseline power measured: %.1fW (%d samples over %.1fs)",
        mean_power,
        len(samples),
        actual_duration,
    )

    return result


def invalidate_baseline_cache(device_index: int | None = None) -> None:
    """Invalidate cached baseline measurements.

    Args:
        device_index: Specific device to invalidate, or None to clear all.
    """
    if device_index is not None:
        _baseline_cache.pop(device_index, None)
        logger.debug("Baseline cache invalidated for device %d", device_index)
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
            baseline_method="cached" if cache_age > 1.0 else "fresh",
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
