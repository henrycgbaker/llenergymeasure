"""Unit tests for core/baseline.py — baseline power measurement and cache logic.

Covers:
- Cache hit path (returns cached without calling pynvml)
- Cache miss / TTL expiry triggers fresh measurement
- Fresh measurement samples, computes mean, stores in cache
- pynvml unavailable returns None
- Device handle failure returns None
- NVMLError during sampling skips bad samples
- No samples collected returns None
- invalidate_baseline_cache (specific GPU set and all devices)
- adjust_energy_for_baseline (positive and floor-at-zero)
- create_energy_breakdown (with and without baseline)
- Multi-GPU baseline measurement sums power across GPUs
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.harness.baseline import (
    BaselineCache,
    _baseline_cache,
    adjust_energy_for_baseline,
    create_energy_breakdown,
    invalidate_baseline_cache,
    measure_baseline_power,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_pynvml_mock(power_mw_values: list[int] | None = None) -> MagicMock:
    """Build a minimal pynvml mock.

    power_mw_values: side_effect list for nvmlDeviceGetPowerUsage.
                     If None, returns a constant 200_000 (200 W).
    """
    mock = MagicMock()
    # Must define NVMLError as an exception class so `except pynvml.NVMLError` works
    mock.NVMLError = Exception

    if power_mw_values is not None:
        mock.nvmlDeviceGetPowerUsage.side_effect = power_mw_values
    else:
        mock.nvmlDeviceGetPowerUsage.return_value = 200_000  # 200 W

    return mock


@contextmanager
def _noop_nvml_context():
    """No-op replacement for nvml_context() — skips nvmlInit/nvmlShutdown."""
    yield


def _make_fresh_cache_entry(
    gpu_indices: list[int] | None = None, power_w: float = 50.0
) -> BaselineCache:
    """Create a cache entry that is within TTL (timestamp = now)."""
    return BaselineCache(
        power_w=power_w,
        timestamp=time.time(),
        gpu_indices=gpu_indices if gpu_indices is not None else [0],
        sample_count=3,
        duration_sec=0.3,
    )


def _make_expired_cache_entry(
    gpu_indices: list[int] | None = None, power_w: float = 50.0
) -> BaselineCache:
    """Create a cache entry that is beyond the default 3600s TTL."""
    return BaselineCache(
        power_w=power_w,
        timestamp=time.time() - 4000.0,  # well past 1-hour TTL
        gpu_indices=gpu_indices if gpu_indices is not None else [0],
        sample_count=3,
        duration_sec=0.3,
    )


# Patch target constants
_MODULE = "llenergymeasure.harness.baseline"


# =============================================================================
# Cache hit tests
# =============================================================================


def test_cache_hit_returns_cached_without_pynvml_call():
    """Pre-populated fresh cache entry is returned without calling pynvml."""
    entry = _make_fresh_cache_entry(gpu_indices=[0], power_w=42.0)
    _baseline_cache[(0,)] = entry

    mock_pynvml = _make_pynvml_mock()

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
    ):
        result = measure_baseline_power(gpu_indices=[0], cache_ttl_sec=3600.0)

    assert result is not entry  # replace() creates a new object
    assert result.from_cache is True
    assert result.power_w == 42.0
    mock_pynvml.nvmlDeviceGetPowerUsage.assert_not_called()


def test_cache_hit_respects_ttl():
    """Fresh entry within TTL is returned; expired entry is not used."""
    fresh_entry = _make_fresh_cache_entry(gpu_indices=[0], power_w=99.0)
    _baseline_cache[(0,)] = fresh_entry

    # Very short TTL — should still hit because timestamp is "now"
    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
    ):
        result = measure_baseline_power(gpu_indices=[0], cache_ttl_sec=3600.0)

    assert result.from_cache is True
    assert result.power_w == fresh_entry.power_w


def test_cache_hit_backward_compat_device_index():
    """device_index kwarg still hits cache keyed by (device_index,) tuple."""
    entry = _make_fresh_cache_entry(gpu_indices=[0], power_w=77.0)
    _baseline_cache[(0,)] = entry

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
    ):
        result = measure_baseline_power(device_index=0, cache_ttl_sec=3600.0)

    assert result.from_cache is True
    assert result.power_w == 77.0


# =============================================================================
# Cache miss / fresh measurement
# =============================================================================


def test_cache_expired_triggers_fresh_measurement():
    """An expired cache entry causes a fresh measurement to be taken."""
    expired = _make_expired_cache_entry(gpu_indices=[0], power_w=10.0)
    _baseline_cache[(0,)] = expired

    mock_pynvml = _make_pynvml_mock(power_mw_values=[200_000])

    # monotonic: start=0.0, loop-check=0.1 (< 1.0 duration, one iteration), then 999.0 exits
    mono_values = [0.0, 0.1, 999.0, 999.0]
    mono_iter = iter(mono_values)

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{_MODULE}.time.monotonic", side_effect=lambda: next(mono_iter)),
        patch(f"{_MODULE}.time.sleep"),
    ):
        result = measure_baseline_power(gpu_indices=[0], duration_sec=1.0, cache_ttl_sec=3600.0)

    # The expired entry should have been replaced
    assert result is not expired
    assert result is not None
    mock_pynvml.nvmlDeviceGetPowerUsage.assert_called()


def test_fresh_measurement_samples_and_caches():
    """Empty cache: samples power, computes mean, stores in _baseline_cache."""
    power_values = [150_000, 155_000, 148_000]  # milliwatts
    mock_pynvml = _make_pynvml_mock(power_mw_values=power_values)

    # monotonic: start=0.0, then 0.1, 0.2, 0.3 (< 0.4 duration) then 1.0 (exits)
    mono_values = [0.0, 0.1, 0.2, 0.3, 1.0, 1.0]
    mono_iter = iter(mono_values)

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{_MODULE}.time.monotonic", side_effect=lambda: next(mono_iter)),
        patch(f"{_MODULE}.time.sleep"),
    ):
        result = measure_baseline_power(gpu_indices=[0], duration_sec=0.4)

    assert result is not None
    expected_mean = sum(v / 1000.0 for v in power_values) / len(power_values)
    assert result.power_w == pytest.approx(expected_mean, rel=1e-6)
    assert result.sample_count == len(power_values)
    assert result.gpu_indices == [0]
    # Must be stored in cache
    assert _baseline_cache.get((0,)) is result


# =============================================================================
# Failure paths: pynvml unavailable
# =============================================================================


def test_pynvml_unavailable_returns_none():
    """When find_spec('pynvml') returns None, measure_baseline_power returns None."""
    with patch(f"{_MODULE}.importlib.util.find_spec", return_value=None):
        result = measure_baseline_power(gpu_indices=[0])

    assert result is None


def test_device_handle_failure_returns_none():
    """nvmlDeviceGetHandleByIndex raises: measure_baseline_power returns None."""
    mock_pynvml = _make_pynvml_mock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("handle error")

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
    ):
        result = measure_baseline_power(gpu_indices=[0])

    assert result is None


def test_nvml_error_during_sampling_skips_bad_samples():
    """NVMLError on some samples is skipped; valid samples are still collected."""
    nvml_error = Exception("nvml error")
    # Alternating: good, bad, good
    power_side_effects = [200_000, nvml_error, 180_000]

    mock_pynvml = _make_pynvml_mock(power_mw_values=power_side_effects)
    mock_pynvml.NVMLError = type(nvml_error)

    mono_values = [0.0, 0.1, 0.2, 0.3, 1.0, 1.0]
    mono_iter = iter(mono_values)

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{_MODULE}.time.monotonic", side_effect=lambda: next(mono_iter)),
        patch(f"{_MODULE}.time.sleep"),
    ):
        result = measure_baseline_power(gpu_indices=[0], duration_sec=0.4)

    # Two valid samples (200 W and 180 W); bad sample skipped
    assert result is not None
    assert result.sample_count == 2
    assert result.power_w == pytest.approx((200.0 + 180.0) / 2)


def test_no_samples_collected_returns_none():
    """All nvmlDeviceGetPowerUsage calls raise NVMLError: returns None."""
    nvml_error = Exception("nvml error")
    mock_pynvml = _make_pynvml_mock()
    mock_pynvml.NVMLError = type(nvml_error)
    mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = nvml_error

    mono_values = [0.0, 0.05, 0.10, 0.15, 1.0, 1.0]
    mono_iter = iter(mono_values)

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{_MODULE}.time.monotonic", side_effect=lambda: next(mono_iter)),
        patch(f"{_MODULE}.time.sleep"),
    ):
        result = measure_baseline_power(gpu_indices=[0], duration_sec=0.2)

    assert result is None


# =============================================================================
# Multi-GPU baseline measurement
# =============================================================================


def test_multi_gpu_baseline_measurement():
    """Baseline power for 2 GPUs sums their per-GPU means."""
    # GPU 0 at 100W, GPU 1 at 150W -> total = 250W
    call_count = [0]

    def power_side_effect(handle):
        call_count[0] += 1
        # Alternate handles: 0 -> 100_000 mW, 1 -> 150_000 mW
        # Handles are mock objects; we use call order to determine GPU
        # handle_0 is the first handle returned, handle_1 second
        if handle is handle_0:
            return 100_000
        return 150_000

    mock_pynvml = MagicMock()
    mock_pynvml.NVMLError = Exception
    handle_0 = MagicMock(name="handle_0")
    handle_1 = MagicMock(name="handle_1")
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [handle_0, handle_1]
    mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = power_side_effect

    # monotonic: start=0.0, one iteration (0.1 < 0.4), then exit (1.0)
    mono_values = [0.0, 0.1, 1.0, 1.0]
    mono_iter = iter(mono_values)

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch.dict(sys.modules, {"pynvml": mock_pynvml}),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
        patch(f"{_MODULE}.time.monotonic", side_effect=lambda: next(mono_iter)),
        patch(f"{_MODULE}.time.sleep"),
    ):
        result = measure_baseline_power(gpu_indices=[0, 1], duration_sec=0.4)

    assert result is not None
    assert result.gpu_indices == [0, 1]
    # 100W + 150W = 250W
    assert result.power_w == pytest.approx(250.0, rel=1e-6)


def test_multi_gpu_cache_keyed_by_sorted_tuple():
    """Multi-GPU baseline is cached under sorted gpu_indices tuple."""
    entry = _make_fresh_cache_entry(gpu_indices=[0, 1], power_w=300.0)
    _baseline_cache[(0, 1)] = entry

    with (
        patch(f"{_MODULE}.nvml_context", _noop_nvml_context),
        patch(f"{_MODULE}.importlib.util.find_spec", return_value=MagicMock()),
    ):
        # Same GPUs different order — should hit cache
        result = measure_baseline_power(gpu_indices=[1, 0], cache_ttl_sec=3600.0)

    assert result.from_cache is True
    assert result.power_w == entry.power_w


# =============================================================================
# invalidate_baseline_cache
# =============================================================================


def test_invalidate_specific_device():
    """invalidate_baseline_cache([0]) removes GPU 0 but leaves GPU 1."""
    _baseline_cache[(0,)] = _make_fresh_cache_entry(gpu_indices=[0])
    _baseline_cache[(1,)] = _make_fresh_cache_entry(gpu_indices=[1])

    invalidate_baseline_cache([0])

    assert (0,) not in _baseline_cache
    assert (1,) in _baseline_cache


def test_invalidate_all_devices():
    """invalidate_baseline_cache(None) clears the entire cache."""
    _baseline_cache[(0,)] = _make_fresh_cache_entry(gpu_indices=[0])
    _baseline_cache[(1,)] = _make_fresh_cache_entry(gpu_indices=[1])
    _baseline_cache[(2,)] = _make_fresh_cache_entry(gpu_indices=[2])

    invalidate_baseline_cache(None)

    assert len(_baseline_cache) == 0


def test_invalidate_missing_device_is_noop():
    """invalidate_baseline_cache for a GPU set not in cache does not raise."""
    _baseline_cache[(0,)] = _make_fresh_cache_entry(gpu_indices=[0])

    invalidate_baseline_cache([99])  # GPU 99 not in cache

    assert (0,) in _baseline_cache  # GPU 0 untouched


def test_invalidate_multi_gpu_set():
    """invalidate_baseline_cache([0, 1]) removes (0, 1) key."""
    _baseline_cache[(0, 1)] = _make_fresh_cache_entry(gpu_indices=[0, 1])
    _baseline_cache[(0,)] = _make_fresh_cache_entry(gpu_indices=[0])

    invalidate_baseline_cache([0, 1])

    assert (0, 1) not in _baseline_cache
    assert (0,) in _baseline_cache  # single-GPU entry untouched


# =============================================================================
# adjust_energy_for_baseline
# =============================================================================


def test_adjust_energy_positive_result():
    """adjust_energy subtracts baseline energy from total; result > 0."""
    # total = 100J, baseline = 10W * 5s = 50J -> adjusted = 50J
    result = adjust_energy_for_baseline(
        total_energy_j=100.0,
        baseline_power_w=10.0,
        duration_sec=5.0,
    )
    assert result == pytest.approx(50.0)


def test_adjust_energy_floors_at_zero():
    """adjust_energy floors at 0.0 when baseline > total (measurement noise)."""
    # total = 5J, baseline = 10W * 5s = 50J -> would be -45J, floors at 0
    result = adjust_energy_for_baseline(
        total_energy_j=5.0,
        baseline_power_w=10.0,
        duration_sec=5.0,
    )
    assert result == 0.0


def test_adjust_energy_exact_zero():
    """adjust_energy returns 0.0 when total exactly equals baseline energy."""
    result = adjust_energy_for_baseline(
        total_energy_j=50.0,
        baseline_power_w=10.0,
        duration_sec=5.0,
    )
    assert result == 0.0


# =============================================================================
# create_energy_breakdown
# =============================================================================


def test_create_energy_breakdown_without_baseline():
    """None baseline produces an EnergyBreakdown with method='unavailable'."""
    breakdown = create_energy_breakdown(
        total_energy_j=75.0,
        baseline=None,
        duration_sec=10.0,
    )

    assert breakdown.raw_j == 75.0
    assert breakdown.adjusted_j is None
    assert breakdown.baseline_power_w is None
    assert breakdown.baseline_method == "unavailable"
    assert breakdown.baseline_timestamp is None
    assert breakdown.baseline_cache_age_sec is None


def test_create_energy_breakdown_with_fresh_baseline():
    """Baseline with timestamp ~now produces method='fresh' (cache_age < 1s)."""
    entry = BaselineCache(
        power_w=20.0,
        timestamp=time.time(),  # age ~ 0s
        gpu_indices=[0],
        sample_count=5,
        duration_sec=0.5,
    )

    breakdown = create_energy_breakdown(
        total_energy_j=100.0,
        baseline=entry,
        duration_sec=10.0,
    )

    assert breakdown.raw_j == 100.0
    # adjusted = 100 - (20W * 10s) = 100 - 200 = -100 -> 0.0
    assert breakdown.adjusted_j == 0.0
    assert breakdown.baseline_power_w == 20.0
    assert breakdown.baseline_method == "fresh"
    assert breakdown.baseline_timestamp is not None
    assert breakdown.baseline_cache_age_sec is not None
    assert breakdown.baseline_cache_age_sec >= 0.0


def test_create_energy_breakdown_with_cached_baseline():
    """Baseline with from_cache=True produces method='cached'."""
    entry = BaselineCache(
        power_w=5.0,
        timestamp=time.time() - 600.0,  # 10 minutes old
        gpu_indices=[0],
        sample_count=5,
        duration_sec=0.5,
        from_cache=True,
    )

    breakdown = create_energy_breakdown(
        total_energy_j=200.0,
        baseline=entry,
        duration_sec=20.0,
    )

    # adjusted = 200 - (5W * 20s) = 200 - 100 = 100J
    assert breakdown.adjusted_j == pytest.approx(100.0)
    assert breakdown.baseline_method == "cached"
    assert breakdown.baseline_cache_age_sec > 1.0


def test_create_energy_breakdown_adjusted_value_correct():
    """EnergyBreakdown.adjusted_j equals adjust_energy_for_baseline result."""
    power_w = 15.0
    duration_sec = 8.0
    total_energy_j = 180.0

    entry = BaselineCache(
        power_w=power_w,
        timestamp=time.time() - 10.0,
        gpu_indices=[0],
        sample_count=3,
        duration_sec=0.3,
    )

    breakdown = create_energy_breakdown(
        total_energy_j=total_energy_j,
        baseline=entry,
        duration_sec=duration_sec,
    )

    expected_adjusted = adjust_energy_for_baseline(total_energy_j, power_w, duration_sec)
    assert breakdown.adjusted_j == pytest.approx(expected_adjusted)
