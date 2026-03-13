"""Unit tests for PowerThermalSampler and PowerThermalResult.

Covers:
- Thermal throttle constant fixes (B1): hw/sw bit distinction, combined bit, deprecated names
- Sampler lifecycle: start/stop, context manager, thread cleanup
- pynvml available: samples collected, is_available True, mean_power correct
- pynvml unavailable (ImportError): is_available False, sample_count 0
- Device handle failure: sample_count 0
- get_power_samples filters None values
- PowerThermalResult.from_sampler class method
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.power_thermal import (
    PowerThermalResult,
    PowerThermalSample,
    PowerThermalSampler,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_pynvml_mock(
    *,
    hw_thermal: int = 0x40,
    sw_thermal: int = 0x20,
    sw_power_cap: int = 0x04,
    hw_power_brake: int = 0x80,
    use_deprecated_names: bool = False,
) -> MagicMock:
    """Build a pynvml mock with the given constant values.

    When use_deprecated_names=True, only the deprecated nvmlClocksThrottleReason*
    names are present (simulating older NVML). When False, the modern
    nvmlClocksEventReason* names are present (and deprecated names may be absent).
    """
    mock = MagicMock()

    if use_deprecated_names:
        # Only deprecated names present — no nvmlClocksEventReason* attrs
        del mock.nvmlClocksEventReasonHwThermalSlowdown
        del mock.nvmlClocksEventReasonSwThermalSlowdown
        del mock.nvmlClocksEventReasonSwPowerCap
        del mock.nvmlClocksEventReasonHwPowerBrakeSlowdown
        mock.nvmlClocksThrottleReasonHwThermalSlowdown = hw_thermal
        mock.nvmlClocksThrottleReasonSwThermalSlowdown = sw_thermal
        mock.nvmlClocksThrottleReasonSwPowerCap = sw_power_cap
        mock.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = hw_power_brake
    else:
        # Modern names present
        mock.nvmlClocksEventReasonHwThermalSlowdown = hw_thermal
        mock.nvmlClocksEventReasonSwThermalSlowdown = sw_thermal
        mock.nvmlClocksEventReasonSwPowerCap = sw_power_cap
        mock.nvmlClocksEventReasonHwPowerBrakeSlowdown = hw_power_brake

    return mock


def _make_sampler_with_samples(throttle_reasons_values: list[int]) -> PowerThermalSampler:
    """Create a PowerThermalSampler with pre-populated samples (no real pynvml needed)."""
    sampler = PowerThermalSampler(device_index=0, sample_interval_ms=100)
    for i, reasons in enumerate(throttle_reasons_values):
        # thermal_throttle flag not used by get_thermal_throttle_info (it reads throttle_reasons)
        sample = PowerThermalSample(
            timestamp=float(i),
            throttle_reasons=reasons,
            thermal_throttle=False,  # will be recalculated in get_thermal_throttle_info
        )
        sampler._samples.append(sample)
    return sampler


# =============================================================================
# Test 1: hw and sw thermal constants are distinct (0x40 vs 0x20)
# =============================================================================


def test_thermal_constants_are_distinct():
    """hw_thermal_bit=0x40 and sw_thermal_bit=0x20; thermal=True for hw-only throttle."""
    mock_pynvml = _make_pynvml_mock(hw_thermal=0x40, sw_thermal=0x20)

    sampler = _make_sampler_with_samples([0x40])  # hw thermal only

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        info = sampler.get_thermal_throttle_info()

    assert info.hw_thermal is True, "hw_thermal should be True (0x40 bit set)"
    assert info.sw_thermal is False, "sw_thermal should be False (0x20 bit not set)"
    assert info.thermal is True, "thermal (combined) should be True when hw_thermal=True"


# =============================================================================
# Test 2: Combined thermal bit detects sw-only throttle
# =============================================================================


def test_thermal_combined_detects_either():
    """thermal=True for sw-only throttle; hw_thermal=False when hw bit not set."""
    mock_pynvml = _make_pynvml_mock(hw_thermal=0x40, sw_thermal=0x20)

    sampler = _make_sampler_with_samples([0x20])  # sw thermal only

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        info = sampler.get_thermal_throttle_info()

    assert info.thermal is True, "thermal (combined) should be True when sw_thermal=True"
    assert info.sw_thermal is True, "sw_thermal should be True (0x20 bit set)"
    assert info.hw_thermal is False, "hw_thermal should be False (0x40 bit not set)"


# =============================================================================
# Test 3: No throttle when reasons=0
# =============================================================================


def test_thermal_no_throttle():
    """All thermal fields False when throttle_reasons=0."""
    mock_pynvml = _make_pynvml_mock(hw_thermal=0x40, sw_thermal=0x20)

    sampler = _make_sampler_with_samples([0])

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        info = sampler.get_thermal_throttle_info()

    assert info.thermal is False, "thermal should be False when no throttle bits set"
    assert info.sw_thermal is False, "sw_thermal should be False"
    assert info.hw_thermal is False, "hw_thermal should be False"
    assert info.detected is False, "detected should be False (no sample had thermal_throttle=True)"


# =============================================================================
# Test 4: Deprecated name fallback
# =============================================================================


def test_deprecated_name_fallback():
    """When nvmlClocksEventReason* attrs absent, nvmlClocksThrottleReason* fallback used."""
    mock_pynvml = _make_pynvml_mock(
        hw_thermal=0x40,
        sw_thermal=0x20,
        use_deprecated_names=True,
    )

    # Inject both hw and sw throttle
    sampler = _make_sampler_with_samples([0x60])  # 0x40 | 0x20

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        info = sampler.get_thermal_throttle_info()

    assert info.hw_thermal is True, "hw_thermal should be True via deprecated fallback"
    assert info.sw_thermal is True, "sw_thermal should be True via deprecated fallback"
    assert info.thermal is True, "thermal (combined) should be True"


# =============================================================================
# Helper: build a fully-mocked pynvml for lifecycle tests
# =============================================================================


def _make_full_pynvml_mock() -> MagicMock:
    """Build a pynvml mock suitable for _sample_loop lifecycle tests."""
    mock = MagicMock()
    mock.NVMLError = Exception

    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.return_value = handle
    mock.nvmlDeviceGetPowerUsage.return_value = 200_000  # 200 W in mW
    mock.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        used=10 * 1024**3,
        total=80 * 1024**3,
    )
    mock.nvmlDeviceGetTemperature.return_value = 75
    mock.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=90)
    mock.nvmlClocksEventReasonSwThermalSlowdown = 0x20
    mock.nvmlClocksEventReasonHwThermalSlowdown = 0x40
    mock.nvmlClocksEventReasonSwPowerCap = 0x04
    mock.nvmlClocksEventReasonHwPowerBrakeSlowdown = 0x80
    mock.nvmlDeviceGetCurrentClocksEventReasons.return_value = 0
    mock.NVML_TEMPERATURE_GPU = 0

    return mock


# =============================================================================
# Test 5: Sampler lifecycle — pynvml available
# =============================================================================


def test_sampler_lifecycle_pynvml_available():
    """Sampler starts, collects samples, stops; is_available=True, mean_power=200W."""
    mock_pynvml = _make_full_pynvml_mock()

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        sampler = PowerThermalSampler(device_index=0, sample_interval_ms=10)
        with sampler:
            time.sleep(0.07)  # ~7 intervals at 10ms; gives thread time to collect samples

    assert sampler.is_available is True
    assert sampler.sample_count > 0
    mean = sampler.get_mean_power()
    assert mean is not None
    assert mean == pytest.approx(200.0, rel=0.01)


# =============================================================================
# Test 6: Sampler — pynvml unavailable (ImportError)
# =============================================================================


def test_sampler_pynvml_unavailable():
    """When pynvml import fails, is_available=False and no samples collected."""
    with patch.dict(sys.modules, {"pynvml": None}):
        sampler = PowerThermalSampler(device_index=0, sample_interval_ms=10)
        with sampler:
            time.sleep(0.03)

    assert sampler.is_available is False
    assert sampler.sample_count == 0
    assert sampler.get_mean_power() is None


# =============================================================================
# Test 7: Sampler — device handle failure
# =============================================================================


def test_sampler_device_handle_failure():
    """nvmlDeviceGetHandleByIndex raises: no samples collected."""
    mock_pynvml = _make_full_pynvml_mock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("handle error")

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        sampler = PowerThermalSampler(device_index=0, sample_interval_ms=10)
        with sampler:
            time.sleep(0.03)

    assert sampler.sample_count == 0


# =============================================================================
# Test 8: Context manager — __enter__ starts, __exit__ stops, thread is None after
# =============================================================================


def test_sampler_context_manager_lifecycle():
    """__enter__ starts the thread, __exit__ stops it and sets _thread to None."""
    mock_pynvml = _make_full_pynvml_mock()

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        sampler = PowerThermalSampler(device_index=0, sample_interval_ms=10)

        # Before start: thread is None
        assert sampler._thread is None

        sampler.__enter__()
        assert sampler._thread is not None  # thread started
        assert sampler._running is True

        sampler.__exit__(None, None, None)

    # After exit: thread cleaned up
    assert sampler._thread is None
    assert sampler._running is False


# =============================================================================
# Test 9: get_power_samples filters None values
# =============================================================================


def test_get_power_samples_filters_none():
    """get_power_samples() excludes samples where power_w is None."""
    sampler = PowerThermalSampler(device_index=0, sample_interval_ms=100)

    # Inject samples: some with power, some without
    sampler._samples = [
        PowerThermalSample(timestamp=0.0, power_w=100.0),
        PowerThermalSample(timestamp=0.1, power_w=None),  # no power reading
        PowerThermalSample(timestamp=0.2, power_w=200.0),
        PowerThermalSample(timestamp=0.3, power_w=None),  # no power reading
        PowerThermalSample(timestamp=0.4, power_w=150.0),
    ]

    power_samples = sampler.get_power_samples()
    assert power_samples == [100.0, 200.0, 150.0]
    assert len(power_samples) == 3


# =============================================================================
# Test 10: PowerThermalResult.from_sampler
# =============================================================================


def test_power_thermal_result_from_sampler():
    """PowerThermalResult.from_sampler() extracts power, memory, temp samples correctly."""
    sampler = PowerThermalSampler(device_index=0, sample_interval_ms=100)
    sampler._pynvml_available = True

    # Inject pre-computed samples
    sampler._samples = [
        PowerThermalSample(
            timestamp=0.0,
            power_w=100.0,
            memory_used_mb=10_000.0,
            temperature_c=70.0,
            throttle_reasons=0,
            thermal_throttle=False,
        ),
        PowerThermalSample(
            timestamp=0.1,
            power_w=200.0,
            memory_used_mb=11_000.0,
            temperature_c=72.0,
            throttle_reasons=0,
            thermal_throttle=False,
        ),
    ]

    mock_pynvml = _make_pynvml_mock(hw_thermal=0x40, sw_thermal=0x20)
    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        result = PowerThermalResult.from_sampler(sampler)

    assert result.available is True
    assert result.sample_count == 2
    assert result.power_samples == [100.0, 200.0]
    assert result.memory_samples == [10_000.0, 11_000.0]
    assert result.temperature_samples == [70.0, 72.0]
    assert result.thermal_throttle_info is not None


# =============================================================================
# Test 11: get_thermal_throttle_info with no samples returns default
# =============================================================================


def test_get_thermal_throttle_info_empty_samples():
    """get_thermal_throttle_info with no samples returns default ThermalThrottleInfo."""
    sampler = PowerThermalSampler(device_index=0, sample_interval_ms=100)
    assert sampler._samples == []

    info = sampler.get_thermal_throttle_info()

    assert info.detected is False
    assert info.thermal is False
    assert info.throttle_duration_sec == 0.0
    assert info.max_temperature_c is None
