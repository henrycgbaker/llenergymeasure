"""Unit tests for thermal throttle constant fixes in PowerThermalSampler.

Tests B1 fix: hw_thermal_bit and sw_thermal_bit resolve to distinct NVML constants
(0x40 and 0x20), thermal_bit = hw | sw (combined), and deprecated name fallback works.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from llenergymeasure.core.power_thermal import PowerThermalSample, PowerThermalSampler

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
