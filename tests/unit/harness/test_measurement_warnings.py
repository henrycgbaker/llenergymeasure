"""Unit tests for collect_measurement_warnings().

Tests all four warning flags: short duration, persistence mode off, thermal drift,
and low NVML sample count.
"""

from __future__ import annotations

from llenergymeasure.harness.measurement_warnings import collect_measurement_warnings

# ---------------------------------------------------------------------------
# Warning 1: Short measurement duration
# ---------------------------------------------------------------------------


def test_short_duration_warning_fires_below_10s():
    """Warning fires when measurement duration is under 10 seconds."""
    warnings = collect_measurement_warnings(5.0, True, 40.0, 40.0, 50)
    assert any("short_measurement_duration" in w for w in warnings)


def test_short_duration_warning_absent_at_10s():
    """Warning is absent when duration equals exactly 10 seconds."""
    warnings = collect_measurement_warnings(10.0, True, 40.0, 40.0, 50)
    assert not any("short_measurement_duration" in w for w in warnings)


def test_short_duration_warning_absent_above_10s():
    """Warning is absent when duration is above 10 seconds."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 50)
    assert not any("short_measurement_duration" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 2: GPU persistence mode off
# ---------------------------------------------------------------------------


def test_persistence_mode_warning_fires_when_off():
    """Warning fires when gpu_persistence_mode=False."""
    warnings = collect_measurement_warnings(30.0, False, 40.0, 40.0, 50)
    assert any("gpu_persistence_mode_off" in w for w in warnings)


def test_persistence_mode_warning_absent_when_on():
    """Warning is absent when gpu_persistence_mode=True."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 50)
    assert not any("gpu_persistence_mode_off" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 3: Thermal drift
# ---------------------------------------------------------------------------


def test_thermal_drift_warning_fires_above_threshold():
    """Warning fires when temperature drift exceeds 10C threshold."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 52.0, 50)
    assert any("thermal_drift_detected" in w for w in warnings)


def test_thermal_drift_warning_absent_within_threshold():
    """Warning is absent when temperature drift is within threshold."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 49.9, 50)
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_thermal_drift_warning_absent_when_temps_unavailable():
    """Warning is absent when temperature readings are unavailable (None)."""
    warnings = collect_measurement_warnings(30.0, True, None, None, 50)
    assert not any("thermal_drift_detected" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning 4: Low NVML sample count
# ---------------------------------------------------------------------------


def test_low_nvml_sample_warning_fires_below_10():
    """Warning fires when NVML sample count is below 10."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 5)
    assert any("nvml_low_sample_count" in w for w in warnings)


def test_low_nvml_sample_warning_absent_at_10():
    """Warning is absent when NVML sample count equals 10."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 40.0, 10)
    assert not any("nvml_low_sample_count" in w for w in warnings)


# ---------------------------------------------------------------------------
# Warning interactions
# ---------------------------------------------------------------------------


def test_no_warnings_when_nvml_inactive_and_all_conditions_good():
    """With nvml_sample_count=0 and good conditions, only the low-sample warning fires."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 0)
    # Only low sample count fires (nvml_sample_count=0 < 10)
    assert len(warnings) == 1
    assert "nvml_low_sample_count" in warnings[0]


def test_no_warnings_when_all_conditions_good():
    """With NVML active (>=10 samples) and all other conditions good, no warnings fire."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 50)
    assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_list_of_strings():
    """collect_measurement_warnings always returns a list of strings."""
    result = collect_measurement_warnings(30.0, True, 40.0, 40.0, 0)
    assert isinstance(result, list)
    assert all(isinstance(w, str) for w in result)
