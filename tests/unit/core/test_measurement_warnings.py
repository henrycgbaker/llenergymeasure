"""Unit tests for collect_measurement_warnings().

Tests all five warning flags: short duration, persistence mode off, thermal drift,
low NVML sample count, and sub-100ms thermal throttle blind spot.
"""

from __future__ import annotations

from llenergymeasure.core.measurement_warnings import collect_measurement_warnings

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
# Warning 5: Sub-100ms thermal throttle blind spot (M4)
# ---------------------------------------------------------------------------


def test_throttle_subsampling_warning_present_when_nvml_active():
    """Warning fires when nvml_sample_count > 0 (NVML is active)."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 50)
    assert any("thermal_throttle_subsampling" in w for w in warnings)


def test_throttle_subsampling_warning_absent_when_nvml_inactive():
    """Warning is absent when nvml_sample_count == 0 (NVML not active)."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 0)
    assert not any("thermal_throttle_subsampling" in w for w in warnings)


def test_throttle_subsampling_warning_present_with_sample_count_1():
    """Warning fires even with a single NVML sample."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 1)
    assert any("thermal_throttle_subsampling" in w for w in warnings)


def test_throttle_subsampling_warning_present_regardless_of_persistence_mode():
    """Warning fires with nvml_sample_count > 0 even when persistence mode is off."""
    warnings = collect_measurement_warnings(30.0, False, 40.0, 42.0, 50)
    assert any("thermal_throttle_subsampling" in w for w in warnings)


def test_throttle_subsampling_warning_contains_methodology_note():
    """Warning text mentions 100ms and methodology limitation."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 50)
    throttle_warnings = [w for w in warnings if "thermal_throttle_subsampling" in w]
    assert len(throttle_warnings) == 1
    assert "100ms" in throttle_warnings[0]
    assert "Methodology limitation" in throttle_warnings[0]


# ---------------------------------------------------------------------------
# Warning interactions
# ---------------------------------------------------------------------------


def test_no_warnings_when_nvml_inactive_and_all_conditions_good():
    """With nvml_sample_count=0 and good conditions, only the low-sample warning fires.

    Note: nvml_sample_count=0 triggers nvml_low_sample_count (0 < 10) but NOT
    thermal_throttle_subsampling (nvml_sample_count is not > 0). There is no
    combination that produces zero warnings once we accept that NVML always either
    gives a low-count warning (inactive) or a throttle-blind-spot warning (active).
    """
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 0)
    # Only low sample count fires (nvml_sample_count=0 < 10)
    assert len(warnings) == 1
    assert "nvml_low_sample_count" in warnings[0]


def test_with_nvml_active_and_good_conditions_only_throttle_warning():
    """With NVML active (>=10 samples) and all other conditions good, only throttle warning fires."""
    warnings = collect_measurement_warnings(30.0, True, 40.0, 42.0, 50)
    assert len(warnings) == 1
    assert "thermal_throttle_subsampling" in warnings[0]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_list_of_strings():
    """collect_measurement_warnings always returns a list of strings."""
    result = collect_measurement_warnings(30.0, True, 40.0, 40.0, 0)
    assert isinstance(result, list)
    assert all(isinstance(w, str) for w in result)
