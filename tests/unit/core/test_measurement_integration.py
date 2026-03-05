"""Unit tests for measurement integration: timeseries, warnings, and PyTorchBackend wiring.

All tests are mocked — no GPU or real model required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.measurement_warnings import collect_measurement_warnings
from llenergymeasure.core.power_thermal import PowerThermalSample
from llenergymeasure.core.timeseries import write_timeseries_parquet

# =============================================================================
# Timeseries writer tests
# =============================================================================


def _make_samples(n_seconds: int, start_ts: float = 0.0) -> list[PowerThermalSample]:
    """Generate synthetic PowerThermalSamples at 100ms intervals spanning n_seconds."""
    samples = []
    for sec in range(n_seconds):
        for ms_offset in range(0, 1000, 100):  # 10 samples per second
            ts = start_ts + sec + ms_offset / 1000.0
            samples.append(
                PowerThermalSample(
                    timestamp=ts,
                    power_w=100.0 + sec * 2.0,
                    temperature_c=45.0 + sec * 0.5,
                    memory_used_mb=8192.0,
                    memory_total_mb=40960.0,
                    sm_utilisation=85.0,
                    throttle_reasons=0,
                )
            )
    return samples


def test_timeseries_parquet_write() -> None:
    """write_timeseries_parquet() produces 1 Hz rows with correct schema."""
    import pyarrow.parquet as pq

    samples = _make_samples(n_seconds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "timeseries.parquet"
        result_path = write_timeseries_parquet(samples, output, gpu_index=0)

        assert result_path == output
        assert output.exists()

        table = pq.read_table(output)
        # Should have 3-4 rows (one per second bucket)
        assert 3 <= len(table) <= 4, f"Expected 3-4 rows, got {len(table)}"

        # Check schema columns
        expected_columns = {
            "timestamp_s",
            "gpu_index",
            "power_w",
            "temperature_c",
            "memory_used_mb",
            "memory_total_mb",
            "sm_utilisation_pct",
            "throttle_reasons",
        }
        assert set(table.schema.names) == expected_columns

        # Check gpu_index column value
        gpu_indices = table.column("gpu_index").to_pylist()
        assert all(g == 0 for g in gpu_indices)

        # Power values should be non-null (we provided real values)
        power_values = table.column("power_w").to_pylist()
        assert all(p is not None for p in power_values)


def test_timeseries_parquet_empty() -> None:
    """write_timeseries_parquet() with empty samples creates 0-row file with correct schema."""
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "timeseries.parquet"
        write_timeseries_parquet([], output, gpu_index=1)

        assert output.exists()
        table = pq.read_table(output)
        assert len(table) == 0

        # Schema must still be correct even with 0 rows
        expected_columns = {
            "timestamp_s",
            "gpu_index",
            "power_w",
            "temperature_c",
            "memory_used_mb",
            "memory_total_mb",
            "sm_utilisation_pct",
            "throttle_reasons",
        }
        assert set(table.schema.names) == expected_columns


def test_timeseries_parquet_creates_parent_dir() -> None:
    """write_timeseries_parquet() creates parent directories as needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "nested" / "dirs" / "timeseries.parquet"
        write_timeseries_parquet([], output)
        assert output.exists()


def test_timeseries_throttle_reasons_ored() -> None:
    """Throttle reason bitmasks are OR'd within each 1s bucket."""
    import pyarrow.parquet as pq

    # Two samples in same second with different throttle bits
    samples = [
        PowerThermalSample(timestamp=0.0, throttle_reasons=0b01),
        PowerThermalSample(timestamp=0.1, throttle_reasons=0b10),
        PowerThermalSample(timestamp=0.2, throttle_reasons=0b00),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "ts.parquet"
        write_timeseries_parquet(samples, output)
        table = pq.read_table(output)
        throttle = table.column("throttle_reasons").to_pylist()
        assert throttle[0] == 0b11  # 0b01 | 0b10 | 0b00


# =============================================================================
# Measurement warnings tests
# =============================================================================


def test_warnings_short_duration() -> None:
    """Short measurement duration triggers short_measurement_duration warning."""
    warnings = collect_measurement_warnings(
        duration_sec=5.0,
        gpu_persistence_mode=True,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=100,
    )
    assert any("short_measurement_duration" in w for w in warnings)


def test_warnings_persistence_mode() -> None:
    """Persistence mode off triggers gpu_persistence_mode_off warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=False,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=100,
    )
    assert any("gpu_persistence_mode_off" in w for w in warnings)


def test_warnings_thermal_drift() -> None:
    """Temperature drift above threshold triggers thermal_drift_detected warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=30.0,
        temp_end_c=45.0,  # 15C drift > 10C threshold
        nvml_sample_count=100,
        thermal_drift_threshold_c=10.0,
    )
    assert any("thermal_drift_detected" in w for w in warnings)
    # Warning should include the actual drift and threshold
    drift_warning = next(w for w in warnings if "thermal_drift_detected" in w)
    assert "15.0C" in drift_warning
    assert "10.0C" in drift_warning


def test_warnings_thermal_drift_below_threshold() -> None:
    """Temperature drift below threshold does NOT trigger thermal_drift_detected."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=44.0,
        temp_end_c=45.0,  # 1C drift < 10C threshold
        nvml_sample_count=100,
        thermal_drift_threshold_c=10.0,
    )
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_warnings_low_sample_count() -> None:
    """Fewer than 10 NVML samples triggers nvml_low_sample_count warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=5,
    )
    assert any("nvml_low_sample_count" in w for w in warnings)


def test_warnings_clean_measurement() -> None:
    """All-good values produce 0 warnings."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=44.0,
        temp_end_c=45.0,
        nvml_sample_count=600,
        thermal_drift_threshold_c=10.0,
    )
    assert len(warnings) == 0


def test_warnings_all_four_triggered() -> None:
    """All four warning conditions triggered simultaneously."""
    warnings = collect_measurement_warnings(
        duration_sec=5.0,  # < 10s
        gpu_persistence_mode=False,  # off
        temp_start_c=30.0,  # 15C drift
        temp_end_c=45.0,
        nvml_sample_count=5,  # < 10 samples
        thermal_drift_threshold_c=10.0,
    )
    assert len(warnings) == 4
    warning_str = " ".join(warnings)
    assert "short_measurement_duration" in warning_str
    assert "gpu_persistence_mode_off" in warning_str
    assert "thermal_drift_detected" in warning_str
    assert "nvml_low_sample_count" in warning_str


def test_warnings_none_temps_no_drift_warning() -> None:
    """When temperatures are None, thermal_drift_detected is NOT triggered."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=None,
        temp_end_c=None,
        nvml_sample_count=100,
    )
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_warnings_custom_threshold() -> None:
    """Custom thermal_drift_threshold_c is applied correctly."""
    # 5C drift should only warn if threshold is 4C
    warnings_4 = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=40.0,
        temp_end_c=45.0,
        nvml_sample_count=100,
        thermal_drift_threshold_c=4.0,
    )
    warnings_6 = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=40.0,
        temp_end_c=45.0,
        nvml_sample_count=100,
        thermal_drift_threshold_c=6.0,
    )
    assert any("thermal_drift_detected" in w for w in warnings_4)
    assert not any("thermal_drift_detected" in w for w in warnings_6)


# =============================================================================
# PyTorchBackend integration wiring tests
# =============================================================================


def test_cuda_sync_called() -> None:
    """_cuda_sync() calls torch.cuda.synchronize() when CUDA is available."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        backend._cuda_sync()

    mock_torch.cuda.synchronize.assert_called_once()


def test_cuda_sync_skipped_when_cuda_unavailable() -> None:
    """_cuda_sync() skips synchronize() when torch.cuda.is_available() is False."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        backend._cuda_sync()

    mock_torch.cuda.synchronize.assert_not_called()


def test_run_warmup_returns_warmup_result() -> None:
    """_run_warmup() returns a WarmupResult (not None)."""
    from llenergymeasure.config.models import ExperimentConfig, WarmupConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend
    from llenergymeasure.domain.metrics import WarmupResult

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="test/model",
        warmup=WarmupConfig(enabled=False),
    )

    # Disabled warmup returns WarmupResult immediately
    result = backend._run_warmup(MagicMock(), MagicMock(), config, ["hello"])
    assert isinstance(result, WarmupResult)
    assert result.iterations_completed == 0
    assert result.converged is True


def test_run_warmup_disabled_path() -> None:
    """_run_warmup() with enabled=False returns WarmupResult with 0 iterations."""
    from llenergymeasure.config.models import ExperimentConfig, WarmupConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="test/model",
        warmup=WarmupConfig(enabled=False),
    )

    result = backend._run_warmup(MagicMock(), MagicMock(), config, ["hello"])
    assert result.iterations_completed == 0
    assert result.converged is True
    assert result.final_cv == 0.0


def test_build_result_uses_real_energy_values() -> None:
    """_build_result() populates total_energy_j from EnergyMeasurement, not 0.0."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend, _MeasurementData
    from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo, WarmupResult

    backend = PyTorchBackend()
    config = ExperimentConfig(model="test/model")
    data = _MeasurementData(
        total_tokens=100,
        total_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
    )
    energy_measurement = EnergyMeasurement(total_j=42.5, duration_sec=10.0)
    flops_result = FlopsResult(
        value=1e12, method="palm_formula", confidence="medium", precision="n/a"
    )
    warmup_result = WarmupResult(
        converged=True,
        final_cv=0.0,
        iterations_completed=5,
        target_cv=0.05,
        max_prompts=20,
    )
    now = datetime.now()
    result = backend._build_result(
        config=config,
        data=data,
        snapshot=None,
        start_time=now,
        end_time=now,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=flops_result,
        warmup_result=warmup_result,
        timeseries_path=None,
        measurement_warnings=[],
    )

    assert result.total_energy_j == 42.5
    assert result.total_flops == 1e12
    assert result.avg_energy_per_token_j == pytest.approx(42.5 / 50)
    assert result.measurement_warnings == []
    assert result.energy_breakdown is not None


def test_build_result_zero_energy_when_no_backend() -> None:
    """_build_result() returns total_energy_j=0.0 when energy_measurement is None."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend, _MeasurementData
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo, WarmupResult

    backend = PyTorchBackend()
    config = ExperimentConfig(model="test/model")
    data = _MeasurementData(total_tokens=100, total_time_sec=10.0)
    flops_result = FlopsResult(value=0.0, method="palm_formula", confidence="low", precision="n/a")
    warmup_result = WarmupResult(
        converged=True,
        final_cv=0.0,
        iterations_completed=0,
        target_cv=0.05,
        max_prompts=20,
    )
    now = datetime.now()

    result = backend._build_result(
        config=config,
        data=data,
        snapshot=None,
        start_time=now,
        end_time=now,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=None,
        baseline=None,
        flops_result=flops_result,
        warmup_result=warmup_result,
        timeseries_path=None,
        measurement_warnings=["short_measurement_duration: ..."],
    )

    assert result.total_energy_j == 0.0
    assert result.avg_energy_per_token_j == 0.0
    assert len(result.measurement_warnings) == 1


def test_measurement_data_tracks_input_output_tokens() -> None:
    """_MeasurementData separates input_tokens and output_tokens."""
    from llenergymeasure.core.backends.pytorch import _MeasurementData

    data = _MeasurementData()
    assert data.input_tokens == 0
    assert data.output_tokens == 0

    data.input_tokens += 50
    data.output_tokens += 30
    data.total_tokens += 80
    assert data.input_tokens == 50
    assert data.output_tokens == 30
    assert data.total_tokens == 80


# =============================================================================
# _build_result() field wiring tests (CM-16, RES-16, RES-06)
# =============================================================================


def _make_build_result_args():
    """Shared helper: return kwargs for PyTorchBackend._build_result() with minimal data."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import _MeasurementData
    from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo, WarmupResult

    config = ExperimentConfig(model="gpt2")
    data = _MeasurementData(
        total_tokens=100,
        total_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
    )
    energy_measurement = EnergyMeasurement(total_j=100.0, duration_sec=10.0)
    flops_result = FlopsResult(
        value=1e12, method="palm_formula", confidence="medium", precision="n/a"
    )
    warmup_result = WarmupResult(
        converged=True,
        final_cv=0.0,
        iterations_completed=3,
        target_cv=0.05,
        max_prompts=20,
    )
    now = datetime(2026, 1, 1, 12, 0, 0)
    return dict(
        config=config,
        data=data,
        snapshot=None,
        start_time=now,
        end_time=now,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=flops_result,
        warmup_result=warmup_result,
        timeseries_path=None,
        measurement_warnings=[],
    )


def test_build_result_populates_timeseries_field() -> None:
    """_build_result() with timeseries_path='timeseries.parquet' sets result.timeseries (CM-16)."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    kwargs = _make_build_result_args()
    kwargs["timeseries_path"] = "timeseries.parquet"

    result = backend._build_result(**kwargs)

    assert result.timeseries == "timeseries.parquet", (
        "timeseries field should be populated from timeseries_path argument"
    )


def test_build_result_populates_effective_config() -> None:
    """_build_result() populates result.effective_config with the model name (RES-16)."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    kwargs = _make_build_result_args()

    result = backend._build_result(**kwargs)

    assert result.effective_config != {}, "effective_config should not be empty"
    assert result.effective_config.get("model") == "gpt2", (
        "effective_config['model'] should be 'gpt2' (the configured model name)"
    )


def test_build_result_propagates_baseline_fields() -> None:
    """_build_result() with a baseline populates baseline_power_w and energy_adjusted_j (RES-06)."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend
    from llenergymeasure.core.baseline import BaselineCache

    backend = PyTorchBackend()
    kwargs = _make_build_result_args()
    kwargs["baseline"] = BaselineCache(
        power_w=30.0,
        timestamp=0.0,
        device_index=0,
        sample_count=300,
        duration_sec=30.0,
    )

    result = backend._build_result(**kwargs)

    assert result.baseline_power_w == pytest.approx(30.0), (
        "baseline_power_w should be populated from EnergyBreakdown.baseline_power_w"
    )
    assert result.energy_adjusted_j is not None, (
        "energy_adjusted_j should be populated when baseline is provided"
    )
