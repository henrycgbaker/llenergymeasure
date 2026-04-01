"""Unit tests for ExperimentResult v2.0 schema.

Tests cover: schema_version="2.0", all RES-01..RES-11 fields, JSON round-trip,
config hash utility, MultiGPUMetrics, frozen model enforcement.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import (
    ExperimentResult,
    compute_measurement_config_hash,
)
from llenergymeasure.domain.metrics import EnergyBreakdown, MultiGPUMetrics

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def make_result():
    """Factory for minimal valid ExperimentResult instances."""

    def _make(**kwargs):
        defaults = dict(
            experiment_id="test-001",
            measurement_config_hash="abcdef0123456789",
            measurement_methodology="total",
            total_tokens=1000,
            total_energy_j=50.0,
            total_inference_time_sec=10.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.05,
            total_flops=1e12,
            start_time=datetime(2026, 2, 26, 14, 0, 0),
            end_time=datetime(2026, 2, 26, 14, 0, 10),
        )
        defaults.update(kwargs)
        return ExperimentResult(**defaults)

    return _make


# ---------------------------------------------------------------------------
# Task 2.1: schema_version
# ---------------------------------------------------------------------------


def test_schema_version_default_is_2_0(make_result):
    """schema_version defaults to '2.0', not '2.0.0'."""
    result = make_result()
    assert result.schema_version == "2.0"


# ---------------------------------------------------------------------------
# Task 2.2: measurement_config_hash
# ---------------------------------------------------------------------------


def test_measurement_config_hash_field(make_result):
    """measurement_config_hash round-trips through construction."""
    result = make_result(measurement_config_hash="abcdef0123456789")
    assert result.measurement_config_hash == "abcdef0123456789"


# ---------------------------------------------------------------------------
# Tasks 2.3-2.5: measurement_methodology
# ---------------------------------------------------------------------------


def test_measurement_methodology_total(make_result):
    """measurement_methodology='total' validates."""
    result = make_result(measurement_methodology="total")
    assert result.measurement_methodology == "total"


def test_measurement_methodology_steady_state(make_result):
    """measurement_methodology='steady_state' validates."""
    result = make_result(measurement_methodology="steady_state")
    assert result.measurement_methodology == "steady_state"


def test_measurement_methodology_windowed(make_result):
    """measurement_methodology='windowed' validates."""
    result = make_result(measurement_methodology="windowed")
    assert result.measurement_methodology == "windowed"


def test_measurement_methodology_invalid(make_result):
    """measurement_methodology='invalid' raises ValidationError."""
    with pytest.raises(ValidationError):
        make_result(measurement_methodology="invalid")


# ---------------------------------------------------------------------------
# Tasks 2.6-2.7: steady_state_window
# ---------------------------------------------------------------------------


def test_steady_state_window_tuple(make_result):
    """steady_state_window=(12.3, 67.8) validates and round-trips."""
    result = make_result(steady_state_window=(12.3, 67.8))
    assert result.steady_state_window == (12.3, 67.8)


def test_steady_state_window_none(make_result):
    """steady_state_window=None validates when methodology=total."""
    result = make_result(measurement_methodology="total", steady_state_window=None)
    assert result.steady_state_window is None


# ---------------------------------------------------------------------------
# Tasks 2.8-2.10: energy detail fields
# ---------------------------------------------------------------------------


def test_baseline_power_w_optional(make_result):
    """baseline_power_w defaults to None, accepts float."""
    result_none = make_result()
    assert result_none.baseline_power_w is None

    result_set = make_result(baseline_power_w=42.5)
    assert result_set.baseline_power_w == 42.5


def test_energy_adjusted_j_optional(make_result):
    """energy_adjusted_j defaults to None, accepts float."""
    result_none = make_result()
    assert result_none.energy_adjusted_j is None

    result_set = make_result(energy_adjusted_j=45.2)
    assert result_set.energy_adjusted_j == 45.2


def test_energy_per_device_j_optional(make_result):
    """energy_per_device_j defaults to None, accepts list[float]."""
    result_none = make_result()
    assert result_none.energy_per_device_j is None

    result_set = make_result(energy_per_device_j=[25.0, 25.0])
    assert result_set.energy_per_device_j == [25.0, 25.0]


# ---------------------------------------------------------------------------
# Task 2.11: energy_breakdown
# ---------------------------------------------------------------------------


def test_energy_breakdown_optional(make_result):
    """Can embed EnergyBreakdown or leave as None."""
    result_none = make_result()
    assert result_none.energy_breakdown is None

    breakdown = EnergyBreakdown(raw_j=50.0, adjusted_j=45.0, baseline_power_w=0.5)
    result_set = make_result(energy_breakdown=breakdown)
    assert result_set.energy_breakdown is not None
    assert result_set.energy_breakdown.raw_j == 50.0


# ---------------------------------------------------------------------------
# Task 2.12: multi_gpu
# ---------------------------------------------------------------------------


def test_multi_gpu_optional(make_result):
    """Can embed MultiGPUMetrics or leave as None."""
    result_none = make_result()
    assert result_none.multi_gpu is None

    mgpu = MultiGPUMetrics(
        num_gpus=2,
        energy_per_gpu_j=[25.0, 25.0],
        energy_total_j=50.0,
        energy_per_output_token_j=0.05,
    )
    result_set = make_result(multi_gpu=mgpu)
    assert result_set.multi_gpu is not None
    assert result_set.multi_gpu.num_gpus == 2


# ---------------------------------------------------------------------------
# Task 2.13: reproducibility_notes
# ---------------------------------------------------------------------------


def test_reproducibility_notes_default(make_result):
    """Default reproducibility_notes contains 'NVML' and 'accuracy'."""
    result = make_result()
    notes = result.reproducibility_notes
    assert "NVML" in notes
    assert "accuracy" in notes.lower()


# ---------------------------------------------------------------------------
# Task 2.14: measurement_warnings
# ---------------------------------------------------------------------------


def test_measurement_warnings_default_empty(make_result):
    """measurement_warnings defaults to empty list."""
    result = make_result()
    assert result.measurement_warnings == []


def test_measurement_warnings_can_be_set(make_result):
    """measurement_warnings accepts a list of strings."""
    warnings = ["Short duration (8.2s < 60s recommended)."]
    result = make_result(measurement_warnings=warnings)
    assert result.measurement_warnings == warnings


# ---------------------------------------------------------------------------
# Task 2.15: warmup_excluded_samples
# ---------------------------------------------------------------------------


def test_warmup_excluded_samples_optional(make_result):
    """warmup_excluded_samples defaults to None, accepts int."""
    result_none = make_result()
    assert result_none.warmup_excluded_samples is None

    result_set = make_result(warmup_excluded_samples=5)
    assert result_set.warmup_excluded_samples == 5


# ---------------------------------------------------------------------------
# Task 2.16: timeseries
# ---------------------------------------------------------------------------


def test_timeseries_field(make_result):
    """timeseries field accepts filename string or None."""
    result_file = make_result(timeseries="timeseries.parquet")
    assert result_file.timeseries == "timeseries.parquet"

    result_none = make_result(timeseries=None)
    assert result_none.timeseries is None


# ---------------------------------------------------------------------------
# Task 2.17: environment_snapshot
# ---------------------------------------------------------------------------


def test_environment_snapshot_optional(make_result):
    """Can embed EnvironmentSnapshot or leave as None."""
    result_none = make_result()
    assert result_none.environment_snapshot is None


# ---------------------------------------------------------------------------
# Task 2.18: frozen model
# ---------------------------------------------------------------------------


def test_frozen_model(make_result):
    """Assignment after construction raises ValidationError (frozen=True)."""
    result = make_result()
    with pytest.raises((ValidationError, TypeError)):
        result.schema_version = "3.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_duration_sec_property(make_result):
    """duration_sec property computes correct duration."""
    result = make_result(
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 30),
    )
    assert result.duration_sec == 30.0


def test_tokens_per_joule_property(make_result):
    """tokens_per_joule property computes correct efficiency."""
    result = make_result(total_tokens=1000, total_energy_j=50.0)
    assert result.tokens_per_joule == pytest.approx(20.0)


def test_tokens_per_joule_zero_energy(make_result):
    """tokens_per_joule returns 0.0 when total_energy_j is zero."""
    result = make_result(total_energy_j=0.0)
    assert result.tokens_per_joule == 0.0


# ---------------------------------------------------------------------------
# Tasks 2.22-2.24: compute_measurement_config_hash
# ---------------------------------------------------------------------------


def test_compute_measurement_config_hash():
    """Known ExperimentConfig produces 16-char hex string."""
    cfg = ExperimentConfig(model="gpt2")
    h = compute_measurement_config_hash(cfg)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_deterministic():
    """Same config always produces same hash."""
    cfg = ExperimentConfig(model="gpt2")
    h1 = compute_measurement_config_hash(cfg)
    h2 = compute_measurement_config_hash(cfg)
    assert h1 == h2


def test_config_hash_different_configs():
    """Different configs produce different hashes."""
    cfg1 = ExperimentConfig(model="gpt2")
    cfg2 = ExperimentConfig(model="meta-llama/Llama-2-7b-hf")
    h1 = compute_measurement_config_hash(cfg1)
    h2 = compute_measurement_config_hash(cfg2)
    assert h1 != h2


# ---------------------------------------------------------------------------
# Task 2.25: MultiGPUMetrics model
# ---------------------------------------------------------------------------


def test_multi_gpu_metrics_model():
    """MultiGPUMetrics validates with all required fields."""
    m = MultiGPUMetrics(
        num_gpus=4,
        energy_per_gpu_j=[10.0, 11.0, 12.0, 13.0],
        energy_total_j=46.0,
        energy_per_output_token_j=0.046,
    )
    assert m.num_gpus == 4
    assert len(m.energy_per_gpu_j) == 4
    assert m.energy_total_j == 46.0
    assert m.energy_per_output_token_j == pytest.approx(0.046)


def test_multi_gpu_metrics_requires_fields():
    """MultiGPUMetrics raises ValidationError when required fields missing."""
    with pytest.raises(ValidationError):
        MultiGPUMetrics(num_gpus=2)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Task 2.26: JSON round-trip
# ---------------------------------------------------------------------------


def test_experiment_result_rejects_unknown_kwargs(make_result):
    """ExperimentResult raises ValidationError for unrecognised kwargs (extra='forbid')."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_result(timeseries_path="ts.parquet")


def test_json_round_trip(make_result):
    """ExperimentResult round-trips through JSON preserving all fields."""
    mgpu = MultiGPUMetrics(
        num_gpus=2,
        energy_per_gpu_j=[25.0, 25.0],
        energy_total_j=50.0,
        energy_per_output_token_j=0.05,
    )
    original = make_result(
        measurement_methodology="steady_state",
        steady_state_window=(10.0, 60.0),
        baseline_power_w=42.5,
        energy_adjusted_j=45.2,
        energy_per_device_j=[25.0, 25.0],
        multi_gpu=mgpu,
        timeseries="timeseries.parquet",
        measurement_warnings=["Short duration detected."],
        warmup_excluded_samples=5,
    )
    json_str = original.model_dump_json()
    restored = ExperimentResult.model_validate_json(json_str)

    assert restored.schema_version == original.schema_version
    assert restored.experiment_id == original.experiment_id
    assert restored.measurement_methodology == original.measurement_methodology
    assert restored.steady_state_window == original.steady_state_window
    assert restored.baseline_power_w == original.baseline_power_w
    assert restored.energy_adjusted_j == original.energy_adjusted_j
    assert restored.energy_per_device_j == original.energy_per_device_j
    assert restored.multi_gpu is not None
    assert restored.multi_gpu.num_gpus == 2
    assert restored.start_time == original.start_time
    assert restored.end_time == original.end_time
    assert restored.timeseries == "timeseries.parquet"
    assert restored.measurement_warnings == ["Short duration detected."]
    assert restored.warmup_excluded_samples == 5


# ---------------------------------------------------------------------------
# experiment_name field on ExperimentConfig
# ---------------------------------------------------------------------------


def test_experiment_name_field_exists():
    """ExperimentConfig has experiment_name defaulting to None."""
    config = ExperimentConfig(model="gpt2")
    assert hasattr(config, "experiment_name")
    assert config.experiment_name is None


def test_experiment_name_can_be_set():
    """experiment_name can be set to a string value."""
    config = ExperimentConfig(model="gpt2", experiment_name="my-run")
    assert config.experiment_name == "my-run"


# ---------------------------------------------------------------------------
# DRY n_prompts default
# ---------------------------------------------------------------------------


def test_n_prompts_default_matches_dataset_config():
    """_N_PROMPTS_DEFAULT in api/_impl.py equals DatasetConfig field default."""
    from llenergymeasure.api._impl import _N_PROMPTS_DEFAULT
    from llenergymeasure.config.models import DatasetConfig

    assert DatasetConfig.model_fields["n_prompts"].default == _N_PROMPTS_DEFAULT
