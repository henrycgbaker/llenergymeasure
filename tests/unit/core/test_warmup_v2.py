"""Unit tests for WarmupConfig v2 and related warmup functionality (no GPU required)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import EnergyConfig, ExperimentConfig, WarmupConfig
from llenergymeasure.core.warmup import warmup_until_converged
from llenergymeasure.domain.metrics import WarmupResult

# =============================================================================
# WarmupConfig defaults and constraints
# =============================================================================


def test_warmup_config_n_warmup_default_is_5() -> None:
    """CM-21: n_warmup default is 5 (not 3)."""
    assert WarmupConfig().n_warmup == 5


def test_warmup_config_thermal_floor_default_60() -> None:
    """CM-22: thermal_floor_seconds defaults to 60."""
    assert WarmupConfig().thermal_floor_seconds == 60.0


def test_warmup_config_thermal_floor_minimum_30() -> None:
    """CM-22: thermal_floor_seconds must be >= 30."""
    with pytest.raises(ValidationError):
        WarmupConfig(thermal_floor_seconds=29.0)


def test_warmup_config_thermal_floor_30_ok() -> None:
    """CM-22: thermal_floor_seconds=30.0 is valid (at the boundary)."""
    cfg = WarmupConfig(thermal_floor_seconds=30.0)
    assert cfg.thermal_floor_seconds == 30.0


def test_warmup_config_thermal_floor_below_minimum_edge() -> None:
    """Values just below 30s are rejected."""
    with pytest.raises(ValidationError):
        WarmupConfig(thermal_floor_seconds=29.9)


def test_warmup_config_convergence_detection_default_false() -> None:
    """CM-23: convergence_detection defaults to False (opt-in)."""
    assert WarmupConfig().convergence_detection is False


def test_warmup_config_cv_threshold_default() -> None:
    """CM-23: cv_threshold defaults to 0.05."""
    assert WarmupConfig().cv_threshold == 0.05


def test_warmup_config_cv_threshold_bounds() -> None:
    """CM-23: cv_threshold is bounded [0.01, 0.5]."""
    with pytest.raises(ValidationError):
        WarmupConfig(cv_threshold=0.009)
    with pytest.raises(ValidationError):
        WarmupConfig(cv_threshold=0.51)
    # Boundary values valid
    WarmupConfig(cv_threshold=0.01)
    WarmupConfig(cv_threshold=0.5)


def test_warmup_config_extra_forbid() -> None:
    """extra=forbid: unknown fields raise ValidationError."""
    with pytest.raises(ValidationError):
        WarmupConfig(unknown_field=1)  # type: ignore[call-arg]


def test_warmup_config_enabled_default_true() -> None:
    """enabled defaults to True."""
    assert WarmupConfig().enabled is True


def test_warmup_config_window_size_default() -> None:
    """window_size defaults to 5."""
    assert WarmupConfig().window_size == 5


def test_warmup_config_min_prompts_default() -> None:
    """min_prompts defaults to 5."""
    assert WarmupConfig().min_prompts == 5


def test_warmup_config_max_prompts_default() -> None:
    """max_prompts (CV safety cap) defaults to 20."""
    assert WarmupConfig().max_prompts == 20


# =============================================================================
# warmup_until_converged — fixed mode
# =============================================================================


def test_warmup_fixed_mode_returns_converged() -> None:
    """Fixed mode (convergence_detection=False): converged=True, iterations_completed=n_warmup."""
    call_count = 0

    def mock_inference() -> float:
        nonlocal call_count
        call_count += 1
        return 100.0  # 100ms constant latency

    config = WarmupConfig(n_warmup=3, thermal_floor_seconds=30.0)
    result = warmup_until_converged(mock_inference, config, show_progress=False)

    assert result.converged is True
    assert result.iterations_completed == 3
    assert call_count == 3


def test_warmup_fixed_mode_respects_n_warmup() -> None:
    """Fixed mode runs exactly n_warmup iterations (not max_prompts)."""
    call_count = 0

    def mock_inference() -> float:
        nonlocal call_count
        call_count += 1
        return 50.0

    config = WarmupConfig(n_warmup=7, max_prompts=20, thermal_floor_seconds=30.0)
    result = warmup_until_converged(mock_inference, config, show_progress=False)

    assert result.iterations_completed == 7
    assert call_count == 7


# =============================================================================
# warmup_until_converged — CV mode
# =============================================================================


def test_warmup_cv_mode_converges() -> None:
    """CV mode: stable latencies should converge within max_prompts."""
    config = WarmupConfig(
        n_warmup=1,
        convergence_detection=True,
        cv_threshold=0.1,
        max_prompts=30,
        window_size=5,
        min_prompts=5,
        thermal_floor_seconds=30.0,
    )

    def stable_inference() -> float:
        return 10.0  # constant latency -> CV=0.0

    result = warmup_until_converged(stable_inference, config, show_progress=False)

    assert result.converged is True
    assert result.iterations_completed <= config.max_prompts


def test_warmup_disabled_skips() -> None:
    """When enabled=False, warmup returns immediately with converged=True."""
    call_count = 0

    def mock_inference() -> float:
        nonlocal call_count
        call_count += 1
        return 100.0

    config = WarmupConfig(enabled=False, thermal_floor_seconds=30.0)
    result = warmup_until_converged(mock_inference, config, show_progress=False)

    assert result.converged is True
    assert result.iterations_completed == 0
    assert call_count == 0


# =============================================================================
# WarmupResult — thermal_floor_wait_s field
# =============================================================================


def test_warmup_result_thermal_floor_wait_default() -> None:
    """thermal_floor_wait_s defaults to 0.0 (set by caller, not warmup fn)."""
    r = WarmupResult(
        converged=True,
        final_cv=0.02,
        iterations_completed=5,
        target_cv=0.05,
        max_prompts=20,
    )
    assert r.thermal_floor_wait_s == 0.0


def test_warmup_result_thermal_floor_wait_settable() -> None:
    """thermal_floor_wait_s can be set by the caller after sleep."""
    r = WarmupResult(
        converged=True,
        final_cv=0.02,
        iterations_completed=5,
        target_cv=0.05,
        max_prompts=20,
        thermal_floor_wait_s=60.5,
    )
    assert r.thermal_floor_wait_s == 60.5


def test_warmup_result_thermal_floor_wait_non_negative() -> None:
    """thermal_floor_wait_s must be non-negative."""
    with pytest.raises(ValidationError):
        WarmupResult(
            converged=True,
            final_cv=0.02,
            iterations_completed=5,
            target_cv=0.05,
            max_prompts=20,
            thermal_floor_wait_s=-1.0,
        )


def test_warmup_until_converged_result_thermal_default() -> None:
    """warmup_until_converged() returns WarmupResult with thermal_floor_wait_s=0.0."""
    config = WarmupConfig(n_warmup=2, thermal_floor_seconds=30.0)
    result = warmup_until_converged(lambda: 10.0, config, show_progress=False)
    assert result.thermal_floor_wait_s == 0.0


# =============================================================================
# EnergyConfig
# =============================================================================


def test_energy_config_default() -> None:
    """EnergyConfig defaults to backend='auto'."""
    assert EnergyConfig().backend == "auto"


def test_energy_config_null() -> None:
    """EnergyConfig(backend=None) disables energy measurement."""
    cfg = EnergyConfig(backend=None)
    assert cfg.backend is None


def test_energy_config_valid_backends() -> None:
    """All backend literal values are accepted."""
    for backend in ("auto", "nvml", "zeus", "codecarbon"):
        cfg = EnergyConfig(backend=backend)
        assert cfg.backend == backend


def test_energy_config_invalid_backend() -> None:
    """Unknown backend values raise ValidationError."""
    with pytest.raises(ValidationError):
        EnergyConfig(backend="unknown_backend")  # type: ignore[arg-type]


def test_energy_config_extra_forbid() -> None:
    """EnergyConfig: extra fields raise ValidationError."""
    with pytest.raises(ValidationError):
        EnergyConfig(backend="auto", extra_field=1)  # type: ignore[call-arg]


# =============================================================================
# ExperimentConfig — energy field
# =============================================================================


def test_experiment_config_has_energy() -> None:
    """ExperimentConfig.energy defaults to EnergyConfig with backend='auto'."""
    cfg = ExperimentConfig(model="gpt2")
    assert cfg.energy.backend == "auto"


def test_experiment_config_energy_override() -> None:
    """ExperimentConfig allows overriding energy backend."""
    cfg = ExperimentConfig(model="gpt2", energy=EnergyConfig(backend="nvml"))
    assert cfg.energy.backend == "nvml"


def test_experiment_config_energy_disabled() -> None:
    """ExperimentConfig allows disabling energy measurement via null."""
    cfg = ExperimentConfig(model="gpt2", energy=EnergyConfig(backend=None))
    assert cfg.energy.backend is None


# =============================================================================
# ExperimentResult — measurement_warnings field
# =============================================================================


def test_experiment_result_has_measurement_warnings() -> None:
    """ExperimentResult.measurement_warnings exists and defaults to empty list."""
    from datetime import datetime

    from llenergymeasure.domain.experiment import AggregationMetadata, ExperimentResult

    result = ExperimentResult(
        experiment_id="test-001",
        measurement_config_hash="abc123def4567890",
        measurement_methodology="total",
        aggregation=AggregationMetadata(num_processes=1),
        total_tokens=100,
        total_energy_j=10.0,
        total_inference_time_sec=5.0,
        avg_tokens_per_second=20.0,
        avg_energy_per_token_j=0.1,
        total_flops=1e12,
        start_time=datetime(2026, 1, 1, 0, 0, 0),
        end_time=datetime(2026, 1, 1, 0, 0, 5),
    )
    assert result.measurement_warnings == []


def test_experiment_result_measurement_warnings_populated() -> None:
    """measurement_warnings can be set to a list of warning strings."""
    from datetime import datetime

    from llenergymeasure.domain.experiment import AggregationMetadata, ExperimentResult

    result = ExperimentResult(
        experiment_id="test-002",
        measurement_config_hash="abc123def4567890",
        measurement_methodology="total",
        aggregation=AggregationMetadata(num_processes=1),
        total_tokens=100,
        total_energy_j=10.0,
        total_inference_time_sec=5.0,
        avg_tokens_per_second=20.0,
        avg_energy_per_token_j=0.1,
        total_flops=1e12,
        start_time=datetime(2026, 1, 1, 0, 0, 0),
        end_time=datetime(2026, 1, 1, 0, 0, 5),
        measurement_warnings=["Short measurement duration (5s < 30s minimum)"],
    )
    assert len(result.measurement_warnings) == 1
    assert "Short measurement duration" in result.measurement_warnings[0]
