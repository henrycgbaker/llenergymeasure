"""Unit tests for WarmupConfig v2 and related warmup functionality (no GPU required)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import WarmupConfig
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
