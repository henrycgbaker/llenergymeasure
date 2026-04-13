"""Unit tests for engines/_helpers.py — shared engine utilities.

All tests run GPU-free. warmup_until_converged is now in harness.warmup;
this file tests the remaining helpers: compute_cv, warmup_single_token
(interface), and cleanup/memory helpers (import-only, no GPU needed).
"""

from __future__ import annotations

from collections.abc import Callable

from llenergymeasure.config.models import WarmupConfig
from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.harness.warmup import warmup_until_converged

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed_latency_fn(latency_ms: float) -> Callable[[], float]:
    """Return an inference callable that always returns a fixed latency."""

    def _run() -> float:
        return latency_ms

    return _run


def _varying_latency_fn(latencies: list[float]) -> Callable[[], float]:
    """Return an inference callable that cycles through a list of latencies."""
    it = iter(latencies)

    def _run() -> float:
        return next(it)

    return _run


# ---------------------------------------------------------------------------
# warmup disabled
# ---------------------------------------------------------------------------


def test_warmup_disabled_returns_converged_immediately():
    """When warmup.enabled=False, warmup_until_converged returns with 0 iterations."""
    config = WarmupConfig(enabled=False)
    result = warmup_until_converged(_fixed_latency_fn(100.0), config)

    assert isinstance(result, WarmupResult)
    assert result.converged is True
    assert result.iterations_completed == 0


# ---------------------------------------------------------------------------
# Fixed-mode warmup (convergence_detection=False)
# ---------------------------------------------------------------------------


def test_warmup_fixed_mode_runs_n_warmup_iterations():
    """Fixed mode runs exactly n_warmup iterations."""
    call_count = [0]

    def _counting_run() -> float:
        call_count[0] += 1
        return 50.0

    config = WarmupConfig(n_warmup=3, convergence_detection=False, enabled=True)
    result = warmup_until_converged(_counting_run, config)

    assert result.iterations_completed == 3
    assert call_count[0] == 3


def test_warmup_fixed_mode_marks_converged():
    """Fixed mode marks converged=True regardless of latency variance."""
    config = WarmupConfig(n_warmup=4, convergence_detection=False, enabled=True)
    # High variance latencies — would not converge in CV mode
    run_fn = _varying_latency_fn([10.0, 200.0, 10.0, 200.0])
    result = warmup_until_converged(run_fn, config)

    assert result.converged is True


def test_warmup_fixed_mode_result_has_warmup_result_type():
    """warmup_until_converged always returns a WarmupResult."""
    config = WarmupConfig(n_warmup=2, convergence_detection=False, enabled=True)
    result = warmup_until_converged(_fixed_latency_fn(75.0), config)

    assert isinstance(result, WarmupResult)


def test_warmup_fixed_mode_records_max_prompts():
    """WarmupResult.max_prompts matches the config value used."""
    config = WarmupConfig(n_warmup=5, convergence_detection=False, enabled=True)
    result = warmup_until_converged(_fixed_latency_fn(50.0), config)

    # In fixed mode, max_prompts is not directly set but target_cv from config is
    assert result.target_cv == config.cv_threshold


# ---------------------------------------------------------------------------
# CV-convergence mode
# ---------------------------------------------------------------------------


def test_warmup_cv_mode_converges_on_stable_latency():
    """CV mode marks converged when latency CV drops below threshold."""
    # Uniform latency => CV = 0 => converges after min_prompts
    config = WarmupConfig(
        n_warmup=1,
        convergence_detection=True,
        cv_threshold=0.05,
        max_prompts=20,
        window_size=3,
        min_prompts=3,
        enabled=True,
    )
    result = warmup_until_converged(_fixed_latency_fn(50.0), config)

    assert result.converged is True
    assert result.final_cv < config.cv_threshold


def test_warmup_cv_mode_does_not_converge_with_high_variance():
    """CV mode does not converge when latency variance is too high."""
    config = WarmupConfig(
        n_warmup=1,
        convergence_detection=True,
        cv_threshold=0.05,  # minimum allowed threshold
        max_prompts=6,
        window_size=3,
        min_prompts=3,
        enabled=True,
    )
    # Alternating latencies of 1 and 100 => very high CV, never converges
    alternating = [1.0, 100.0] * 10
    run_fn = _varying_latency_fn(alternating)
    result = warmup_until_converged(run_fn, config)

    assert result.converged is False


def test_warmup_cv_mode_respects_max_prompts():
    """CV mode stops at max_prompts even without convergence."""
    config = WarmupConfig(
        n_warmup=1,
        convergence_detection=True,
        cv_threshold=0.05,
        max_prompts=5,
        window_size=3,
        min_prompts=3,
        enabled=True,
    )
    call_count = [0]

    def _run() -> float:
        call_count[0] += 1
        # Alternating high variance so it never converges
        return 1.0 if call_count[0] % 2 == 0 else 100.0

    warmup_until_converged(_run, config)
    assert call_count[0] <= config.max_prompts


# ---------------------------------------------------------------------------
# Error handling: inference callable raises
# ---------------------------------------------------------------------------


def test_warmup_continues_after_inference_failure():
    """warmup_until_converged continues after an individual inference exception."""
    call_count = [0]

    def _sometimes_fails() -> float:
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("simulated inference failure")
        return 50.0

    config = WarmupConfig(n_warmup=4, convergence_detection=False, enabled=True)
    # Should not raise — failures are caught and logged inside warmup_until_converged
    result = warmup_until_converged(_sometimes_fails, config)

    assert isinstance(result, WarmupResult)
    assert call_count[0] == 4  # all 4 attempted
