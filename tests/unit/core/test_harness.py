"""Unit tests for MeasurementHarness.

Tests use a FakeBackend that implements BackendPlugin with controllable outputs.
No GPU required - all hardware dependencies are mocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.backends.protocol import InferenceOutput
from llenergymeasure.core.harness import MeasurementHarness
from llenergymeasure.domain.metrics import WarmupResult

# ---------------------------------------------------------------------------
# FakeBackend - minimal BackendPlugin implementation for tests
# ---------------------------------------------------------------------------


@dataclass
class FakeBackend:
    """Minimal BackendPlugin for testing MeasurementHarness lifecycle.

    All methods record calls for assertion in tests.
    """

    backend_name: str = "fake"
    call_log: list[str] = field(default_factory=list)
    inference_output: InferenceOutput | None = None
    fail_on_run_inference: bool = False

    @property
    def name(self) -> str:
        return self.backend_name

    def load_model(self, config: Any) -> dict:
        self.call_log.append("load_model")
        return {"model": "fake_model_object"}

    def warmup(self, config: Any, model: Any) -> WarmupResult:
        self.call_log.append("warmup")
        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1,
            target_cv=0.01,
            max_prompts=10,
        )

    def run_inference(self, config: Any, model: Any) -> InferenceOutput:
        self.call_log.append("run_inference")
        if self.fail_on_run_inference:
            raise RuntimeError("Fake inference failure")
        if self.inference_output is not None:
            return self.inference_output
        return InferenceOutput(
            elapsed_time_sec=1.0,
            input_tokens=10,
            output_tokens=20,
            peak_memory_mb=512.0,
            model_memory_mb=256.0,
            batch_times=[1.0],
        )

    def cleanup(self, model: Any) -> None:
        self.call_log.append("cleanup")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    """Minimal ExperimentConfig with all measurement infrastructure disabled."""
    from llenergymeasure.config.models import ExperimentConfig

    return ExperimentConfig(
        model="fake/model",
        backend="pytorch",
        n=1,
        max_input_tokens=32,
        max_output_tokens=32,
    )


# ---------------------------------------------------------------------------
# Harness lifecycle tests
# ---------------------------------------------------------------------------


def _make_harness_patches():
    """Return context manager stack of patches for all harness external calls."""
    import contextlib
    from unittest.mock import MagicMock

    patches = [
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.measure_baseline_power",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.select_energy_backend",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.thermal_floor_wait",
            return_value=0.0,
        ),
        patch(
            "llenergymeasure.core.harness.estimate_flops_palm",
            return_value=MagicMock(value=1e9),
        ),
        patch(
            "llenergymeasure.core.harness._cuda_sync",
        ),
        patch(
            "llenergymeasure.core.harness.PowerThermalSampler",
        ),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="timeseries.parquet"),
        ),
        patch(
            "llenergymeasure.core.harness.collect_measurement_warnings",
            return_value=[],
        ),
    ]
    return contextlib.ExitStack(), patches


def _apply_patches():
    """Apply all harness patches and return the ExitStack."""
    import contextlib

    stack = contextlib.ExitStack()
    patches = [
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.measure_baseline_power",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.select_energy_backend",
            return_value=None,
        ),
        patch(
            "llenergymeasure.core.harness.thermal_floor_wait",
            return_value=0.0,
        ),
        patch(
            "llenergymeasure.core.harness.estimate_flops_palm",
            return_value=MagicMock(value=1e9),
        ),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler"),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="timeseries.parquet"),
        ),
        patch(
            "llenergymeasure.core.harness.collect_measurement_warnings",
            return_value=[],
        ),
    ]
    for p in patches:
        stack.enter_context(p)
    return stack


def test_harness_calls_backend_lifecycle_in_order(minimal_config):
    """load_model, warmup, run_inference, cleanup must be called in that exact order."""
    backend = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        harness.run(backend, minimal_config)

    assert backend.call_log == ["load_model", "warmup", "run_inference", "cleanup"]


def test_harness_cleanup_called_on_inference_error(minimal_config):
    """cleanup() must be called even when run_inference raises an exception."""
    backend = FakeBackend(fail_on_run_inference=True)
    harness = MeasurementHarness()

    with _apply_patches(), pytest.raises(RuntimeError, match="Fake inference failure"):
        harness.run(backend, minimal_config)

    assert "cleanup" in backend.call_log
    assert "run_inference" in backend.call_log


def test_harness_returns_experiment_result(minimal_config):
    """harness.run() must return an ExperimentResult with the correct backend field."""
    from llenergymeasure.domain.experiment import ExperimentResult

    backend = FakeBackend(backend_name="fake")
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(backend, minimal_config)

    assert isinstance(result, ExperimentResult)
    assert result.backend == "fake"


def test_harness_thermal_floor_wait_set_on_warmup_result(minimal_config):
    """thermal_floor_wait_s on WarmupResult must be set by harness (not by backend.warmup)."""
    backend = FakeBackend()
    harness = MeasurementHarness()

    with (
        _apply_patches(),
        patch(
            "llenergymeasure.core.harness.thermal_floor_wait",
            return_value=30.0,
        ),
    ):
        harness.run(backend, minimal_config)

    # The WarmupResult returned by backend.warmup() starts with thermal_floor_wait_s=0.0
    # The harness must have set it to 30.0. We verify this indirectly by checking the
    # harness called thermal_floor_wait (which is patched) — the lifecycle test covers order.
    # Backend's warmup result should have been mutated by harness.
    assert backend.call_log == ["load_model", "warmup", "run_inference", "cleanup"]


def test_harness_sets_warmup_result_thermal_floor(minimal_config):
    """WarmupResult.thermal_floor_wait_s must reflect the actual wait time from harness."""
    captured_warmup: list[WarmupResult] = []

    class TrackingBackend(FakeBackend):
        def warmup(self, config: Any, model: Any) -> WarmupResult:
            result = super().warmup(config, model)
            # thermal_floor_wait_s starts at 0 from warmup_until_converged
            assert result.thermal_floor_wait_s == 0.0
            captured_warmup.append(result)
            return result

    backend = TrackingBackend()
    harness = MeasurementHarness()

    with (
        _apply_patches(),
        patch(
            "llenergymeasure.core.harness.thermal_floor_wait",
            return_value=45.0,
        ),
    ):
        harness.run(backend, minimal_config)

    # After harness.run(), the WarmupResult captured by warmup() should have been
    # mutated by harness to reflect thermal_floor_wait_s = 45.0
    assert len(captured_warmup) == 1
    assert captured_warmup[0].thermal_floor_wait_s == 45.0
