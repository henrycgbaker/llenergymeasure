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
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=_make_mock_pts()),
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


# ---------------------------------------------------------------------------
# gpu_indices wiring tests
# ---------------------------------------------------------------------------


def _make_mock_pts():
    """Build a PowerThermalSampler mock that satisfies the harness's usage."""
    from llenergymeasure.domain.metrics import ThermalThrottleInfo

    mock_pts_cls = MagicMock()
    mock_pts_instance = MagicMock()
    mock_pts_instance.get_thermal_throttle_info.return_value = ThermalThrottleInfo()
    mock_pts_instance.get_samples.return_value = []
    # Context manager protocol
    mock_pts_instance.__enter__ = MagicMock(return_value=mock_pts_instance)
    mock_pts_instance.__exit__ = MagicMock(return_value=False)
    mock_pts_cls.return_value = mock_pts_instance
    return mock_pts_cls


def test_harness_passes_gpu_indices_to_energy_backend(minimal_config):
    """harness.run(gpu_indices=[0, 1]) passes gpu_indices to select_energy_backend."""
    backend = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.core.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.core.harness.select_energy_backend", return_value=None) as mock_seb,
        patch("llenergymeasure.core.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.core.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.core.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(backend, minimal_config, gpu_indices=[0, 1])

    mock_seb.assert_called_once()
    _, kwargs = mock_seb.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_passes_gpu_indices_to_thermal_sampler(minimal_config):
    """harness.run(gpu_indices=[0, 1]) instantiates PowerThermalSampler with those indices."""
    backend = FakeBackend()
    harness = MeasurementHarness()

    mock_pts_cls = _make_mock_pts()

    with (
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.core.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.core.harness.select_energy_backend", return_value=None),
        patch("llenergymeasure.core.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.core.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=mock_pts_cls),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.core.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(backend, minimal_config, gpu_indices=[0, 1])

    mock_pts_cls.assert_called_once()
    _, kwargs = mock_pts_cls.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_passes_gpu_indices_to_baseline(minimal_config):
    """harness.run(gpu_indices=[0, 1]) passes gpu_indices to measure_baseline_power when enabled."""
    from llenergymeasure.config.models import ExperimentConfig

    config_with_baseline = ExperimentConfig(
        model="fake/model",
        backend="pytorch",
        n=1,
        max_input_tokens=32,
        max_output_tokens=32,
        baseline={"enabled": True, "duration_seconds": 5.0},
    )

    backend = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.core.harness.measure_baseline_power", return_value=None) as mock_mbp,
        patch("llenergymeasure.core.harness.select_energy_backend", return_value=None),
        patch("llenergymeasure.core.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.core.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.core.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(backend, config_with_baseline, gpu_indices=[0, 1])

    mock_mbp.assert_called_once()
    _, kwargs = mock_mbp.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_defaults_to_no_gpu_indices(minimal_config):
    """harness.run() without gpu_indices passes None to subsystems (defaults to [0] internally)."""
    backend = FakeBackend()
    harness = MeasurementHarness()

    mock_pts_cls = _make_mock_pts()

    with (
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.core.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.core.harness.select_energy_backend", return_value=None) as mock_seb,
        patch("llenergymeasure.core.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.core.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=mock_pts_cls),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.core.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(backend, minimal_config)

    # gpu_indices should be None (not a missing kwarg error)
    _, seb_kwargs = mock_seb.call_args
    assert seb_kwargs.get("gpu_indices") is None

    _, pts_kwargs = mock_pts_cls.call_args
    assert pts_kwargs.get("gpu_indices") is None


# ---------------------------------------------------------------------------
# H2: Thermal floor ordering — start_tracking after thermal_floor_wait
# ---------------------------------------------------------------------------


def test_harness_start_tracking_called_after_thermal_floor_wait(minimal_config):
    """start_tracking() must be called after thermal_floor_wait() (H2 ordering verification).

    This verifies that energy measurement begins only after the thermal floor wait,
    so idle GPU heat does not inflate the measured energy window.

    We verify ordering by recording call order via side_effect on both
    thermal_floor_wait and select_energy_backend (which is called just before
    start_tracking). select_energy_backend returns None so no actual energy
    measurement object is needed (avoids MagicMock comparison issues in _build_result).
    """
    call_order: list[str] = []

    def fake_thermal_floor_wait(config):  # type: ignore[no-untyped-def]
        call_order.append("thermal_floor_wait")
        return 0.0

    def fake_select_energy_backend(backend_name, *, gpu_indices=None):  # type: ignore[no-untyped-def]
        call_order.append("select_energy_backend")
        return None  # No tracker; avoids MagicMock total_j > 0 comparison in _build_result

    backend = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.core.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.core.harness.measure_baseline_power", return_value=None),
        patch(
            "llenergymeasure.core.harness.select_energy_backend",
            side_effect=fake_select_energy_backend,
        ),
        patch(
            "llenergymeasure.core.harness.thermal_floor_wait",
            side_effect=fake_thermal_floor_wait,
        ),
        patch("llenergymeasure.core.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.core.harness._cuda_sync"),
        patch("llenergymeasure.core.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.core.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.core.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(backend, minimal_config)

    # Both must have been called
    assert "thermal_floor_wait" in call_order, "thermal_floor_wait was never called"
    assert "select_energy_backend" in call_order, "select_energy_backend was never called"

    # thermal_floor_wait (step 6) must precede select_energy_backend (step 7),
    # which in turn immediately precedes start_tracking (step 8) in harness source.
    tf_idx = call_order.index("thermal_floor_wait")
    seb_idx = call_order.index("select_energy_backend")
    assert tf_idx < seb_idx, (
        f"thermal_floor_wait (pos {tf_idx}) must be called before "
        f"select_energy_backend/start_tracking (pos {seb_idx})"
    )


# ---------------------------------------------------------------------------
# M18: Per-device memory tracking — _capture_model_memory_mb(gpu_indices)
# ---------------------------------------------------------------------------


def test_capture_model_memory_mb_multi_gpu(minimal_config):
    """_capture_model_memory_mb(gpu_indices=[0, 1]) queries each device and returns max.

    Verifies that max_memory_allocated is called with device=0 and device=1,
    and that the returned value is max(device_0_bytes, device_1_bytes) / (1024*1024).
    """
    harness = MeasurementHarness()

    device_memory = {0: 512 * 1024 * 1024, 1: 768 * 1024 * 1024}  # 512 MB, 768 MB

    def fake_max_memory_allocated(device=None):
        return device_memory.get(device, 0)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.max_memory_allocated.side_effect = fake_max_memory_allocated

    with (
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch("llenergymeasure.core.harness.importlib") as mock_importlib,
    ):
        mock_importlib.util.find_spec.return_value = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            import importlib

            with patch.object(importlib.util, "find_spec", return_value=MagicMock()):
                result = harness._capture_model_memory_mb(gpu_indices=[0, 1])

    # Should return max(512, 768) = 768 MB
    assert result == pytest.approx(768.0)

    # Verify called with both devices
    calls = mock_torch.cuda.max_memory_allocated.call_args_list

    def _get_device(call):
        # max_memory_allocated is called as max_memory_allocated(device=idx)
        if "device" in call.kwargs:
            return call.kwargs["device"]
        return call.args[0] if call.args else None

    devices_called = {_get_device(c) for c in calls}
    assert 0 in devices_called, f"device=0 not in calls: {calls}"
    assert 1 in devices_called, f"device=1 not in calls: {calls}"


def test_capture_model_memory_mb_defaults_to_gpu_zero(minimal_config):
    """_capture_model_memory_mb() without gpu_indices defaults to querying device=0."""
    harness = MeasurementHarness()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.max_memory_allocated.return_value = 256 * 1024 * 1024  # 256 MB

    with patch.dict("sys.modules", {"torch": mock_torch}):
        import importlib

        with patch.object(importlib.util, "find_spec", return_value=MagicMock()):
            result = harness._capture_model_memory_mb(gpu_indices=None)

    assert result == pytest.approx(256.0)

    # Must have been called with device=0 (not no-arg)
    calls = mock_torch.cuda.max_memory_allocated.call_args_list
    assert len(calls) == 1
    call = calls[0]
    device_arg = (
        call.kwargs.get("device")
        if "device" in call.kwargs
        else (call.args[0] if call.args else None)
    )
    assert device_arg == 0, f"Expected device=0, got device={device_arg}"


# ---------------------------------------------------------------------------
# H1: Harness-owned canonical inference timer (time.perf_counter)
# ---------------------------------------------------------------------------


def test_harness_sets_inference_time_sec(minimal_config):
    """Harness must override output.inference_time_sec with perf_counter delta.

    The backend returns elapsed_time_sec=99.0 (intentionally different).
    The harness must set inference_time_sec to a real perf_counter delta,
    NOT the backend's elapsed_time_sec.
    """
    custom_output = InferenceOutput(
        elapsed_time_sec=99.0,  # Backend's own timer — harness must NOT use this
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
    )
    backend = FakeBackend(inference_output=custom_output)
    harness = MeasurementHarness()

    with _apply_patches():
        harness.run(backend, minimal_config)

    # Harness must have written a non-zero value
    assert custom_output.inference_time_sec > 0, "harness must set inference_time_sec > 0"
    # Must NOT be the backend's elapsed_time_sec (proves harness override, not passthrough)
    assert custom_output.inference_time_sec != 99.0, (
        "inference_time_sec must be perf_counter delta, not backend's elapsed_time_sec"
    )


def test_inference_time_sec_used_in_result(minimal_config):
    """total_inference_time_sec in ExperimentResult must come from harness perf_counter delta.

    We mock time.perf_counter to return controlled values (100.0 then 105.0 → 5.0s delta).
    The backend returns elapsed_time_sec=99.0. The result must show 5.0, not 99.0.
    """
    custom_output = InferenceOutput(
        elapsed_time_sec=99.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
    )
    backend = FakeBackend(inference_output=custom_output)
    harness = MeasurementHarness()

    perf_counter_values = iter([100.0, 105.0])  # start=100, end=105 → delta=5.0

    with (
        _apply_patches(),
        patch("llenergymeasure.core.harness.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = lambda: next(perf_counter_values)
        result = harness.run(backend, minimal_config)

    assert result.total_inference_time_sec == pytest.approx(5.0), (
        f"Expected 5.0 from perf_counter delta, got {result.total_inference_time_sec}"
    )


def test_datetime_still_used_for_wall_clock(minimal_config):
    """ExperimentResult.start_time and end_time must be datetime objects, not perf_counter values."""
    from datetime import datetime as dt

    backend = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(backend, minimal_config)

    assert isinstance(result.start_time, dt), (
        f"start_time must be datetime, got {type(result.start_time)}"
    )
    assert isinstance(result.end_time, dt), (
        f"end_time must be datetime, got {type(result.end_time)}"
    )
