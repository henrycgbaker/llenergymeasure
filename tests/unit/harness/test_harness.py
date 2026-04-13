"""Unit tests for MeasurementHarness.

Tests use a FakeBackend that implements EnginePlugin with controllable outputs.
No GPU required - all hardware dependencies are mocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.harness import MeasurementHarness

# ---------------------------------------------------------------------------
# FakeBackend - minimal EnginePlugin implementation for tests
# ---------------------------------------------------------------------------


@dataclass
class FakeBackend:
    """Minimal EnginePlugin for testing MeasurementHarness lifecycle.

    All methods record calls for assertion in tests.
    """

    engine_name: str = "fake"
    call_log: list[str] = field(default_factory=list)
    inference_output: InferenceOutput | None = None
    fail_on_run_inference: bool = False

    @property
    def name(self) -> str:
        return self.engine_name

    def load_model(self, config: Any, **kwargs: Any) -> dict:
        self.call_log.append("load_model")
        return {"model": "fake_model_object"}

    def warmup(self, config: Any, model: Any, prompts: list[str]) -> WarmupResult:
        self.call_log.append("warmup")
        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1,
            target_cv=0.01,
            max_prompts=10,
        )

    def run_warmup_prompt(self, config: Any, model: Any, prompt: str) -> float:
        self.call_log.append("run_warmup_prompt")
        # Return 0.0 = simple kernel warmup (no CV convergence loop needed in tests)
        return 0.0

    def run_inference(self, config: Any, model: Any, prompts: list[str]) -> InferenceOutput:
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
    from llenergymeasure.config.models import DatasetConfig, ExperimentConfig

    return ExperimentConfig(
        model="fake/model",
        engine="transformers",
        dataset=DatasetConfig(n_prompts=1),
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
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch(
            "llenergymeasure.harness.measure_baseline_power",
            return_value=None,
        ),
        patch(
            "llenergymeasure.harness.load_prompts",
            return_value=["test prompt"],
        ),
        patch(
            "llenergymeasure.harness.select_energy_sampler",
            return_value=None,
        ),
        patch(
            "llenergymeasure.harness.thermal_floor_wait",
            return_value=0.0,
        ),
        patch(
            "llenergymeasure.harness.estimate_flops_palm",
            return_value=MagicMock(value=1e9),
        ),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="timeseries.parquet"),
        ),
        patch(
            "llenergymeasure.harness.collect_measurement_warnings",
            return_value=[],
        ),
    ]
    for p in patches:
        stack.enter_context(p)
    return stack


def test_harness_calls_engine_lifecycle_in_order(minimal_config):
    """load_model, run_warmup_prompt, run_inference, cleanup must be called in that order.

    The harness now calls run_warmup_prompt() directly (not warmup()). The warmup
    convergence loop is owned by the harness via warmup_until_converged().
    """
    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        harness.run(engine, minimal_config)

    assert "load_model" in engine.call_log
    assert "run_warmup_prompt" in engine.call_log
    assert "run_inference" in engine.call_log
    assert "cleanup" in engine.call_log
    # Order: load_model before run_warmup_prompt, run_warmup_prompt before run_inference
    lm_idx = engine.call_log.index("load_model")
    wp_idx = engine.call_log.index("run_warmup_prompt")
    ri_idx = engine.call_log.index("run_inference")
    assert lm_idx < wp_idx < ri_idx


def test_harness_cleanup_called_on_inference_error(minimal_config):
    """cleanup() must be called even when run_inference raises an exception."""
    engine = FakeBackend(fail_on_run_inference=True)
    harness = MeasurementHarness()

    with _apply_patches(), pytest.raises(RuntimeError, match="Fake inference failure"):
        harness.run(engine, minimal_config)

    assert "cleanup" in engine.call_log
    assert "run_inference" in engine.call_log


def test_harness_returns_experiment_result(minimal_config):
    """harness.run() must return an ExperimentResult with the correct engine field."""
    from llenergymeasure.domain.experiment import ExperimentResult

    engine = FakeBackend(engine_name="fake")
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert isinstance(result, ExperimentResult)
    assert result.engine == "fake"


def test_harness_sets_llenergymeasure_version(minimal_config):
    """harness.run() populates llenergymeasure_version from _version.__version__."""
    from llenergymeasure._version import __version__

    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert result.llenergymeasure_version == __version__


def test_harness_sets_engine_version_from_engine(minimal_config):
    """harness.run() populates engine_version via getattr(engine, 'version')."""
    engine = FakeBackend()
    engine.version = "1.2.3"  # type: ignore[attr-defined]
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert result.engine_version == "1.2.3"


def test_harness_engine_version_none_when_missing(minimal_config):
    """engine_version is None when engine has no version property."""
    engine = FakeBackend()
    # FakeBackend has no .version attribute by default
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert result.engine_version is None


def test_harness_thermal_floor_wait_set_on_warmup_result(minimal_config):
    """thermal_floor_wait_s on WarmupResult must be set by harness (not by engine).

    The harness creates WarmupResult and sets thermal_floor_wait_s from thermal_floor_wait().
    """
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        _apply_patches(),
        patch(
            "llenergymeasure.harness.thermal_floor_wait",
            return_value=30.0,
        ),
    ):
        result = harness.run(engine, minimal_config)

    # thermal_floor_wait_s must be set in the result's warmup_result
    assert result.warmup_result is not None
    assert result.warmup_result.thermal_floor_wait_s == pytest.approx(30.0)


def test_harness_sets_warmup_result_thermal_floor(minimal_config):
    """WarmupResult.thermal_floor_wait_s must reflect the actual wait time from harness.

    The harness creates WarmupResult after calling run_warmup_prompt(), then sets
    thermal_floor_wait_s from the return value of thermal_floor_wait().
    """
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        _apply_patches(),
        patch(
            "llenergymeasure.harness.thermal_floor_wait",
            return_value=45.0,
        ),
    ):
        result = harness.run(engine, minimal_config)

    # The harness must set thermal_floor_wait_s on the WarmupResult
    assert result.warmup_result is not None
    assert result.warmup_result.thermal_floor_wait_s == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# gpu_indices wiring tests
# ---------------------------------------------------------------------------


def _make_mock_pts(*, samples: list | None = None):
    """Build a PowerThermalSampler mock that satisfies the harness's usage."""
    from llenergymeasure.domain.metrics import ThermalThrottleInfo

    mock_pts_cls = MagicMock()
    mock_pts_instance = MagicMock()
    mock_pts_instance.get_thermal_throttle_info.return_value = ThermalThrottleInfo()
    mock_pts_instance.get_samples.return_value = samples if samples is not None else []
    # Context manager protocol
    mock_pts_instance.__enter__ = MagicMock(return_value=mock_pts_instance)
    mock_pts_instance.__exit__ = MagicMock(return_value=False)
    mock_pts_cls.return_value = mock_pts_instance
    return mock_pts_cls


def test_harness_passes_gpu_indices_to_energy_sampler(minimal_config):
    """harness.run(gpu_indices=[0, 1]) passes gpu_indices to select_energy_sampler."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None) as mock_seb,
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, minimal_config, gpu_indices=[0, 1])

    mock_seb.assert_called_once()
    _, kwargs = mock_seb.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_passes_gpu_indices_to_thermal_sampler(minimal_config):
    """harness.run(gpu_indices=[0, 1]) instantiates PowerThermalSampler with those indices."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    mock_pts_cls = _make_mock_pts()

    with (
        patch(
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=mock_pts_cls),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, minimal_config, gpu_indices=[0, 1])

    mock_pts_cls.assert_called_once()
    _, kwargs = mock_pts_cls.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_passes_gpu_indices_to_baseline(minimal_config):
    """harness.run(gpu_indices=[0, 1]) passes gpu_indices to measure_baseline_power when enabled."""
    from llenergymeasure.config.models import DatasetConfig, ExperimentConfig

    config_with_baseline = ExperimentConfig(
        model="fake/model",
        engine="transformers",
        dataset=DatasetConfig(n_prompts=1),
        max_input_tokens=32,
        max_output_tokens=32,
        baseline={"enabled": True, "duration_seconds": 5.0},
    )

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None) as mock_mbp,
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, config_with_baseline, gpu_indices=[0, 1])

    mock_mbp.assert_called_once()
    _, kwargs = mock_mbp.call_args
    assert kwargs.get("gpu_indices") == [0, 1]


def test_harness_defaults_to_no_gpu_indices(minimal_config):
    """harness.run() without gpu_indices passes None to subsystems (defaults to [0] internally)."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    mock_pts_cls = _make_mock_pts()

    with (
        patch(
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None) as mock_seb,
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=mock_pts_cls),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, minimal_config)

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
    thermal_floor_wait and select_energy_sampler (which is called just before
    start_tracking). select_energy_sampler returns None so no actual energy
    measurement object is needed (avoids MagicMock comparison issues in _build_result).
    """
    call_order: list[str] = []

    def fake_thermal_floor_wait(config):  # type: ignore[no-untyped-def]
        call_order.append("thermal_floor_wait")
        return 0.0

    def fake_select_energy_sampler(engine_name, *, gpu_indices=None):  # type: ignore[no-untyped-def]
        call_order.append("select_energy_sampler")
        return None  # No tracker; avoids MagicMock total_j > 0 comparison in _build_result

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch(
            "llenergymeasure.harness.collect_environment_snapshot",
            return_value=None,
        ),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch(
            "llenergymeasure.harness.select_energy_sampler",
            side_effect=fake_select_energy_sampler,
        ),
        patch(
            "llenergymeasure.harness.thermal_floor_wait",
            side_effect=fake_thermal_floor_wait,
        ),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, minimal_config)

    # Both must have been called
    assert "thermal_floor_wait" in call_order, "thermal_floor_wait was never called"
    assert "select_energy_sampler" in call_order, "select_energy_sampler was never called"

    # thermal_floor_wait (step 6) must precede select_energy_sampler (step 7),
    # which in turn immediately precedes start_tracking (step 8) in harness source.
    tf_idx = call_order.index("thermal_floor_wait")
    seb_idx = call_order.index("select_energy_sampler")
    assert tf_idx < seb_idx, (
        f"thermal_floor_wait (pos {tf_idx}) must be called before "
        f"select_energy_sampler/start_tracking (pos {seb_idx})"
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
        patch("llenergymeasure.harness.importlib") as mock_importlib,
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

    The engine returns elapsed_time_sec=99.0 (intentionally different).
    The harness must set inference_time_sec to a real perf_counter delta,
    NOT the engine's elapsed_time_sec.
    """
    custom_output = InferenceOutput(
        elapsed_time_sec=99.0,  # Backend's own timer — harness must NOT use this
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
    )
    engine = FakeBackend(inference_output=custom_output)
    harness = MeasurementHarness()

    with _apply_patches():
        harness.run(engine, minimal_config)

    # Harness must have written a non-zero value
    assert custom_output.inference_time_sec > 0, "harness must set inference_time_sec > 0"
    # Must NOT be the engine's elapsed_time_sec (proves harness override, not passthrough)
    assert custom_output.inference_time_sec != 99.0, (
        "inference_time_sec must be perf_counter delta, not engine's elapsed_time_sec"
    )


def test_inference_time_sec_used_in_result(minimal_config):
    """total_inference_time_sec in ExperimentResult must come from harness perf_counter delta.

    We mock time.perf_counter to return controlled values (100.0 then 105.0 → 5.0s delta).
    The engine returns elapsed_time_sec=99.0. The result must show 5.0, not 99.0.
    """
    custom_output = InferenceOutput(
        elapsed_time_sec=99.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
    )
    engine = FakeBackend(inference_output=custom_output)
    harness = MeasurementHarness()

    perf_counter_values = iter([100.0, 105.0])  # start=100, end=105 → delta=5.0

    with (
        _apply_patches(),
        patch("llenergymeasure.harness.time") as mock_time,
    ):
        mock_time.perf_counter.side_effect = lambda: next(perf_counter_values)
        result = harness.run(engine, minimal_config)

    assert result.total_inference_time_sec == pytest.approx(5.0), (
        f"Expected 5.0 from perf_counter delta, got {result.total_inference_time_sec}"
    )


def test_datetime_still_used_for_wall_clock(minimal_config):
    """ExperimentResult.start_time and end_time must be datetime objects, not perf_counter values."""
    from datetime import datetime as dt

    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert isinstance(result.start_time, dt), (
        f"start_time must be datetime, got {type(result.start_time)}"
    )
    assert isinstance(result.end_time, dt), (
        f"end_time must be datetime, got {type(result.end_time)}"
    )


# ---------------------------------------------------------------------------
# Dropped field wiring tests — warmup_result, energy_per_device_j, multi_gpu
# ---------------------------------------------------------------------------


def test_harness_warmup_result_wired_to_experiment_result(minimal_config):
    """WarmupResult must reach ExperimentResult.warmup_result with thermal_floor_wait_s set.

    The harness owns warmup result creation (via run_warmup_prompt + warmup_until_converged)
    and sets thermal_floor_wait_s from thermal_floor_wait().
    """
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        _apply_patches(),
        patch(
            "llenergymeasure.harness.thermal_floor_wait",
            return_value=15.0,
        ),
    ):
        result = harness.run(engine, minimal_config)

    assert result.warmup_result is not None
    assert result.warmup_result.converged is True
    assert result.warmup_result.thermal_floor_wait_s == pytest.approx(15.0)


def test_harness_per_gpu_j_wired_to_energy_per_device_j(minimal_config):
    """per_gpu_j from EnergyMeasurement must reach ExperimentResult.energy_per_device_j and multi_gpu."""
    from llenergymeasure.energy.nvml import EnergyMeasurement

    fake_energy_measurement = EnergyMeasurement(
        total_j=150.0,
        duration_sec=10.0,
        per_gpu_j={0: 70.0, 1: 80.0},
    )

    mock_energy_sampler = MagicMock()
    mock_energy_sampler.start_tracking.return_value = MagicMock()
    mock_energy_sampler.stop_tracking.return_value = fake_energy_measurement

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch(
            "llenergymeasure.harness.select_energy_sampler",
            return_value=mock_energy_sampler,
        ),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        result = harness.run(engine, minimal_config)

    assert result.energy_per_device_j == [70.0, 80.0]
    assert result.multi_gpu is not None
    assert result.multi_gpu.num_gpus == 2
    assert result.multi_gpu.energy_per_gpu_j == [70.0, 80.0]
    assert result.multi_gpu.energy_total_j == pytest.approx(150.0)


def test_harness_single_gpu_no_multi_gpu_metrics(minimal_config):
    """Single-GPU per_gpu_j must populate energy_per_device_j but leave multi_gpu as None."""
    from llenergymeasure.energy.nvml import EnergyMeasurement

    fake_energy_measurement = EnergyMeasurement(
        total_j=100.0,
        duration_sec=8.0,
        per_gpu_j={0: 100.0},
    )

    mock_energy_sampler = MagicMock()
    mock_energy_sampler.start_tracking.return_value = MagicMock()
    mock_energy_sampler.stop_tracking.return_value = fake_energy_measurement

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch(
            "llenergymeasure.harness.select_energy_sampler",
            return_value=mock_energy_sampler,
        ),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        result = harness.run(engine, minimal_config)

    assert result.energy_per_device_j == [100.0]
    assert result.multi_gpu is None


# ---------------------------------------------------------------------------
# Error recovery: energy sampler returns None
# ---------------------------------------------------------------------------


def test_harness_energy_sampler_none_placeholder(minimal_config):
    """When select_energy_sampler returns None, result still has zero energy (placeholder)."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    # select_energy_sampler returns None in _apply_patches, so total_energy_j should be 0.0
    assert result.total_energy_j == pytest.approx(0.0)
    assert result.avg_energy_per_token_j == pytest.approx(0.0)


def test_harness_stop_tracking_raises_cleanup_still_called(minimal_config):
    """If energy sampler stop_tracking raises, cleanup is still called and model released."""
    mock_energy = MagicMock()
    mock_energy.start_tracking.return_value = MagicMock()
    mock_energy.stop_tracking.side_effect = RuntimeError("NVML error")

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=mock_energy),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
        pytest.raises(RuntimeError, match="NVML error"),
    ):
        harness.run(engine, minimal_config)

    # cleanup was called despite the exception (try/finally in harness)
    assert "cleanup" in engine.call_log


def test_harness_flops_estimation_raises_returns_zero_flops(minimal_config):
    """If FLOPs estimation raises, total_flops should be 0, not crash."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch(
            "llenergymeasure.harness.estimate_flops_palm_from_config",
            side_effect=RuntimeError("model not found"),
        ),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        result = harness.run(engine, minimal_config)

    assert result.total_flops == pytest.approx(0.0)


def test_harness_energy_none_still_produces_valid_result(minimal_config):
    """Full lifecycle with None energy produces a valid ExperimentResult."""
    from llenergymeasure.domain.experiment import ExperimentResult

    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert isinstance(result, ExperimentResult)
    assert result.total_tokens > 0
    assert result.energy_breakdown is None or result.energy_breakdown.raw_j == 0.0


def test_build_result_includes_model_name(minimal_config):
    """_build_result() populates model_name from config.model."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    assert result.model_name == minimal_config.model


def test_build_result_populates_mj_per_tok(minimal_config):
    """_build_result() sets mj_per_tok_total when total_tokens > 0."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, minimal_config)

    # FakeBackend returns total_tokens = input(10) + output(20) = 30
    assert result.total_tokens > 0
    assert result.mj_per_tok_total is not None
    # With no energy sampler (returns None), total_energy_j = 0.0 → 0.0 mJ/tok
    assert result.mj_per_tok_total == pytest.approx(0.0)
    # No baseline → mj_per_tok_adjusted must be None
    assert result.mj_per_tok_adjusted is None


def test_harness_flops_none_result_has_none_derived(minimal_config):
    """When FLOPs estimation returns None, derived fields are None."""
    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch(
            "llenergymeasure.harness.estimate_flops_palm_from_config",
            return_value=None,
        ),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ),
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        result = harness.run(engine, minimal_config)

    assert result.total_flops == pytest.approx(0.0)
    assert result.flops_per_output_token is None
    assert result.flops_per_second is None


# ---------------------------------------------------------------------------
# save_timeseries: parquet sidecar gating
# ---------------------------------------------------------------------------


def test_harness_save_timeseries_false_skips_parquet(tmp_path):
    """save_timeseries=False prevents write_timeseries_parquet from being called."""
    from llenergymeasure.config.models import DatasetConfig, ExperimentConfig

    config = ExperimentConfig(
        model="fake/model",
        engine="transformers",
        dataset=DatasetConfig(n_prompts=1),
        max_input_tokens=32,
        max_output_tokens=32,
    )

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch("llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=MagicMock(name="f"),
        ) as mock_write_ts,
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        result = harness.run(engine, config, output_dir=str(tmp_path), save_timeseries=False)

    mock_write_ts.assert_not_called()
    assert result.timeseries is None


def test_harness_save_timeseries_true_writes_parquet(tmp_path):
    """save_timeseries=True (default) writes timeseries parquet when output_dir is set."""
    from llenergymeasure.config.models import DatasetConfig, ExperimentConfig

    config = ExperimentConfig(
        model="fake/model",
        engine="transformers",
        dataset=DatasetConfig(n_prompts=1),
        max_input_tokens=32,
        max_output_tokens=32,
    )

    engine = FakeBackend()
    harness = MeasurementHarness()

    with (
        patch("llenergymeasure.harness.collect_environment_snapshot", return_value=None),
        patch("llenergymeasure.harness.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.thermal_floor_wait", return_value=0.0),
        patch("llenergymeasure.harness.estimate_flops_palm", return_value=MagicMock(value=0)),
        patch("llenergymeasure.harness._cuda_sync"),
        patch(
            "llenergymeasure.harness.PowerThermalSampler", new=_make_mock_pts(samples=[MagicMock()])
        ),
        patch(
            "llenergymeasure.harness.write_timeseries_parquet",
            return_value=Path(tmp_path / "timeseries.parquet"),
        ) as mock_write_ts,
        patch("llenergymeasure.harness.collect_measurement_warnings", return_value=[]),
    ):
        harness.run(engine, config, output_dir=str(tmp_path), save_timeseries=True)

    mock_write_ts.assert_called_once()
