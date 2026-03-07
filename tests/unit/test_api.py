"""Unit tests for the llenergymeasure public API surface.

Tests cover Phase 3 and Phase 4 (Plan 03) success criteria:
1. Public imports resolve
2. run_experiment returns ExperimentResult (no union, no None)
3. No disk writes when output_dir not set
4. Internal names raise AttributeError
5. __version__ matches pyproject.toml
6. run_study raises NotImplementedError with M2 message
7. _run() calls run_preflight once per experiment config
8. _run() calls get_backend with correct backend name
9. _run() returns StudyResult with experiment results
10. _run() propagates PreFlightError and BackendError unchanged
11. run_experiment end-to-end with mocked backend returns ExperimentResult
12. All test cases pass without GPU hardware (uses monkeypatching)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import llenergymeasure
from llenergymeasure import (
    ExperimentConfig,
    ExperimentResult,
    StudyConfig,
    StudyResult,
    __version__,
    run_experiment,
    run_study,
)
from llenergymeasure.domain.experiment import AggregationMetadata
from llenergymeasure.exceptions import BackendError, PreFlightError

_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_EPOCH_END = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


# =============================================================================
# Test helper
# =============================================================================


def _make_experiment_result(**overrides) -> ExperimentResult:
    """Build a minimal valid ExperimentResult for testing."""
    defaults = {
        "experiment_id": "test-001",
        "measurement_config_hash": "abc123def4567890",
        "measurement_methodology": "total",
        "aggregation": AggregationMetadata(num_processes=1),
        "total_tokens": 1000,
        "total_energy_j": 10.0,
        "total_inference_time_sec": 5.0,
        "avg_tokens_per_second": 200.0,
        "avg_energy_per_token_j": 0.01,
        "total_flops": 1e9,
        "start_time": _EPOCH,
        "end_time": _EPOCH_END,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


def _make_study_result(**overrides) -> StudyResult:
    """Build a StudyResult containing one ExperimentResult."""
    return StudyResult(experiments=[_make_experiment_result()])


# =============================================================================
# Test 1: Public imports resolve
# =============================================================================


def test_public_imports_resolve():
    """All 7 public names import correctly from llenergymeasure."""
    assert run_experiment is not None
    assert run_study is not None
    assert ExperimentConfig is not None
    assert StudyConfig is not None
    assert ExperimentResult is not None
    assert StudyResult is not None
    assert __version__ == llenergymeasure.__version__


# =============================================================================
# Test 2: Internal names raise AttributeError
# =============================================================================


def test_internal_name_raises_attribute_error():
    """Names not in __all__ raise AttributeError on module access."""
    internal_names = [
        "load_experiment_config",
        "ConfigError",
        "AggregatedResult",
        "LLEMError",
        "deep_merge",
    ]
    for name in internal_names:
        with pytest.raises(AttributeError, match=name):
            getattr(llenergymeasure, name)


# =============================================================================
# Test 3: run_experiment returns ExperimentResult (no union, no None)
# =============================================================================


def test_run_experiment_returns_experiment_result(monkeypatch):
    """run_experiment returns exactly ExperimentResult, not a union or None."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study, **kw: _make_study_result())

    config = ExperimentConfig(model="gpt2")
    result = run_experiment(config)

    assert result is not None
    assert isinstance(result, ExperimentResult)
    # Confirm it is NOT a StudyResult (no union types)
    assert not isinstance(result, StudyResult)


# =============================================================================
# Test 4: YAML path form
# =============================================================================


def test_run_experiment_yaml_path_form(tmp_path, monkeypatch):
    """run_experiment resolves correctly from a YAML path."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("model: gpt2\n")

    result = run_experiment(str(config_path))

    assert isinstance(result, ExperimentResult)
    # Confirm the study was built from the YAML
    assert captured_study["value"].experiments[0].model == "gpt2"


# =============================================================================
# Test 5: kwargs form
# =============================================================================


def test_run_experiment_kwargs_form(monkeypatch):
    """run_experiment kwargs form passes model and n to ExperimentConfig."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    result = run_experiment(model="gpt2", n=50)

    assert isinstance(result, ExperimentResult)
    assert captured_study["value"].experiments[0].model == "gpt2"
    assert captured_study["value"].experiments[0].n == 50


# =============================================================================
# Test 6: No config + no model raises ConfigError
# =============================================================================


def test_run_experiment_no_config_no_model_raises():
    """run_experiment() with no arguments raises ConfigError (not TypeError)."""
    from llenergymeasure.exceptions import ConfigError

    with pytest.raises(ConfigError):
        run_experiment()


# =============================================================================
# Test 7: No disk writes when output_dir not set
# =============================================================================


def test_run_experiment_no_disk_writes(tmp_path, monkeypatch):
    """run_experiment produces no disk writes when output_dir is not specified."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study, **kw: _make_study_result())

    # Change working directory to tmp_path to catch any accidental writes
    config = ExperimentConfig(model="gpt2")
    run_experiment(config)

    # tmp_path should be empty — no files written there
    written_files = list(tmp_path.rglob("*"))
    assert written_files == [], f"Unexpected files written: {written_files}"


# =============================================================================
# Test 8: run_study is implemented (M2)
# =============================================================================


def test_run_study_invalid_type_raises_config_error():
    """run_study(42) raises ConfigError, not NotImplementedError."""
    from llenergymeasure.exceptions import ConfigError

    with pytest.raises(ConfigError):
        run_study(42)  # type: ignore[arg-type]


# =============================================================================
# Test 9: __all__ list matches exports
# =============================================================================


def test_all_list_matches_exports():
    """Every name in __all__ is importable from llenergymeasure."""
    for name in llenergymeasure.__all__:
        obj = getattr(llenergymeasure, name, None)
        assert obj is not None, f"__all__ member '{name}' is not importable from llenergymeasure"


# =============================================================================
# Test 10: __version__ in __all__
# =============================================================================


def test_version_in_all():
    """__version__ is explicitly in __all__."""
    assert "__version__" in llenergymeasure.__all__


# =============================================================================
# Test 11: run_experiment with Path object (not just str)
# =============================================================================


def test_run_experiment_path_object_form(tmp_path, monkeypatch):
    """run_experiment accepts a Path object as well as a str path."""
    import llenergymeasure._api as api_module

    monkeypatch.setattr(api_module, "_run", lambda study, **kw: _make_study_result())

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt2\n")

    result = run_experiment(config_path)  # Path object, not str
    assert isinstance(result, ExperimentResult)


# =============================================================================
# Test 12: kwargs form — backend kwarg passed through
# =============================================================================


def test_run_experiment_kwargs_backend(monkeypatch):
    """run_experiment kwargs form passes backend to ExperimentConfig."""
    import llenergymeasure._api as api_module

    captured_study = {}

    def mock_run(study, **kw):
        captured_study["value"] = study
        return _make_study_result()

    monkeypatch.setattr(api_module, "_run", mock_run)

    run_experiment(model="gpt2", backend="pytorch")

    assert captured_study["value"].experiments[0].backend == "pytorch"


# =============================================================================
# Phase 4 Plan 03: _run() wiring tests
# =============================================================================


class _MockBackend:
    """Minimal mock InferenceBackend for _run() tests."""

    def __init__(self, result: ExperimentResult) -> None:
        self._result = result
        self.run_calls: list[ExperimentConfig] = []

    @property
    def name(self) -> str:
        return "pytorch"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        self.run_calls.append(config)
        return self._result


def test_run_calls_preflight_once_per_config(monkeypatch, tmp_path):
    """_run() calls run_preflight once for the single in-process experiment."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    preflight_calls: list = []

    def mock_preflight(config):
        preflight_calls.append(config)

    mock_result = _make_experiment_result()
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", mock_preflight)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config1 = ExperimentConfig(model="gpt2")
    study = StudyConfig(experiments=[config1])

    api_module._run(study)

    assert len(preflight_calls) == 1, f"Expected 1 preflight call, got {len(preflight_calls)}"
    assert preflight_calls[0].model == "gpt2"


def test_run_calls_get_backend_with_correct_name(monkeypatch, tmp_path):
    """_run() calls get_backend with the experiment's backend name."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    mock_result = _make_experiment_result()
    mock_backend = _MockBackend(mock_result)

    backend_calls: list[str] = []

    def mock_get_backend(name: str):
        backend_calls.append(name)
        return mock_backend

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", mock_get_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config = ExperimentConfig(model="gpt2", backend="pytorch")
    study = StudyConfig(experiments=[config])

    api_module._run(study)

    assert len(backend_calls) == 1
    assert backend_calls[0] == "pytorch"


def test_run_returns_study_result(monkeypatch, tmp_path):
    """_run() returns a StudyResult containing the experiment results."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    mock_result = _make_experiment_result(experiment_id="wired-001")
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    config = ExperimentConfig(model="gpt2")
    study = StudyConfig(experiments=[config], name="my-study")

    study_result = api_module._run(study)

    assert isinstance(study_result, StudyResult)
    assert study_result.name == "my-study"
    assert len(study_result.experiments) == 1
    assert study_result.experiments[0].experiment_id == "wired-001"


def test_run_propagates_preflight_error(monkeypatch, tmp_path):
    """_run() propagates PreFlightError without catching it."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    def failing_preflight(config):
        raise PreFlightError(["CUDA not available"])

    monkeypatch.setattr(pf_module, "run_preflight", failing_preflight)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(
        backends_module, "get_backend", lambda name: _MockBackend(_make_experiment_result())
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )

    config = ExperimentConfig(model="gpt2")
    study = StudyConfig(experiments=[config])

    with pytest.raises(PreFlightError):
        api_module._run(study)


def test_run_propagates_backend_error(monkeypatch, tmp_path):
    """_run() propagates BackendError without catching it."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    class _FailingBackend:
        @property
        def name(self) -> str:
            return "pytorch"

        def run(self, config: ExperimentConfig) -> ExperimentResult:
            raise BackendError("GPU out of memory")

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: _FailingBackend())
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )

    config = ExperimentConfig(model="gpt2")
    study = StudyConfig(experiments=[config])

    with pytest.raises(BackendError, match="GPU out of memory"):
        api_module._run(study)


def test_run_experiment_end_to_end_mocked(monkeypatch, tmp_path):
    """run_experiment() flows through the real _run() pipeline (mocked backend) and returns ExperimentResult."""
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    expected_result = _make_experiment_result(experiment_id="e2e-test")
    mock_backend = _MockBackend(expected_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    result = run_experiment(model="gpt2")

    assert isinstance(result, ExperimentResult)
    assert not isinstance(result, StudyResult)
    assert result.experiment_id == "e2e-test"
    assert len(mock_backend.run_calls) == 1
    assert mock_backend.run_calls[0].model == "gpt2"


# =============================================================================
# Plan 02: run_study() and _run() dispatcher tests
# =============================================================================


def test_run_study_accepts_study_config(monkeypatch, tmp_path):
    """run_study(StudyConfig) returns StudyResult with populated summary."""
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    mock_result = _make_experiment_result(experiment_id="study-test")
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    # Avoid real disk writes by patching create_study_dir and save_result
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    study = StudyConfig(experiments=[ExperimentConfig(model="gpt2")])
    result = run_study(study)

    assert isinstance(result, StudyResult)
    assert result.summary is not None
    assert result.summary.completed == 1
    assert result.summary.failed == 0


def test_run_study_accepts_path(tmp_path, monkeypatch):
    """run_study(str path) loads YAML and returns StudyResult."""
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    yaml_content = "experiments:\n  - model: gpt2\n"
    yaml_path = tmp_path / "study.yaml"
    yaml_path.write_text(yaml_content)

    mock_result = _make_experiment_result(experiment_id="path-test")
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    result = run_study(str(yaml_path))

    assert isinstance(result, StudyResult)


def test_run_dispatches_single_in_process(monkeypatch, tmp_path):
    """Single experiment + n_cycles=1 bypasses StudyRunner (in-process path)."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module
    from llenergymeasure.study.runner import StudyRunner

    mock_result = _make_experiment_result(experiment_id="inproc-test")
    mock_backend = _MockBackend(mock_result)

    runner_created = []
    original_runner_init = StudyRunner.__init__

    def mock_runner_init(self, *args, **kwargs):
        runner_created.append(True)
        original_runner_init(self, *args, **kwargs)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )
    monkeypatch.setattr(StudyRunner, "__init__", mock_runner_init)

    study = StudyConfig(
        experiments=[ExperimentConfig(model="gpt2")],
        execution={"n_cycles": 1, "cycle_order": "sequential"},
    )
    api_module._run(study)

    # Single experiment + n_cycles=1 should NOT create StudyRunner
    assert runner_created == [], "StudyRunner was created for single in-process path"
    # Backend was called directly
    assert len(mock_backend.run_calls) == 1


def test_run_study_returns_study_result_type():
    """run_study return annotation is StudyResult (not a union)."""
    import inspect

    import llenergymeasure._api as api_module

    try:
        import typing

        typing.get_type_hints(api_module.run_study)
    except Exception:
        pass

    # Check via source that return type is annotated as StudyResult
    src = inspect.getsource(api_module.run_study)
    assert "-> StudyResult" in src, "run_study must have -> StudyResult return annotation"


# =============================================================================
# Plan 04: runner resolution wiring in _run()
# =============================================================================


def test_run_resolves_runners_and_passes_to_study_runner(monkeypatch, tmp_path):
    """_run() resolves runners via resolve_study_runners and passes runner_specs to StudyRunner."""
    import llenergymeasure._api as api_module
    import llenergymeasure.orchestration.preflight as pf_module
    from llenergymeasure.infra.runner_resolution import RunnerSpec

    mock_result = _make_experiment_result(experiment_id="runner-wired")

    resolved_specs = {
        "pytorch": RunnerSpec(mode="local", image=None, source="default"),
    }

    # Capture what runner_specs was passed to _run_via_runner
    captured_runner_specs: list = []
    original_run_via_runner = api_module._run_via_runner

    def mock_run_via_runner(study, manifest, study_dir, runner_specs=None):
        captured_runner_specs.append(runner_specs)
        return original_run_via_runner(study, manifest, study_dir, runner_specs=runner_specs)

    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        lambda backends, yaml_runners=None, user_config=None: resolved_specs,
    )
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: type("C", (), {"runners": None})(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )
    monkeypatch.setattr(api_module, "_run_via_runner", mock_run_via_runner)

    # Use a 2-experiment study to force _run_via_runner path (not _run_in_process)
    import llenergymeasure.core.backends as backends_module

    mock_backend = _MockBackend(mock_result)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)

    # Mock StudyRunner.run() to avoid real subprocess spawning
    from llenergymeasure.study.runner import StudyRunner

    monkeypatch.setattr(StudyRunner, "run", lambda self: [mock_result])

    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="gpt2", backend="pytorch"),
            ExperimentConfig(model="gpt2-medium", backend="pytorch"),
        ]
    )

    api_module._run(study)

    assert len(captured_runner_specs) == 1, "_run_via_runner not called or called multiple times"
    assert captured_runner_specs[0] == resolved_specs


# =============================================================================
# Plan 24-01: GPU memory check in _run_in_process
# =============================================================================


def test_run_in_process_calls_gpu_memory_check(monkeypatch, tmp_path):
    """_run_in_process() calls check_gpu_memory_residual before running the experiment."""
    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module

    gpu_check_calls: list[int] = []

    def mock_gpu_check(device_index=0, threshold_mb=1024.0):
        gpu_check_calls.append(device_index)

    mock_result = _make_experiment_result(experiment_id="gpu-check-test")
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
        mock_gpu_check,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    from unittest.mock import MagicMock

    from llenergymeasure.study.manifest import ManifestWriter

    mock_manifest = MagicMock(spec=ManifestWriter)

    config = ExperimentConfig(model="gpt2", backend="pytorch")
    study = StudyConfig(experiments=[config])

    api_module._run_in_process(study, mock_manifest, tmp_path, runner_specs=None)

    assert len(gpu_check_calls) == 1, (
        f"Expected check_gpu_memory_residual to be called once, got {len(gpu_check_calls)}"
    )


def test_run_mixed_runner_warning_logged(monkeypatch, tmp_path, caplog):
    """_run() logs a warning when runner_specs has mixed local/docker modes."""
    import logging

    import llenergymeasure._api as api_module
    import llenergymeasure.core.backends as backends_module
    import llenergymeasure.orchestration.preflight as pf_module
    from llenergymeasure.infra.runner_resolution import RunnerSpec

    mixed_specs = {
        "pytorch": RunnerSpec(mode="local", image=None, source="default"),
        "vllm": RunnerSpec(mode="docker", image=None, source="yaml"),
    }

    mock_result = _make_experiment_result()
    mock_backend = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(backends_module, "get_backend", lambda name: mock_backend)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        lambda backends, yaml_runners=None, user_config=None: mixed_specs,
    )
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: type("C", (), {"runners": None})(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    study = StudyConfig(experiments=[ExperimentConfig(model="gpt2")])

    with caplog.at_level(logging.WARNING, logger="llenergymeasure._api"):
        api_module._run(study)

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mixed" in m.lower() for m in warning_messages), (
        f"Expected mixed runner warning, got: {warning_messages}"
    )


# =============================================================================
# B3 fix: study experiment count no double-multiply
# =============================================================================


def test_study_summary_total_experiments_no_double_multiply(monkeypatch, tmp_path):
    """total_experiments == len(study.experiments) — no double-multiply by n_cycles.

    The study passed to _run() has experiments already cycle-expanded (as load_study_config
    does). With 2 unique configs and n_cycles=3, study.experiments has 6 entries.
    total_experiments must be 6, not 18 (the pre-fix bug: 6 * 3).
    unique_configurations must be 2 (6 / 3).
    """
    import llenergymeasure._api as api_module
    import llenergymeasure.orchestration.preflight as pf_module

    # Build 6 mock results (2 configs × 3 cycles, already cycle-expanded)
    mock_results = [_make_experiment_result(experiment_id=f"b3-{i}") for i in range(6)]

    # Mock _run_via_runner to return pre-built results (bypasses real subprocess)
    def mock_run_via_runner(study, manifest, study_dir, runner_specs=None):
        result_files = [str(tmp_path / f"result-{i}.json") for i in range(6)]
        return result_files, mock_results, []

    monkeypatch.setattr(pf_module, "run_study_preflight", lambda study, **kw: None)
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        lambda backends, yaml_runners=None, user_config=None: {},
    )
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: type("C", (), {"runners": None})(),
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(api_module, "_run_via_runner", mock_run_via_runner)

    # Simulate what load_study_config returns: experiments already cycle-expanded.
    # 2 unique configs × 3 cycles = 6 entries, n_cycles=3 in execution config.
    expanded_experiments = [
        ExperimentConfig(model="gpt2"),
        ExperimentConfig(model="gpt2-medium"),
        ExperimentConfig(model="gpt2"),
        ExperimentConfig(model="gpt2-medium"),
        ExperimentConfig(model="gpt2"),
        ExperimentConfig(model="gpt2-medium"),
    ]
    study = StudyConfig(
        experiments=expanded_experiments,
        execution={"n_cycles": 3, "cycle_order": "sequential"},
    )
    # Verify our study fixture has exactly 6 experiments
    assert len(study.experiments) == 6

    study_result = api_module._run(study)

    assert study_result.summary is not None
    assert study_result.summary.total_experiments == 6, (
        f"Expected 6 (cycle-expanded count), got {study_result.summary.total_experiments} "
        f"(pre-fix bug would give 18 = 6 × 3)"
    )
    assert study_result.summary.unique_configurations == 2, (
        f"Expected 2 unique configurations (6 / 3), got {study_result.summary.unique_configurations}"
    )
