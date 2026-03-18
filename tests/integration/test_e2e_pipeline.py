"""E2E integration tests for the full pipeline: YAML -> config -> runner -> result.

These tests exercise real YAML parsing, real config validation, and the real
orchestrator -- only the inference backend and GPU-specific calls are faked.

CLI E2E tests use CliRunner (thin wrapper verification per CONTEXT.md).

No GPU required. All tests run under the default 'not gpu and not docker' marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_result, make_study_result

# =============================================================================
# Shared patch helpers
# =============================================================================


def _patch_infra(monkeypatch: Any, tmp_path: Path, mock_result: Any) -> None:
    """Patch infrastructure calls common to all pipeline tests.

    Patches out:
    - run_study_preflight (multi-backend guard, Docker pre-flight)
    - run_preflight (per-experiment GPU check)
    - check_gpu_memory_residual (NVML residual check)
    - load_user_config (user config file access)
    - resolve_study_runners (runner resolution -- forces local path, no Docker dispatch)
    - create_study_dir + ManifestWriter (disk manifest writes)
    - save_result (per-experiment disk writes)
    """
    import llenergymeasure.harness.preflight
    import llenergymeasure.study.preflight
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter

    monkeypatch.setattr(
        llenergymeasure.study.preflight, "run_study_preflight", lambda study, **kw: None
    )
    monkeypatch.setattr(llenergymeasure.harness.preflight, "run_preflight", lambda config: None)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "llenergymeasure.config.user_config.load_user_config",
        lambda **kwargs: type("C", (), {"runners": None})(),
    )
    # Force all backends to use the local in-process path.
    # Without this, resolve_study_runners may return docker specs on Docker-enabled hosts.
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        lambda backends, yaml_runners=None, user_config=None: {
            b: RunnerSpec(mode="local", image=None, source="default") for b in backends
        },
    )
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.create_study_dir",
        lambda name, output_dir: tmp_path,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    mock_manifest = MagicMock(spec=ManifestWriter)
    monkeypatch.setattr(
        "llenergymeasure.study.manifest.ManifestWriter",
        lambda *args, **kwargs: mock_manifest,
    )


def _patch_backend(monkeypatch: Any, mock_result: Any) -> None:
    """Patch get_backend and MeasurementHarness to return a pre-built result."""
    import llenergymeasure.backends as backends_module
    import llenergymeasure.harness as harness_module

    monkeypatch.setattr(backends_module, "get_backend", lambda name: MagicMock())
    monkeypatch.setattr(
        harness_module.MeasurementHarness,
        "run",
        lambda self, backend, config, **kw: mock_result,
    )


# =============================================================================
# Pipeline E2E tests
# =============================================================================


class TestPipelineSingleExperiment:
    """Test 1: YAML -> ExperimentConfig -> backend -> ExperimentResult."""

    def test_pipeline_single_experiment(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Full pipeline: real YAML parse, real config validation, fake backend."""
        from llenergymeasure import run_experiment
        from llenergymeasure.config.loader import load_experiment_config
        from llenergymeasure.domain.experiment import ExperimentResult

        # Write minimal experiment YAML
        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nbackend: pytorch\nn: 5\n")

        mock_result = make_result()
        _patch_infra(monkeypatch, tmp_path, mock_result)
        _patch_backend(monkeypatch, mock_result)

        # Real YAML parsing and config validation (not mocked)
        experiment_config = load_experiment_config(path=yaml_path)
        assert experiment_config.model == "gpt2"
        assert experiment_config.backend == "pytorch"
        assert experiment_config.n == 5

        # Full run_experiment call
        result = run_experiment(experiment_config, skip_preflight=True)

        assert isinstance(result, ExperimentResult)
        assert result.total_energy_j == mock_result.total_energy_j
        assert result.total_tokens == mock_result.total_tokens

    def test_pipeline_single_experiment_from_yaml_path(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """run_experiment accepts YAML path directly (str form)."""
        from llenergymeasure import run_experiment
        from llenergymeasure.domain.experiment import ExperimentResult

        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nn: 3\n")

        mock_result = make_result()
        _patch_infra(monkeypatch, tmp_path, mock_result)
        _patch_backend(monkeypatch, mock_result)

        result = run_experiment(str(yaml_path), skip_preflight=True)

        assert isinstance(result, ExperimentResult)


class TestPipelineMultiExperimentSweep:
    """Test 2: Study YAML with sweep -> StudyConfig -> N experiment configs."""

    def test_study_yaml_sweep_produces_correct_experiment_count(self, tmp_path: Path) -> None:
        """YAML with precision sweep -> StudyConfig with 2 experiments (fp16, bf16).

        Tests real YAML parsing + config validation. Does not exercise the runner
        (StudyRunner uses multiprocessing spawn; subprocess patching is not feasible
        for cross-process injection). Runner integration is covered by CLI E2E tests.
        """
        from llenergymeasure.config.loader import load_study_config

        yaml_content = """\
model: gpt2
sweep:
  precision: [fp16, bf16]
execution:
  n_cycles: 1
  cycle_order: sequential
  experiment_gap_seconds: 0
warmup:
  enabled: false
baseline:
  enabled: false
"""
        yaml_path = tmp_path / "study.yaml"
        yaml_path.write_text(yaml_content)

        study_config = load_study_config(yaml_path)

        # Sweep over 2 precisions, n_cycles=1 → 2 experiments total
        assert len(study_config.experiments) == 2
        precisions = {exp.precision for exp in study_config.experiments}
        assert precisions == {"fp16", "bf16"}

    def test_study_yaml_model_sweep(self, tmp_path: Path) -> None:
        """YAML with model sweep produces correct number of experiment configs."""
        from llenergymeasure.config.loader import load_study_config

        yaml_content = """\
sweep:
  model: [gpt2, distilgpt2]
  backend: [pytorch]
execution:
  n_cycles: 1
  cycle_order: sequential
"""
        yaml_path = tmp_path / "study.yaml"
        yaml_path.write_text(yaml_content)

        study_config = load_study_config(yaml_path)

        assert len(study_config.experiments) == 2
        models = {exp.model for exp in study_config.experiments}
        assert models == {"gpt2", "distilgpt2"}

    def test_study_config_study_design_hash_set(self, tmp_path: Path) -> None:
        """StudyConfig.study_design_hash is populated after loading."""
        from llenergymeasure.config.loader import load_study_config

        yaml_content = "model: gpt2\nexecution:\n  n_cycles: 1\n  cycle_order: sequential\n"
        yaml_path = tmp_path / "study.yaml"
        yaml_path.write_text(yaml_content)

        study_config = load_study_config(yaml_path)

        assert study_config.study_design_hash
        assert len(study_config.study_design_hash) == 16


class TestPipelineErrorPropagation:
    """Test 3: BackendError propagates from backend through run_experiment to caller."""

    def test_backend_error_propagates(self, tmp_path: Path, monkeypatch: Any) -> None:
        """BackendError raised in backend propagates unchanged to the caller.

        _patch_infra forces resolve_study_runners to return a local spec so the
        in-process path is exercised (not Docker dispatch, which catches errors differently).
        """
        import llenergymeasure.backends as backends_module
        import llenergymeasure.harness as harness_module
        from llenergymeasure import run_experiment
        from llenergymeasure.utils.exceptions import BackendError

        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nbackend: pytorch\n")

        _patch_infra(monkeypatch, tmp_path, make_result())

        # Backend raises on run
        def _failing_harness_run(self: Any, backend: Any, config: Any, **kw: Any) -> None:
            raise BackendError("GPU OOM")

        monkeypatch.setattr(backends_module, "get_backend", lambda name: MagicMock())
        monkeypatch.setattr(harness_module.MeasurementHarness, "run", _failing_harness_run)

        with pytest.raises(BackendError, match="GPU OOM"):
            run_experiment(str(yaml_path), skip_preflight=True)

    def test_config_error_on_invalid_yaml(self, tmp_path: Path) -> None:
        """ConfigError raised for YAML with unknown fields (real loader validation)."""
        from llenergymeasure.config.loader import load_experiment_config
        from llenergymeasure.utils.exceptions import ConfigError

        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("model: gpt2\nunknown_field: value\n")

        with pytest.raises(ConfigError, match="Unknown field"):
            load_experiment_config(path=yaml_path)

    def test_config_error_on_missing_file(self, tmp_path: Path) -> None:
        """ConfigError raised when config file does not exist."""
        from llenergymeasure.config.loader import load_experiment_config
        from llenergymeasure.utils.exceptions import ConfigError

        with pytest.raises(ConfigError, match="not found"):
            load_experiment_config(path=tmp_path / "nonexistent.yaml")


class TestPipelineDryRun:
    """Test 4: CLI dry-run path -- validates config and estimates VRAM without running backend."""

    def test_pipeline_dry_run_no_backend_call(self, tmp_path: Path, monkeypatch: Any) -> None:
        """CLI --dry-run validates config and prints output without calling backend."""
        from typer.testing import CliRunner

        import llenergymeasure.backends as backends_module
        from llenergymeasure.cli import app

        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nbackend: pytorch\nn: 5\n")

        backend_call_count = []

        def _tracking_get_backend(name: str) -> Any:
            backend_call_count.append(name)
            return MagicMock()

        monkeypatch.setattr(backends_module, "get_backend", _tracking_get_backend)

        # Patch VRAM utilities to avoid GPU hardware dependency.
        # estimate_vram returns dict[str, float] | None (not a plain float).
        _fake_vram = {"weights_gb": 1.0, "kv_cache_gb": 0.5, "overhead_gb": 0.2, "total_gb": 1.7}
        monkeypatch.setattr(
            "llenergymeasure.cli.run.estimate_vram",
            lambda config: _fake_vram,
        )
        monkeypatch.setattr(
            "llenergymeasure.cli.run.get_gpu_vram_gb",
            lambda: 40.0,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(yaml_path), "--dry-run"])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        # Dry-run shows config without running backend
        assert "Config" in result.output or "gpt2" in result.output
        # Backend must NOT be called in dry-run
        assert backend_call_count == [], f"Backend was called during dry-run: {backend_call_count}"


# =============================================================================
# CLI E2E tests (thin wrapper verification)
# =============================================================================


class TestCLIE2ESingleExperiment:
    """Test 5: llem run <yaml> via CliRunner with patched run_experiment."""

    def test_cli_e2e_single_experiment(self, tmp_path: Path, monkeypatch: Any) -> None:
        """CLI run command exits 0 and prints result summary for a single experiment."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nbackend: pytorch\nn: 5\n")

        mock_result = make_result(experiment_id="cli-e2e-001")

        # Patch run_experiment at the CLI module import (tqdm requires the real sys.stderr)
        monkeypatch.setattr(
            "llenergymeasure.cli.run.run_experiment",
            lambda config, **kw: mock_result,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(yaml_path)])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}:\n{result.output}"
        assert "Result:" in result.output

    def test_cli_e2e_model_flag(self, tmp_path: Path, monkeypatch: Any) -> None:
        """CLI run --model gpt2 via CliRunner exits 0."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        mock_result = make_result(experiment_id="cli-flag-001")

        monkeypatch.setattr(
            "llenergymeasure.cli.run.run_experiment",
            lambda config, **kw: mock_result,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--model", "gpt2", "--backend", "pytorch"])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}:\n{result.output}"
        assert "Result:" in result.output


class TestCLIE2EStudy:
    """Test 6: llem run <study.yaml> via CliRunner with patched run_study."""

    def test_cli_e2e_study(self, tmp_path: Path, monkeypatch: Any) -> None:
        """CLI run command with study YAML exits 0 and prints study summary."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        yaml_path = tmp_path / "study.yaml"
        yaml_path.write_text(
            "model: gpt2\nsweep:\n  precision: [fp16, bf16]\n"
            "execution:\n  n_cycles: 1\n  cycle_order: sequential\n"
        )

        mock_study_result = make_study_result()

        # run_study is imported lazily inside _run_study_impl via `from llenergymeasure import run_study`
        # Patch at the source module so the lazy import picks up our mock
        import llenergymeasure
        import llenergymeasure.api
        import llenergymeasure.utils

        monkeypatch.setattr(llenergymeasure, "run_study", lambda config, **kw: mock_study_result)

        # Also patch format_preflight_summary to avoid real config.grid dependency
        monkeypatch.setattr(
            "llenergymeasure.config.grid.format_preflight_summary",
            lambda study_config: "Preflight: 2 experiments",
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(yaml_path)])

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}:\n{result.output}"
        assert "Study:" in result.output


# =============================================================================
# CLI E2E error exit codes (parametrised)
# =============================================================================


class TestCLIE2EErrorExitCodes:
    """Test 7: CLI exits with correct codes for different error types."""

    @pytest.mark.parametrize(
        "error_class, expected_exit_code",
        [
            pytest.param("ConfigError", 2, id="config-error-exit-2"),
            pytest.param("ExperimentError", 1, id="experiment-error-exit-1"),
            pytest.param("BackendError", 1, id="backend-error-exit-1"),
            pytest.param("PreFlightError", 1, id="preflight-error-exit-1"),
        ],
    )
    def test_cli_error_exit_codes(
        self,
        tmp_path: Path,
        monkeypatch: Any,
        error_class: str,
        expected_exit_code: int,
    ) -> None:
        """llem run exits with correct code when run_experiment raises an error."""
        from typer.testing import CliRunner

        import llenergymeasure.utils.exceptions
        from llenergymeasure.cli import app

        yaml_path = tmp_path / "experiment.yaml"
        yaml_path.write_text("model: gpt2\nbackend: pytorch\nn: 5\n")

        error_cls = getattr(llenergymeasure.utils.exceptions, error_class)
        exc_instance = error_cls(f"test {error_class}")

        monkeypatch.setattr(
            "llenergymeasure.cli.run.run_experiment",
            lambda config, **kw: (_ for _ in ()).throw(exc_instance),
        )

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(yaml_path)])

        assert result.exit_code == expected_exit_code, (
            f"For {error_class}, expected exit code {expected_exit_code}, "
            f"got {result.exit_code}.\nOutput:\n{result.output}"
        )

    def test_cli_missing_config_and_model_exits_2(self, tmp_path: Path) -> None:
        """llem run with no config file and no --model exits with code 2 (ConfigError)."""
        from typer.testing import CliRunner

        from llenergymeasure.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 2, f"Expected exit 2, got {result.exit_code}:\n{result.output}"
