"""Unit tests for the llem run CLI command.

Tests use typer.testing.CliRunner to invoke the CLI without loading models or
touching GPU hardware. All heavy operations are mocked.

Note: typer's CliRunner routes all output (stdout + stderr) to .output.
Error messages printed to sys.stderr are captured in .output for assertions.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from llenergymeasure.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result() -> MagicMock:
    """Return a minimal mock ExperimentResult with required attributes."""
    from llenergymeasure.domain.experiment import ExperimentResult

    result = MagicMock(spec=ExperimentResult)
    result.experiment_id = "test-exp-001"
    result.total_energy_j = 100.0
    result.avg_tokens_per_second = 42.0
    result.duration_sec = 5.0
    result.measurement_warnings = []
    result.baseline_power_w = None
    result.energy_adjusted_j = None
    result.total_flops = 0.0
    result.latency_stats = None
    result.warmup_excluded_samples = None
    result.process_results = []
    result.output_dir = None
    return result


def _make_mock_config() -> MagicMock:
    """Return a minimal mock ExperimentConfig."""
    from llenergymeasure.config.models import ExperimentConfig

    config = MagicMock(spec=ExperimentConfig)
    config.model = "gpt2"
    config.backend = "pytorch"
    config.precision = "bf16"
    config.n = 100
    config.dataset = "aienergyscore"
    config.output_dir = None
    config.max_input_tokens = 512
    config.max_output_tokens = 128
    config.pytorch = None
    return config


# ---------------------------------------------------------------------------
# Basic flag tests
# ---------------------------------------------------------------------------


def test_run_help():
    """llem run --help exits 0 and shows expected flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    assert "--model" in plain
    assert "--backend" in plain
    assert "--dry-run" in plain


def test_run_version():
    """llem --version exits 0 and prints version string."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llem v" in result.output


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


def test_run_no_args_exits_2():
    """llem run with no args (no config, no --model) exits with code 2."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )


def test_run_config_error_exits_2():
    """ConfigError raised by load_experiment_config exits with code 2."""
    from llenergymeasure.exceptions import ConfigError

    with patch("llenergymeasure.cli.run.load_experiment_config") as mock_load:
        mock_load.side_effect = ConfigError("bad config: unknown field 'foop'")
        result = runner.invoke(app, ["run", "nonexistent.yaml"])

    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "ConfigError" in result.output


def test_run_validation_error_exits_2():
    """Pydantic ValidationError from a bad field value exits with code 2."""
    # "pytorh" is a misspelled backend — Pydantic will raise ValidationError
    result = runner.invoke(app, ["run", "--model", "gpt2", "--backend", "pytorh"])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "Config validation failed" in result.output


def test_run_preflight_error_exits_1():
    """PreFlightError raised by run_experiment exits with code 1."""
    from llenergymeasure.exceptions import PreFlightError

    mock_config = _make_mock_config()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment") as mock_run,
    ):
        mock_run.side_effect = PreFlightError("no GPU available")
        result = runner.invoke(app, ["run", "--model", "gpt2"])

    assert result.exit_code == 1, (
        f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
    )
    assert "PreFlightError" in result.output


def test_run_experiment_error_exits_1():
    """ExperimentError raised during run exits with code 1."""
    from llenergymeasure.exceptions import ExperimentError

    mock_config = _make_mock_config()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment") as mock_run,
    ):
        mock_run.side_effect = ExperimentError("inference crashed")
        result = runner.invoke(app, ["run", "--model", "gpt2"])

    assert result.exit_code == 1, (
        f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
    )
    assert "ExperimentError" in result.output


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_run_dry_run_exits_0():
    """--dry-run exits 0 and calls print_dry_run with resolved config."""
    mock_config = _make_mock_config()
    mock_vram = {
        "weights_gb": 0.24,
        "kv_cache_gb": 0.01,
        "overhead_gb": 0.04,
        "total_gb": 0.29,
    }

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.estimate_vram", return_value=mock_vram),
        patch("llenergymeasure.cli.run.get_gpu_vram_gb", return_value=None),
        patch("llenergymeasure.cli.run.print_dry_run") as mock_print_dry,
    ):
        result = runner.invoke(app, ["run", "--model", "gpt2", "--dry-run"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_print_dry.assert_called_once()


def test_run_dry_run_calls_estimate_vram():
    """--dry-run calls estimate_vram and get_gpu_vram_gb with the resolved config."""
    mock_config = _make_mock_config()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.estimate_vram", return_value=None) as mock_vram,
        patch("llenergymeasure.cli.run.get_gpu_vram_gb", return_value=None) as mock_gpu_vram,
        patch("llenergymeasure.cli.run.print_dry_run"),
    ):
        result = runner.invoke(app, ["run", "--model", "gpt2", "--dry-run"])

    assert result.exit_code == 0
    mock_vram.assert_called_once_with(mock_config)
    mock_gpu_vram.assert_called_once()


# ---------------------------------------------------------------------------
# Quiet flag test
# ---------------------------------------------------------------------------


def test_run_quiet_flag_accepted():
    """--quiet suppresses the tqdm progress spinner (disable=True)."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment", return_value=mock_result),
        patch("llenergymeasure.cli.run.print_experiment_header"),
        patch("llenergymeasure.cli.run.print_result_summary"),
        patch("llenergymeasure.cli.run.tqdm") as mock_tqdm,
    ):
        # Make tqdm context manager work
        mock_tqdm.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["run", "--model", "gpt2", "--quiet"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    # Verify tqdm was called with disable=True when --quiet is set
    call_kwargs = mock_tqdm.call_args
    assert call_kwargs is not None, "tqdm was not called"
    # quiet=True so disable must be True (quiet or not isatty -> True or x -> True)
    disable_val = call_kwargs.kwargs.get("disable")
    assert disable_val is True, f"Expected disable=True in tqdm call, got disable={disable_val!r}"


# ---------------------------------------------------------------------------
# Successful run test
# ---------------------------------------------------------------------------


def test_run_success_prints_summary():
    """Successful run calls print_result_summary with the returned result."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment", return_value=mock_result),
        patch("llenergymeasure.cli.run.print_experiment_header"),
        patch("llenergymeasure.cli.run.print_result_summary") as mock_summary,
        patch("llenergymeasure.cli.run.tqdm") as mock_tqdm,
    ):
        mock_tqdm.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)

        result = runner.invoke(app, ["run", "--model", "gpt2"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_summary.assert_called_once_with(mock_result)


# ---------------------------------------------------------------------------
# Study CLI tests (Phase 12)
# ---------------------------------------------------------------------------


def test_study_detection_with_sweep_key(tmp_path):
    """YAML with sweep: key is detected as study mode."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text("""
name: test
model: test/model
sweep:
  precision: [fp32, fp16]
""")
    import yaml

    raw = yaml.safe_load(study_yaml.read_text())
    assert "sweep" in raw


def test_study_detection_with_experiments_key(tmp_path):
    """YAML with experiments: key is detected as study mode."""
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text("""
name: test
experiments:
  - model: test/model-a
  - model: test/model-b
""")
    import yaml

    raw = yaml.safe_load(study_yaml.read_text())
    assert "experiments" in raw


def test_cli_flags_present():
    """llem run has --cycles, --order, --no-gaps flags."""
    import inspect

    from llenergymeasure.cli.run import run

    sig = inspect.signature(run)
    params = set(sig.parameters.keys())
    assert "cycles" in params
    assert "order" in params
    assert "no_gaps" in params


def test_print_study_summary_basic():
    """print_study_summary runs without error on a minimal StudyResult."""
    from io import StringIO
    from unittest.mock import MagicMock, patch

    from llenergymeasure.cli._display import print_study_summary
    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    # Use model_construct to bypass Pydantic validation for the container —
    # experiments list contains a MagicMock, which is not a valid ExperimentResult.
    exp = MagicMock()
    exp.effective_config = {"model": "test/model", "precision": "fp16"}
    exp.backend = "pytorch"
    exp.duration_sec = 45.2
    exp.total_energy_j = 123.4
    exp.avg_tokens_per_second = 42.5

    summary = StudySummary(
        total_experiments=1,
        completed=1,
        failed=0,
        total_wall_time_s=50.0,
        total_energy_j=123.4,
    )
    result = StudyResult.model_construct(
        experiments=[exp],
        name="test-study",
        study_design_hash="abcd1234",
        summary=summary,
        result_files=["results/exp1/result.json"],
        measurement_protocol={},
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        print_study_summary(result)
    output = mock_stdout.getvalue()
    assert "test-study" in output
    assert "abcd1234" in output


def test_print_study_progress():
    """print_study_progress produces a formatted line to stderr."""
    from io import StringIO
    from unittest.mock import patch

    from llenergymeasure.cli._display import print_study_progress
    from llenergymeasure.config.models import ExperimentConfig

    config = ExperimentConfig(model="test/model", backend="pytorch")
    with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
        print_study_progress(1, 4, config, status="completed", elapsed=30.5, energy=100.0)
    output = mock_stderr.getvalue()
    assert "[1/4]" in output
    assert "OK" in output
    assert "test/model" in output
