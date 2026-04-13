"""Unit tests for the llem run CLI command.

Tests use typer.testing.CliRunner to invoke the CLI without loading models or
touching GPU hardware. All heavy operations are mocked.

Note: typer's CliRunner routes all output (stdout + stderr) to .output.
Error messages printed to sys.stderr are captured in .output for assertions.
"""

from __future__ import annotations

import re
from pathlib import Path
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
    return result


def _make_mock_config() -> MagicMock:
    """Return a minimal mock ExperimentConfig."""
    from llenergymeasure.config.models import ExperimentConfig

    config = MagicMock(spec=ExperimentConfig)
    config.model = "gpt2"
    config.engine = "transformers"
    config.dtype = "bfloat16"
    config.dataset = MagicMock()
    config.dataset.source = "aienergyscore"
    config.dataset.n_prompts = 100
    config.dataset.order = "interleaved"
    config.max_input_tokens = 256
    config.max_output_tokens = 256
    config.transformers = None
    config.baseline = MagicMock()
    config.baseline.enabled = False
    return config


def _make_capture_load() -> tuple:
    """Return (capture_fn, captured_overrides) for study routing tests.

    The capture function mimics load_study_config: records cli_overrides
    and returns a MagicMock with properly configured study attributes.
    """
    captured: list = []

    def _capture(path, cli_overrides=None):
        captured.append(cli_overrides)
        mock = MagicMock()
        mock.experiments = [MagicMock()]
        mock.study_execution.n_cycles = 1
        mock.skipped_configs = []
        return mock

    return _capture, captured


# ---------------------------------------------------------------------------
# _build_header unit tests
# ---------------------------------------------------------------------------


def test_build_header_strips_hf_org_prefix():
    """_build_header strips the HuggingFace org prefix from model name."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "meta-llama/Llama-3.2-1B-Instruct"
    config.engine = "vllm"
    config.dtype = "bfloat16"
    config.dataset.n_prompts = 100  # default — should not appear

    header = _build_header(config, runner_tag="docker")
    assert "Llama-3.2-1B-Instruct" in header
    assert "meta-llama" not in header
    assert "[docker]" in header


def test_build_header_default_dtype_omitted():
    """_build_header omits dtype when it is the default 'bfloat16'."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "gpt2"
    config.engine = "transformers"
    config.dtype = "bfloat16"  # default — should not appear
    config.dataset.n_prompts = 100  # default — should not appear

    header = _build_header(config, runner_tag="local")
    assert "bfloat16" not in header
    assert header == "gpt2 | transformers [local]"


def test_build_header_nondefault_fields_shown():
    """_build_header includes dtype and n when non-default."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "gpt2"
    config.engine = "transformers"
    config.dtype = "float16"
    config.dataset.n_prompts = 50  # non-default — should appear

    header = _build_header(config, runner_tag="local")
    assert "float16" in header
    assert "n_prompts=50" in header


# ---------------------------------------------------------------------------
# Basic flag tests
# ---------------------------------------------------------------------------


def test_run_help():
    """llem run --help exits 0 and shows expected flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    assert "--model" in plain
    assert "--engine" in plain
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
    from llenergymeasure.utils.exceptions import ConfigError

    with patch("llenergymeasure.cli.run.load_experiment_config") as mock_load:
        mock_load.side_effect = ConfigError("bad config: unknown field 'foop'")
        result = runner.invoke(app, ["run", "nonexistent.yaml"])

    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "ConfigError" in result.output


def test_run_validation_error_exits_2():
    """Pydantic ValidationError from a bad field value exits with code 2."""
    # "pytorh" is a misspelled engine — Pydantic will raise ValidationError
    result = runner.invoke(app, ["run", "--model", "gpt2", "--engine", "pytorh"])
    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )
    assert "Config validation failed" in result.output


def test_run_preflight_error_exits_1():
    """PreFlightError raised by run_experiment exits with code 1."""
    from llenergymeasure.utils.exceptions import PreFlightError

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
    from llenergymeasure.utils.exceptions import ExperimentError

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
    """--quiet suppresses step progress display (progress=None passed to run_experiment)."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment", return_value=mock_result) as mock_run,
        patch("llenergymeasure.cli.run.print_result_summary"),
    ):
        result = runner.invoke(app, ["run", "--model", "gpt2", "--quiet"])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    # In quiet mode, progress callback must be None
    call_kwargs = mock_run.call_args
    assert call_kwargs is not None, "run_experiment was not called"
    assert call_kwargs.kwargs.get("progress") is None, "Expected progress=None in quiet mode"


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
        patch("llenergymeasure.cli.run.print_result_summary") as mock_summary,
    ):
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
  dtype: [float32, float16]
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
    """llem run --help output includes --cycles, --order, and --no-gaps flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    assert "--cycles" in plain
    assert "--order" in plain
    assert "--no-gaps" in plain


def test_print_study_summary_basic():
    """print_study_summary runs without error on a minimal StudyResult."""
    from io import StringIO
    from unittest.mock import MagicMock, patch

    from llenergymeasure.cli._display import print_study_summary
    from llenergymeasure.domain.experiment import StudyResult, StudySummary

    # Use model_construct to bypass Pydantic validation for the container —
    # experiments list contains a MagicMock, which is not a valid ExperimentResult.
    exp = MagicMock()
    exp.model_name = "test/model"
    exp.engine = "transformers"
    exp.duration_sec = 45.2
    exp.total_energy_j = 123.4
    exp.avg_tokens_per_second = 42.5
    exp.total_inference_time_sec = 40.0
    exp.energy_adjusted_j = None
    exp.mj_per_tok_adjusted = None
    exp.mj_per_tok_total = None

    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="test-study",
        study_design_hash="abcd1234",
        summary=StudySummary(
            total_experiments=1, completed=1, failed=0, total_wall_time_s=50.0, total_energy_j=123.4
        ),
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

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.study._progress import print_study_progress

    config = ExperimentConfig(model="test/model", engine="transformers")
    with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
        print_study_progress(1, 4, config, status="completed", elapsed=30.5, energy=100.0)
    output = mock_stderr.getvalue()
    assert "[1/4]" in output
    assert "OK" in output
    assert "test/model" in output


# ---------------------------------------------------------------------------
# Study routing tests — verify CLI actually invokes run_study for study YAMLs
# ---------------------------------------------------------------------------


def test_run_study_routing_sweep_yaml(tmp_path):
    """YAML with sweep: key routes to run_study via _run_study_impl."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()

    # run_study and load_study_config are lazily imported inside _run_study_impl;
    # patch at the source modules, not at llenergymeasure.cli.run
    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.config.loader.load_study_config") as mock_load,
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        mock_config = MagicMock()
        mock_config.experiments = [MagicMock(), MagicMock()]
        mock_config.study_execution.n_cycles = 1
        mock_config.skipped_configs = []
        mock_load.return_value = mock_config
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_run.assert_called_once()


def test_run_study_routing_experiments_yaml(tmp_path):
    """YAML with experiments: key routes to run_study."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nexperiments:\n  - model: test/model-a\n    engine: transformers\n  - model: test/model-b\n    engine: transformers\n"
    )
    mock_study_result = make_study_result()

    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.config.loader.load_study_config") as mock_load,
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        mock_config = MagicMock()
        mock_config.experiments = [MagicMock(), MagicMock()]
        mock_config.study_execution.n_cycles = 1
        mock_config.skipped_configs = []
        mock_load.return_value = mock_config
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_run.assert_called_once()


def test_run_saves_to_output_dir(tmp_path):
    """When --output CLI flag is passed, run_experiment receives output_dir."""
    mock_config = _make_mock_config()
    mock_result = _make_mock_result()
    mock_result.timeseries = None
    output_dir = tmp_path / "out"

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment", return_value=mock_result) as mock_run,
        patch("llenergymeasure.cli.run.print_result_summary"),
    ):
        result = runner.invoke(app, ["run", "--model", "gpt2", "--output", str(output_dir)])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs.get("output_dir") == str(output_dir)


def test_run_study_cli_defaults_applied(tmp_path):
    """Study YAML without execution block receives CLI effective defaults: n_cycles=3, experiment_order=shuffle."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()
    _capture_load, captured_overrides = _make_capture_load()

    # load_study_config, run_study, and build_preflight_panel are all lazily
    # imported inside _run_study_impl — patch at source modules
    with (
        patch("llenergymeasure.config.loader.load_study_config", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0
    assert len(captured_overrides) == 1
    overrides = captured_overrides[0]
    assert overrides is not None
    assert overrides["study_execution"]["n_cycles"] == 3
    assert overrides["study_execution"]["experiment_order"] == "shuffle"


def test_run_no_model_no_config_error_message():
    """Error message for missing model/config mentions 'Provide a config file or --model flag'."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "Provide a config file or --model flag" in result.output


def test_run_engine_error_exits_1():
    """EngineError raised by run_experiment exits with code 1."""
    from llenergymeasure.utils.exceptions import EngineError

    mock_config = _make_mock_config()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment") as mock_run,
    ):
        mock_run.side_effect = EngineError("OOM during forward pass")
        result = runner.invoke(app, ["run", "--model", "gpt2"])

    assert result.exit_code == 1
    assert "EngineError" in result.output


# ---------------------------------------------------------------------------
# New robustness flag tests (Phase 40.2)
# ---------------------------------------------------------------------------


def _make_study_yaml(tmp_path, content: str | None = None) -> Path:
    """Write a minimal study YAML to tmp_path and return its path."""

    study_yaml = tmp_path / "study.yaml"
    if content is None:
        content = "name: test\nmodel: test/model\nengine: transformers\nsweep:\n  dtype: [float32, float16]\n"
    study_yaml.write_text(content)
    return study_yaml


def _make_mock_study_result():
    """Return a minimal mock StudyResult."""
    from tests.conftest import make_study_result

    return make_study_result()


def test_fail_fast_sets_max_consecutive_failures(tmp_path):
    """--fail-fast sets max_consecutive_failures=1 in study_execution overrides."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    _capture_load, captured_overrides = _make_capture_load()

    with (
        patch("llenergymeasure.config.loader.load_study_config", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--fail-fast"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    overrides = captured_overrides[0]
    assert overrides is not None
    assert overrides["study_execution"]["max_consecutive_failures"] == 1
    assert overrides["study_execution"]["circuit_breaker_cooldown_seconds"] == 0


def test_no_circuit_breaker_sets_max_failures_zero(tmp_path):
    """--no-circuit-breaker sets max_consecutive_failures=0 in study_execution overrides."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    _capture_load, captured_overrides = _make_capture_load()

    with (
        patch("llenergymeasure.config.loader.load_study_config", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--no-circuit-breaker"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    overrides = captured_overrides[0]
    assert overrides is not None
    assert overrides["study_execution"]["max_consecutive_failures"] == 0


def test_timeout_flag_sets_wall_clock_timeout(tmp_path):
    """--timeout 24 sets wall_clock_timeout_hours=24.0 in study_execution overrides."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    _capture_load, captured_overrides = _make_capture_load()

    with (
        patch("llenergymeasure.config.loader.load_study_config", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--timeout", "24"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    overrides = captured_overrides[0]
    assert overrides is not None
    assert overrides["study_execution"]["wall_clock_timeout_hours"] == 24.0


def test_timeout_flag_fractional(tmp_path):
    """--timeout 1.5 sets wall_clock_timeout_hours=1.5."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    _capture_load, captured_overrides = _make_capture_load()

    with (
        patch("llenergymeasure.config.loader.load_study_config", side_effect=_capture_load),
        patch("llenergymeasure.run_study", return_value=mock_study_result),
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--timeout", "1.5"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    overrides = captured_overrides[0]
    assert overrides["study_execution"]["wall_clock_timeout_hours"] == 1.5


def test_resume_flag_passes_resume_to_api(tmp_path):
    """--resume flag passes resume=True to run_study (API handles auto-detect)."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    mock_study_config = MagicMock()
    mock_study_config.study_design_hash = "abc123"
    mock_study_config.skipped_configs = []
    mock_study_config.experiments = [MagicMock()]
    mock_study_config.study_name = "test"
    mock_study_config.study_execution = MagicMock()
    mock_study_config.study_execution.n_cycles = 3

    with (
        patch("llenergymeasure.config.loader.load_study_config", return_value=mock_study_config),
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
        patch(
            "llenergymeasure.api.find_resumable_study",
            return_value=tmp_path / "fake-study",
        ),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--resume"])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["resume"] is True


def test_resume_dir_flag_passes_path_to_api(tmp_path):
    """--resume-dir passes the explicit directory to run_study."""
    study_yaml = _make_study_yaml(tmp_path)
    mock_study_result = _make_mock_study_result()
    mock_study_config = MagicMock()
    mock_study_config.study_design_hash = "abc123"
    mock_study_config.skipped_configs = []
    mock_study_config.experiments = [MagicMock()]
    mock_study_config.study_name = "test"
    mock_study_config.study_execution = MagicMock()
    mock_study_config.study_execution.n_cycles = 3

    explicit_dir = tmp_path / "my_study"
    explicit_dir.mkdir()
    (explicit_dir / "manifest.json").write_text("{}")

    with (
        patch("llenergymeasure.config.loader.load_study_config", return_value=mock_study_config),
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        result = runner.invoke(app, ["run", str(study_yaml), "--resume-dir", str(explicit_dir)])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["resume_dir"] == explicit_dir


def test_new_flags_visible_in_help():
    """--resume, --fail-fast, --no-circuit-breaker, --timeout, --no-lock appear in --help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    plain = _strip_ansi(result.output)
    assert "--resume" in plain
    assert "--fail-fast" in plain
    assert "--no-circuit-breaker" in plain
    assert "--timeout" in plain
    assert "--no-lock" in plain
