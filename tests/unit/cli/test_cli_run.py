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
    config.dtype = "bfloat16"
    config.dataset = MagicMock()
    config.dataset.source = "aienergyscore"
    config.dataset.n_prompts = 100
    config.dataset.order = "interleaved"
    config.output_dir = None
    config.max_input_tokens = 256
    config.max_output_tokens = 256
    config.pytorch = None
    config.baseline = MagicMock()
    config.baseline.enabled = False
    return config


# ---------------------------------------------------------------------------
# _build_header unit tests
# ---------------------------------------------------------------------------


def test_build_header_strips_hf_org_prefix():
    """_build_header strips the HuggingFace org prefix from model name."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "meta-llama/Llama-3.2-1B-Instruct"
    config.backend = "vllm"
    config.dtype = "bfloat16"
    config.dataset.n_prompts = 100

    header = _build_header(config, runner_tag="docker")
    assert "Llama-3.2-1B-Instruct" in header
    assert "meta-llama" not in header
    assert "[docker]" in header


def test_build_header_default_dtype_omitted():
    """_build_header omits dtype when it is the default 'bfloat16'."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "gpt2"
    config.backend = "pytorch"
    config.dtype = "bfloat16"  # default — should not appear
    config.dataset.n_prompts = 100

    header = _build_header(config, runner_tag="local")
    assert "bfloat16" not in header
    assert header == "gpt2 | pytorch [local]"


def test_build_header_nondefault_fields_shown():
    """_build_header includes dtype and n when non-default."""
    from llenergymeasure.cli.run import _build_header

    config = _make_mock_config()
    config.model = "gpt2"
    config.backend = "pytorch"
    config.dtype = "float16"
    config.dataset.n_prompts = 50

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
    # "pytorh" is a misspelled backend — Pydantic will raise ValidationError
    result = runner.invoke(app, ["run", "--model", "gpt2", "--backend", "pytorh"])
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
    exp.effective_config = {"model": "test/model", "dtype": "float16"}
    exp.backend = "pytorch"
    exp.duration_sec = 45.2
    exp.total_energy_j = 123.4
    exp.avg_tokens_per_second = 42.5
    exp.total_inference_time_sec = 40.0

    summary = StudySummary(
        total_experiments=1,
        completed=1,
        failed=0,
        total_wall_time_s=50.0,
        total_energy_j=123.4,
    )
    result = StudyResult.model_construct(
        experiments=[exp],
        study_name="test-study",
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

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.study._progress import print_study_progress

    config = ExperimentConfig(model="test/model", backend="pytorch")
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
        "name: test\nmodel: test/model\nbackend: pytorch\nsweep:\n  dtype: [float32, float16]\n"
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
        mock_load.return_value = MagicMock()
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
        "name: test\nexperiments:\n  - model: test/model-a\n    backend: pytorch\n  - model: test/model-b\n    backend: pytorch\n"
    )
    mock_study_result = make_study_result()

    with (
        patch("llenergymeasure.run_study", return_value=mock_study_result) as mock_run,
        patch("llenergymeasure.config.loader.load_study_config") as mock_load,
        patch("llenergymeasure.config.grid.build_preflight_panel"),
        patch("llenergymeasure.cli._display.print_study_summary"),
    ):
        mock_load.return_value = MagicMock()
        result = runner.invoke(app, ["run", str(study_yaml)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )
    mock_run.assert_called_once()


def test_run_saves_to_output_dir(tmp_path):
    """When output_dir is set on the config, save_result is called."""
    mock_config = _make_mock_config()
    mock_config.output_dir = str(tmp_path / "out")
    mock_result = _make_mock_result()
    mock_result.timeseries = None

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment", return_value=mock_result),
        patch("llenergymeasure.cli.run.print_result_summary"),
        patch("llenergymeasure.api.save_result") as mock_save_result,
    ):
        result = runner.invoke(app, ["run", "--model", "gpt2", "--output", str(tmp_path / "out")])

    assert result.exit_code == 0, f"Expected exit 0. Output: {result.output}"
    mock_save_result.assert_called_once()


def test_run_study_cli_defaults_applied(tmp_path):
    """Study YAML without execution block receives CLI effective defaults: n_cycles=3, experiment_order=shuffle."""
    from tests.conftest import make_study_result

    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        "name: test\nmodel: test/model\nbackend: pytorch\nsweep:\n  dtype: [float32, float16]\n"
    )
    mock_study_result = make_study_result()

    captured_overrides: list = []

    def _capture_load(path, cli_overrides=None):
        captured_overrides.append(cli_overrides)
        return MagicMock()

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


def test_run_backend_error_exits_1():
    """BackendError raised by run_experiment exits with code 1."""
    from llenergymeasure.utils.exceptions import BackendError

    mock_config = _make_mock_config()

    with (
        patch("llenergymeasure.cli.run.load_experiment_config", return_value=mock_config),
        patch("llenergymeasure.cli.run.run_experiment") as mock_run,
    ):
        mock_run.side_effect = BackendError("OOM during forward pass")
        result = runner.invoke(app, ["run", "--model", "gpt2"])

    assert result.exit_code == 1
    assert "BackendError" in result.output
