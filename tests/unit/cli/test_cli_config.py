"""Unit tests for the llem config CLI command.

Tests use typer.testing.CliRunner to invoke the CLI without touching GPU hardware
or the file system. All external probes are mocked.

Note: typer's CliRunner routes all output to .output for assertions.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from llenergymeasure.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_config_help() -> None:
    """config --help exits 0 and mentions --verbose."""
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in _strip_ansi(result.output)


def test_config_basic_output() -> None:
    """Basic config output shows GPU name, engine status, and correct install hints."""
    mock_gpu = [{"name": "NVIDIA A100", "vram_gb": 80.0}]

    def fake_find_spec(name: str) -> MagicMock | None:
        # transformers (pytorch) installed; vllm and tensorrt_llm not
        if name == "transformers":
            return MagicMock()
        return None

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=mock_gpu),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "GPU" in result.output
    assert "A100" in result.output
    assert "transformers: installed" in result.output
    assert "vllm: not installed" in result.output


def test_config_no_gpu() -> None:
    """When _probe_gpu returns None, output contains 'No GPU detected'."""
    with patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "No GPU detected" in result.output


def test_config_verbose_shows_python_version() -> None:
    """--verbose output contains Python and a version number."""
    with patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None):
        result = runner.invoke(app, ["config", "--verbose"])

    assert result.exit_code == 0
    assert "Python" in result.output
    # Version string like "3.12.1" must appear somewhere after the Python header
    import sys

    short_version = sys.version.split()[0]
    assert short_version in result.output


def test_config_user_config_path_shown() -> None:
    """User config path is printed in the Config section."""
    fake_path = Path("/home/user/.config/llenergymeasure/config.yaml")
    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("llenergymeasure.config.user_config.get_user_config_path", return_value=fake_path),
    ):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert str(fake_path) in result.output


def test_config_exits_0() -> None:
    """Config command always exits 0 regardless of environment state."""
    with patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None):
        result = runner.invoke(app, ["config"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Verbose mode tests
# ---------------------------------------------------------------------------


def test_config_verbose_gpu_driver() -> None:
    """With -v and a GPU present, the driver version is shown."""
    mock_gpu = [{"name": "NVIDIA A100", "vram_gb": 80.0}]
    mock_driver_raw = b"535.129.03"

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlSystemGetDriverVersion.return_value = mock_driver_raw
    mock_pynvml.__spec__ = MagicMock()  # Must be non-None — sys.modules insertion requires it

    mock_nvml_context = MagicMock()
    mock_nvml_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_nvml_context.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=mock_gpu),
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch("llenergymeasure.device.gpu_info.nvml_context", mock_nvml_context),
    ):
        result = runner.invoke(app, ["config", "-v"])

    assert result.exit_code == 0
    assert "Driver: 535.129.03" in result.output


def test_config_verbose_engine_versions() -> None:
    """With -v and transformers installed, version string is shown in parentheses."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.2.0"

    def fake_find_spec(name: str) -> MagicMock | None:
        if name in ("transformers", "pynvml"):
            return MagicMock()
        return None

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("llenergymeasure.cli.config_cmd._probe_engine_version", return_value="2.2.0"),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        result = runner.invoke(app, ["config", "-v"])

    assert result.exit_code == 0
    assert "(2.2.0)" in result.output


# ---------------------------------------------------------------------------
# Energy sampler detection
# ---------------------------------------------------------------------------


def test_config_energy_samplers_zeus() -> None:
    """When zeus is installed, Energy section shows 'zeus'."""

    def fake_find_spec_zeus(name: str) -> MagicMock | None:
        if name == "zeus":
            return MagicMock()
        return None

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("importlib.util.find_spec", side_effect=fake_find_spec_zeus),
    ):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Sampler: zeus" in result.output


def test_config_energy_samplers_none() -> None:
    """When no energy samplers are installed, Energy section shows 'Sampler: none'."""

    def fake_find_spec_none(name: str) -> None:
        return None

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("importlib.util.find_spec", side_effect=fake_find_spec_none),
    ):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Sampler: none" in result.output


# ---------------------------------------------------------------------------
# User config verbose non-defaults / not found
# ---------------------------------------------------------------------------


def test_config_user_config_not_found() -> None:
    """When config file does not exist, Status shows 'using defaults'."""
    fake_path = Path("/nonexistent/.config/llenergymeasure/config.yaml")

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("llenergymeasure.config.user_config.get_user_config_path", return_value=fake_path),
    ):
        result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "using defaults" in result.output


def test_config_user_config_loaded_verbose_non_defaults() -> None:
    """With -v and a loaded config that has non-default values, 'Non-default values:' appears."""
    from pathlib import Path as _Path

    from llenergymeasure.config.user_config import UserConfig

    # Build a UserConfig with a non-default value
    defaults_cfg = UserConfig()
    defaults_dump = defaults_cfg.model_dump()

    # We need a config object that differs from defaults -- create a mock
    mock_user_cfg = MagicMock(spec=UserConfig)
    modified_dump = {section: dict(vals) for section, vals in defaults_dump.items()}
    # Inject a non-default value in the first section we find that has dict values
    for section, vals in modified_dump.items():
        if isinstance(vals, dict) and vals:
            first_key = next(iter(vals))
            modified_dump[section][first_key] = "__non_default_value__"
            break
    mock_user_cfg.model_dump.return_value = modified_dump

    fake_path = MagicMock(spec=_Path)
    fake_path.exists.return_value = True
    fake_path.__str__ = lambda self: "/fake/config.yaml"

    # load_user_config and UserConfig are lazy-imported inside the function body —
    # patch at the source module, not at config_cmd
    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None),
        patch("llenergymeasure.config.user_config.get_user_config_path", return_value=fake_path),
        patch("llenergymeasure.config.user_config.load_user_config", return_value=mock_user_cfg),
        patch("llenergymeasure.config.user_config.UserConfig", return_value=defaults_cfg),
    ):
        result = runner.invoke(app, ["config", "-v"])

    assert result.exit_code == 0
    assert "Non-default values:" in result.output


# ---------------------------------------------------------------------------
# Direct helper function tests
# ---------------------------------------------------------------------------


def test_probe_gpu_returns_list_with_gpu_info() -> None:
    """_probe_gpu returns a list of dicts when pynvml works correctly."""
    from types import SimpleNamespace

    from llenergymeasure.cli.config_cmd import _probe_gpu

    mock_pynvml = MagicMock()
    mock_pynvml.__spec__ = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA A100"
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = SimpleNamespace(total=80_000_000_000)

    mock_nvml_context = MagicMock()
    mock_nvml_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_nvml_context.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch("llenergymeasure.device.gpu_info.nvml_context", mock_nvml_context),
    ):
        result = _probe_gpu()

    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "NVIDIA A100"
    assert result[0]["vram_gb"] == pytest.approx(80.0, rel=1e-3)


def test_probe_gpu_returns_none_on_error() -> None:
    """_probe_gpu returns None when pynvml raises an exception."""
    from llenergymeasure.cli.config_cmd import _probe_gpu

    with patch.dict("sys.modules", {"pynvml": None}):
        result = _probe_gpu()

    assert result is None


def test_probe_engine_version_transformers() -> None:
    """_probe_engine_version returns torch.__version__ for transformers engine."""
    from llenergymeasure.cli.config_cmd import _probe_engine_version

    mock_torch = MagicMock()
    mock_torch.__version__ = "2.2.0"

    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = _probe_engine_version("transformers")

    assert result == "2.2.0"


def test_probe_engine_version_returns_none_on_import_error() -> None:
    """_probe_engine_version returns None when the engine package is not installed."""
    from llenergymeasure.cli.config_cmd import _probe_engine_version

    with patch.dict("sys.modules", {"vllm": None}):
        result = _probe_engine_version("vllm")

    assert result is None


def test_config_verbose_driver_exception_handled() -> None:
    """With -v and driver query raising, command still exits 0 (exception suppressed)."""
    mock_gpu = [{"name": "NVIDIA A100", "vram_gb": 80.0}]

    mock_pynvml = MagicMock()
    mock_pynvml.__spec__ = MagicMock()
    mock_pynvml.nvmlSystemGetDriverVersion.side_effect = RuntimeError("driver error")

    mock_nvml_context = MagicMock()
    mock_nvml_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_nvml_context.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=mock_gpu),
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch("llenergymeasure.device.gpu_info.nvml_context", mock_nvml_context),
    ):
        result = runner.invoke(app, ["config", "-v"])

    # Driver exception is swallowed — command must still succeed
    assert result.exit_code == 0


def test_config_cmd_renders_without_rich() -> None:
    """config_cmd renders correctly when rich is absent from sys.modules.

    If config_cmd imported rich at the top-level, removing it from sys.modules
    and importing the module would raise ImportError. The command must run
    without rich available.
    """
    import sys

    # Temporarily hide rich from sys.modules to verify config_cmd does not
    # depend on it at module load or execution time.
    rich_modules = {k: v for k, v in sys.modules.items() if k == "rich" or k.startswith("rich.")}
    for k in rich_modules:
        sys.modules.pop(k)

    try:
        with patch("llenergymeasure.cli.config_cmd._probe_gpu", return_value=None):
            result = runner.invoke(app, ["config"])
        assert result.exit_code == 0, f"config command failed: {result.output}"
    finally:
        sys.modules.update(rich_modules)
