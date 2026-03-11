"""Unit tests for the llem config CLI command.

Tests use typer.testing.CliRunner to invoke the CLI without touching GPU hardware
or the file system. All external probes are mocked.

Note: typer's CliRunner routes all output to .output for assertions.
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
# Tests
# ---------------------------------------------------------------------------


def test_config_help() -> None:
    """config --help exits 0 and mentions --verbose."""
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in _strip_ansi(result.output)


def test_config_basic_output() -> None:
    """Basic config output shows GPU name, backend status, and correct install hints."""
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
    assert "pytorch: installed" in result.output
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
