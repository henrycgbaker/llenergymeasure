"""Unit tests for user configuration loading (llenergymeasure.config.user_config).

Tests XDG path, missing file graceful defaults, valid file loading, and
env var overrides.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.config.user_config import UserConfig, get_user_config_path, load_user_config

# ---------------------------------------------------------------------------
# get_user_config_path
# ---------------------------------------------------------------------------


def test_get_user_config_path_returns_path():
    """get_user_config_path() returns a Path object."""
    path = get_user_config_path()
    assert isinstance(path, Path)


def test_get_user_config_path_ends_with_config_yaml():
    """get_user_config_path() ends with 'config.yaml'."""
    path = get_user_config_path()
    assert path.name == "config.yaml"


def test_get_user_config_path_contains_llenergymeasure():
    """get_user_config_path() includes 'llenergymeasure' in the path."""
    path = get_user_config_path()
    assert "llenergymeasure" in str(path)


# ---------------------------------------------------------------------------
# Missing file → defaults
# ---------------------------------------------------------------------------


def test_load_user_config_missing_file_returns_defaults(tmp_path):
    """load_user_config() with nonexistent file returns UserConfig with all defaults."""
    nonexistent = tmp_path / "nonexistent.yaml"
    config = load_user_config(config_path=nonexistent)
    assert isinstance(config, UserConfig)
    # Default values from the model
    assert config.output.results_dir == "./results"
    assert config.measurement.energy_backend == "auto"
    assert config.ui.verbosity == "standard"


def test_load_user_config_missing_file_no_error(tmp_path):
    """load_user_config() with missing file does not raise any exception."""
    nonexistent = tmp_path / "missing.yaml"
    # Should not raise
    config = load_user_config(config_path=nonexistent)
    assert config is not None


# ---------------------------------------------------------------------------
# Valid file loading
# ---------------------------------------------------------------------------


def test_load_user_config_valid_file(tmp_path):
    """load_user_config() with valid YAML returns UserConfig with overridden values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("output:\n  results_dir: /custom/results\nui:\n  verbosity: verbose\n")
    config = load_user_config(config_path=config_file)
    assert config.output.results_dir == "/custom/results"
    assert config.ui.verbosity == "verbose"


def test_load_user_config_partial_file_uses_defaults_for_missing(tmp_path):
    """Partial user config file merges with defaults for unspecified fields."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ui:\n  verbosity: quiet\n")
    config = load_user_config(config_path=config_file)
    assert config.ui.verbosity == "quiet"
    # Unspecified fields retain defaults
    assert config.output.results_dir == "./results"


def test_load_user_config_energy_backend_override(tmp_path):
    """User config can override energy backend."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("measurement:\n  energy_backend: nvml\n")
    config = load_user_config(config_path=config_file)
    assert config.measurement.energy_backend == "nvml"


def test_load_user_config_advanced_section(tmp_path):
    """User config can set advanced NVML poll interval."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("advanced:\n  nvml_poll_interval_ms: 200\n")
    config = load_user_config(config_path=config_file)
    assert config.advanced.nvml_poll_interval_ms == 200


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


def test_env_var_llem_carbon_intensity(tmp_path, monkeypatch):
    """LLEM_CARBON_INTENSITY env var sets carbon_intensity_gco2_kwh."""
    monkeypatch.setenv("LLEM_CARBON_INTENSITY", "450.0")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.measurement.carbon_intensity_gco2_kwh == pytest.approx(450.0)


def test_env_var_llem_datacenter_pue(tmp_path, monkeypatch):
    """LLEM_DATACENTER_PUE env var sets datacenter_pue."""
    monkeypatch.setenv("LLEM_DATACENTER_PUE", "1.5")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.measurement.datacenter_pue == pytest.approx(1.5)


def test_env_var_llem_no_prompt(tmp_path, monkeypatch):
    """LLEM_NO_PROMPT env var disables interactive prompts."""
    monkeypatch.setenv("LLEM_NO_PROMPT", "1")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.ui.prompt is False


def test_env_var_runner_pytorch(tmp_path, monkeypatch):
    """LLEM_RUNNER_PYTORCH env var overrides pytorch runner."""
    monkeypatch.setenv("LLEM_RUNNER_PYTORCH", "docker:nvcr.io/nvidia/pytorch:latest")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.runners.pytorch == "docker:nvcr.io/nvidia/pytorch:latest"


def test_env_var_overrides_take_precedence_over_file(tmp_path, monkeypatch):
    """Env var overrides take precedence over config file values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("measurement:\n  datacenter_pue: 1.2\n")
    monkeypatch.setenv("LLEM_DATACENTER_PUE", "1.8")
    config = load_user_config(config_path=config_file)
    assert config.measurement.datacenter_pue == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Silent ignore of invalid env var values
# ---------------------------------------------------------------------------


def test_silent_ignore_invalid_float_carbon_intensity(tmp_path, monkeypatch):
    """LLEM_CARBON_INTENSITY='not_a_number' is silently ignored (treated as not set)."""
    monkeypatch.setenv("LLEM_CARBON_INTENSITY", "not_a_number")
    # Should not raise
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    # Value is not set (treated as None — same as not providing the env var)
    assert config.measurement.carbon_intensity_gco2_kwh is None


def test_silent_ignore_invalid_float_datacenter_pue(tmp_path, monkeypatch):
    """LLEM_DATACENTER_PUE='abc' is silently ignored (treated as not set)."""
    monkeypatch.setenv("LLEM_DATACENTER_PUE", "abc")
    # Should not raise — value is silently ignored
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    # Falls back to default when invalid
    assert config.measurement.datacenter_pue == pytest.approx(1.0)
