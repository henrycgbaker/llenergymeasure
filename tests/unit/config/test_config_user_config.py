"""Unit tests for user configuration loading (llenergymeasure.config.user_config).

Tests XDG path, missing file graceful defaults, valid file loading, and
env var overrides.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.config.ssot import (
    ENV_CARBON_INTENSITY,
    ENV_DATACENTER_PUE,
    ENV_NO_PROMPT,
    ENV_RUNNER_PREFIX,
)
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
    assert config.measurement.energy_sampler == "auto"
    assert config.ui.log_level == "WARNING"


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
    config_file.write_text("output:\n  results_dir: /custom/results\nui:\n  log_level: DEBUG\n")
    config = load_user_config(config_path=config_file)
    assert config.output.results_dir == "/custom/results"
    assert config.ui.log_level == "DEBUG"


def test_load_user_config_partial_file_uses_defaults_for_missing(tmp_path):
    """Partial user config file merges with defaults for unspecified fields."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ui:\n  log_level: INFO\n")
    config = load_user_config(config_path=config_file)
    assert config.ui.log_level == "INFO"
    # Unspecified fields retain defaults
    assert config.output.results_dir == "./results"


def test_load_user_config_energy_sampler_override(tmp_path):
    """User config can override energy sampler preference."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("measurement:\n  energy_sampler: nvml\n")
    config = load_user_config(config_path=config_file)
    assert config.measurement.energy_sampler == "nvml"


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


def test_env_var_llem_carbon_intensity(tmp_path, monkeypatch):
    """LLEM_CARBON_INTENSITY env var sets carbon_intensity_gco2_kwh."""
    monkeypatch.setenv(ENV_CARBON_INTENSITY, "450.0")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.measurement.carbon_intensity_gco2_kwh == pytest.approx(450.0)


def test_env_var_llem_datacenter_pue(tmp_path, monkeypatch):
    """LLEM_DATACENTER_PUE env var sets datacenter_pue."""
    monkeypatch.setenv(ENV_DATACENTER_PUE, "1.5")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.measurement.datacenter_pue == pytest.approx(1.5)


def test_env_var_llem_no_prompt(tmp_path, monkeypatch):
    """LLEM_NO_PROMPT env var disables interactive prompts."""
    monkeypatch.setenv(ENV_NO_PROMPT, "1")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.ui.prompt is False


def test_env_var_runner_transformers(tmp_path, monkeypatch):
    """LLEM_RUNNER_TRANSFORMERS env var overrides transformers runner."""
    monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "docker:nvcr.io/nvidia/pytorch:latest")
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    assert config.runners.transformers == "docker:nvcr.io/nvidia/pytorch:latest"


def test_env_var_overrides_take_precedence_over_file(tmp_path, monkeypatch):
    """Env var overrides take precedence over config file values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("measurement:\n  datacenter_pue: 1.2\n")
    monkeypatch.setenv(ENV_DATACENTER_PUE, "1.8")
    config = load_user_config(config_path=config_file)
    assert config.measurement.datacenter_pue == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Silent ignore of invalid env var values
# ---------------------------------------------------------------------------


def test_silent_ignore_invalid_float_carbon_intensity(tmp_path, monkeypatch):
    """LLEM_CARBON_INTENSITY='not_a_number' is silently ignored (treated as not set)."""
    monkeypatch.setenv(ENV_CARBON_INTENSITY, "not_a_number")
    # Should not raise
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    # Value is not set (treated as None — same as not providing the env var)
    assert config.measurement.carbon_intensity_gco2_kwh is None


def test_silent_ignore_invalid_float_datacenter_pue(tmp_path, monkeypatch):
    """LLEM_DATACENTER_PUE='abc' is silently ignored (treated as not set)."""
    monkeypatch.setenv(ENV_DATACENTER_PUE, "abc")
    # Should not raise — value is silently ignored
    config = load_user_config(config_path=tmp_path / "nonexistent.yaml")
    # Falls back to default when invalid
    assert config.measurement.datacenter_pue == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UserRunnersConfig "auto" default behaviour
# ---------------------------------------------------------------------------


def test_user_runners_config_defaults_to_auto():
    """UserRunnersConfig() with no args has all three runner fields default to 'auto'."""
    from llenergymeasure.config.user_config import UserRunnersConfig

    config = UserRunnersConfig()
    assert config.transformers == "auto"
    assert config.vllm == "auto"
    assert config.tensorrt == "auto"


def test_user_runners_config_accepts_auto_in_file(tmp_path):
    """Config file with runners.transformers: auto loads without error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("runners:\n  transformers: auto\n")
    config = load_user_config(config_path=config_file)
    assert config.runners.transformers == "auto"


def test_user_runners_config_validator_accepts_auto():
    """UserRunnersConfig(transformers='auto') does not raise ValidationError."""
    from llenergymeasure.config.user_config import UserRunnersConfig

    config = UserRunnersConfig(transformers="auto")
    assert config.transformers == "auto"
