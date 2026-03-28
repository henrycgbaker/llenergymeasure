"""User preferences configuration loading.

Loads optional user preferences from ~/.config/llenergymeasure/config.yaml
(XDG-compliant path via platformdirs). Missing file silently applies all
defaults. Invalid YAML or schema raises ConfigError.

Precedence (low → high):
  built-in defaults < config file < env vars < CLI flags (handled elsewhere)
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from llenergymeasure.config.models import EnergySamplerName


class UserOutputConfig(BaseModel):
    """Output path preferences."""

    model_config = {"extra": "forbid"}

    results_dir: str = Field(default="./results", description="Default results output location")
    model_cache_dir: str = Field(
        default="~/.cache/huggingface", description="HuggingFace model cache"
    )


class UserRunnersConfig(BaseModel):
    """Runner selection per backend."""

    model_config = {"extra": "forbid"}

    pytorch: str = Field(
        default="auto",
        description="PyTorch runner: 'auto', 'local', 'docker' (built-in image), or 'docker:<image>'",
    )
    vllm: str = Field(
        default="auto",
        description="vLLM runner: 'auto', 'local', 'docker' (built-in image), or 'docker:<image>'",
    )
    tensorrt: str = Field(
        default="auto",
        description="TensorRT runner: 'auto', 'local', 'docker' (built-in image), or 'docker:<image>'",
    )

    @model_validator(mode="after")
    def validate_runner_format(self) -> UserRunnersConfig:
        for field_name in ("pytorch", "vllm", "tensorrt"):
            value = getattr(self, field_name)
            if value.startswith("singularity:"):
                raise ValueError(
                    f"Singularity runner not yet supported (runners.{field_name}='{value}'). "
                    "Use 'auto', 'local', 'docker', or 'docker:<image>'."
                )
            if (
                value != "auto"
                and value != "local"
                and value != "docker"
                and not value.startswith("docker:")
            ):
                raise ValueError(
                    f"runners.{field_name}: expected 'auto', 'local', 'docker', or 'docker:<image>', "
                    f"got '{value}'"
                )
        return self


class UserMeasurementConfig(BaseModel):
    """Energy measurement preferences."""

    model_config = {"extra": "forbid"}

    energy_sampler: EnergySamplerName = Field(
        default="auto", description="Energy sampler: auto=best available (Zeus>NVML>CodeCarbon)"
    )
    carbon_intensity_gco2_kwh: float | None = Field(
        default=None, ge=0.0, description="gCO2/kWh for local electricity grid"
    )
    datacenter_pue: float = Field(default=1.0, ge=1.0, description="Power Usage Effectiveness")


class UserUIConfig(BaseModel):
    """User interface preferences."""

    model_config = {"extra": "forbid"}

    log_level: Literal["WARNING", "INFO", "DEBUG"] = Field(default="WARNING")
    prompt: bool = Field(default=True, description="Enable interactive prompts (False for CI/HPC)")
    progress_mode: Literal["auto", "plain", "quiet"] = Field(
        default="auto",
        description="Progress output mode: auto=Rich Live TTY, plain=sequential print, quiet=silent",
    )


class UserExecutionConfig(BaseModel):
    """Execution gap preferences (machine-local thermal defaults)."""

    model_config = {"extra": "forbid"}

    experiment_gap_seconds: float = Field(
        default=60.0, ge=0.0, description="Thermal gap between experiments"
    )
    cycle_gap_seconds: float = Field(
        default=300.0, ge=0.0, description="Thermal gap between cycles"
    )


class UserConfig(BaseModel):
    """User preferences loaded from ~/.config/llenergymeasure/config.yaml.

    All fields are optional — missing file or missing fields fall back to
    built-in defaults. Invalid values raise ConfigError via load_user_config().
    """

    model_config = {"extra": "forbid"}

    output: UserOutputConfig = Field(default_factory=UserOutputConfig)
    runners: UserRunnersConfig = Field(default_factory=UserRunnersConfig)
    images: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-backend Docker image overrides (orthogonal to runners). "
            "Keys are backend names, values are image references. "
            "Empty dict = use smart default (local build → registry fallback)."
        ),
    )
    measurement: UserMeasurementConfig = Field(default_factory=UserMeasurementConfig)
    ui: UserUIConfig = Field(default_factory=UserUIConfig)
    execution: UserExecutionConfig = Field(default_factory=UserExecutionConfig)


def get_user_config_path() -> Path:
    """Return the XDG-compliant user config path.

    Linux:   ~/.config/llenergymeasure/config.yaml
    macOS:   ~/Library/Application Support/llenergymeasure/config.yaml
    Windows: %APPDATA%\\llenergymeasure\\config.yaml
    """
    from platformdirs import user_config_dir

    return Path(user_config_dir("llenergymeasure")) / "config.yaml"


def _apply_env_overrides(config: UserConfig) -> UserConfig:
    """Apply LLEM_* environment variable overrides to user config.

    Env vars sit above config file and below CLI flags in precedence.
    Returns updated UserConfig (Pydantic models are immutable; use model_copy).
    """
    runners_updates: dict[str, str] = {}
    for backend in ("pytorch", "vllm", "tensorrt"):
        env_key = f"LLEM_RUNNER_{backend.upper()}"
        if val := os.environ.get(env_key):
            runners_updates[backend] = val

    measurement_updates: dict[str, Any] = {}
    if val := os.environ.get("LLEM_CARBON_INTENSITY"):
        with contextlib.suppress(ValueError):
            measurement_updates["carbon_intensity_gco2_kwh"] = float(val)
    if val := os.environ.get("LLEM_DATACENTER_PUE"):
        with contextlib.suppress(ValueError):
            measurement_updates["datacenter_pue"] = float(val)

    ui_updates: dict[str, Any] = {}
    if os.environ.get("LLEM_NO_PROMPT"):
        ui_updates["prompt"] = False

    # Apply updates using model_copy(update=...) for immutable Pydantic models
    new_runners = (
        config.runners.model_copy(update=runners_updates) if runners_updates else config.runners
    )
    new_measurement = (
        config.measurement.model_copy(update=measurement_updates)
        if measurement_updates
        else config.measurement
    )
    new_ui = config.ui.model_copy(update=ui_updates) if ui_updates else config.ui

    if runners_updates or measurement_updates or ui_updates:
        return config.model_copy(
            update={
                "runners": new_runners,
                "measurement": new_measurement,
                "ui": new_ui,
            }
        )
    return config


def load_user_config(config_path: Path | None = None) -> UserConfig:
    """Load user configuration from ~/.config/llenergymeasure/config.yaml.

    Missing file: silently applies all defaults — no error.
    Invalid YAML: raises ConfigError with parse error detail.
    Invalid schema: raises ConfigError with field path context.

    Args:
        config_path: Explicit path override (for testing). None = XDG default.

    Returns:
        UserConfig with file values merged over defaults, env vars applied on top.
    """
    from llenergymeasure.utils.exceptions import ConfigError

    path = config_path or get_user_config_path()

    if not path.exists():
        # Missing file — zero-config, apply all defaults + env var overrides
        return _apply_env_overrides(UserConfig())

    try:
        content = path.read_text()
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"User config must be a YAML mapping: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in user config {path}: {e}") from e

    try:
        config = UserConfig.model_validate(data)
    except ValidationError as e:
        # Format Pydantic errors as ConfigError with field paths for researcher clarity
        errors = [f"  {err['loc']}: {err['msg']}" for err in e.errors()]
        raise ConfigError(f"Invalid user config {path}:\n" + "\n".join(errors)) from e

    return _apply_env_overrides(config)


__all__ = ["UserConfig", "get_user_config_path", "load_user_config"]
