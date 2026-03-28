"""Runner resolution — determine local vs Docker execution mode for each backend.

Precedence chain (highest wins):
  env var > study/experiment YAML > user config > auto-detection > default

Auto-detection: if Docker + NVIDIA Container Toolkit are available on the host,
default to Docker for best measurement isolation. Otherwise fall back to local
with a nudge message.

User config: non-"auto" values in UserRunnersConfig are treated as explicit.
"auto" (the default) falls through to auto-detection, allowing Docker to be
picked up automatically when available. Pass ``user_config=None`` to skip
the user config step entirely.

This module is intentionally free of Docker dispatch mechanics — it only decides
*what* should run *where*. Dispatch is handled by DockerRunner (Plan 03).
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserRunnersConfig

from llenergymeasure.config.ssot import RUNNER_DOCKER, RUNNER_LOCAL, RunnerMode

# Re-exported from image_registry for convenience — parse_runner_value is defined
# there (canonical home) but used heavily in this module and its tests.
from llenergymeasure.infra.docker_preflight import _NVIDIA_TOOLKIT_BINS
from llenergymeasure.infra.image_registry import parse_runner_value

__all__ = [
    "RunnerSpec",
    "is_docker_available",
    "parse_runner_value",
    "resolve_runner",
    "resolve_study_runners",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env file loading
# ---------------------------------------------------------------------------


@functools.cache
def _load_dotenv() -> None:
    """Load ``.env`` from the working directory if present.

    Uses ``override=False`` so shell environment variables always win.
    Cached so the filesystem scan happens at most once per process.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Docker availability detection
# ---------------------------------------------------------------------------


@functools.cache
def is_docker_available() -> bool:
    """Return True if Docker CLI and NVIDIA Container Toolkit are both on PATH.

    This is a quick host-level check (PATH inspection only). Container-level GPU
    validation is done at pre-flight time.

    Checks:
        1. ``docker`` CLI is on PATH (via shutil.which)
        2. At least one NVIDIA Container Toolkit binary is on PATH:
           ``nvidia-container-runtime``, ``nvidia-ctk``, or ``nvidia-container-cli``
    """
    if shutil.which("docker") is None:
        return False

    return any(shutil.which(tool) is not None for tool in _NVIDIA_TOOLKIT_BINS)


# ---------------------------------------------------------------------------
# RunnerSpec dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunnerSpec:
    """Resolved runner specification for a single experiment execution.

    Attributes:
        mode:         Execution mode — "local" or "docker".
        image:        Docker image to use. None for local mode or when the
                      default should be resolved at dispatch time.
        source:       Which layer of the precedence chain produced this spec:
                      "env", "yaml", "user_config", "auto_detected", "default".
        image_source: Where the Docker image was resolved from:
                      "env", "yaml", "runner_override", "user_config",
                      "local_build", "registry", or None (local mode / unresolved).
    """

    mode: RunnerMode
    image: str | None
    source: str
    image_source: str | None = None
    extra_mounts: list[tuple[str, str]] = field(default_factory=list)

    def to_runner_info(self) -> dict[str, str | None]:
        """Build runner info dict for progress display callbacks."""
        return {
            "mode": self.mode,
            "source": self.source,
            "image": self.image,
            "image_source": self.image_source,
        }


# ---------------------------------------------------------------------------
# Core resolution function
# ---------------------------------------------------------------------------


def resolve_runner(
    backend: str,
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
) -> RunnerSpec:
    """Resolve the runner for a single backend using the full precedence chain.

    Precedence (highest to lowest):
        1. Env var   ``LLEM_RUNNER_{BACKEND}``  — source="env"
        2. Study YAML ``runners:`` section       — source="yaml"
        3. User config ``runners.{backend}``     — source="user_config"
           Only non-"auto" values are treated as explicit. "auto" falls through
           to step 4. Pass ``user_config=None`` to allow auto-detection.
        4. Auto-detection: Docker + NVIDIA CT    — source="auto_detected"
        5. Built-in default: local              — source="default"

    When mode is "docker" and image is None, the caller (DockerRunner) should
    resolve the image via ``get_default_image(backend)`` from image_registry.

    Args:
        backend:      Backend name, e.g. "pytorch", "vllm", "tensorrt".
        yaml_runners: Runners dict from study YAML ``runners:`` section.
                      Keys are backend names, values are runner strings.
        user_config:  UserRunnersConfig from loaded user preferences.
                      None = no user config present (enables auto-detection).
                      When provided, "auto" values fall through to auto-detection;
                      explicit values ("local", "docker", "docker:<img>") are honoured.

    Returns:
        RunnerSpec with mode, image, and source fields populated.
    """
    # Load .env (idempotent, shell env wins via override=False)
    _load_dotenv()

    # 1. Env var: LLEM_RUNNER_{BACKEND} (highest precedence)
    env_key = f"LLEM_RUNNER_{backend.upper()}"
    if env_val := os.environ.get(env_key):
        mode, image = parse_runner_value(env_val)
        return RunnerSpec(mode=mode, image=image, source="env")
    # 2. Study/experiment YAML runners section
    if yaml_runners is not None and backend in yaml_runners:
        yaml_val = yaml_runners[backend]
        if yaml_val is not None:
            mode, image = parse_runner_value(yaml_val)
            return RunnerSpec(mode=mode, image=image, source="yaml")
    # 3. User config — "auto" means no explicit preference, fall through to auto-detection.
    #    Passing user_config=None means "no user config file present" → auto-detect.
    if user_config is not None:
        user_val: str = getattr(user_config, backend, "auto")
        if user_val != "auto":
            mode, image = parse_runner_value(user_val)
            return RunnerSpec(
                mode=mode, image=image, source="user_config"
            )  # "auto" -> fall through to auto-detection

    # 4. Auto-detection: Docker + NVIDIA Container Toolkit available?
    if is_docker_available():
        logger.info("Docker detected. Using containerised execution for reproducible measurements.")
        return RunnerSpec(mode=RUNNER_DOCKER, image=None, source="auto_detected")

    # 5. Default: local with nudge message
    logger.info(
        "Docker not detected. Install Docker + NVIDIA Container Toolkit "
        "for reproducible isolated measurements."
    )
    return RunnerSpec(mode=RUNNER_LOCAL, image=None, source="default")


# ---------------------------------------------------------------------------
# Study-level runner resolution
# ---------------------------------------------------------------------------


def resolve_study_runners(
    backends: list[str],
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
) -> dict[str, RunnerSpec]:
    """Resolve runners for all backends in a study.

    Calls ``resolve_runner`` for each unique backend and returns a mapping of
    backend name → RunnerSpec. The ``yaml_runners`` dict (from the study YAML
    ``runners:`` section) and ``user_config`` are passed through unchanged.

    Args:
        backends:     Unique backend names present in the study's experiments.
        yaml_runners: Study-level ``runners:`` section from YAML (optional).
        user_config:  Loaded UserRunnersConfig (optional, None = auto-detect).

    Returns:
        Dict mapping each backend name to its resolved RunnerSpec.
    """
    return {
        backend: resolve_runner(backend, yaml_runners=yaml_runners, user_config=user_config)
        for backend in backends
    }
