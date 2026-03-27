"""Pre-flight validation for study configurations.

Runs before any Docker dispatch or experiment execution. Handles multi-backend
study validation and Docker pre-flight checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.ssot import RUNNER_DOCKER, SOURCE_MULTI_BACKEND_ELEVATION
from llenergymeasure.utils.exceptions import PreFlightError

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserRunnersConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec

logger = logging.getLogger(__name__)


def run_study_preflight(
    study: StudyConfig,
    skip_preflight: bool = False,
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
) -> dict[str, RunnerSpec]:
    """Pre-flight checks for a study configuration.

    Single-backend studies pass through — per-experiment pre-flight runs later
    in the subprocess. Multi-backend studies auto-elevate to Docker when Docker
    is available; raise PreFlightError otherwise.

    When any experiment in the study will use a Docker runner, runs Docker
    pre-flight checks (GPU visibility, CUDA/driver compat) unless skipped.

    Args:
        study: Resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks. The effective skip value
            is ``skip_preflight OR study.execution.skip_preflight`` — CLI flag
            takes priority, then YAML config.
        yaml_runners: Runner config from the study YAML ``runners:`` section.
            Forwarded to ``resolve_study_runners()`` so pre-flight uses the same
            runner resolution as the actual dispatch path.
        user_config: Loaded UserRunnersConfig. Forwarded to
            ``resolve_study_runners()`` to match actual dispatch precedence.

    Returns:
        Resolved runner specs dict (backend -> RunnerSpec) for reuse by caller.

    Raises:
        PreFlightError: Multi-backend study and Docker is not available.
        DockerPreFlightError: Docker pre-flight check failed (inherits PreFlightError).
    """
    from llenergymeasure.infra.runner_resolution import (
        RunnerSpec,
        is_docker_available,
        resolve_study_runners,
    )

    backends = {exp.backend for exp in study.experiments}
    is_multi_backend = len(backends) > 1

    if is_multi_backend:
        backend_list = ", ".join(sorted(backends))
        if not is_docker_available():
            raise PreFlightError(
                f"Multi-backend study requires Docker isolation. "
                f"Found backends: {backend_list}. "
                "Install Docker + NVIDIA Container Toolkit, or use a single backend."
            )
        logger.info(
            "Multi-backend study detected (%s). Auto-elevating all backends to Docker "
            "for isolation.",
            backend_list,
        )

    # Resolve runners using normal precedence chain.
    runner_specs = resolve_study_runners(
        list(backends), yaml_runners=yaml_runners, user_config=user_config
    )

    # Multi-backend enforcement: override any local runners to Docker.
    # The YAML may say pytorch: local, but multi-backend studies require
    # Docker isolation for all backends.
    if is_multi_backend:
        for backend, spec in runner_specs.items():
            if spec.mode != RUNNER_DOCKER:
                logger.info(
                    "Overriding %s runner from %s to docker (multi-backend isolation).",
                    backend,
                    spec.mode,
                )
                runner_specs[backend] = RunnerSpec(
                    mode=RUNNER_DOCKER, image=spec.image, source=SOURCE_MULTI_BACKEND_ELEVATION
                )

    # Docker pre-flight: run once if any backend resolves to a Docker runner.
    # Effective skip = CLI flag (skip_preflight param) OR YAML config value.
    effective_skip = skip_preflight or study.execution.skip_preflight
    if any(spec.mode == RUNNER_DOCKER for spec in runner_specs.values()):
        from llenergymeasure.infra.docker_preflight import run_docker_preflight

        run_docker_preflight(skip=effective_skip)

    return runner_specs
