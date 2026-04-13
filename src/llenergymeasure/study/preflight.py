"""Pre-flight validation for study configurations.

Runs before any Docker dispatch or experiment execution. Handles multi-engine
study validation and Docker pre-flight checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.ssot import RUNNER_DOCKER, SOURCE_MULTI_ENGINE_ELEVATION
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
    yaml_images: dict[str, str] | None = None,
    user_config_images: dict[str, str] | None = None,
) -> tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]]:
    """Pre-flight checks for a study configuration.

    Single-engine studies pass through — per-experiment pre-flight runs later
    in the subprocess. Multi-engine studies auto-elevate to Docker when Docker
    is available; raise PreFlightError otherwise.

    When any experiment in the study will use a Docker runner, runs Docker
    pre-flight checks (GPU visibility, CUDA/driver compat) unless skipped.

    After runner mode resolution, resolves Docker images for all Docker
    engines using the orthogonal image precedence chain (env > YAML > user
    config > local build > registry).

    Args:
        study: Resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks. The effective skip value
            is ``skip_preflight OR study.study_execution.skip_preflight`` — CLI flag
            takes priority, then YAML config.
        yaml_runners: Runner config from the study YAML ``runners:`` section.
            Forwarded to ``resolve_study_runners()`` so pre-flight uses the same
            runner resolution as the actual dispatch path.
        user_config: Loaded UserRunnersConfig. Forwarded to
            ``resolve_study_runners()`` to match actual dispatch precedence.
        yaml_images: Image overrides from the study YAML ``images:`` section.
        user_config_images: Image overrides from the user config ``images:``
            section.

    Returns:
        Tuple of (runner_specs, system_overrides):
        - runner_specs: Resolved runner specs dict (engine -> RunnerSpec).
          Docker runner specs have ``image`` and ``image_source`` populated.
        - system_overrides: Dict of overrides applied during preflight, keyed
          by override target (e.g. ``"runner.pytorch"``). Each value has
          ``declared``, ``effective``, and ``reason`` keys.

    Raises:
        PreFlightError: Multi-engine study and Docker is not available.
        DockerPreFlightError: Docker pre-flight check failed (inherits PreFlightError).
    """
    from llenergymeasure.infra.runner_resolution import (
        RunnerSpec,
        is_docker_available,
        resolve_study_runners,
    )

    engines = {exp.engine for exp in study.experiments}
    is_multi_engine = len(engines) > 1
    system_overrides: dict[str, dict[str, str]] = {}

    if is_multi_engine:
        engine_list = ", ".join(sorted(engines))
        if not is_docker_available():
            raise PreFlightError(
                f"Multi-engine study requires Docker isolation. "
                f"Found engines: {engine_list}. "
                "Install Docker + NVIDIA Container Toolkit, or use a single engine."
            )
        logger.info(
            "Multi-engine study detected (%s). Auto-elevating all engines to Docker for isolation.",
            engine_list,
        )

    # Resolve runners using normal precedence chain.
    runner_specs = resolve_study_runners(
        list(engines), yaml_runners=yaml_runners, user_config=user_config
    )

    # Multi-engine enforcement: override any local runners to Docker.
    # The YAML may say pytorch: local, but multi-engine studies require
    # Docker isolation for all engines.
    if is_multi_engine:
        for engine_name, spec in runner_specs.items():
            if spec.mode != RUNNER_DOCKER:
                logger.info(
                    "Overriding %s runner from %s to docker (multi-engine isolation).",
                    engine_name,
                    spec.mode,
                )
                system_overrides[f"runner.{engine_name}"] = {
                    "declared": spec.mode,
                    "effective": RUNNER_DOCKER,
                    "reason": "auto-elevated (multi-engine study)",
                }
                runner_specs[engine_name] = RunnerSpec(
                    mode=RUNNER_DOCKER, image=spec.image, source=SOURCE_MULTI_ENGINE_ELEVATION
                )

    # Resolve Docker images for all Docker engines (orthogonal to runner mode).
    from llenergymeasure.infra.image_registry import resolve_image

    for engine_name, spec in runner_specs.items():
        if spec.mode == RUNNER_DOCKER:
            image, image_source = resolve_image(
                engine_name,
                spec_image=spec.image,
                yaml_images=yaml_images,
                user_config_images=user_config_images,
            )
            runner_specs[engine_name] = RunnerSpec(
                mode=spec.mode,
                image=image,
                source=spec.source,
                image_source=image_source,
                extra_mounts=spec.extra_mounts,
            )

    # Docker pre-flight: run once if any engine resolves to a Docker runner.
    # Effective skip = CLI flag (skip_preflight param) OR YAML config value.
    effective_skip = skip_preflight or study.study_execution.skip_preflight
    if any(spec.mode == RUNNER_DOCKER for spec in runner_specs.values()):
        from llenergymeasure.infra.docker_preflight import run_docker_preflight

        run_docker_preflight(skip=effective_skip)

    return runner_specs, system_overrides
