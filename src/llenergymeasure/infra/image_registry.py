"""Docker image resolution for engine containers.

Two image sources
-----------------
**Local images** (``llenergymeasure:{engine}``) are produced by
``docker compose build`` / ``make docker-build-all`` and always reflect the
current source tree.  They are preferred for fast local iteration.

**Registry images** (``ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}``)
are published by CI on release tags.  Versioned and immutable, they are used
in CI, by pip-install users, and whenever no local image is present.

``get_default_image(engine)`` checks for a local image first, then falls back
to the registry tag built from the current package version (or ``"latest"`` if
version detection fails).

Overriding the image
--------------------
The ``runners:`` section in the study YAML accepts explicit image references::

    runners:
      pytorch: local                    # host execution (no Docker)
      vllm: docker                      # default resolution (local → registry)
      tensorrt: "docker:my/custom:tag"  # explicit image override

``parse_runner_value()`` converts these into a ``(runner_type, image_override)``
tuple consumed by the runner resolution chain.
"""

from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache

from llenergymeasure.config.ssot import (
    ENGINE_TENSORRT,
    ENGINE_TRANSFORMERS,
    ENGINE_VLLM,
    ENV_IMAGE_PREFIX,
    RUNNER_DOCKER,
    RUNNER_LOCAL,
    TIMEOUT_DOCKER_CLI,
    TIMEOUT_NVCC,
    RunnerMode,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_IMAGE_TEMPLATE",
    "get_cuda_major_version",
    "get_default_image",
    "parse_runner_value",
    "resolve_image",
    "show_image_resolution",
]

# ---------------------------------------------------------------------------
# Template and registry
# ---------------------------------------------------------------------------

# {version} is filled at runtime; matches tags pushed by docker-publish.yml.
DEFAULT_IMAGE_TEMPLATE = "ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}"

# Local image tag produced by `docker compose build` (no registry prefix).
LOCAL_IMAGE_TEMPLATE = "llenergymeasure:{engine}"

# Engines that have a Docker image in the registry.
_SUPPORTED_ENGINES = frozenset({ENGINE_TRANSFORMERS, ENGINE_VLLM, ENGINE_TENSORRT})


# ---------------------------------------------------------------------------
# CUDA version detection
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_cuda_major_version() -> str | None:
    """Detect the host CUDA major version.

    Tries ``nvcc --version`` first, falls back to pynvml.

    Returns:
        CUDA major version string, e.g. ``"12"``, or ``None`` if detection fails.
    """
    # --- Primary: nvcc ---
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_NVCC,
        )
        if result.returncode == 0:
            major = _parse_cuda_major_from_nvcc(result.stdout)
            if major:
                return major
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # --- Secondary: pynvml ---
    try:
        import pynvml  # type: ignore[import-untyped]

        from llenergymeasure.device.gpu_info import nvml_context

        cuda_version_raw: int | None = None
        with nvml_context():
            cuda_version_raw = pynvml.nvmlSystemGetCudaDriverVersion()
        if cuda_version_raw is not None:
            # nvmlSystemGetCudaDriverVersion returns an integer like 12030 → major 12
            major_int = cuda_version_raw // 1000
            if major_int > 0:
                return str(major_int)
    except Exception:
        pass

    return None


def _parse_cuda_major_from_nvcc(output: str) -> str | None:
    """Extract CUDA major version from ``nvcc --version`` stdout.

    Example line::
        Cuda compilation tools, release 12.3, V12.3.107

    Args:
        output: Full stdout string from ``nvcc --version``.

    Returns:
        Major version string (e.g. ``"12"``), or ``None`` if not parseable.
    """
    import re

    match = re.search(r"release\s+(\d+)\.\d+", output)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_default_image(engine: str) -> str:
    """Resolve the default Docker image for *engine*.

    Resolution order:

    1. **Local image** (``llenergymeasure:{engine}``): produced by
       ``docker compose build`` / ``make docker-build-all``.  Always reflects
       current source code.  Preferred for local development iteration.
    2. **Registry image** (``ghcr.io/…/{engine}:v{version}``): published by
       CI on release tags.  Versioned and immutable.  Used in CI, by pip-install
       users, and whenever no local image exists.

    To force a specific image, use ``runners: {engine}: "docker:<image:tag>"``
    in the study YAML.

    Args:
        engine: Engine name, e.g. ``"vllm"``, ``"transformers"``, ``"tensorrt"``.

    Returns:
        Image reference string, e.g. ``"llenergymeasure:vllm"`` (local) or
        ``"ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0"`` (registry).
    """
    # Prefer local image from docker compose build
    local_image = LOCAL_IMAGE_TEMPLATE.format(engine=engine)
    if _image_exists_locally(local_image):
        logger.info("Using local image %s (from docker compose build)", local_image)
        return local_image

    # Fall back to versioned GHCR tag
    from llenergymeasure._version import __version__

    version = __version__ if __version__ else "latest"
    ghcr_image = DEFAULT_IMAGE_TEMPLATE.format(engine=engine, version=version)
    logger.info("No local image found; using registry image %s", ghcr_image)
    return ghcr_image


@lru_cache(maxsize=8)
def _image_exists_locally(image: str) -> bool:
    """Check whether a Docker image tag exists in the local cache."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def resolve_image(
    engine: str,
    *,
    spec_image: str | None = None,
    yaml_images: dict[str, str] | None = None,
    user_config_images: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the Docker image for *engine* using the full precedence chain.

    This is the **image axis** of the orthogonal runner/image resolution system.
    The runner axis (local vs docker) is handled by ``resolve_runner()`` in
    ``runner_resolution.py``.

    Precedence (highest to lowest):

    1. ``LLEM_IMAGE_{ENGINE}`` env var (from shell or ``.env`` file)
    2. Study YAML ``images:`` section
    3. Explicit image from runner spec (``docker:<image>`` shorthand)
    4. User config ``images:`` section
    5. Smart default: local image → registry fallback

    Args:
        engine:              Engine name (e.g. ``"vllm"``).
        spec_image:          Image override from ``docker:<image>`` runner
                             shorthand.  None when runner was bare ``"docker"``.
        yaml_images:         ``images:`` dict from the study YAML (optional).
        user_config_images:  ``images:`` dict from user config (optional).

    Returns:
        ``(image, image_source)`` tuple where *image_source* indicates provenance:
        ``"env"``, ``"yaml"``, ``"runner_override"``, ``"user_config"``,
        ``"local_build"``, or ``"registry"``.
    """
    # Load .env so LLEM_IMAGE_* vars are available
    from llenergymeasure.infra.runner_resolution import _load_dotenv

    _load_dotenv()

    # 1. Env var (includes .env via python-dotenv)
    env_key = f"{ENV_IMAGE_PREFIX}{engine.upper()}"
    if env_val := os.environ.get(env_key):
        logger.info("Image for %s resolved from env var %s: %s", engine, env_key, env_val)
        return (env_val, "env")

    # 2. Study YAML images: section
    if yaml_images and engine in yaml_images:
        img = yaml_images[engine]
        logger.info("Image for %s resolved from study YAML images: %s", engine, img)
        return (img, "yaml")

    # 3. Explicit image from runner spec (docker:<image> shorthand)
    if spec_image is not None:
        logger.info("Image for %s resolved from runner override: %s", engine, spec_image)
        return (spec_image, "runner_override")

    # 4. User config images: section
    if user_config_images and engine in user_config_images:
        img = user_config_images[engine]
        logger.info("Image for %s resolved from user config images: %s", engine, img)
        return (img, "user_config")

    # 5. Smart default: delegate to get_default_image() (local build → registry)
    image = get_default_image(engine)
    local_image = LOCAL_IMAGE_TEMPLATE.format(engine=engine)
    if image == local_image:
        source = "local_build"
    elif _image_exists_locally(image):
        source = "registry_cached"
    else:
        source = "registry"
    return (image, source)


def show_image_resolution() -> None:
    """Print which Docker image each engine will resolve to.

    Shows local vs registry source for each engine.  Used by
    ``make docker-images`` for quick diagnostics.
    """
    print("=== Image resolution ===")
    for engine in sorted(_SUPPORTED_ENGINES):
        image, source = resolve_image(engine)
        print(f"  {engine:10s} -> {image}  ({source})")


def parse_runner_value(value: str) -> tuple[RunnerMode, str | None]:
    """Parse a runner config value into ``(runner_type, image_override)``.

    Accepted forms::

        "local"                → ("local", None)
        "docker"               → ("docker", None)
        "docker:image/name:tag" → ("docker", "image/name:tag")

    Args:
        value: Raw string from ``runners.{engine}`` in YAML config.

    Returns:
        Tuple of ``(runner_type, image_override)`` where ``image_override`` is
        ``None`` when the built-in default image should be used.

    Raises:
        ValueError: If ``"docker:"`` is given with an empty image name, or if
                    the value is not one of the recognised runner types.
    """
    if value == RUNNER_LOCAL:
        return (RUNNER_LOCAL, None)

    if value == RUNNER_DOCKER:
        return (RUNNER_DOCKER, None)

    if value.startswith("docker:"):
        image = value[len("docker:") :]
        if not image:
            raise ValueError(
                "empty image name in runner value 'docker:' — "
                "use 'docker' (bare) to select the built-in default image, "
                "or 'docker:full/image:tag' for an explicit image."
            )
        return (RUNNER_DOCKER, image)

    raise ValueError(
        f"Unrecognised runner value {value!r}. "
        "Accepted values: 'local', 'docker', or 'docker:<image-tag>'."
    )
