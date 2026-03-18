"""Built-in registry mapping backend names to default Docker images.

Images follow the naming convention::

    ghcr.io/henrycgbaker/llenergymeasure/{backend}:v{version}

where:
    {version} — current package version (e.g. "0.9.0")

Falls back to "latest" tag if version detection fails.

Runner value parsing
--------------------
``parse_runner_value()`` converts the ``runners.{backend}`` YAML field into a
``(runner_type, image_override)`` tuple::

    "local"                → ("local", None)
    "docker"               → ("docker", None)   # use built-in default image
    "docker:custom/img:v1" → ("docker", "custom/img:v1")  # explicit override
"""

from __future__ import annotations

import subprocess
from functools import lru_cache

__all__ = [
    "DEFAULT_IMAGE_TEMPLATE",
    "get_cuda_major_version",
    "get_default_image",
    "parse_runner_value",
]

# ---------------------------------------------------------------------------
# Template and registry
# ---------------------------------------------------------------------------

# {version} is filled at runtime; matches tags pushed by docker-publish.yml.
DEFAULT_IMAGE_TEMPLATE = "ghcr.io/henrycgbaker/llenergymeasure/{backend}:v{version}"

# Backends that have a Docker image in the registry.
_SUPPORTED_BACKENDS = frozenset({"pytorch", "vllm", "tensorrt"})


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
            timeout=5,
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


def get_default_image(backend: str) -> str:
    """Resolve the default Docker image for *backend*.

    Uses the current package version to build a tag matching what
    ``docker-publish.yml`` pushes (``v{version}``).
    Falls back to ``"latest"`` if version detection fails.

    Args:
        backend: Backend name, e.g. ``"vllm"``, ``"pytorch"``, ``"tensorrt"``.

    Returns:
        Full image reference string, e.g.
        ``"ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0"``.
    """
    from llenergymeasure import __version__

    version = __version__ if __version__ else "latest"

    return DEFAULT_IMAGE_TEMPLATE.format(
        backend=backend,
        version=version,
    )


def parse_runner_value(value: str) -> tuple[str, str | None]:
    """Parse a runner config value into ``(runner_type, image_override)``.

    Accepted forms::

        "local"                → ("local", None)
        "docker"               → ("docker", None)
        "docker:image/name:tag" → ("docker", "image/name:tag")

    Args:
        value: Raw string from ``runners.{backend}`` in YAML config.

    Returns:
        Tuple of ``(runner_type, image_override)`` where ``image_override`` is
        ``None`` when the built-in default image should be used.

    Raises:
        ValueError: If ``"docker:"`` is given with an empty image name, or if
                    the value is not one of the recognised runner types.
    """
    if value == "local":
        return ("local", None)

    if value == "docker":
        return ("docker", None)

    if value.startswith("docker:"):
        image = value[len("docker:") :]
        if not image:
            raise ValueError(
                "empty image name in runner value 'docker:' — "
                "use 'docker' (bare) to select the built-in default image, "
                "or 'docker:full/image:tag' for an explicit image."
            )
        return ("docker", image)

    raise ValueError(
        f"Unrecognised runner value {value!r}. "
        "Accepted values: 'local', 'docker', or 'docker:<image-tag>'."
    )
