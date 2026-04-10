"""Environment snapshot collection for experiment reproducibility.

Captures the full software and hardware environment context before an experiment,
enabling post-hoc analysis of environmental factors affecting measurements.
"""

import importlib.metadata
import logging
import platform
from concurrent.futures import Future, ThreadPoolExecutor

from llenergymeasure._version import __version__
from llenergymeasure.device.environment import collect_environment_metadata
from llenergymeasure.domain.environment import (
    EnvironmentSnapshot,
    detect_cuda_version_with_source,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Package enumeration (in-process, replaces subprocess pip freeze / conda list)
# ---------------------------------------------------------------------------


def _collect_installed_packages() -> list[str]:
    """Enumerate installed packages in-process. Covers pip, conda, manual installs."""
    result = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        if name and version:
            result.append(f"{name}=={version}")
    return sorted(result)


# ---------------------------------------------------------------------------
# Collection function
# ---------------------------------------------------------------------------


def collect_environment_snapshot() -> EnvironmentSnapshot:
    """Capture a full environment snapshot before experiment start.

    Collects hardware metadata (via device/environment.py), Python version,
    pip freeze, optional conda list, tool version, and CUDA version.
    """
    hardware = collect_environment_metadata()
    cuda_version, cuda_version_source = detect_cuda_version_with_source()

    return EnvironmentSnapshot(
        hardware=hardware,
        python_version=platform.python_version(),
        tool_version=__version__,
        cuda_version=cuda_version,
        cuda_version_source=cuda_version_source,
    )


def collect_software_environment() -> dict[str, object]:
    """Collect study-level software environment for the environment.json artefact.

    This is a study-level constant (the container environment doesn't change
    between experiments), so it's collected once per study and written to
    ``_study-artefacts/environment.json``.

    Returns:
        Dict with python_version, installed_packages, llenergymeasure_version,
        cuda_version, and cuda_version_source.
    """
    cuda_version, cuda_version_source = detect_cuda_version_with_source()
    return {
        "python_version": platform.python_version(),
        "installed_packages": _collect_installed_packages(),
        "llenergymeasure_version": __version__,
        "cuda_version": cuda_version,
        "cuda_version_source": cuda_version_source,
    }


# ---------------------------------------------------------------------------
# Background-threaded snapshot collection
# ---------------------------------------------------------------------------

_snapshot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="env-snapshot")


def collect_environment_snapshot_async() -> Future[EnvironmentSnapshot]:
    """Start background collection. Call .result(timeout=10) before writing results."""
    return _snapshot_executor.submit(collect_environment_snapshot)
