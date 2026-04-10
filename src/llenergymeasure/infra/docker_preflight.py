"""Docker pre-flight checks — validate the host environment before container launch.

Tiered execution model:
  Tier 1 (host-level checks, no container needed):
    1. Docker CLI on PATH
    2. NVIDIA Container Toolkit installed (any of nvidia-container-runtime,
       nvidia-ctk, nvidia-container-cli)
    3. Host nvidia-smi — warns if missing, does NOT hard-block (remote daemon support)

  Tier 2 (requires a running container — only reached if Tier 1 passes):
    4. GPU visibility inside container (docker run --gpus all nvidia-smi)
    5. CUDA/driver compatibility (parsed from the same container probe)

Multiple failures within a tier are reported together as a numbered list before
aborting. Silent on success — no output when all checks pass.

Call ``run_docker_preflight(skip=True)`` to bypass all checks (CLI --skip-preflight
flag or YAML execution.skip_preflight: true).
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from llenergymeasure.config.ssot import TIMEOUT_NVIDIA_SMI
from llenergymeasure.utils.exceptions import DockerPreFlightError

__all__ = ["run_docker_preflight"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROBE_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"
_PROBE_TIMEOUT = 30  # seconds for container probe
_NVIDIA_TOOLKIT_INSTALL_URL = (
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
)
_CUDA_COMPAT_URL = "https://docs.nvidia.com/deploy/cuda-compatibility/"
_DOCKER_INSTALL_URL = "https://docs.docker.com/engine/install/"

_NVIDIA_TOOLKIT_BINS = (
    "nvidia-container-runtime",
    "nvidia-ctk",
    "nvidia-container-cli",
)


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------


def _check_docker_cli() -> str | None:
    """Return an error string if docker CLI is not on PATH, else None."""
    if shutil.which("docker") is None:
        return f"Docker not found on PATH\n     Fix: Install Docker Engine — {_DOCKER_INSTALL_URL}"
    return None


def _check_nvidia_toolkit() -> str | None:
    """Return an error string if no NVIDIA Container Toolkit binary is on PATH, else None."""
    if not any(shutil.which(tool) is not None for tool in _NVIDIA_TOOLKIT_BINS):
        return (
            "NVIDIA Container Toolkit not found\n"
            f"     Fix: Install NVIDIA Container Toolkit — {_NVIDIA_TOOLKIT_INSTALL_URL}"
        )
    return None


def _get_host_driver_version() -> str | None:
    """Run host nvidia-smi and return the driver version string, or None.

    Warns (does not raise) if nvidia-smi is missing or fails — this supports
    remote Docker daemon scenarios where the local host has no GPU driver.
    """
    if shutil.which("nvidia-smi") is None:
        logger.warning(
            "Host nvidia-smi not found. This may be a remote Docker daemon setup. "
            "Skipping host driver version check — GPU validation will proceed via "
            "container probe only."
        )
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_NVIDIA_SMI,
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0].strip()
            if version:
                return version
        logger.warning(
            "Host nvidia-smi returned non-zero exit code %d. "
            "Proceeding without host driver version check.",
            result.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Host nvidia-smi timed out. Proceeding without host driver version check.")
    except FileNotFoundError:
        logger.warning(
            "Host nvidia-smi not found. This may be a remote Docker daemon setup. "
            "GPU validation will proceed via container probe only."
        )
    except Exception as exc:
        logger.warning("Host nvidia-smi check failed: %s. Proceeding.", exc)

    return None


def _probe_container_gpu(host_driver_version: str | None) -> list[str]:
    """Run a lightweight container probe to validate GPU visibility and CUDA compat.

    Combines GPU name and driver version queries into a single docker run invocation.
    Returns a list of error strings (empty = all OK).
    """
    errors: list[str] = []

    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                _PROBE_IMAGE,
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        errors.append(
            "Container GPU probe timed out after "
            f"{_PROBE_TIMEOUT}s\n"
            f"     Fix: Check Docker daemon and GPU availability — {_NVIDIA_TOOLKIT_INSTALL_URL}"
        )
        return errors
    except FileNotFoundError:
        # docker not on PATH — should have been caught by tier 1, but guard defensively
        errors.append(
            "Docker not found when launching container probe\n"
            f"     Fix: Install Docker Engine — {_DOCKER_INSTALL_URL}"
        )
        return errors

    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        # CUDA/driver version mismatch: typically "cuda" + "version" or
        # "driver/library version mismatch" (NVML), or "nvml" + "driver".
        # Plain "device driver" in GPU access errors must NOT trigger this path.
        _is_cuda_compat_error = (
            (
                "cuda" in stderr_lower
                and ("version" in stderr_lower or "incompatible" in stderr_lower)
            )
            or "driver/library version mismatch" in stderr_lower
            or ("nvml" in stderr_lower and "driver" in stderr_lower)
            or ("initialize nvml" in stderr_lower and "driver" in stderr_lower)
        )
        # GPU quota / resource allocation failure: server-side quota systems
        # (e.g. shared GPU servers) reject the container with a clear allocation
        # error rather than a toolkit misconfiguration.
        _is_quota_error = (
            "quota" in stderr_lower
            or "allocation failed" in stderr_lower
            or "gpu allocation" in stderr_lower
        )
        if _is_cuda_compat_error:
            host_info = (
                f"Host driver: {host_driver_version}"
                if host_driver_version
                else "Host driver: unknown"
            )
            errors.append(
                f"CUDA/driver compatibility error inside container. {host_info}\n"
                "     The container CUDA version may require a newer host driver.\n"
                f"     See: {_CUDA_COMPAT_URL}"
            )
        elif _is_quota_error:
            stderr_detail = result.stderr.strip()
            errors.append(
                "GPU quota or allocation limit reached — the probe container could not "
                "acquire a GPU.\n"
                "     Free up GPU quota (e.g. stop a running container) and retry.\n"
                f"     Docker stderr: {stderr_detail}"
            )
        else:
            stderr_detail = result.stderr.strip()
            detail_suffix = f"\n     Docker stderr: {stderr_detail}" if stderr_detail else ""
            errors.append(
                "GPU not accessible inside Docker container\n"
                "     Possible cause: NVIDIA Container Toolkit not configured correctly.\n"
                f"     Fix: {_NVIDIA_TOOLKIT_INSTALL_URL}"
                f"{detail_suffix}"
            )
        return errors

    # Container probe succeeded — optionally check for major driver version mismatch
    output = result.stdout.strip()
    if output and host_driver_version:
        lines = output.splitlines()
        if lines:
            parts = lines[0].split(",")
            if len(parts) >= 2:
                container_driver = parts[1].strip()
                host_major = host_driver_version.split(".")[0]
                container_major = container_driver.split(".")[0]
                if host_major != container_major:
                    logger.warning(
                        "Host driver major version (%s) differs from container driver major "
                        "version (%s). This is unusual and may indicate a version mismatch.",
                        host_driver_version,
                        container_driver,
                    )

    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_docker_preflight(skip: bool = False) -> None:
    """Run tiered Docker pre-flight checks before any container is launched.

    Validates:
      Tier 1 — Docker CLI on PATH, NVIDIA Container Toolkit installed,
                host nvidia-smi (warn-only if missing)
      Tier 2 — GPU visibility inside container, CUDA/driver compatibility

    Silent on success. Raises DockerPreFlightError with actionable numbered
    list on failure.

    Args:
        skip: If True, log a warning and return immediately without running any checks.

    Raises:
        DockerPreFlightError: One or more tier-1 or tier-2 checks failed.
    """
    if skip:
        logger.warning(
            "Docker pre-flight checks skipped (--skip-preflight). "
            "Container GPU visibility and CUDA/driver compatibility are not validated."
        )
        return

    # -----------------------------------------------------------------------
    # Tier 1: host-level checks (no container needed)
    # -----------------------------------------------------------------------
    tier1_failures: list[str] = []

    docker_error = _check_docker_cli()
    if docker_error is not None:
        tier1_failures.append(docker_error)

    toolkit_error = _check_nvidia_toolkit()
    if toolkit_error is not None:
        tier1_failures.append(toolkit_error)

    # Host nvidia-smi — warn-only (supports remote Docker daemon)
    host_driver_version = _get_host_driver_version()

    if tier1_failures:
        n = len(tier1_failures)
        numbered = "\n".join(f"  {i + 1}. {msg}" for i, msg in enumerate(tier1_failures))
        raise DockerPreFlightError(f"Docker pre-flight failed: {n} issue(s) found\n{numbered}")

    # -----------------------------------------------------------------------
    # Tier 2: container-level checks (only reached if tier 1 passes)
    # -----------------------------------------------------------------------
    tier2_failures = _probe_container_gpu(host_driver_version)

    if tier2_failures:
        n = len(tier2_failures)
        numbered = "\n".join(f"  {i + 1}. {msg}" for i, msg in enumerate(tier2_failures))
        raise DockerPreFlightError(f"Docker pre-flight failed: {n} issue(s) found\n{numbered}")
