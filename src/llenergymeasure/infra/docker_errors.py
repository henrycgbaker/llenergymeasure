"""Docker error hierarchy with categorised error translation.

Translates cryptic Docker / NVIDIA Container Toolkit stderr into actionable
error types, each carrying a `fix_suggestion` string so callers can surface
useful guidance to the researcher without them needing to understand Docker
internals.

Error categories:
    DockerImagePullError    — image not found, pull failed
    DockerGPUAccessError    — NVIDIA Container Toolkit / GPU access problem
    DockerOOMError          — out-of-memory inside container
    DockerPermissionError   — permission denied (docker daemon socket)
    DockerTimeoutError      — container hit timeout or was killed
    DockerContainerError    — generic fallback with stderr snippet
"""

from __future__ import annotations

from llenergymeasure.exceptions import DockerError

__all__ = [
    "DockerContainerError",
    "DockerGPUAccessError",
    "DockerImagePullError",
    "DockerOOMError",
    "DockerPermissionError",
    "DockerTimeoutError",
    "capture_stderr_snippet",
    "translate_docker_error",
]

# ---------------------------------------------------------------------------
# Error subclasses
# ---------------------------------------------------------------------------


class DockerImagePullError(DockerError):
    """Docker image could not be found or pulled."""


class DockerGPUAccessError(DockerError):
    """NVIDIA Container Toolkit or GPU access error inside Docker."""


class DockerOOMError(DockerError):
    """Container ran out of memory (GPU or host)."""


class DockerPermissionError(DockerError):
    """Permission denied when communicating with Docker daemon."""


class DockerTimeoutError(DockerError):
    """Container exceeded the allowed wall-clock time and was killed."""


class DockerContainerError(DockerError):
    """Generic Docker container failure (unrecognised stderr pattern)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def capture_stderr_snippet(stderr: str, max_lines: int = 20) -> str:
    """Return the last *max_lines* lines of *stderr*.

    Args:
        stderr: Raw stderr string from the container.
        max_lines: Maximum number of lines to return (default 20).

    Returns:
        Truncated string with at most *max_lines* lines.
    """
    lines = stderr.splitlines()
    if len(lines) <= max_lines:
        return stderr
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

# Patterns checked in order; first match wins.
# Each tuple: (list-of-substrings-to-match, error-factory).
# All pattern matching is case-insensitive.

_IMAGE_PULL_PATTERNS = [
    "no such image",
    "not found",
    "manifest unknown",
    "pull access denied",
    "unable to find image",
    "repository does not exist",
    "name unknown",
]

_GPU_ACCESS_PATTERNS = [
    "could not select device driver",
    "nvidia-container-cli",
    "failed to initialize nvml",
    "no gpus available",
    "error response from daemon: could not find device",
    "driver procedures are not found",
    "nvml error",
]

_OOM_PATTERNS = [
    "oom",
    "out of memory",
    "cuda out of memory",
    "outofmemoryerror",
    "memory error",
    "killed",  # Linux OOM killer sends SIGKILL
]

_PERMISSION_PATTERNS = [
    "permission denied",
    "got permission denied",
    "connect: permission denied",
    "is the docker daemon running",
]


def translate_docker_error(returncode: int, stderr: str, image: str) -> DockerError:
    """Translate a Docker container exit into a categorised DockerError.

    Inspects *returncode* and *stderr* to determine the most likely failure
    category, then returns an appropriate subclass with a `fix_suggestion`.

    Args:
        returncode: Exit code from the ``docker run`` process.
            Use 124 for timeout-kill, -9 / -15 for signal-based kills.
        stderr: Full stderr captured from the container / docker CLI.
        image: Image name that was used, embedded in fix suggestions.

    Returns:
        A DockerError subclass with ``fix_suggestion`` and ``stderr_snippet``.
    """
    snippet = capture_stderr_snippet(stderr)
    lower = stderr.lower()

    # --- Timeout: returncode 124 (GNU timeout) or -9 / -15 (SIGKILL/SIGTERM) ---
    if returncode in (124, -9, -15):
        return DockerTimeoutError(
            message=f"Container was killed (exit code {returncode}). Possible timeout.",
            fix_suggestion="Increase timeout or reduce experiment size.",
            stderr_snippet=snippet,
        )

    # --- Image pull / not found ---
    if any(pat in lower for pat in _IMAGE_PULL_PATTERNS):
        return DockerImagePullError(
            message=f"Image not found or could not be pulled: {image}",
            fix_suggestion=f"docker pull {image}",
            stderr_snippet=snippet,
        )

    # --- GPU / NVIDIA Container Toolkit ---
    if any(pat in lower for pat in _GPU_ACCESS_PATTERNS):
        return DockerGPUAccessError(
            message="GPU access error inside Docker. NVIDIA Container Toolkit may not be configured.",
            fix_suggestion=(
                "Install NVIDIA Container Toolkit: "
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            ),
            stderr_snippet=snippet,
        )

    # --- OOM: check before permission so "killed" doesn't collide ---
    if any(pat in lower for pat in _OOM_PATTERNS):
        return DockerOOMError(
            message="Container ran out of memory.",
            fix_suggestion="Reduce batch size or use a smaller model.",
            stderr_snippet=snippet,
        )

    # --- Permission denied ---
    if any(pat in lower for pat in _PERMISSION_PATTERNS):
        return DockerPermissionError(
            message="Permission denied communicating with the Docker daemon.",
            fix_suggestion="Add user to docker group: sudo usermod -aG docker $USER",
            stderr_snippet=snippet,
        )

    # --- Fallback: unrecognised error ---
    return DockerContainerError(
        message=f"Container exited with code {returncode}.",
        fix_suggestion="Check container logs above for details.",
        stderr_snippet=snippet,
    )
