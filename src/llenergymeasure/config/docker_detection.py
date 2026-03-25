"""Docker container detection."""

from __future__ import annotations

from pathlib import Path


def is_inside_docker() -> bool:
    """Check if code is running inside a Docker container.

    Uses two detection methods:
    1. Presence of /.dockerenv file
    2. Docker/containerd strings in /proc/1/cgroup

    Returns:
        True if running inside Docker container, False otherwise.
    """
    if Path("/.dockerenv").exists():
        return True

    try:
        with open("/proc/1/cgroup") as f:
            content = f.read()
            if "docker" in content or "containerd" in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    return False
