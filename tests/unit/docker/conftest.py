"""Shared fixtures and factories for docker/ tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def make_subprocess_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess.

    Replaces the identical _make_proc (test_docker_runner) and
    _make_subprocess_result (test_docker_preflight) factories.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc
