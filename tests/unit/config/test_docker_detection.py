"""Unit tests for config/docker_detection.py.

Covers:
- is_inside_docker() — detects Docker environment via /.dockerenv and /proc/1/cgroup
- should_use_docker_for_campaign() — decides local vs Docker dispatch

All filesystem and subprocess I/O is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import llenergymeasure.config.docker_detection as detection_mod
from llenergymeasure.config.docker_detection import is_inside_docker, should_use_docker_for_campaign


def _make_dockerenv_patch(exists_result: bool):
    """Return a selective Path.exists patch that only affects /.dockerenv checks.

    Uses the detection module's Path so the patch is narrowed to that module only.
    """
    original_exists = Path.exists

    def patched_exists(self: Path) -> bool:
        if str(self) == "/.dockerenv":
            return exists_result
        return original_exists(self)

    return patch.object(detection_mod.Path, "exists", patched_exists)


# ---------------------------------------------------------------------------
# is_inside_docker
# ---------------------------------------------------------------------------


class TestIsInsideDocker:
    def test_returns_true_when_dockerenv_file_exists(self):
        """Detection method 1: /.dockerenv present."""
        with _make_dockerenv_patch(True):
            assert is_inside_docker() is True

    def test_returns_false_when_dockerenv_absent_and_cgroup_clean(self):
        """Both methods return negative → not in Docker."""
        cgroup_content = "11:cpuset:/ \n10:memory:/\n"
        with (
            _make_dockerenv_patch(False),
            # Narrow patch: only affects open() within the docker_detection module
            patch(
                "llenergymeasure.config.docker_detection.open",
                mock_open(read_data=cgroup_content),
                create=True,
            ),
        ):
            assert is_inside_docker() is False

    def test_returns_true_when_cgroup_contains_docker(self):
        """Detection method 2: 'docker' string in /proc/1/cgroup."""
        cgroup_content = "11:cpuset:/docker/abc123\n"
        with (
            _make_dockerenv_patch(False),
            patch(
                "llenergymeasure.config.docker_detection.open",
                mock_open(read_data=cgroup_content),
                create=True,
            ),
        ):
            assert is_inside_docker() is True

    def test_returns_true_when_cgroup_contains_containerd(self):
        """Detection method 2: 'containerd' string in /proc/1/cgroup."""
        cgroup_content = "11:cpuset:/containerd/abc123\n"
        with (
            _make_dockerenv_patch(False),
            patch(
                "llenergymeasure.config.docker_detection.open",
                mock_open(read_data=cgroup_content),
                create=True,
            ),
        ):
            assert is_inside_docker() is True

    def test_returns_false_when_cgroup_file_not_found(self):
        """/proc/1/cgroup missing (e.g. macOS) → not Docker."""
        with (
            _make_dockerenv_patch(False),
            patch(
                "llenergymeasure.config.docker_detection.open",
                side_effect=FileNotFoundError,
                create=True,
            ),
        ):
            assert is_inside_docker() is False

    def test_returns_false_when_cgroup_permission_denied(self):
        """/proc/1/cgroup not readable → not Docker."""
        with (
            _make_dockerenv_patch(False),
            patch(
                "llenergymeasure.config.docker_detection.open",
                side_effect=PermissionError,
                create=True,
            ),
        ):
            assert is_inside_docker() is False

    def test_dockerenv_takes_priority_over_cgroup(self):
        """/.dockerenv check short-circuits — cgroup is never read."""
        with (
            _make_dockerenv_patch(True),
            # Narrow patch: cgroup open within detection module should never be called
            patch(
                "llenergymeasure.config.docker_detection.open",
                side_effect=AssertionError("cgroup should not be read"),
                create=True,
            ),
        ):
            # Should return True without reading cgroup
            assert is_inside_docker() is True


# ---------------------------------------------------------------------------
# should_use_docker_for_campaign
# ---------------------------------------------------------------------------


class TestShouldUseDockerForCampaign:
    def test_returns_false_when_already_in_docker(self):
        """Already in Docker → no nested containers."""
        with patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=True):
            result = should_use_docker_for_campaign(["pytorch"])
        assert result is False

    def test_returns_false_for_single_installed_backend(self):
        """Single backend installed locally → run locally."""
        with (
            patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=False),
            patch(
                "llenergymeasure.config.backend_detection.is_backend_available", return_value=True
            ),
        ):
            result = should_use_docker_for_campaign(["pytorch"])
        assert result is False

    def test_returns_true_for_single_uninstalled_backend(self):
        """Backend not installed → dispatch to Docker."""
        with (
            patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=False),
            patch(
                "llenergymeasure.config.backend_detection.is_backend_available", return_value=False
            ),
        ):
            result = should_use_docker_for_campaign(["vllm"])
        assert result is True

    def test_returns_true_for_multiple_backends(self):
        """Multi-backend → always use Docker."""
        with (
            patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=False),
            patch(
                "llenergymeasure.config.backend_detection.is_backend_available", return_value=True
            ),
        ):
            result = should_use_docker_for_campaign(["pytorch", "vllm"])
        assert result is True

    def test_inside_docker_overrides_multi_backend(self):
        """Already in Docker → False even with multiple backends."""
        with patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=True):
            result = should_use_docker_for_campaign(["pytorch", "vllm"])
        assert result is False

    def test_empty_backend_list_uses_docker(self):
        """No backends specified → len != 1 → Docker."""
        with (
            patch("llenergymeasure.config.docker_detection.is_inside_docker", return_value=False),
            patch(
                "llenergymeasure.config.backend_detection.is_backend_available",
                return_value=True,
            ),
        ):
            result = should_use_docker_for_campaign([])
        assert result is True
