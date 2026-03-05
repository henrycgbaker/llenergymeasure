"""Tests for Docker error hierarchy and stderr translation."""

from __future__ import annotations

import pytest

from llenergymeasure.exceptions import DockerError, LLEMError
from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerGPUAccessError,
    DockerImagePullError,
    DockerOOMError,
    DockerPermissionError,
    DockerTimeoutError,
    capture_stderr_snippet,
    translate_docker_error,
)

# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_docker_error_inherits_llem_error(self):
        assert issubclass(DockerError, LLEMError)

    def test_subclasses_inherit_docker_error(self):
        for cls in [
            DockerImagePullError,
            DockerGPUAccessError,
            DockerOOMError,
            DockerPermissionError,
            DockerTimeoutError,
            DockerContainerError,
        ]:
            assert issubclass(cls, DockerError), f"{cls.__name__} should inherit DockerError"
            assert issubclass(cls, LLEMError), f"{cls.__name__} should inherit LLEMError"


# ---------------------------------------------------------------------------
# fix_suggestion attribute on every subclass
# ---------------------------------------------------------------------------


class TestFixSuggestion:
    @pytest.mark.parametrize(
        "cls",
        [
            DockerImagePullError,
            DockerGPUAccessError,
            DockerOOMError,
            DockerPermissionError,
            DockerTimeoutError,
            DockerContainerError,
        ],
    )
    def test_has_fix_suggestion(self, cls):
        err = cls(message="test", fix_suggestion="do something")
        assert hasattr(err, "fix_suggestion")
        assert err.fix_suggestion == "do something"

    @pytest.mark.parametrize(
        "cls",
        [
            DockerImagePullError,
            DockerGPUAccessError,
            DockerOOMError,
            DockerPermissionError,
            DockerTimeoutError,
            DockerContainerError,
        ],
    )
    def test_has_stderr_snippet(self, cls):
        err = cls(message="test", fix_suggestion="do something", stderr_snippet="snippet here")
        assert err.stderr_snippet == "snippet here"

    @pytest.mark.parametrize(
        "cls",
        [
            DockerImagePullError,
            DockerGPUAccessError,
            DockerOOMError,
            DockerPermissionError,
            DockerTimeoutError,
            DockerContainerError,
        ],
    )
    def test_stderr_snippet_defaults_to_none(self, cls):
        err = cls(message="test", fix_suggestion="do something")
        assert err.stderr_snippet is None


# ---------------------------------------------------------------------------
# capture_stderr_snippet
# ---------------------------------------------------------------------------


class TestCaptureStderrSnippet:
    def test_short_output_returned_unchanged(self):
        text = "line1\nline2\nline3"
        result = capture_stderr_snippet(text, max_lines=20)
        assert result == text

    def test_long_output_truncated_to_last_n_lines(self):
        lines = [f"line{i}" for i in range(50)]
        text = "\n".join(lines)
        result = capture_stderr_snippet(text, max_lines=10)
        result_lines = result.splitlines()
        assert len(result_lines) == 10
        assert result_lines[-1] == "line49"
        assert result_lines[0] == "line40"

    def test_exactly_max_lines_returned_unchanged(self):
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = capture_stderr_snippet(text, max_lines=20)
        assert result == text

    def test_empty_string(self):
        result = capture_stderr_snippet("", max_lines=20)
        assert result == ""

    def test_custom_max_lines(self):
        lines = [f"line{i}" for i in range(100)]
        text = "\n".join(lines)
        result = capture_stderr_snippet(text, max_lines=5)
        assert len(result.splitlines()) == 5


# ---------------------------------------------------------------------------
# translate_docker_error — category matching
# ---------------------------------------------------------------------------


class TestTranslateDockerError:
    IMAGE = "ghcr.io/llem/vllm:1.19.0-cuda12"

    def test_image_not_found_matches_image_pull_error(self):
        stderr = "Error response from daemon: No such image: ghcr.io/llem/vllm:1.19.0-cuda12"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerImagePullError)
        assert "docker pull" in err.fix_suggestion
        assert self.IMAGE in err.fix_suggestion

    def test_manifest_unknown_matches_image_pull_error(self):
        stderr = "manifest unknown: manifest unknown"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerImagePullError)

    def test_not_found_matches_image_pull_error(self):
        stderr = "Error: image not found in repository"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerImagePullError)

    def test_pull_access_denied_matches_image_pull_error(self):
        stderr = "pull access denied for ghcr.io/llem/vllm, repository does not exist"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerImagePullError)

    def test_nvidia_container_cli_matches_gpu_access_error(self):
        stderr = "docker: Error response from daemon: failed to create shim: nvidia-container-cli"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerGPUAccessError)
        assert "NVIDIA Container Toolkit" in err.fix_suggestion

    def test_could_not_select_device_driver_matches_gpu_access_error(self):
        stderr = 'could not select device driver "" with capabilities: [[gpu]]'
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerGPUAccessError)

    def test_cuda_out_of_memory_matches_oom_error(self):
        stderr = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerOOMError)
        assert "batch size" in err.fix_suggestion.lower() or "model" in err.fix_suggestion.lower()

    def test_out_of_memory_matches_oom_error(self):
        stderr = "out of memory"
        err = translate_docker_error(137, stderr, self.IMAGE)
        assert isinstance(err, DockerOOMError)

    def test_permission_denied_matches_permission_error(self):
        stderr = "permission denied while trying to connect to the Docker daemon socket"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerPermissionError)
        assert "docker group" in err.fix_suggestion.lower() or "usermod" in err.fix_suggestion

    def test_got_permission_denied_matches_permission_error(self):
        stderr = "Got permission denied while trying to connect to the Docker daemon"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerPermissionError)

    def test_returncode_124_matches_timeout_error(self):
        stderr = ""
        err = translate_docker_error(124, stderr, self.IMAGE)
        assert isinstance(err, DockerTimeoutError)
        assert "timeout" in err.fix_suggestion.lower() or "Increase" in err.fix_suggestion

    def test_returncode_minus9_matches_timeout_error(self):
        stderr = "Killed"
        err = translate_docker_error(-9, stderr, self.IMAGE)
        assert isinstance(err, DockerTimeoutError)

    def test_returncode_minus15_matches_timeout_error(self):
        stderr = "Terminated"
        err = translate_docker_error(-15, stderr, self.IMAGE)
        assert isinstance(err, DockerTimeoutError)

    def test_unknown_error_produces_container_error(self):
        stderr = "Something very unexpected happened\nLine 2\nLine 3"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerContainerError)
        # snippet should be present
        assert err.stderr_snippet is not None

    def test_unknown_error_contains_stderr_snippet(self):
        lines = [f"line{i}" for i in range(30)]
        stderr = "\n".join(lines)
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerContainerError)
        snippet_lines = err.stderr_snippet.splitlines()
        assert len(snippet_lines) <= 20

    def test_returned_error_is_instance_of_docker_error(self):
        for returncode, stderr in [
            (1, "No such image: foo"),
            (1, "nvidia-container-cli: initialization error"),
            (1, "CUDA out of memory"),
            (1, "permission denied"),
            (124, ""),
            (1, "unexpected error xyz"),
        ]:
            err = translate_docker_error(returncode, stderr, self.IMAGE)
            assert isinstance(err, DockerError), f"Expected DockerError for stderr={stderr!r}"

    def test_case_insensitive_matching(self):
        stderr = "NO SUCH IMAGE: GHCR.IO/LLEM/TEST:LATEST"
        err = translate_docker_error(1, stderr, self.IMAGE)
        assert isinstance(err, DockerImagePullError)
