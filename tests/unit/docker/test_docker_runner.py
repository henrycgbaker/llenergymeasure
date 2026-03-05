"""Unit tests for DockerRunner dispatch lifecycle.

All tests mock subprocess.run — no real Docker invocations.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerImagePullError,
    DockerOOMError,
    DockerPermissionError,
    DockerTimeoutError,
)
from llenergymeasure.infra.docker_runner import DockerRunner
from tests.conftest import make_config, make_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE = "ghcr.io/llenergymeasure/vllm:1.19.0-cuda12"


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock CompletedProcess."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# Test 1: Success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_returns_experiment_result_on_success(self, tmp_path):
        """Successful container exit returns an ExperimentResult and cleans up."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-abc123"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", return_value=_make_proc(0)),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            # Write a valid result JSON before runner reads it
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned = runner.run(config)

        assert returned.experiment_id == result.experiment_id
        # Exchange dir should be cleaned up on success
        mock_rmtree.assert_called_once()

    def test_runner_metadata_injected_into_effective_config(self, tmp_path):
        """Runner metadata is injected into the result's effective_config."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-meta"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", return_value=_make_proc(0)),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="yaml")
            returned = runner.run(config)

        assert returned.effective_config["runner_type"] == "docker"
        assert returned.effective_config["runner_image"] == IMAGE
        assert returned.effective_config["runner_source"] == "yaml"


# ---------------------------------------------------------------------------
# Test 2: Container failure — image not found
# ---------------------------------------------------------------------------


class TestContainerFailure:
    def test_image_pull_error_raised_on_no_such_image(self, tmp_path):
        """Non-zero exit with 'No such image' raises DockerImagePullError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-pull"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(1, stderr="Error: No such image: ghcr.io/example:latest"),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

    def test_exchange_dir_preserved_on_failure(self, tmp_path):
        """Exchange dir is NOT cleaned up when the container fails."""
        config = make_config()
        exchange_dir = tmp_path / "llem-preserved"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(1, stderr="Error: No such image"),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        # rmtree must NOT be called — dir preserved for debugging
        mock_rmtree.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: OOM error
# ---------------------------------------------------------------------------


class TestOOMError:
    def test_oom_error_raised_on_exit_137(self, tmp_path):
        """Exit 137 with OOM keyword raises DockerOOMError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-oom"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(137, stderr="OOM killer invoked: killed by container OOM"),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerOOMError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 4: Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_docker_timeout_error(self, tmp_path):
        """subprocess.TimeoutExpired is translated to DockerTimeoutError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-timeout"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=30),
            ),
        ):
            runner = DockerRunner(image=IMAGE, timeout=30)
            with pytest.raises(DockerTimeoutError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 5: Permission error
# ---------------------------------------------------------------------------


class TestPermissionError:
    def test_permission_error_raised(self, tmp_path):
        """Exit 1 with 'permission denied' raises DockerPermissionError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-perm"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(
                    1,
                    stderr="Got permission denied while trying to connect to the Docker daemon socket",
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerPermissionError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 6: Missing result file
# ---------------------------------------------------------------------------


class TestMissingResultFile:
    def test_missing_result_raises_container_error(self, tmp_path):
        """Container exits 0 but writes no result file → DockerContainerError."""
        config = make_config()
        exchange_dir = tmp_path / "llem-nofile"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(0),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError, match="no result file"):
                runner.run(config)


# ---------------------------------------------------------------------------
# Test 7: Error payload from container
# ---------------------------------------------------------------------------


class TestErrorPayloadFromContainer:
    def test_error_dict_returned_when_container_writes_error_json(self, tmp_path):
        """Container exits 0 but result file contains an error payload → return dict."""
        config = make_config()
        exchange_dir = tmp_path / "llem-errpayload"
        exchange_dir.mkdir()

        error_payload = {
            "type": "RuntimeError",
            "message": "CUDA out of memory inside container",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: CUDA OOM",
        }

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                return_value=_make_proc(0),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(json.dumps(error_payload), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned = runner.run(config)

        # Error payloads are returned as dicts, not ExperimentResult
        assert isinstance(returned, dict)
        assert returned["type"] == "RuntimeError"
        assert "traceback" in returned


# ---------------------------------------------------------------------------
# Test 8: Docker command structure
# ---------------------------------------------------------------------------


class TestDockerCommandStructure:
    def test_command_includes_required_flags(self, tmp_path):
        """docker run command includes --rm, --gpus all, --shm-size 8g, mount, env var, entrypoint."""
        config = make_config()
        exchange_dir = tmp_path / "llem-cmd"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            return _make_proc(1, stderr="No such image")  # fail fast

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]

        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--gpus" in cmd
        assert "all" in cmd
        assert "--shm-size" in cmd
        assert "8g" in cmd

        # Volume mount: exchange_dir:/run/llem
        joined = " ".join(cmd)
        assert ":/run/llem" in joined

        # LLEM_CONFIG_PATH env var
        assert any("LLEM_CONFIG_PATH" in arg for arg in cmd)

        # Entrypoint module
        assert "llenergymeasure.infra.container_entrypoint" in joined

        # Image tag
        assert IMAGE in cmd


# ---------------------------------------------------------------------------
# Test 9: HF_TOKEN propagation
# ---------------------------------------------------------------------------


class TestHFTokenPropagation:
    def test_hf_token_added_to_docker_command(self, tmp_path, monkeypatch):
        """HF_TOKEN env var is forwarded into the docker run command."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
        config = make_config()
        exchange_dir = tmp_path / "llem-hftoken"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            return _make_proc(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        joined = " ".join(cmd)
        assert "HF_TOKEN=hf_test_secret_token" in joined

    def test_hf_token_absent_when_not_set(self, tmp_path, monkeypatch):
        """HF_TOKEN is not added to docker command when env var is absent."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = make_config()
        exchange_dir = tmp_path / "llem-nohf"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            return _make_proc(1, stderr="No such image")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", side_effect=fake_run),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerImagePullError):
                runner.run(config)

        cmd = captured_cmds[0]
        joined = " ".join(cmd)
        assert "HF_TOKEN" not in joined


# ---------------------------------------------------------------------------
# Test 10: Runner metadata in effective_config (explicit verification)
# ---------------------------------------------------------------------------


class TestRunnerMetadata:
    def test_effective_config_has_runner_keys(self, tmp_path):
        """Result effective_config contains runner_type, runner_image, runner_source."""
        config = make_config()
        result = make_result()
        exchange_dir = tmp_path / "llem-runnerkeys"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", return_value=_make_proc(0)),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="auto_detected")
            returned = runner.run(config)

        ec = returned.effective_config
        assert ec["runner_type"] == "docker"
        assert ec["runner_image"] == IMAGE
        assert ec["runner_source"] == "auto_detected"

    def test_existing_effective_config_preserved(self, tmp_path):
        """Runner metadata does not overwrite existing effective_config keys."""
        config = make_config()
        result = make_result(effective_config={"model_dtype": "float16", "batch_size": 8})
        exchange_dir = tmp_path / "llem-preserve-ec"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", return_value=_make_proc(0)),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="yaml")
            returned = runner.run(config)

        ec = returned.effective_config
        assert ec["model_dtype"] == "float16"
        assert ec["batch_size"] == 8
        assert ec["runner_type"] == "docker"


# ---------------------------------------------------------------------------
# Test 11: Cleanup warning on rmtree failure
# ---------------------------------------------------------------------------


class TestCleanupWarning:
    def test_cleanup_failure_logs_warning_not_raises(self, tmp_path, caplog):
        """shutil.rmtree failure on cleanup emits a warning but does not raise."""
        import logging

        config = make_config()
        result = make_result()
        exchange_dir = tmp_path / "llem-cleanupwarn"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch("llenergymeasure.infra.docker_runner.subprocess.run", return_value=_make_proc(0)),
            patch(
                "llenergymeasure.infra.docker_runner.shutil.rmtree",
                side_effect=PermissionError("permission denied"),
            ),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            from llenergymeasure.domain.experiment import compute_measurement_config_hash

            config_hash = compute_measurement_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            # Should NOT raise even though rmtree fails
            returned = runner.run(config)

        # Still returns a valid result
        assert returned.experiment_id == result.experiment_id
        # Warning was logged
        assert any("Could not remove" in record.message for record in caplog.records)
