"""Unit tests for DockerRunner dispatch lifecycle.

All tests mock subprocess.run — no real Docker invocations.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerImagePullError,
    DockerOOMError,
    DockerPermissionError,
    DockerTimeoutError,
)
from llenergymeasure.infra.docker_runner import DockerRunner, _env_file, _mask_secrets
from tests.conftest import make_config, make_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE = "ghcr.io/llenergymeasure/vllm:1.19.0-cuda12"


def _docker_config_hash(config) -> str:
    """Return the config hash as DockerRunner computes it: from the clean config.

    DockerRunner computes the hash from the unmodified config (output_dir is no
    longer a config field). Tests that write result files must use this helper
    to match the actual file name.
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    return compute_measurement_config_hash(config)


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock CompletedProcess."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _subprocess_run_with_image_cached(*run_results: MagicMock):
    """Create a subprocess.run side_effect that handles _ensure_image's inspect call.

    The first subprocess.run call is always ``docker image inspect`` from
    ``_ensure_image`` — returns exit 0 (image cached). Subsequent calls
    return the provided results in order.
    """
    results = iter([_make_proc(0), *run_results])  # image inspect → 0, then user results
    return lambda *a, **k: next(results)


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
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree") as mock_rmtree,
        ):
            # Write a valid result JSON before runner reads it
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

        assert returned.experiment_id == result.experiment_id
        # Exchange dir should be cleaned up on success
        mock_rmtree.assert_called_once()

    def test_hash_independent_of_output_settings(self, tmp_path):
        """Config hash is computed from the clean config without output path influence.

        output_dir has been removed from ExperimentConfig entirely. The hash is
        deterministic and stable regardless of any output settings (which are now
        passed via env vars, not config fields).
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config = make_config()
        result = make_result()

        # Hash is the same whether computed directly or via the helper
        direct_hash = compute_measurement_config_hash(config)
        helper_hash = _docker_config_hash(config)
        assert direct_hash == helper_hash

        exchange_dir = tmp_path / "llem-hashfix"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            # Write result under the clean hash — both host and container agree
            result_path = exchange_dir / f"{direct_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

        assert returned.experiment_id == result.experiment_id

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
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="yaml")
            returned, _ts_dir = runner.run(config)

        metadata = runner.get_runner_metadata()
        assert metadata["runner_type"] == "docker"
        assert metadata["runner_image"] == IMAGE
        assert metadata["runner_source"] == "yaml"


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

        # First subprocess.run call is _ensure_image (docker image inspect) → return 0 (cached).
        # Second call is the actual docker run → return 137 (OOM).
        calls = iter(
            [
                _make_proc(0),  # docker image inspect — image cached
                _make_proc(137, stderr="OOM killer invoked: killed by container OOM"),
            ]
        )

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=lambda *a, **k: next(calls),
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

        # First call is image inspect (return 0), second is docker run (timeout)
        results = iter(
            [
                _make_proc(0),  # docker image inspect
                None,  # sentinel: raise TimeoutExpired
            ]
        )

        def fake_run(*a, **k):
            r = next(results)
            if r is None:
                raise subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=30)
            return r

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=fake_run,
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
                side_effect=_subprocess_run_with_image_cached(
                    _make_proc(
                        1,
                        stderr="Got permission denied while trying to connect to the Docker daemon socket",
                    )
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
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(json.dumps(error_payload), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            returned, _ts = runner.run(config)

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
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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
        assert "llenergymeasure.entrypoints.container" in joined

        # Image tag
        assert IMAGE in cmd


# ---------------------------------------------------------------------------
# Test 9: HF_TOKEN propagation
# ---------------------------------------------------------------------------


class TestHFTokenPropagation:
    def test_hf_token_uses_env_file_not_cli_arg(self, tmp_path, monkeypatch):
        """HF_TOKEN is forwarded via --env-file, not as a -e CLI argument."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token")
        config = make_config()
        exchange_dir = tmp_path / "llem-hftoken"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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
        # Token value must NOT appear in the command args (security requirement)
        assert "hf_test_secret_token" not in joined
        # Must use --env-file instead
        assert "--env-file" in cmd

    def test_hf_token_absent_when_not_set(self, tmp_path, monkeypatch):
        """HF_TOKEN is not added to docker command when env var is absent."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = make_config()
        exchange_dir = tmp_path / "llem-nohf"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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
    def test_get_runner_metadata_returns_correct_keys(self):
        """get_runner_metadata() returns runner_type, runner_image, runner_source."""
        runner = DockerRunner(image=IMAGE, source="auto_detected")
        metadata = runner.get_runner_metadata()
        assert metadata["runner_type"] == "docker"
        assert metadata["runner_image"] == IMAGE
        assert metadata["runner_source"] == "auto_detected"

    def test_run_returns_tuple(self, tmp_path):
        """DockerRunner.run() returns (result, ts_tmpdir) tuple."""
        config = make_config()
        result = make_result()
        exchange_dir = tmp_path / "llem-tuple"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE, source="yaml")
            returned, ts_dir = runner.run(config)

        assert returned.experiment_id == result.experiment_id
        assert ts_dir is None  # no timeseries.parquet in exchange dir


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
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.shutil.rmtree",
                side_effect=PermissionError("permission denied"),
            ),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            # Should NOT raise even though rmtree fails
            returned, _ts = runner.run(config)

        # Still returns a valid result
        assert returned.experiment_id == result.experiment_id
        # Warning was logged
        assert any("Could not remove" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Test 12: S1 security — HF_TOKEN env-file pattern
# ---------------------------------------------------------------------------


class TestHFTokenSecure:
    def test_hf_token_not_in_cmd_args(self, tmp_path, monkeypatch):
        """HF_TOKEN value never appears in docker run command args (S1 security fix)."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token_12345")
        config = make_config()
        exchange_dir = tmp_path / "llem-s1-cmd"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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
        # Token value must not appear in any cmd element
        assert not any("hf_test_secret_token_12345" in arg for arg in cmd)
        # Token must not be passed as -e KEY=VALUE
        assert not any("HF_TOKEN" in arg and "=" in arg for arg in cmd)
        # --env-file must be present
        assert "--env-file" in cmd

    def test_hf_token_env_file_content(self):
        """_env_file creates a file with KEY=VALUE format; file is deleted after context exits."""
        secrets = {"HF_TOKEN": "hf_test_secret_value"}

        captured_path: list[Path] = []

        with _env_file(secrets) as path:
            assert path is not None
            captured_path.append(path)
            content = path.read_text()
            assert content == "HF_TOKEN=hf_test_secret_value\n"

        # File must be deleted after context exits
        assert not captured_path[0].exists()

    def test_env_file_without_hf_token_still_has_output_vars(self, tmp_path, monkeypatch):
        """When HF_TOKEN is absent, env-file still exists (LLEM_OUTPUT_DIR etc.) but has no HF_TOKEN."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = make_config()
        exchange_dir = tmp_path / "llem-s1-notoken"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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
        # Env-file is still used for LLEM_OUTPUT_DIR and LLEM_SAVE_TIMESERIES
        assert "--env-file" in cmd
        # But HF_TOKEN must not appear anywhere in the command
        joined = " ".join(cmd)
        assert "HF_TOKEN" not in joined

    def test_env_file_cleanup_on_failure(self):
        """Temp env-file is deleted even when an exception is raised inside the context."""
        secrets = {"HF_TOKEN": "hf_cleanup_test_value"}
        captured_path: list[Path] = []

        try:
            with _env_file(secrets) as path:
                assert path is not None
                captured_path.append(path)
                # File exists inside context
                assert path.exists()
                raise RuntimeError("Simulated failure inside context")
        except RuntimeError:
            pass

        # File must be deleted even after the exception
        assert not captured_path[0].exists()

    def test_mask_secrets(self):
        """_mask_secrets replaces long secret values with *** in strings."""
        # Long values (>4 chars) are masked
        result = _mask_secrets(
            "docker run -e HF_TOKEN=abc123xyz",
            {"HF_TOKEN": "abc123xyz"},
        )
        assert result == "docker run -e HF_TOKEN=***"

        # Short values (<=4 chars) are NOT masked — avoids false positives
        result_short = _mask_secrets(
            "some text with abcd",
            {"KEY": "abcd"},
        )
        assert "abcd" in result_short


# ---------------------------------------------------------------------------
# Test 13: mpirun injection for TRT-LLM tensor parallelism
# ---------------------------------------------------------------------------


class TestMpirunInjection:
    """Verify mpirun is injected iff backend=tensorrt and tp_size > 1."""

    def _capture_cmd(self, config, tmp_path) -> list[str]:
        """Run DockerRunner.run() with a fake subprocess and capture the docker cmd."""
        exchange_dir = tmp_path / "llem-mpi"
        exchange_dir.mkdir()

        captured_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            # Skip the _ensure_image "docker image inspect" call
            if cmd[:3] == ["docker", "image", "inspect"]:
                return _make_proc(0)
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

        return captured_cmds[0]

    def test_mpirun_injected_for_tensorrt_tp2(self, tmp_path):
        """TRT-LLM with tp_size=2 gets mpirun -n 2 --allow-run-as-root before python3."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=2))
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" in cmd
        assert "-n" in cmd
        assert "2" in cmd
        assert "--allow-run-as-root" in cmd

        # mpirun must appear after the image name and before python3
        image_idx = cmd.index(IMAGE)
        mpirun_idx = cmd.index("mpirun")
        python_idx = cmd.index("python3")
        assert image_idx < mpirun_idx < python_idx

    def test_mpirun_injected_for_tensorrt_tp4(self, tmp_path):
        """TRT-LLM with tp_size=4 gets mpirun -n 4 with correct stringified count."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=4))
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" in cmd
        n_idx = cmd.index("-n")
        assert cmd[n_idx + 1] == "4"

    def test_no_mpirun_for_tensorrt_tp1(self, tmp_path):
        """TRT-LLM with tp_size=1 does NOT get mpirun (single-GPU path)."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=1))
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" not in cmd

    def test_no_mpirun_for_tensorrt_tp_none(self, tmp_path):
        """TRT-LLM with tp_size=None (default) does NOT get mpirun."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=None))
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" not in cmd

    def test_no_mpirun_for_vllm(self, tmp_path):
        """vLLM backend never gets mpirun regardless of any config."""
        config = make_config(backend="vllm")
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" not in cmd

    def test_no_mpirun_for_pytorch(self, tmp_path):
        """PyTorch backend (default) never gets mpirun."""
        config = make_config()  # backend="pytorch" by default
        cmd = self._capture_cmd(config, tmp_path)

        assert "mpirun" not in cmd


# ---------------------------------------------------------------------------
# Test 14: Extra volume mounts
# ---------------------------------------------------------------------------


class TestExtraMounts:
    """Verify extra_mounts produce correct -v flags and TRT-LLM auto-cache-mount works."""

    def _build_cmd(self, config, tmp_path, runner: DockerRunner) -> list[str]:
        """Call _build_docker_cmd directly on the runner (no subprocess needed)."""
        return runner._build_docker_cmd(config, "abc123", "/tmp/llem-test")

    def test_extra_mounts_appear_in_docker_cmd(self, tmp_path):
        """-v flags for extra_mounts appear in command before image name."""
        config = make_config()
        runner = DockerRunner(image=IMAGE, extra_mounts=[("/host/cache", "/root/.cache")])
        cmd = self._build_cmd(config, tmp_path, runner)

        assert "-v" in cmd
        assert "/host/cache:/root/.cache" in cmd

        # Mount must appear BEFORE the image name
        mount_idx = cmd.index("/host/cache:/root/.cache")
        image_idx = cmd.index(IMAGE)
        assert mount_idx < image_idx

    def test_multiple_extra_mounts(self, tmp_path):
        """Multiple extra mounts all appear in the command."""
        config = make_config()
        runner = DockerRunner(
            image=IMAGE,
            extra_mounts=[("/host/a", "/container/a"), ("/host/b", "/container/b")],
        )
        cmd = self._build_cmd(config, tmp_path, runner)

        assert "/host/a:/container/a" in cmd
        assert "/host/b:/container/b" in cmd

    def test_no_extra_mounts_no_extra_volumes(self, tmp_path):
        """Without extra_mounts, only the exchange_dir -v mount is present."""
        config = make_config()
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        # Only one -v flag: the exchange dir
        v_count = cmd.count("-v")
        assert v_count == 1
        assert "/tmp/llem-test:/run/llem" in cmd

    def test_tensorrt_auto_cache_mount(self, tmp_path):
        """TRT-LLM backend auto-mounts ~/.cache/trt-llm:/root/.cache/trt-llm."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=1))
        runner = DockerRunner(image=IMAGE)

        with patch(
            "llenergymeasure.infra.docker_runner.Path.home",
            return_value=Path("/home/testuser"),
        ):
            cmd = self._build_cmd(config, tmp_path, runner)

        assert "-v" in cmd
        assert "/home/testuser/.cache/trt-llm:/root/.cache/trt-llm" in cmd

    def test_tensorrt_auto_cache_mount_not_duplicated(self, tmp_path):
        """User's custom /root/.cache/trt-llm path prevents auto-mount duplication."""
        from llenergymeasure.config.backend_configs import TensorRTConfig

        config = make_config(backend="tensorrt", tensorrt=TensorRTConfig(tp_size=1))
        runner = DockerRunner(
            image=IMAGE,
            extra_mounts=[("/custom/cache", "/root/.cache/trt-llm")],
        )

        with patch(
            "llenergymeasure.infra.docker_runner.Path.home",
            return_value=Path("/home/testuser"),
        ):
            cmd = self._build_cmd(config, tmp_path, runner)

        # /root/.cache/trt-llm appears as a container path exactly once
        trt_mounts = [arg for arg in cmd if ":/root/.cache/trt-llm" in arg]
        assert len(trt_mounts) == 1
        # And it's the user's custom host path, not the auto-generated one
        assert "/custom/cache:/root/.cache/trt-llm" in cmd

    def test_non_tensorrt_no_auto_cache_mount(self, tmp_path):
        """PyTorch backend does not get the TRT-LLM auto-cache mount."""
        config = make_config()  # backend="pytorch"
        runner = DockerRunner(image=IMAGE)
        cmd = self._build_cmd(config, tmp_path, runner)

        assert not any("trt-llm" in arg for arg in cmd)


# ---------------------------------------------------------------------------
# Test 15: container.log persistence on failure
# ---------------------------------------------------------------------------


class TestContainerLogPersistence:
    def test_container_log_written_on_failure(self, tmp_path):
        """Non-zero exit writes stderr_text to container.log in exchange dir."""
        config = make_config()
        exchange_dir = tmp_path / "llem-logtest"
        exchange_dir.mkdir()

        stderr_content = "ERROR: CUDA out of memory\nKilled by OOM\nstack trace here"

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    _make_proc(137, stderr=stderr_content)
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerOOMError):
                runner.run(config)

        log_path = exchange_dir / "container.log"
        assert log_path.exists()
        assert log_path.read_text(encoding="utf-8") == stderr_content

    def test_exchange_dir_set_on_error(self, tmp_path):
        """Error carries exchange_dir attribute for log discovery."""
        config = make_config()
        exchange_dir = tmp_path / "llem-logmsg"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    _make_proc(1, stderr="some error text")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError) as exc_info:
                runner.run(config)

            assert exc_info.value.exchange_dir == str(exchange_dir)


# ---------------------------------------------------------------------------
# Test: Timeseries parquet rescue before exchange dir cleanup
# ---------------------------------------------------------------------------


class TestTimeseriesParquetRescue:
    """DockerRunner must rescue timeseries.parquet from the exchange dir before cleanup.

    The harness inside the container writes timeseries.parquet to /run/llem
    (= exchange_dir on host). DockerRunner must move it to a temp dir and
    inject effective_config["output_dir"] pointing there, so the study
    runner's _save_and_record can find it.
    """

    def test_parquet_rescued_to_temp_dir(self, tmp_path):
        """When timeseries.parquet exists, it is rescued to a temp dir and output_dir set."""
        config = make_config()
        result = make_result(
            timeseries="timeseries.parquet",
            model_name="gpt2",
        )

        exchange_dir = tmp_path / "llem-ts-rescue"
        exchange_dir.mkdir()
        rescue_dir = tmp_path / "llem-ts-rescued"
        rescue_dir.mkdir()

        # tempfile.mkdtemp is called twice: once for exchange dir, once for rescue dir
        mkdtemp_returns = iter([str(exchange_dir), str(rescue_dir)])

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                side_effect=lambda **kw: next(mkdtemp_returns),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.shutil.rmtree",
            ) as mock_rmtree,
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            # Simulate harness writing timeseries.parquet inside container
            ts_path = exchange_dir / "timeseries.parquet"
            ts_path.write_bytes(b"PARQUET_CONTENT")

            runner = DockerRunner(image=IMAGE)
            returned, ts_dir = runner.run(config)

        # ts_dir must point to the rescue dir containing the parquet sidecar
        assert ts_dir == rescue_dir
        assert (rescue_dir / "timeseries.parquet").exists()
        assert (rescue_dir / "timeseries.parquet").read_bytes() == b"PARQUET_CONTENT"

        # Exchange dir must still be cleaned up
        mock_rmtree.assert_called_once()

    def test_no_parquet_no_rescue(self, tmp_path):
        """When no timeseries.parquet exists, ts_dir is None."""
        config = make_config()
        result = make_result()

        exchange_dir = tmp_path / "llem-no-ts"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            _returned, ts_dir = runner.run(config)

        # No parquet rescue means ts_dir is None
        assert ts_dir is None

    def test_config_serialised_without_output_dir(self, tmp_path):
        """Config JSON written to exchange dir must not contain output_dir (passed via env)."""
        config = make_config()

        result = make_result()
        exchange_dir = tmp_path / "llem-cfg-check"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        # Read back the config JSON that was written for the container
        config_path = exchange_dir / f"{config_hash}_config.json"
        written_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "output_dir" not in written_config


# ---------------------------------------------------------------------------
# Test 16: Error JSON on non-zero exit
# ---------------------------------------------------------------------------


class TestErrorJsonOnNonZeroExit:
    def test_error_json_raises_with_payload(self, tmp_path):
        """When container exits non-zero AND error JSON exists, raises DockerContainerError
        with error_payload attribute containing the structured traceback."""
        config = make_config()
        exchange_dir = tmp_path / "llem-errorjson"
        exchange_dir.mkdir()

        config_hash = _docker_config_hash(config)
        error_payload = {
            "type": "RuntimeError",
            "message": "CUDA out of memory inside container",
            "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: CUDA OOM",
        }

        error_json_path = exchange_dir / f"{config_hash}_error.json"
        error_json_path.write_text(json.dumps(error_payload), encoding="utf-8")

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    _make_proc(1, stderr="some docker stderr noise")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError) as exc_info:
                runner.run(config)

        exc = exc_info.value
        # Structured payload attached to the exception
        assert hasattr(exc, "error_payload")
        assert exc.error_payload["type"] == "RuntimeError"
        assert exc.error_payload["message"] == "CUDA out of memory inside container"
        assert "traceback" in exc.error_payload
        # Error message includes the Python exception info
        assert "RuntimeError" in str(exc)
        assert "CUDA out of memory" in str(exc)
        # exchange_dir preserved for debugging
        assert exc.exchange_dir == str(exchange_dir)
        # container.log still written
        assert (exchange_dir / "container.log").exists()
        assert (exchange_dir / "container.log").read_text(encoding="utf-8") == (
            "some docker stderr noise"
        )

    def test_stderr_fallback_when_no_error_json(self, tmp_path):
        """When container exits non-zero and NO error JSON exists, DockerError is raised."""
        config = make_config()
        exchange_dir = tmp_path / "llem-noerrorjson"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(
                    _make_proc(1, stderr="Error: container failed with unknown reason")
                ),
            ),
        ):
            runner = DockerRunner(image=IMAGE)
            with pytest.raises(DockerContainerError):
                runner.run(config)


# ---------------------------------------------------------------------------
# Version mismatch warning
# ---------------------------------------------------------------------------


class TestVersionMismatchWarning:
    def test_warns_when_container_version_none(self, tmp_path, caplog):
        """Warning emitted when container result has no llenergymeasure_version."""
        import logging

        config = make_config()
        result = make_result()  # llenergymeasure_version defaults to None

        exchange_dir = tmp_path / "llem-version-none"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert any("rebuild Docker images" in r.message for r in caplog.records)

    def test_warns_when_container_version_differs(self, tmp_path, caplog):
        """Warning emitted when container version differs from host version."""
        import logging

        config = make_config()
        result = make_result(llenergymeasure_version="0.1.0")

        exchange_dir = tmp_path / "llem-version-diff"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert any("rebuild Docker images" in r.message for r in caplog.records)

    def test_no_warning_when_versions_match(self, tmp_path, caplog):
        """No warning when container version matches host version."""
        import logging

        from llenergymeasure._version import __version__

        config = make_config()
        result = make_result(llenergymeasure_version=__version__)

        exchange_dir = tmp_path / "llem-version-match"
        exchange_dir.mkdir()

        with (
            patch(
                "llenergymeasure.infra.docker_runner.tempfile.mkdtemp",
                return_value=str(exchange_dir),
            ),
            patch(
                "llenergymeasure.infra.docker_runner.subprocess.run",
                side_effect=_subprocess_run_with_image_cached(_make_proc(0)),
            ),
            patch("llenergymeasure.infra.docker_runner.shutil.rmtree"),
            caplog.at_level(logging.WARNING, logger="llenergymeasure.infra.docker_runner"),
        ):
            config_hash = _docker_config_hash(config)
            result_path = exchange_dir / f"{config_hash}_result.json"
            result_path.write_text(result.model_dump_json(), encoding="utf-8")

            runner = DockerRunner(image=IMAGE)
            runner.run(config)

        assert not any("rebuild Docker images" in r.message for r in caplog.records)
