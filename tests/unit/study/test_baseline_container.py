"""Tests for the host-side baseline container dispatch helper.

All tests patch ``subprocess.run`` at the module boundary — nothing actually
talks to Docker.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llenergymeasure.config.ssot import ENV_BASELINE_SPEC_PATH
from llenergymeasure.study import baseline_container

_MODULE = "llenergymeasure.study.baseline_container"


def _write_result(exchange_dir: Path, power_w: float = 42.59) -> None:
    (exchange_dir / "baseline_result.json").write_text(
        json.dumps(
            {
                "power_w": power_w,
                "timestamp": time.time(),
                "gpu_indices": [0],
                "sample_count": 288,
                "duration_sec": 30.0,
                "mode": "measure",
            }
        ),
        encoding="utf-8",
    )


def _make_subprocess_result(returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["docker"], returncode=returncode, stderr=stderr)


class TestBuildBaselineDockerCmd:
    def test_cmd_contains_mode_env_and_gpu_filter(self, tmp_path: Path):
        cmd = baseline_container.build_baseline_docker_cmd(
            image="ghcr.io/foo/bar:v1",
            exchange_dir=str(tmp_path),
            gpu_indices=[0, 2],
        )
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--gpus" in cmd
        # env vars
        assert any(f"{ENV_BASELINE_SPEC_PATH}=" in part for part in cmd)
        assert any("CUDA_VISIBLE_DEVICES=0,2" in part for part in cmd)
        # image
        assert "ghcr.io/foo/bar:v1" in cmd
        # entrypoint
        assert "python3" in cmd
        assert "-m" in cmd
        assert "llenergymeasure.entrypoints.baseline_measure" in cmd

    def test_cmd_empty_gpu_indices(self, tmp_path: Path):
        cmd = baseline_container.build_baseline_docker_cmd(
            image="img:latest",
            exchange_dir=str(tmp_path),
            gpu_indices=[],
        )
        assert any("CUDA_VISIBLE_DEVICES=" in part for part in cmd)


class TestRunBaselineContainerSuccess:
    def test_run_success_returns_baseline_cache(self, tmp_path: Path, monkeypatch):
        # Force mkdtemp to land inside tmp_path so the test can write the result file.
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _fake_run(cmd, **kwargs):
            _write_result(exchange_dir, power_w=42.6)
            return _make_subprocess_result(returncode=0)

        with patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=30.0,
                gpu_indices=[0],
            )

        assert result is not None
        assert result.power_w == pytest.approx(42.6)
        assert result.gpu_indices == [0]

    def test_exchange_dir_cleaned_on_success(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _fake_run(cmd, **kwargs):
            _write_result(exchange_dir)
            return _make_subprocess_result(returncode=0)

        with patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run):
            baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert not exchange_dir.exists()


class TestRunBaselineContainerFailure:
    def test_nonzero_exit_reads_error_json(self, tmp_path: Path, monkeypatch, caplog):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _fake_run(cmd, **kwargs):
            (exchange_dir / "baseline_error.json").write_text(
                json.dumps(
                    {
                        "type": "RuntimeError",
                        "message": "NVML unavailable",
                        "traceback": "...",
                    }
                ),
                encoding="utf-8",
            )
            return _make_subprocess_result(returncode=1, stderr="nvml error\n")

        with (
            caplog.at_level("WARNING"),
            patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run),
        ):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert result is None
        assert "NVML unavailable" in caplog.text
        # Exchange dir preserved for post-mortem
        assert exchange_dir.exists()

    def test_missing_result_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        with patch(
            f"{_MODULE}.subprocess.run",
            return_value=_make_subprocess_result(returncode=0),
        ):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert result is None

    def test_timeout_returns_none(self, tmp_path: Path, monkeypatch, caplog):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _raise_timeout(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1.0)

        with (
            caplog.at_level("WARNING"),
            patch(f"{_MODULE}.subprocess.run", side_effect=_raise_timeout),
        ):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert result is None
        assert "timed out" in caplog.text.lower()

    def test_malformed_result_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        def _fake_run(cmd, **kwargs):
            (exchange_dir / "baseline_result.json").write_text(
                "not valid json{{{", encoding="utf-8"
            )
            return _make_subprocess_result(returncode=0)

        with patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert result is None

    def test_docker_binary_missing_returns_none(self, tmp_path: Path, monkeypatch):
        exchange_dir = tmp_path / "exchange"
        exchange_dir.mkdir()
        monkeypatch.setattr(f"{_MODULE}.tempfile.mkdtemp", lambda prefix="": str(exchange_dir))

        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError("docker")):
            result = baseline_container.run_baseline_container(
                image="img:latest",
                mode="measure",
                duration_sec=1.0,
                gpu_indices=[0],
            )

        assert result is None
        # FileNotFoundError path cleans up (no diagnostic value in keeping dir)
        assert not exchange_dir.exists()
