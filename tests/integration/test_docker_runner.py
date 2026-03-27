"""Docker runner integration tests — real container dispatch.

These tests exercise the full Docker runner path: config serialisation →
docker run → container entrypoint → result JSON → host read-back.

They require Docker with NVIDIA Container Toolkit and a pre-built
``llenergymeasure-ci:pytorch`` image (built by gpu-ci.yml).

Run: pytest tests/integration/test_docker_runner.py -m docker -v
Requires: Docker, nvidia-ctk, GPU, pre-built llenergymeasure-ci:pytorch image
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

IMAGE = "llenergymeasure-ci:pytorch"


def _docker_available() -> bool:
    """Check Docker daemon and NVIDIA Container Toolkit are reachable."""
    if shutil.which("docker") is None:
        return False
    # Check daemon is running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    # Check NVIDIA runtime
    return any(
        shutil.which(tool) is not None
        for tool in ("nvidia-container-runtime", "nvidia-ctk", "nvidia-container-cli")
    )


def _image_exists(image: str) -> bool:
    """Check if a Docker image is available locally."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", image],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


pytestmark = pytest.mark.docker


@pytest.mark.skipif(not _docker_available(), reason="Docker + NVIDIA toolkit not available")
@pytest.mark.skipif(not _image_exists(IMAGE), reason=f"Image {IMAGE!r} not built")
class TestDockerRunnerIntegration:
    """Full round-trip tests through the real Docker runner."""

    def _make_config(self, tmp_path, **overrides):
        """Create a minimal ExperimentConfig for Docker dispatch."""
        from llenergymeasure.config.models import (
            BaselineConfig,
            DatasetConfig,
            ExperimentConfig,
            WarmupConfig,
        )

        defaults = dict(
            model="gpt2",
            backend="pytorch",
            dataset=DatasetConfig(n_prompts=3),
            output_dir=str(tmp_path),
            warmup=WarmupConfig(enabled=False),
            baseline=BaselineConfig(enabled=False),
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_round_trip_single_experiment(self, tmp_path):
        """DockerRunner.run() dispatches to container and returns ExperimentResult."""
        from llenergymeasure.domain.experiment import ExperimentResult
        from llenergymeasure.infra.docker_runner import DockerRunner

        config = self._make_config(tmp_path)
        runner = DockerRunner(image=IMAGE, timeout=300, source="test")
        result = runner.run(config)

        assert isinstance(result, ExperimentResult)
        assert result.total_energy_j > 0
        assert result.avg_tokens_per_second > 0
        assert result.total_tokens > 0

    def test_runner_metadata_injected(self, tmp_path):
        """Result effective_config contains Docker runner metadata."""
        from llenergymeasure.infra.docker_runner import DockerRunner

        config = self._make_config(tmp_path)
        runner = DockerRunner(image=IMAGE, timeout=300, source="test")
        result = runner.run(config)

        ec = result.effective_config or {}
        assert ec.get("runner_mode") == "docker"
        assert ec.get("runner_image") == IMAGE
        assert ec.get("runner_source") == "test"

    def test_exchange_dir_cleaned_on_success(self, tmp_path):
        """Exchange dir is removed after successful run (no temp dir leak)."""
        import glob

        from llenergymeasure.infra.docker_runner import DockerRunner

        # Count llem- temp dirs before
        before = set(glob.glob("/tmp/llem-*"))

        config = self._make_config(tmp_path)
        runner = DockerRunner(image=IMAGE, timeout=300, source="test")
        runner.run(config)

        # After success, no new llem- dirs should remain
        after = set(glob.glob("/tmp/llem-*"))
        new_dirs = after - before
        assert len(new_dirs) == 0, f"Leaked exchange dirs: {new_dirs}"

    def test_docker_error_on_missing_image(self, tmp_path):
        """Non-existent image raises DockerContainerError or DockerImagePullError."""
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.utils.exceptions import DockerError

        config = self._make_config(tmp_path)
        runner = DockerRunner(image="llenergymeasure-nonexistent:latest", timeout=60, source="test")

        with pytest.raises(DockerError):
            runner.run(config)

    def test_study_runner_dispatches_to_docker(self, tmp_path):
        """StudyRunner routes to Docker path when runner spec has mode='docker'."""
        from unittest.mock import patch

        from llenergymeasure.config.models import (
            BaselineConfig,
            DatasetConfig,
            ExecutionConfig,
            ExperimentConfig,
            StudyConfig,
            WarmupConfig,
        )
        from llenergymeasure.domain.experiment import ExperimentResult
        from llenergymeasure.study.runner import StudyRunner

        config = ExperimentConfig(
            model="gpt2",
            backend="pytorch",
            dataset=DatasetConfig(n_prompts=3),
            output_dir=str(tmp_path),
            warmup=WarmupConfig(enabled=False),
            baseline=BaselineConfig(enabled=False),
        )
        study = StudyConfig(
            experiments=[config],
            study_execution=ExecutionConfig(
                n_cycles=1,
                experiment_order="sequential",
                experiment_gap_seconds=0,
            ),
        )

        # Patch resolve_study_runners to force Docker dispatch
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        docker_spec = RunnerSpec(mode="docker", image=IMAGE, source="test")

        with patch(
            "llenergymeasure.study.runner.resolve_study_runners",
            return_value={"pytorch": docker_spec},
        ):
            runner = StudyRunner(study, output_dir=tmp_path)
            result = runner.run()

        assert len(result.experiments) == 1
        exp = result.experiments[0]
        assert isinstance(exp, ExperimentResult)
        assert exp.total_energy_j > 0
