"""Tests for study-level pre-flight checks (CM-10, DOCK-05)."""

import pytest

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.exceptions import PreFlightError
from llenergymeasure.orchestration.preflight import run_study_preflight


def test_single_backend_passes():
    """Single-backend study passes pre-flight without error."""
    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="m1", backend="pytorch"),
            ExperimentConfig(model="m2", backend="pytorch"),
        ]
    )
    run_study_preflight(study)  # should not raise


def test_multi_backend_raises_preflight_error(monkeypatch):
    """Multi-backend study raises PreFlightError when Docker is not available."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="m1", backend="pytorch"),
            ExperimentConfig(model="m2", backend="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Multi-backend"):
        run_study_preflight(study)


def test_multi_backend_error_mentions_docker(monkeypatch):
    """Error message directs user to Docker runner."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="m1", backend="pytorch"),
            ExperimentConfig(model="m2", backend="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Docker"):
        run_study_preflight(study)


def test_multi_backend_error_lists_backends(monkeypatch):
    """Error message lists the detected backends."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="m1", backend="pytorch"),
            ExperimentConfig(model="m2", backend="vllm"),
        ]
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study)
    assert "pytorch" in str(exc_info.value)
    assert "vllm" in str(exc_info.value)
