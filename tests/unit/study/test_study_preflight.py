"""Tests for study-level pre-flight checks (CM-10, DOCK-05)."""

from unittest.mock import MagicMock

import pytest

from llenergymeasure.api.preflight import run_study_preflight
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.utils.exceptions import PreFlightError


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


def test_preflight_forwards_runner_context(monkeypatch):
    """run_study_preflight forwards yaml_runners and user_config to resolve_study_runners."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(backends, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        # Return local specs so no Docker preflight is triggered
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in backends}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    mock_user_config = MagicMock()
    study = StudyConfig(experiments=[ExperimentConfig(model="m1", backend="pytorch")])

    run_study_preflight(study, yaml_runners={"pytorch": "local"}, user_config=mock_user_config)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] == {"pytorch": "local"}
    assert captured_calls[0]["user_config"] is mock_user_config


def test_preflight_defaults_to_auto_detect_without_context(monkeypatch):
    """Calling run_study_preflight without yaml_runners/user_config passes None for both."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(backends, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in backends}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    study = StudyConfig(experiments=[ExperimentConfig(model="m1", backend="pytorch")])

    run_study_preflight(study)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] is None
    assert captured_calls[0]["user_config"] is None
