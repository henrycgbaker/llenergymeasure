"""Tests for study-level pre-flight checks (CM-10, DOCK-05)."""

from unittest.mock import MagicMock

import pytest

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.utils.exceptions import PreFlightError


def test_single_engine_passes(monkeypatch):
    """Single-engine study passes pre-flight without error."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="transformers"),
        ]
    )
    run_study_preflight(study)  # should not raise


def test_multi_engine_raises_preflight_error(monkeypatch):
    """Multi-engine study raises PreFlightError when Docker is not available."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Multi-engine"):
        run_study_preflight(study)


def test_multi_engine_error_mentions_docker(monkeypatch):
    """Error message directs user to Docker runner."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Docker"):
        run_study_preflight(study)


def test_multi_engine_error_lists_engines(monkeypatch):
    """Error message lists the detected engines."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study)
    assert "transformers" in str(exc_info.value)
    assert "vllm" in str(exc_info.value)


def test_multi_engine_auto_elevates_local_to_docker(monkeypatch):
    """Multi-engine study overrides local runners to Docker when Docker is available."""
    monkeypatch.setattr("llenergymeasure.infra.runner_resolution.is_docker_available", lambda: True)
    # Mock docker preflight to avoid real Docker checks
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", lambda skip=False: None
    )

    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    specs, overrides = run_study_preflight(
        study, yaml_runners={"transformers": "local", "vllm": "docker"}
    )
    assert specs["transformers"].mode == "docker"
    assert specs["transformers"].source == "multi_engine_elevation"
    assert specs["vllm"].mode == "docker"
    # System overrides should capture the auto-elevation
    assert "runner.transformers" in overrides
    assert overrides["runner.transformers"]["declared"] == "local"
    assert overrides["runner.transformers"]["effective"] == "docker"


def test_preflight_forwards_runner_context(monkeypatch):
    """run_study_preflight forwards yaml_runners and user_config to resolve_study_runners."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        # Return local specs so no Docker preflight is triggered
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    mock_user_config = MagicMock()
    study = StudyConfig(experiments=[ExperimentConfig(task={"model": "m1"}, engine="transformers")])

    run_study_preflight(study, yaml_runners={"transformers": "local"}, user_config=mock_user_config)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] == {"transformers": "local"}
    assert captured_calls[0]["user_config"] is mock_user_config


def test_preflight_defaults_to_auto_detect_without_context(monkeypatch):
    """Calling run_study_preflight without yaml_runners/user_config passes None for both."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    study = StudyConfig(experiments=[ExperimentConfig(task={"model": "m1"}, engine="transformers")])

    run_study_preflight(study)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] is None
    assert captured_calls[0]["user_config"] is None
