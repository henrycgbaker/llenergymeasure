"""Tests for the container entrypoint module.

All tests run without GPU hardware or real Docker containers.
Backend execution is replaced via unittest.mock.patch targeting
the source modules (since container_entrypoint uses lazy local imports).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_config, make_result

# Patch targets: patch the source modules, not container_entrypoint references,
# because container_entrypoint imports these inside function scope.
_PATCH_PREFLIGHT = "llenergymeasure.orchestration.preflight.run_preflight"
_PATCH_GET_BACKEND = "llenergymeasure.core.backends.get_backend"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path):
    """A minimal ExperimentConfig serialised to a JSON file."""
    cfg = make_config(model="gpt2", backend="pytorch")
    config_json = tmp_path / "abc123_config.json"
    config_json.write_text(cfg.model_dump_json(), encoding="utf-8")
    return cfg, config_json


@pytest.fixture
def result_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# run_container_experiment
# ---------------------------------------------------------------------------


class TestRunContainerExperiment:
    def test_writes_result_json_with_config_hash_name(
        self, config, result_dir: Path, tmp_path: Path
    ):
        _cfg, config_path = config
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT) as mock_preflight,
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_preflight.return_value = None
            mock_backend = MagicMock()
            mock_backend.run.return_value = fake_result
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import run_container_experiment

            result_path = run_container_experiment(config_path, result_dir)

        # File must exist
        assert result_path.exists()

        # Name must follow {config_hash}_result.json pattern
        assert result_path.name.endswith("_result.json")
        assert result_path.parent == result_dir

        # Content must be valid JSON
        data = json.loads(result_path.read_text())
        assert "experiment_id" in data

    def test_calls_preflight_before_backend(self, config, result_dir: Path):
        _cfg, config_path = config
        fake_result = make_result()
        call_order = []

        with (
            patch(_PATCH_PREFLIGHT) as mock_preflight,
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):

            def track_preflight(c):
                call_order.append("preflight")

            def track_get_backend(name):
                call_order.append("get_backend")
                b = MagicMock()
                b.run.return_value = fake_result
                return b

            mock_preflight.side_effect = track_preflight
            mock_get_backend.side_effect = track_get_backend

            from llenergymeasure.infra.container_entrypoint import run_container_experiment

            run_container_experiment(config_path, result_dir)

        assert call_order == ["preflight", "get_backend"]

    def test_result_dir_created_if_missing(self, config, tmp_path: Path):
        _cfg, config_path = config
        new_result_dir = tmp_path / "nonexistent" / "deep"
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.run.return_value = fake_result
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import run_container_experiment

            result_path = run_container_experiment(config_path, new_result_dir)

        assert new_result_dir.exists()
        assert result_path.exists()

    def test_backend_failure_propagates(self, config, result_dir: Path):
        _cfg, config_path = config

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.run.side_effect = RuntimeError("GPU exploded")
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import run_container_experiment

            with pytest.raises(RuntimeError, match="GPU exploded"):
                run_container_experiment(config_path, result_dir)


# ---------------------------------------------------------------------------
# Error handling — main() writes error JSON on failure
# ---------------------------------------------------------------------------


class TestMainErrorHandling:
    def test_main_writes_error_json_on_failure(self, tmp_path: Path, monkeypatch):
        cfg = make_config(model="gpt2", backend="pytorch")
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")

        monkeypatch.setenv("LLEM_CONFIG_PATH", str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.run.side_effect = ValueError("model not found")
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import main

            # main() re-raises the exception after writing the error file
            with pytest.raises(ValueError, match="model not found"):
                main()

        # Error JSON should have been written in the same dir as the config
        error_files = list(tmp_path.glob("*_error.json"))
        assert len(error_files) == 1
        error_data = json.loads(error_files[0].read_text())
        assert error_data["type"] == "ValueError"
        assert "model not found" in error_data["message"]
        assert "traceback" in error_data

    def test_main_reads_llem_config_path_env_var(self, tmp_path: Path, monkeypatch):
        cfg = make_config()
        config_path = tmp_path / "xyz789_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")
        fake_result = make_result()

        monkeypatch.setenv("LLEM_CONFIG_PATH", str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.run.return_value = fake_result
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import main

            main()  # should not raise

        result_files = list(tmp_path.glob("*_result.json"))
        assert len(result_files) == 1

    def test_main_raises_if_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("LLEM_CONFIG_PATH", raising=False)

        from llenergymeasure.infra.container_entrypoint import main

        with pytest.raises(RuntimeError, match="LLEM_CONFIG_PATH"):
            main()

    def test_error_json_has_required_keys(self, tmp_path: Path, monkeypatch):
        cfg = make_config()
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")

        monkeypatch.setenv("LLEM_CONFIG_PATH", str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_BACKEND) as mock_get_backend,
        ):
            mock_backend = MagicMock()
            mock_backend.run.side_effect = Exception("boom")
            mock_get_backend.return_value = mock_backend

            from llenergymeasure.infra.container_entrypoint import main

            with pytest.raises(Exception, match="boom"):
                main()

        error_files = list(tmp_path.glob("*_error.json"))
        assert error_files
        data = json.loads(error_files[0].read_text())
        assert set(data.keys()) >= {"type", "message", "traceback"}
