"""Tests for the container entrypoint module.

All tests run without GPU hardware or real Docker containers.
Engine execution is replaced via unittest.mock.patch targeting
the source modules (since container_entrypoint uses lazy local imports).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.ssot import CONTAINER_EXCHANGE_DIR, ENV_CONFIG_PATH
from tests.conftest import make_config, make_result

# Patch targets: patch the source modules, not container_entrypoint references,
# because container_entrypoint imports these inside function scope.
_PATCH_PREFLIGHT = "llenergymeasure.harness.preflight.run_preflight"
_PATCH_GET_ENGINE = "llenergymeasure.engines.get_engine"
_PATCH_HARNESS_RUN = "llenergymeasure.harness.MeasurementHarness.run"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path):
    """A minimal ExperimentConfig serialised to a JSON file."""
    cfg = make_config(model="gpt2", engine="pytorch")
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
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result),
        ):
            mock_preflight.return_value = None
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import run_container_experiment

            result_path = run_container_experiment(config_path, result_dir)

        # File must exist
        assert result_path.exists()

        # Name must follow {config_hash}_result.json pattern
        assert result_path.name.endswith("_result.json")
        assert result_path.parent == result_dir

        # Content must be valid JSON
        data = json.loads(result_path.read_text())
        assert "experiment_id" in data

    def test_calls_preflight_before_engine(self, config, result_dir: Path):
        _cfg, config_path = config
        fake_result = make_result()
        call_order = []

        with (
            patch(_PATCH_PREFLIGHT) as mock_preflight,
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result),
        ):

            def track_preflight(c):
                call_order.append("preflight")

            def track_get_engine(name):
                call_order.append("get_engine")
                return MagicMock()

            mock_preflight.side_effect = track_preflight
            mock_get_engine.side_effect = track_get_engine

            from llenergymeasure.entrypoints.container import run_container_experiment

            run_container_experiment(config_path, result_dir)

        assert call_order == ["preflight", "get_engine"]

    def test_result_dir_created_if_missing(self, config, tmp_path: Path):
        _cfg, config_path = config
        new_result_dir = tmp_path / "nonexistent" / "deep"
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result),
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import run_container_experiment

            result_path = run_container_experiment(config_path, new_result_dir)

        assert new_result_dir.exists()
        assert result_path.exists()

    def test_engine_failure_propagates(self, config, result_dir: Path):
        _cfg, config_path = config

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, side_effect=RuntimeError("GPU exploded")),
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import run_container_experiment

            with pytest.raises(RuntimeError, match="GPU exploded"):
                run_container_experiment(config_path, result_dir)


# ---------------------------------------------------------------------------
# Error handling — main() writes error JSON on failure
# ---------------------------------------------------------------------------


class TestMainErrorHandling:
    def test_main_writes_error_json_on_failure(self, tmp_path: Path, monkeypatch):
        cfg = make_config(model="gpt2", engine="pytorch")
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")

        monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, side_effect=ValueError("model not found")),
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import main

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

        monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result),
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import main

            main()  # should not raise

        result_files = list(tmp_path.glob("*_result.json"))
        assert len(result_files) == 1

    def test_main_raises_if_env_var_missing(self, monkeypatch):
        monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)

        from llenergymeasure.entrypoints.container import main

        with pytest.raises(RuntimeError, match=ENV_CONFIG_PATH):
            main()

    def test_error_json_has_required_keys(self, tmp_path: Path, monkeypatch):
        cfg = make_config()
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")

        monkeypatch.setenv(ENV_CONFIG_PATH, str(config_path))

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, side_effect=Exception("boom")),
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import main

            with pytest.raises(Exception, match="boom"):
                main()

        error_files = list(tmp_path.glob("*_error.json"))
        assert error_files
        data = json.loads(error_files[0].read_text())
        assert set(data.keys()) >= {"type", "message", "traceback"}


# ---------------------------------------------------------------------------
# Container-side baseline loading (PR #242)
# ---------------------------------------------------------------------------


_PATCH_LOAD_BASELINE = "llenergymeasure.harness.baseline.load_baseline_cache"


class TestContainerBaselineLoading:
    """Tests for baseline cache loading inside Docker containers.

    The container entrypoint loads a host-persisted baseline from
    /run/llem/baseline_cache.json when it exists and baseline is enabled.
    """

    def test_loads_baseline_when_cache_file_exists(self, config, result_dir: Path, tmp_path: Path):
        """Baseline loaded from disk when cache file exists and baseline enabled."""
        _cfg, config_path = config
        fake_result = make_result()
        fake_baseline = MagicMock()
        fake_baseline.power_w = 55.0

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result) as mock_run,
            patch(_PATCH_LOAD_BASELINE, return_value=fake_baseline) as mock_load,
            patch("llenergymeasure.entrypoints.container.Path") as mock_path_cls,
        ):
            mock_get_engine.return_value = MagicMock()

            # Make the baseline cache path report as existing
            mock_cache_path = MagicMock()
            mock_cache_path.exists.return_value = True

            def path_factory(p):
                if p == f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json":
                    return mock_cache_path
                return Path(p)

            mock_path_cls.side_effect = path_factory

            from llenergymeasure.entrypoints.container import run_container_experiment

            run_container_experiment(config_path, result_dir)

        # Baseline was loaded from disk
        mock_load.assert_called_once()
        # Baseline was passed to harness.run
        _, kwargs = mock_run.call_args
        assert kwargs.get("baseline") is fake_baseline

    def test_no_baseline_when_cache_file_missing(self, config, result_dir: Path, tmp_path: Path):
        """No baseline loaded when cache file does not exist."""
        _cfg, config_path = config
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result) as mock_run,
            patch(_PATCH_LOAD_BASELINE) as mock_load,
        ):
            mock_get_engine.return_value = MagicMock()

            from llenergymeasure.entrypoints.container import run_container_experiment

            # /run/llem/baseline_cache.json doesn't exist (default in test env)
            run_container_experiment(config_path, result_dir)

        mock_load.assert_not_called()
        _, kwargs = mock_run.call_args
        assert kwargs.get("baseline") is None

    def test_no_baseline_when_disabled(self, tmp_path: Path, result_dir: Path):
        """Baseline not loaded even if cache exists when baseline.enabled=False."""
        cfg = make_config(model="gpt2", engine="pytorch", baseline={"enabled": False})
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result) as mock_run,
            patch(_PATCH_LOAD_BASELINE) as mock_load,
            patch("llenergymeasure.entrypoints.container.Path") as mock_path_cls,
        ):
            mock_get_engine.return_value = MagicMock()
            mock_cache_path = MagicMock()
            mock_cache_path.exists.return_value = True

            def path_factory(p):
                if p == f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json":
                    return mock_cache_path
                return Path(p)

            mock_path_cls.side_effect = path_factory

            from llenergymeasure.entrypoints.container import run_container_experiment

            run_container_experiment(config_path, result_dir)

        mock_load.assert_not_called()
        _, kwargs = mock_run.call_args
        assert kwargs.get("baseline") is None

    def test_uses_config_ttl_for_loading(self, tmp_path: Path, result_dir: Path):
        """load_baseline_cache is called with the config's cache_ttl_seconds."""
        cfg = make_config(
            model="gpt2",
            engine="pytorch",
            baseline={"enabled": True, "cache_ttl_seconds": 900.0},
        )
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result),
            patch(_PATCH_LOAD_BASELINE, return_value=MagicMock()) as mock_load,
            patch("llenergymeasure.entrypoints.container.Path") as mock_path_cls,
        ):
            mock_get_engine.return_value = MagicMock()
            mock_cache_path = MagicMock()
            mock_cache_path.exists.return_value = True

            def path_factory(p):
                if p == f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json":
                    return mock_cache_path
                return Path(p)

            mock_path_cls.side_effect = path_factory

            from llenergymeasure.entrypoints.container import run_container_experiment

            run_container_experiment(config_path, result_dir)

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("ttl") == 900.0

    def test_expired_cache_passes_none_baseline(self, tmp_path: Path, result_dir: Path):
        """When disk cache is expired (load returns None), baseline=None."""
        cfg = make_config(model="gpt2", engine="pytorch")
        config_path = tmp_path / "abc123_config.json"
        config_path.write_text(cfg.model_dump_json(), encoding="utf-8")
        fake_result = make_result()

        with (
            patch(_PATCH_PREFLIGHT),
            patch(_PATCH_GET_ENGINE) as mock_get_engine,
            patch(_PATCH_HARNESS_RUN, return_value=fake_result) as mock_run,
            patch(_PATCH_LOAD_BASELINE, return_value=None),
            patch("llenergymeasure.entrypoints.container.Path") as mock_path_cls,
        ):
            mock_get_engine.return_value = MagicMock()
            mock_cache_path = MagicMock()
            mock_cache_path.exists.return_value = True

            def path_factory(p):
                if p == f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json":
                    return mock_cache_path
                return Path(p)

            mock_path_cls.side_effect = path_factory

            from llenergymeasure.entrypoints.container import run_container_experiment

            run_container_experiment(config_path, result_dir)

        _, kwargs = mock_run.call_args
        assert kwargs.get("baseline") is None


# ---------------------------------------------------------------------------
# __main__ guard — MPI safety contract
# ---------------------------------------------------------------------------


class TestMainGuard:
    def test_main_guard_exists_at_module_level(self):
        """The __main__ guard prevents MPI workers from re-executing main().

        When mpirun spawns worker ranks, they re-import the module but
        __name__ is the dotted module path (not "__main__"), so main()
        is not called. This test documents that contract.
        """
        import inspect

        from llenergymeasure.entrypoints import container

        source = inspect.getsource(container)
        assert 'if __name__ == "__main__":' in source
