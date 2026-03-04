"""Unit tests for cli/experiment.py.

This module is legacy v1.x dead code (not registered in the v2.0 CLI), but
its utility functions are still importable and have testable logic.

Tests cover:
- _is_json_output_mode() — env var detection
- _display_measurement_summary() — JSON mode path (repo is mocked)
- resolve_prompts() — priority/precedence logic (undefined names injected)
- _run_experiment_in_docker() — command construction logic
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import llenergymeasure.cli.experiment as _exp_mod

# ---------------------------------------------------------------------------
# _is_json_output_mode
# ---------------------------------------------------------------------------


class TestIsJsonOutputMode:
    def test_returns_true_when_env_var_set_to_true(self, monkeypatch):
        monkeypatch.setenv("LLM_ENERGY_JSON_OUTPUT", "true")
        assert _exp_mod._is_json_output_mode() is True

    def test_returns_false_when_env_var_unset(self, monkeypatch):
        monkeypatch.delenv("LLM_ENERGY_JSON_OUTPUT", raising=False)
        assert _exp_mod._is_json_output_mode() is False

    def test_returns_false_when_env_var_set_to_other_value(self, monkeypatch):
        monkeypatch.setenv("LLM_ENERGY_JSON_OUTPUT", "1")
        assert _exp_mod._is_json_output_mode() is False

    def test_returns_false_when_env_var_set_to_false(self, monkeypatch):
        monkeypatch.setenv("LLM_ENERGY_JSON_OUTPUT", "false")
        assert _exp_mod._is_json_output_mode() is False


# ---------------------------------------------------------------------------
# _display_measurement_summary — JSON mode path
# ---------------------------------------------------------------------------


class TestDisplayMeasurementSummaryJsonMode:
    """Test the JSON output path, which exercises _output_result_json."""

    def test_prints_json_when_json_mode_enabled(self, monkeypatch, capsys):
        monkeypatch.setenv("LLM_ENERGY_JSON_OUTPUT", "true")

        fake_result = MagicMock()
        fake_result.model_dump.return_value = {"total_energy_j": 10.0, "total_tokens": 100}
        repo = MagicMock()
        repo.load_aggregated.return_value = fake_result

        _exp_mod._display_measurement_summary(repo, "exp-001")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["experiment_id"] == "exp-001"
        assert data["total_energy_j"] == 10.0

    def test_outputs_no_result_status_when_none(self, monkeypatch, capsys):
        monkeypatch.setenv("LLM_ENERGY_JSON_OUTPUT", "true")

        repo = MagicMock()
        repo.load_aggregated.return_value = None

        _exp_mod._display_measurement_summary(repo, "exp-002")

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["experiment_id"] == "exp-002"
        assert data["status"] == "no_aggregated_result"

    def test_non_json_mode_returns_silently_on_none(self, monkeypatch):
        """Non-JSON mode with None result should not raise."""
        monkeypatch.delenv("LLM_ENERGY_JSON_OUTPUT", raising=False)

        repo = MagicMock()
        repo.load_aggregated.return_value = None
        # Should not raise
        _exp_mod._display_measurement_summary(repo, "exp-003")

    def test_non_json_mode_swallows_exceptions(self, monkeypatch):
        """Non-JSON mode wraps display in try/except — must not propagate."""
        monkeypatch.delenv("LLM_ENERGY_JSON_OUTPUT", raising=False)

        repo = MagicMock()
        repo.load_aggregated.side_effect = RuntimeError("display error")
        # Must not raise
        _exp_mod._display_measurement_summary(repo, "exp-004")


# ---------------------------------------------------------------------------
# resolve_prompts — priority logic
#
# resolve_prompts() references undefined names (v1.x dead code, F821 suppressed).
# We inject those names into the module's global namespace before calling,
# then restore the originals. This is the only way to test the function
# without the deleted imports.
# ---------------------------------------------------------------------------


class TestResolvePrompts:
    """Test resolve_prompts() priority ordering with injected fakes."""

    def _make_config(self, dataset=None, prompts=None):
        """Build a minimal config-like object for testing."""
        cfg = MagicMock()
        cfg.dataset = dataset
        cfg.prompts = prompts
        return cfg

    def _inject_fakes(self, monkeypatch, source_factory, load_from_source, load_from_file=None):
        """Inject fake callables into the module's globals."""
        monkeypatch.setattr(_exp_mod, "HuggingFacePromptSource", source_factory, raising=False)
        monkeypatch.setattr(_exp_mod, "load_prompts_from_source", load_from_source, raising=False)
        if load_from_file is not None:
            monkeypatch.setattr(_exp_mod, "load_prompts_from_file", load_from_file, raising=False)
        # Prevent DEFAULT_DATASET lookup from failing in the fallback branch
        monkeypatch.setattr(_exp_mod, "DEFAULT_DATASET", "ai-energy-score", raising=False)
        # Prevent console.print calls from failing
        monkeypatch.setattr(_exp_mod, "console", MagicMock(), raising=False)

    def test_cli_dataset_takes_highest_priority(self, monkeypatch):
        fake_source = MagicMock()
        fake_load = MagicMock(return_value=["cli-prompt"])
        self._inject_fakes(monkeypatch, lambda **kw: fake_source, fake_load)

        config = self._make_config()
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=None,
            dataset="some-dataset",
            dataset_split="train",
            dataset_column=None,
            sample_size=None,
        )

        assert result == ["cli-prompt"]
        fake_load.assert_called_once_with(fake_source)

    def test_cli_prompts_file_used_when_no_dataset(self, monkeypatch, tmp_path):
        fake_load_from_file = MagicMock(return_value=["file-prompt-1", "file-prompt-2"])
        self._inject_fakes(monkeypatch, MagicMock(), MagicMock(), fake_load_from_file)

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.touch()

        config = self._make_config()
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=prompts_file,
            dataset=None,
            dataset_split="train",
            dataset_column=None,
            sample_size=None,
        )

        assert result == ["file-prompt-1", "file-prompt-2"]
        fake_load_from_file.assert_called_once_with(prompts_file)

    def test_sample_size_truncates_prompts_file_results(self, monkeypatch, tmp_path):
        fake_load_from_file = MagicMock(return_value=["p1", "p2", "p3", "p4"])
        self._inject_fakes(monkeypatch, MagicMock(), MagicMock(), fake_load_from_file)

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.touch()

        config = self._make_config()
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=prompts_file,
            dataset=None,
            dataset_split="train",
            dataset_column=None,
            sample_size=2,
        )

        assert result == ["p1", "p2"]

    def test_config_dataset_used_when_no_cli_override(self, monkeypatch):
        fake_source = MagicMock()
        fake_load = MagicMock(return_value=["config-dataset-prompt"])
        self._inject_fakes(monkeypatch, lambda **kw: fake_source, fake_load)

        cfg_dataset = MagicMock()
        cfg_dataset.name = "config-dataset"
        cfg_dataset.split = "test"
        cfg_dataset.column = "text"
        cfg_dataset.sample_size = 10

        config = self._make_config(dataset=cfg_dataset)
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=None,
            dataset=None,
            dataset_split="train",
            dataset_column=None,
            sample_size=None,
        )

        assert result == ["config-dataset-prompt"]

    def test_config_prompts_source_used_as_fallback(self, monkeypatch):
        fake_source = MagicMock()
        fake_load = MagicMock(return_value=["config-prompts-result"])
        self._inject_fakes(monkeypatch, MagicMock(), fake_load)

        config = self._make_config(prompts=fake_source)
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=None,
            dataset=None,
            dataset_split="train",
            dataset_column=None,
            sample_size=None,
        )

        assert result == ["config-prompts-result"]
        fake_load.assert_called_once_with(fake_source)

    def test_dataset_priority_over_prompts_file(self, monkeypatch, tmp_path):
        """CLI --dataset takes priority over CLI --prompts file."""
        fake_source = MagicMock()
        fake_load_from_source = MagicMock(return_value=["dataset-result"])
        fake_load_from_file = MagicMock(return_value=["file-result"])
        self._inject_fakes(
            monkeypatch,
            lambda **kw: fake_source,
            fake_load_from_source,
            fake_load_from_file,
        )

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.touch()

        config = self._make_config()
        result = _exp_mod.resolve_prompts(
            config=config,
            prompts_file=prompts_file,
            dataset="override-dataset",
            dataset_split="train",
            dataset_column=None,
            sample_size=None,
        )

        assert result == ["dataset-result"]
        fake_load_from_file.assert_not_called()


# ---------------------------------------------------------------------------
# _run_experiment_in_docker — command construction
# ---------------------------------------------------------------------------


class TestRunExperimentInDocker:
    """Test Docker command construction without actually running Docker.

    _run_experiment_in_docker() references 'console' — an undefined name
    (deleted v1.x import). We inject a MagicMock into the module's globals
    before calling, then let monkeypatch restore the original (absent) state.
    """

    def _setup_docker_mocks(self, monkeypatch):
        """Inject missing v1.x globals needed by _run_experiment_in_docker."""
        monkeypatch.setattr(_exp_mod, "console", MagicMock(), raising=False)

    def test_builds_docker_compose_run_command(self, monkeypatch):
        self._setup_docker_mocks(monkeypatch)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("LLM_ENERGY_VERBOSITY", "normal")

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with (
            patch("llenergymeasure.config.env_setup.ensure_env_file"),
            patch("llenergymeasure.cli.experiment.subprocess.run", mock_run),
        ):
            result = _exp_mod._run_experiment_in_docker(
                config_path=None,
                backend="pytorch",
                dataset=None,
                sample_size=None,
                results_dir=None,
            )

        assert result == 0
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[:3] == ["docker", "compose", "run"]
        assert "pytorch" in called_cmd

    def test_passes_hf_token_env_var_when_set(self, monkeypatch):
        self._setup_docker_mocks(monkeypatch)
        monkeypatch.setenv("HF_TOKEN", "test-token-xyz")
        monkeypatch.setenv("LLM_ENERGY_VERBOSITY", "normal")

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with (
            patch("llenergymeasure.config.env_setup.ensure_env_file"),
            patch("llenergymeasure.cli.experiment.subprocess.run", mock_run),
        ):
            _exp_mod._run_experiment_in_docker(
                config_path=None,
                backend="vllm",
                dataset=None,
                sample_size=None,
                results_dir=None,
            )

        called_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(called_cmd)
        assert "HF_TOKEN=test-token-xyz" in cmd_str

    def test_includes_dataset_flag_when_provided(self, monkeypatch):
        self._setup_docker_mocks(monkeypatch)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("LLM_ENERGY_VERBOSITY", "normal")

        mock_run = MagicMock(return_value=MagicMock(returncode=0))

        with (
            patch("llenergymeasure.config.env_setup.ensure_env_file"),
            patch("llenergymeasure.cli.experiment.subprocess.run", mock_run),
        ):
            _exp_mod._run_experiment_in_docker(
                config_path=None,
                backend="vllm",
                dataset="alpaca",
                sample_size=50,
                results_dir=None,
            )

        called_cmd = mock_run.call_args[0][0]
        assert "--dataset" in called_cmd
        assert "alpaca" in called_cmd
        assert "--sample-size" in called_cmd
        assert "50" in called_cmd

    def test_returns_nonzero_exit_code_from_docker(self, monkeypatch):
        self._setup_docker_mocks(monkeypatch)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("LLM_ENERGY_VERBOSITY", "normal")

        mock_run = MagicMock(return_value=MagicMock(returncode=1))

        with (
            patch("llenergymeasure.config.env_setup.ensure_env_file"),
            patch("llenergymeasure.cli.experiment.subprocess.run", mock_run),
        ):
            result = _exp_mod._run_experiment_in_docker(
                config_path=None,
                backend="vllm",
                dataset=None,
                sample_size=None,
                results_dir=None,
            )

        assert result == 1

    def test_includes_config_path_in_container_command(self, monkeypatch):
        self._setup_docker_mocks(monkeypatch)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("LLM_ENERGY_VERBOSITY", "normal")

        mock_run = MagicMock(return_value=MagicMock(returncode=0))
        config_path = Path("configs/experiment.yaml")

        with (
            patch("llenergymeasure.config.env_setup.ensure_env_file"),
            patch("llenergymeasure.cli.experiment.subprocess.run", mock_run),
        ):
            _exp_mod._run_experiment_in_docker(
                config_path=config_path,
                backend="pytorch",
                dataset=None,
                sample_size=None,
                results_dir=None,
            )

        called_cmd = mock_run.call_args[0][0]
        # Config path should be prefixed with /app/
        assert any("/app/" in arg for arg in called_cmd)
