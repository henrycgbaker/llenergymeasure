"""Unit tests for runner resolution precedence chain.

Tests cover:
  - parse_runner_value: "local", "docker", "docker:image" forms
  - is_docker_available: PATH inspection for docker + NVIDIA CT tools
  - resolve_runner: full precedence chain (env > yaml > user_config > auto > default)
  - resolve_study_runners: multi-engine resolution
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llenergymeasure.config.ssot import ENV_RUNNER_PREFIX
from llenergymeasure.config.user_config import UserRunnersConfig
from llenergymeasure.infra.runner_resolution import (
    RunnerSpec,
    is_docker_available,
    parse_runner_value,
    resolve_runner,
    resolve_study_runners,
)


@pytest.fixture(autouse=True)
def _clear_docker_cache():
    is_docker_available.cache_clear()
    yield
    is_docker_available.cache_clear()


# ---------------------------------------------------------------------------
# parse_runner_value
# ---------------------------------------------------------------------------


class TestParseRunnerValue:
    def test_local_returns_local_mode_no_image(self):
        mode, image = parse_runner_value("local")
        assert mode == "local"
        assert image is None

    def test_bare_docker_returns_docker_mode_no_image(self):
        mode, image = parse_runner_value("docker")
        assert mode == "docker"
        assert image is None

    def test_docker_with_image_returns_docker_mode_and_image(self):
        mode, image = parse_runner_value("docker:ghcr.io/custom/img:v1")
        assert mode == "docker"
        assert image == "ghcr.io/custom/img:v1"

    def test_docker_with_complex_image_tag(self):
        mode, image = parse_runner_value("docker:nvcr.io/nvidia/pytorch:23.10-py3")
        assert mode == "docker"
        assert image == "nvcr.io/nvidia/pytorch:23.10-py3"

    def test_docker_colon_empty_raises(self):
        """'docker:' with empty image raises ValueError."""
        with pytest.raises(ValueError, match="empty image name"):
            parse_runner_value("docker:")

    def test_unknown_value_raises(self):
        """Unrecognised runner value raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("singularity:myimage")

    def test_unknown_value_plain_raises(self):
        """Unrecognised plain string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("kubernetes")

    def test_docker_with_image_containing_colon(self):
        """Image tag containing colon (e.g. ghcr.io/org/img:1.0) is preserved."""
        mode, image = parse_runner_value("docker:ghcr.io/llem/vllm:1.19.0-cuda12")
        assert mode == "docker"
        assert image == "ghcr.io/llem/vllm:1.19.0-cuda12"


# ---------------------------------------------------------------------------
# is_docker_available
# ---------------------------------------------------------------------------


class TestIsDockerAvailable:
    def test_returns_true_when_docker_and_nvidia_ctk_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-ctk") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_true_when_docker_and_nvidia_container_runtime_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-container-runtime") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_true_when_docker_and_nvidia_container_cli_on_path(self):
        def mock_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in ("docker", "nvidia-container-cli") else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is True

    def test_returns_false_when_docker_not_on_path(self):
        with patch("llenergymeasure.infra.runner_resolution.shutil.which", return_value=None):
            assert is_docker_available() is False

    def test_returns_false_when_docker_present_but_no_nvidia_tool(self):
        def mock_which(name: str) -> str | None:
            return "/usr/bin/docker" if name == "docker" else None

        with patch("llenergymeasure.infra.runner_resolution.shutil.which", side_effect=mock_which):
            assert is_docker_available() is False


# ---------------------------------------------------------------------------
# resolve_runner — precedence chain
# ---------------------------------------------------------------------------


class TestResolveRunner:
    """Test resolve_runner with each precedence layer."""

    # --- Env var (highest) ---

    def test_env_var_wins_over_everything(self, monkeypatch):
        """LLEM_RUNNER_VLLM=docker:custom/img wins over yaml and user_config."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}VLLM", "docker:custom/img")
        yaml_runners = {"vllm": "local"}
        user_config = UserRunnersConfig(vllm="local")

        spec = resolve_runner("vllm", yaml_runners=yaml_runners, user_config=user_config)

        assert spec.source == "env"
        assert spec.mode == "docker"
        assert spec.image == "custom/img"

    def test_env_var_bare_docker(self, monkeypatch):
        """LLEM_RUNNER_PYTORCH=docker (bare) sets mode=docker, image=None."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "docker")

        spec = resolve_runner("transformers")

        assert spec.source == "env"
        assert spec.mode == "docker"
        assert spec.image is None

    def test_env_var_local_overrides_yaml_docker(self, monkeypatch):
        """Env var 'local' takes precedence even when yaml says 'docker'."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "local")
        spec = resolve_runner("transformers", yaml_runners={"transformers": "docker"})
        assert spec.source == "env"
        assert spec.mode == "local"

    # --- YAML runners ---

    def test_yaml_runners_wins_over_user_config(self):
        """yaml_runners={'transformers': 'docker'} wins over user_config with 'local'."""
        user_config = UserRunnersConfig(transformers="local")

        spec = resolve_runner(
            "transformers", yaml_runners={"transformers": "docker"}, user_config=user_config
        )

        assert spec.source == "yaml"
        assert spec.mode == "docker"
        assert spec.image is None

    def test_yaml_runners_docker_with_image(self):
        """yaml_runners with docker:image resolves image correctly."""
        spec = resolve_runner(
            "vllm",
            yaml_runners={"vllm": "docker:ghcr.io/myorg/vllm:latest"},
        )
        assert spec.source == "yaml"
        assert spec.mode == "docker"
        assert spec.image == "ghcr.io/myorg/vllm:latest"

    def test_yaml_runners_missing_engine_falls_through(self):
        """If engine not in yaml_runners, falls through to lower layers."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner(
                "tensorrt",
                yaml_runners={"transformers": "docker"},  # tensorrt not listed
            )
        assert spec.source == "default"

    def test_yaml_runners_none_falls_through(self):
        """yaml_runners=None skips YAML layer entirely."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers", yaml_runners=None)
        assert spec.source == "default"

    # --- User config ---

    def test_user_config_docker_with_image_wins_over_auto_detection(self):
        """user_config.transformers='docker:myimg' wins over auto-detection."""
        user_config = UserRunnersConfig(transformers="docker:myimg")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "docker"
        assert spec.image == "myimg"

    def test_explicit_local_in_user_config_respected_not_overridden_by_auto_detect(self):
        """Explicit 'local' in user_config wins; auto-detection is not applied."""
        user_config = UserRunnersConfig(transformers="local")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "local"

    def test_user_config_bare_docker_sets_mode_docker_image_none(self):
        """user_config.vllm='docker' resolves to mode=docker, image=None."""
        user_config = UserRunnersConfig(vllm="docker")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("vllm", user_config=user_config)

        assert spec.source == "user_config"
        assert spec.mode == "docker"
        assert spec.image is None

    # --- Auto-detection ---

    def test_auto_detected_when_docker_available_and_no_config(self):
        """When Docker available and no explicit config, source='auto_detected'."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers")  # no yaml_runners, no user_config

        assert spec.source == "auto_detected"
        assert spec.mode == "docker"
        assert spec.image is None

    def test_auto_user_config_default_falls_through_to_auto_detection(self):
        """user_config=UserRunnersConfig() (all defaults to 'auto') falls through to auto-detection."""
        user_config = UserRunnersConfig()  # all fields default to "auto"

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        # "auto" falls through — Docker auto-detection applies
        assert spec.source == "auto_detected"
        assert spec.mode == "docker"

    def test_explicit_auto_in_user_config_falls_through_to_auto_detection(self):
        """Explicit 'auto' in user_config falls through to auto-detection."""
        user_config = UserRunnersConfig(transformers="auto")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "auto_detected"
        assert spec.mode == "docker"

    def test_auto_user_config_no_docker_falls_to_default(self):
        """user_config defaults to 'auto', Docker unavailable → falls to default."""
        user_config = UserRunnersConfig()  # all fields default to "auto"

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers", user_config=user_config)

        assert spec.source == "default"
        assert spec.mode == "local"

    # --- Default (local fallback) ---

    def test_default_local_when_docker_unavailable_and_no_config(self):
        """When Docker not available and no config, source='default', mode='local'."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            spec = resolve_runner("transformers")

        assert spec.source == "default"
        assert spec.mode == "local"
        assert spec.image is None

    # --- Parse integration ---

    def test_parse_runner_value_integration_docker_custom_image(self, monkeypatch):
        """parse_runner_value integration: 'docker:ghcr.io/custom:v1' resolves image."""
        monkeypatch.setenv(f"{ENV_RUNNER_PREFIX}TRANSFORMERS", "docker:ghcr.io/custom:v1")
        spec = resolve_runner("transformers")
        assert spec.mode == "docker"
        assert spec.image == "ghcr.io/custom:v1"
        assert spec.source == "env"


# ---------------------------------------------------------------------------
# resolve_study_runners
# ---------------------------------------------------------------------------


class TestResolveStudyRunners:
    def test_resolves_each_engine(self):
        """resolve_study_runners returns spec for each engine."""
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            result = resolve_study_runners(["transformers", "vllm"])

        assert set(result.keys()) == {"transformers", "vllm"}
        assert all(isinstance(v, RunnerSpec) for v in result.values())

    def test_yaml_runners_applied_per_engine(self):
        """yaml_runners are applied to each engine correctly."""
        yaml_runners = {"transformers": "local", "vllm": "docker:myimg"}

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=False,
        ):
            result = resolve_study_runners(["transformers", "vllm"], yaml_runners=yaml_runners)

        assert result["transformers"].mode == "local"
        assert result["transformers"].source == "yaml"
        assert result["vllm"].mode == "docker"
        assert result["vllm"].image == "myimg"

    def test_empty_engines_list_returns_empty_dict(self):
        result = resolve_study_runners([])
        assert result == {}

    def test_single_engine(self):
        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            result = resolve_study_runners(["tensorrt"])

        assert "tensorrt" in result
        assert result["tensorrt"].source == "auto_detected"

    def test_mixed_auto_and_explicit_per_engine(self):
        """Study with one engine explicitly set and another using auto-detection.

        Simulates a researcher who forces transformers=local but lets vllm auto-detect
        to Docker. Each engine resolves independently through the precedence chain.
        """
        user_config = UserRunnersConfig(transformers="local", vllm="auto", tensorrt="auto")

        with patch(
            "llenergymeasure.infra.runner_resolution.is_docker_available",
            return_value=True,
        ):
            result = resolve_study_runners(
                ["transformers", "vllm", "tensorrt"], user_config=user_config
            )

        # pytorch: explicit "local" -> user_config source
        assert result["transformers"].mode == "local"
        assert result["transformers"].source == "user_config"
        # vllm: "auto" falls through -> Docker auto-detected
        assert result["vllm"].mode == "docker"
        assert result["vllm"].source == "auto_detected"
        # tensorrt: "auto" falls through -> Docker auto-detected
        assert result["tensorrt"].mode == "docker"
        assert result["tensorrt"].source == "auto_detected"


# ---------------------------------------------------------------------------
# RunnerSpec.extra_mounts
# ---------------------------------------------------------------------------


class TestRunnerSpecExtraMounts:
    def test_extra_mounts_defaults_to_empty_list(self):
        """RunnerSpec.extra_mounts defaults to empty list when not specified."""
        spec = RunnerSpec(mode="local", image=None, source="default")
        assert spec.extra_mounts == []

    def test_extra_mounts_populated(self):
        """RunnerSpec.extra_mounts stores user-provided mount pairs correctly."""
        spec = RunnerSpec(
            mode="docker",
            image="img",
            source="yaml",
            extra_mounts=[("/host/a", "/container/a")],
        )
        assert spec.extra_mounts == [("/host/a", "/container/a")]
