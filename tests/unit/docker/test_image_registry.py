"""Tests for the built-in Docker image registry and runner value parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llenergymeasure.config.ssot import ENV_IMAGE_PREFIX


@pytest.fixture(autouse=True)
def clear_cuda_version_cache():
    """Ensure lru_cache for get_cuda_major_version is cleared before and after each test."""
    from llenergymeasure.infra.image_registry import get_cuda_major_version

    get_cuda_major_version.cache_clear()
    yield
    get_cuda_major_version.cache_clear()


# ---------------------------------------------------------------------------
# parse_runner_value
# ---------------------------------------------------------------------------


class TestParseRunnerValue:
    def test_local_returns_local_none(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("local") == ("local", None)

    def test_docker_returns_docker_none(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("docker") == ("docker", None)

    def test_docker_colon_image_returns_docker_with_image(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        assert parse_runner_value("docker:custom/img:v1") == ("docker", "custom/img:v1")

    def test_docker_colon_ghcr_image(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        result = parse_runner_value("docker:ghcr.io/org/vllm:1.19.0-cuda12")
        assert result == ("docker", "ghcr.io/org/vllm:1.19.0-cuda12")

    def test_docker_colon_empty_string_raises(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        with pytest.raises(ValueError, match="empty image name"):
            parse_runner_value("docker:")

    def test_unknown_runner_type_raises(self):
        from llenergymeasure.infra.image_registry import parse_runner_value

        with pytest.raises(ValueError, match="Unrecognised runner value"):
            parse_runner_value("kubernetes")


# ---------------------------------------------------------------------------
# get_default_image
# ---------------------------------------------------------------------------


class TestGetDefaultImage:
    def test_prefers_local_image_when_available(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image = get_default_image("vllm")

        assert image == "llenergymeasure:vllm"

    def test_falls_back_to_ghcr_when_no_local_image(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("vllm")

        assert image.startswith("ghcr.io/henrycgbaker/llenergymeasure/vllm:v")

    def test_fallback_to_latest_when_version_empty(self):
        from llenergymeasure.infra.image_registry import get_default_image

        with (
            patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=False),
            patch("llenergymeasure._version.__version__", ""),
        ):
            image = get_default_image("pytorch")

        assert image.endswith(":vlatest")

    def test_backend_name_included_in_image(self):
        from llenergymeasure.infra.image_registry import get_default_image

        for backend in ("pytorch", "vllm", "tensorrt"):
            image = get_default_image(backend)
            assert backend in image, f"Expected backend {backend!r} in image {image!r}"

    def test_ghcr_image_includes_package_version(self):
        from llenergymeasure import __version__
        from llenergymeasure.infra.image_registry import get_default_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image = get_default_image("vllm")

        assert f"v{__version__}" in image


# ---------------------------------------------------------------------------
# show_image_resolution
# ---------------------------------------------------------------------------


class TestShowImageResolution:
    def test_prints_all_backends(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "pytorch" in output
        assert "vllm" in output
        assert "tensorrt" in output

    def test_shows_local_source(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "(local_build)" in output

    def test_shows_registry_source(self, capsys):
        from llenergymeasure.infra.image_registry import show_image_resolution

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            show_image_resolution()

        output = capsys.readouterr().out
        assert "(registry)" in output


# ---------------------------------------------------------------------------
# resolve_image
# ---------------------------------------------------------------------------


class TestResolveImage:
    def test_env_var_takes_highest_precedence(self, monkeypatch):
        from llenergymeasure.infra.image_registry import resolve_image

        monkeypatch.setenv(f"{ENV_IMAGE_PREFIX}VLLM", "custom/env-image:v1")

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            yaml_images={"vllm": "yaml-image:v1"},
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "custom/env-image:v1"
        assert source == "env"

    def test_yaml_images_second_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            yaml_images={"vllm": "yaml-image:v1"},
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "yaml-image:v1"
        assert source == "yaml"

    def test_spec_image_third_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        image, source = resolve_image(
            "vllm",
            spec_image="spec-image:v1",
            user_config_images={"vllm": "uc-image:v1"},
        )

        assert image == "spec-image:v1"
        assert source == "runner_override"

    def test_user_config_images_fourth_precedence(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image, source = resolve_image(
                "vllm",
                user_config_images={"vllm": "uc-image:v1"},
            )

        assert image == "uc-image:v1"
        assert source == "user_config"

    def test_smart_default_local_build(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image, source = resolve_image("vllm")

        assert image == "llenergymeasure:vllm"
        assert source == "local_build"

    def test_smart_default_registry_fallback(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch(
            "llenergymeasure.infra.image_registry._image_exists_locally", return_value=False
        ):
            image, source = resolve_image("vllm")

        assert image.startswith("ghcr.io/henrycgbaker/llenergymeasure/vllm:v")
        assert source == "registry"

    def test_env_var_case_insensitive_backend(self, monkeypatch):
        from llenergymeasure.infra.image_registry import resolve_image

        monkeypatch.setenv(f"{ENV_IMAGE_PREFIX}PYTORCH", "my/pytorch:v1")
        image, source = resolve_image("pytorch")
        assert image == "my/pytorch:v1"
        assert source == "env"

    def test_yaml_images_ignores_other_backends(self):
        from llenergymeasure.infra.image_registry import resolve_image

        with patch("llenergymeasure.infra.image_registry._image_exists_locally", return_value=True):
            image, source = resolve_image(
                "vllm",
                yaml_images={"pytorch": "pytorch-image:v1"},
            )

        assert image == "llenergymeasure:vllm"
        assert source == "local_build"


# ---------------------------------------------------------------------------
# get_cuda_major_version
# ---------------------------------------------------------------------------


class TestGetCudaMajorVersion:
    def test_parses_nvcc_output_correctly(self):
        from llenergymeasure.infra.image_registry import _parse_cuda_major_from_nvcc

        sample = (
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Copyright (c) 2005-2023 NVIDIA Corporation\n"
            "Built on Mon_Apr__3_17:16:06_PDT_2023\n"
            "Cuda compilation tools, release 12.1, V12.1.105\n"
            "Build cuda_12.1.r12.1/compiler.32688072_0\n"
        )
        assert _parse_cuda_major_from_nvcc(sample) == "12"

    def test_parses_nvcc_cuda_11(self):
        from llenergymeasure.infra.image_registry import _parse_cuda_major_from_nvcc

        sample = "Cuda compilation tools, release 11.8, V11.8.89\n"
        assert _parse_cuda_major_from_nvcc(sample) == "11"

    def test_returns_none_for_unrecognised_output(self):
        from llenergymeasure.infra.image_registry import _parse_cuda_major_from_nvcc

        assert _parse_cuda_major_from_nvcc("not a cuda output") is None

    def test_get_cuda_major_version_uses_nvcc_result(self):
        """get_cuda_major_version() should return major version from nvcc when available."""
        from llenergymeasure.infra.image_registry import get_cuda_major_version

        nvcc_output = "Cuda compilation tools, release 12.3, V12.3.107\n"
        mock_result = type("R", (), {"returncode": 0, "stdout": nvcc_output})()

        with patch("subprocess.run", return_value=mock_result):
            version = get_cuda_major_version()

        assert version == "12"

    def test_cuda_major_no_nvidia_smi_subprocess(self, monkeypatch):
        """Verify nvidia-smi subprocess is never called in get_cuda_major_version()."""
        import subprocess as real_subprocess

        from llenergymeasure.infra.image_registry import get_cuda_major_version

        get_cuda_major_version.cache_clear()
        original_run = real_subprocess.run
        calls = []

        def spy_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if cmd[0] == "nvidia-smi":
                raise AssertionError("nvidia-smi should not be called")
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(real_subprocess, "run", spy_run)
        get_cuda_major_version()
        assert not any("nvidia-smi" in str(c) for c in calls)
        get_cuda_major_version.cache_clear()

    def test_get_cuda_major_version_returns_none_when_nvcc_missing(self):
        from llenergymeasure.infra.image_registry import get_cuda_major_version

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            patch.dict("sys.modules", {"pynvml": None}),
        ):
            version = get_cuda_major_version()

        assert version is None
