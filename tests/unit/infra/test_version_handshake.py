"""Tests for ``llenergymeasure.infra.version_handshake``."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra.version_handshake import (
    ENV_SKIP_IMAGE_CHECK,
    LABEL_IMAGE_VERSION,
    LABEL_SCHEMA_FINGERPRINT,
    ImageStamp,
    compute_expconf_fingerprint,
    inspect_image_stamp,
    skip_check_enabled,
)

# ---------------------------------------------------------------------------
# compute_expconf_fingerprint
# ---------------------------------------------------------------------------


class TestComputeExpConfFingerprint:
    def test_deterministic_across_calls(self) -> None:
        assert compute_expconf_fingerprint() == compute_expconf_fingerprint()

    def test_hex_length(self) -> None:
        fp = compute_expconf_fingerprint()
        assert len(fp) == 64
        int(fp, 16)  # must be valid hex

    def test_sensitive_to_model_changes(self) -> None:
        """Adding a field to a throwaway model must change its fingerprint.

        We don't mutate ExperimentConfig itself (too invasive for a unit
        test), but we verify the serialiser is sensitive to schema shape by
        hashing two synthetic pydantic models with and without an extra field.
        """
        import hashlib

        from pydantic import BaseModel

        class _A(BaseModel):
            x: int = 0

        class _B(BaseModel):
            x: int = 0
            y: int = 0

        def _hash(model_cls: type[BaseModel]) -> str:
            payload = json.dumps(
                model_cls.model_json_schema(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        assert _hash(_A) != _hash(_B)


# ---------------------------------------------------------------------------
# inspect_image_stamp
# ---------------------------------------------------------------------------


def _fake_inspect_result(*, labels: dict[str, str] | None, returncode: int = 0) -> MagicMock:
    body = [{"Id": "sha256:abc", "Config": {"Labels": labels}}] if labels is not None else [{}]
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = json.dumps(body).encode("utf-8")
    result.stderr = b""
    return result


class TestInspectImageStamp:
    def test_valid_labels(self) -> None:
        labels = {
            LABEL_SCHEMA_FINGERPRINT: "a" * 64,
            LABEL_IMAGE_VERSION: "0.9.0",
        }
        with patch("subprocess.run", return_value=_fake_inspect_result(labels=labels)):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version="0.9.0", expconf_fingerprint="a" * 64)

    def test_missing_labels(self) -> None:
        with patch("subprocess.run", return_value=_fake_inspect_result(labels={})):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp.pkg_version is None
        assert stamp.expconf_fingerprint is None

    def test_partial_labels(self) -> None:
        labels = {LABEL_IMAGE_VERSION: "0.9.0"}
        with patch("subprocess.run", return_value=_fake_inspect_result(labels=labels)):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp.pkg_version == "0.9.0"
        assert stamp.expconf_fingerprint is None

    def test_timeout_returns_empty_stamp(self) -> None:
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("docker", 5),
        ):
            stamp = inspect_image_stamp("my/img:latest", timeout=1.0)
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_docker_not_installed(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError("docker")):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_nonzero_returncode(self) -> None:
        bad = MagicMock(spec=subprocess.CompletedProcess)
        bad.returncode = 1
        bad.stdout = b""
        bad.stderr = b"No such image"
        with patch("subprocess.run", return_value=bad):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)

    def test_malformed_json(self) -> None:
        bad = MagicMock(spec=subprocess.CompletedProcess)
        bad.returncode = 0
        bad.stdout = b"not json"
        bad.stderr = b""
        with patch("subprocess.run", return_value=bad):
            stamp = inspect_image_stamp("my/img:latest")
        assert stamp == ImageStamp(pkg_version=None, expconf_fingerprint=None)


# ---------------------------------------------------------------------------
# skip_check_enabled
# ---------------------------------------------------------------------------


class TestSkipCheckEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_SKIP_IMAGE_CHECK, value)
        assert skip_check_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "random"])
    def test_falsy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_SKIP_IMAGE_CHECK, value)
        assert skip_check_enabled() is False

    def test_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_SKIP_IMAGE_CHECK, raising=False)
        assert skip_check_enabled() is False
