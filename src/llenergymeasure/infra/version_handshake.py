"""Host/container schema-skew detection via OCI image labels.

Images are stamped at build time with ``org.opencontainers.image.version`` (the
llenergymeasure package version, for display) and
``llem.expconf.schema.fingerprint`` (a SHA-256 over
``ExperimentConfig.model_json_schema()``, the actual blocking signal).

The host computes its own fingerprint at runtime and compares it to the label
on each resolved Docker image in ``StudyRunner._prepare_images``. A mismatch
raises ``VersionMismatchError`` before any experiment runs.

Bypass with ``LLEM_SKIP_IMAGE_CHECK=1`` when the skew is known harmless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from functools import cache

from llenergymeasure.config.ssot import TIMEOUT_DOCKER_INSPECT
from llenergymeasure.utils.compat import StrEnum

__all__ = [
    "ENV_SKIP_IMAGE_CHECK",
    "LABEL_IMAGE_VERSION",
    "LABEL_SCHEMA_FINGERPRINT",
    "ImageStamp",
    "SchemaStatus",
    "VersionMismatchError",
    "classify_stamp",
    "compute_expconf_fingerprint",
    "inspect_image_stamp",
    "parse_image_stamp",
    "rebuild_hint",
    "skip_check_enabled",
]

logger = logging.getLogger(__name__)

ENV_SKIP_IMAGE_CHECK = "LLEM_SKIP_IMAGE_CHECK"
LABEL_SCHEMA_FINGERPRINT = "llem.expconf.schema.fingerprint"
LABEL_IMAGE_VERSION = "org.opencontainers.image.version"


class VersionMismatchError(RuntimeError):
    """Raised when a Docker image's schema fingerprint differs from the host's."""


@dataclass(frozen=True)
class ImageStamp:
    """OCI labels relevant to the schema handshake, pulled from a Docker image."""

    pkg_version: str | None
    expconf_fingerprint: str | None


_EMPTY_STAMP = ImageStamp(pkg_version=None, expconf_fingerprint=None)


class SchemaStatus(StrEnum):
    """Outcome of comparing a Docker image's stamp to the host fingerprint."""

    OK = "OK"
    MISMATCH = "MISMATCH"
    UNVERIFIED = "UNVERIFIED"
    UNREACHABLE = "UNREACHABLE"
    BYPASSED = "BYPASSED"


@cache
def compute_expconf_fingerprint() -> str:
    """Return the SHA-256 hex digest of the ExperimentConfig JSON schema.

    The schema is frozen per-process, so the result is memoised. Callers
    typically display the first 12 hex characters for readability.
    """
    from llenergymeasure.config.introspection import get_experiment_config_schema

    payload = json.dumps(
        get_experiment_config_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_stamp(stamp: ImageStamp, host_fingerprint: str) -> SchemaStatus:
    """Classify an image stamp against the host fingerprint.

    Pure: does not read environment or log. Callers that honour the
    ``LLEM_SKIP_IMAGE_CHECK`` bypass should short-circuit to
    :attr:`SchemaStatus.BYPASSED` themselves before calling this.
    """
    if stamp.expconf_fingerprint is None and stamp.pkg_version is None:
        return SchemaStatus.UNREACHABLE
    if stamp.expconf_fingerprint is None:
        return SchemaStatus.UNVERIFIED
    if stamp.expconf_fingerprint == host_fingerprint:
        return SchemaStatus.OK
    return SchemaStatus.MISMATCH


def inspect_image_stamp(image: str, *, timeout: float = TIMEOUT_DOCKER_INSPECT) -> ImageStamp:
    """Parse handshake labels from ``docker image inspect`` on *image*.

    Returns an empty stamp on any failure (docker not installed, inspect
    timeout, JSON parse error, missing labels). The caller decides whether the
    absence of labels is a warning or an error.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("docker image inspect failed for %s: %s", image, exc)
        return _EMPTY_STAMP

    if result.returncode != 0:
        logger.debug(
            "docker image inspect returned %s for %s: %s",
            result.returncode,
            image,
            result.stderr.decode("utf-8", errors="replace") if result.stderr else "",
        )
        return _EMPTY_STAMP

    return parse_image_stamp(result.stdout)


def parse_image_stamp(inspect_stdout: bytes) -> ImageStamp:
    """Extract an ``ImageStamp`` from raw ``docker image inspect`` JSON."""
    try:
        data = json.loads(inspect_stdout)
    except (json.JSONDecodeError, ValueError):
        return _EMPTY_STAMP
    if not data:
        return _EMPTY_STAMP
    labels = data[0].get("Config", {}).get("Labels") or {}
    return ImageStamp(
        pkg_version=labels.get(LABEL_IMAGE_VERSION),
        expconf_fingerprint=labels.get(LABEL_SCHEMA_FINGERPRINT),
    )


def rebuild_hint(engine: str) -> str:
    """Return the user-facing rebuild command for *engine*."""
    return f"make docker-build-{engine}"


def skip_check_enabled() -> bool:
    """True iff ``LLEM_SKIP_IMAGE_CHECK`` is set to a truthy value."""
    from llenergymeasure.utils.env_config import parse_bool_env

    return parse_bool_env(ENV_SKIP_IMAGE_CHECK)
