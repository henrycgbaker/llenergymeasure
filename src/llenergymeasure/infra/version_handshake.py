"""Host/container schema-skew detection via OCI image labels.

Images are stamped at build time with two identifiers baked in as OCI labels:

- ``org.opencontainers.image.version`` — the llenergymeasure package version
  (e.g. ``0.9.0``), sourced from ``_version.py``. Displayed alongside the
  fingerprint in error messages for human readability. Not the blocking signal,
  because during intra-version dev work both host and image report the same
  version through an entire milestone of schema churn.
- ``llem.expconf.schema.fingerprint`` — a SHA-256 hex digest of
  ``ExperimentConfig.model_json_schema()`` serialised with ``sort_keys=True`` and
  compact separators. This is the actual signal: it changes on every
  ``ExperimentConfig`` (or nested model) structural change, catching the exact
  failure mode where the host adds fields the container image does not know
  about.

The host computes its own fingerprint at runtime and compares it to the label
baked into each resolved Docker image as part of ``StudyRunner._prepare_images``.
A mismatch raises ``VersionMismatchError`` with an actionable rebuild hint
before any experiment runs.

Bypass with ``LLEM_SKIP_IMAGE_CHECK=1`` when the skew is known harmless (e.g. a
new optional field the container will silently ignore).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass

from llenergymeasure.config.models import ExperimentConfig

__all__ = [
    "ENV_SKIP_IMAGE_CHECK",
    "LABEL_IMAGE_VERSION",
    "LABEL_SCHEMA_FINGERPRINT",
    "ImageStamp",
    "VersionMismatchError",
    "compute_expconf_fingerprint",
    "inspect_image_stamp",
    "skip_check_enabled",
]

logger = logging.getLogger(__name__)

ENV_SKIP_IMAGE_CHECK = "LLEM_SKIP_IMAGE_CHECK"
LABEL_SCHEMA_FINGERPRINT = "llem.expconf.schema.fingerprint"
LABEL_IMAGE_VERSION = "org.opencontainers.image.version"

# Docker inspect never takes long; keep the handshake tight so a dead docker
# daemon doesn't stall study startup.
_INSPECT_TIMEOUT_SEC = 10.0


class VersionMismatchError(RuntimeError):
    """Raised when a Docker image's schema fingerprint differs from the host's."""


@dataclass(frozen=True)
class ImageStamp:
    """OCI labels relevant to the schema handshake, pulled from a Docker image."""

    pkg_version: str | None
    expconf_fingerprint: str | None


def compute_expconf_fingerprint() -> str:
    """Return the SHA-256 hex digest of ``ExperimentConfig.model_json_schema()``.

    Uses ``sort_keys=True`` and compact separators so the serialisation is
    deterministic across Python processes and machines. The result is 64 hex
    characters; callers typically display the first 12 for readability.
    """
    payload = json.dumps(
        ExperimentConfig.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_image_stamp(image: str, *, timeout: float = _INSPECT_TIMEOUT_SEC) -> ImageStamp:
    """Parse handshake labels from ``docker image inspect`` on *image*.

    Returns ``ImageStamp(None, None)`` on any failure (docker not installed,
    inspect timeout, JSON parse error, missing labels). The caller decides
    whether the absence of labels is a warning or an error.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("docker image inspect failed for %s: %s", image, exc)
        return ImageStamp(pkg_version=None, expconf_fingerprint=None)

    if result.returncode != 0:
        logger.debug(
            "docker image inspect returned %s for %s: %s",
            result.returncode,
            image,
            result.stderr.decode("utf-8", errors="replace") if result.stderr else "",
        )
        return ImageStamp(pkg_version=None, expconf_fingerprint=None)

    return _parse_stamp_from_inspect_json(result.stdout)


def _parse_stamp_from_inspect_json(inspect_stdout: bytes) -> ImageStamp:
    try:
        data = json.loads(inspect_stdout)
    except (json.JSONDecodeError, ValueError):
        return ImageStamp(pkg_version=None, expconf_fingerprint=None)
    if not data:
        return ImageStamp(pkg_version=None, expconf_fingerprint=None)
    labels = data[0].get("Config", {}).get("Labels") or {}
    return ImageStamp(
        pkg_version=labels.get(LABEL_IMAGE_VERSION),
        expconf_fingerprint=labels.get(LABEL_SCHEMA_FINGERPRINT),
    )


def skip_check_enabled() -> bool:
    """True iff the user has set ``LLEM_SKIP_IMAGE_CHECK`` to a truthy value.

    Accepts ``1``, ``true``, ``yes`` (case-insensitive) as on; anything else is
    treated as off. The variable exists as a break-glass bypass — no CLI flag
    yet because we want to see whether it's ever needed before promoting it.
    """
    raw = os.environ.get(ENV_SKIP_IMAGE_CHECK, "")
    return raw.strip().lower() in {"1", "true", "yes"}
