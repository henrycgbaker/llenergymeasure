"""Image-health doctor checks for the ``llem doctor`` command.

This is the API-layer implementation of the schema-skew handshake. It resolves
each supported backend to its default image (local build if present, else the
versioned GHCR tag), reads the OCI labels off each image, and compares the
``llem.expconf.schema.fingerprint`` label to the host's computed fingerprint.

The output is structured (``BackendDoctorResult``) so the CLI layer can render
it as a table and callers in tests/scripts can consume the same data without
parsing terminal output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import BACKEND_PYTORCH, BACKEND_TENSORRT, BACKEND_VLLM
from llenergymeasure.infra.image_registry import get_default_image
from llenergymeasure.infra.version_handshake import (
    compute_expconf_fingerprint,
    inspect_image_stamp,
    skip_check_enabled,
)

__all__ = [
    "BackendDoctorResult",
    "DoctorReport",
    "DoctorStatus",
    "run_doctor_checks",
]

SUPPORTED_BACKENDS: tuple[str, ...] = (BACKEND_PYTORCH, BACKEND_VLLM, BACKEND_TENSORRT)


class DoctorStatus(str, Enum):
    """Per-backend image status."""

    OK = "OK"
    MISMATCH = "MISMATCH"
    UNVERIFIED = "UNVERIFIED"
    UNREACHABLE = "UNREACHABLE"


@dataclass(frozen=True)
class BackendDoctorResult:
    """One row of the doctor table."""

    backend: str
    image: str
    pkg_version: str | None
    image_fingerprint: str | None
    status: DoctorStatus
    detail: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """Full doctor output: per-backend rows plus host-side context."""

    host_pkg_version: str
    host_fingerprint: str
    skip_check_active: bool
    results: list[BackendDoctorResult]

    @property
    def any_mismatch(self) -> bool:
        return any(r.status is DoctorStatus.MISMATCH for r in self.results)


def run_doctor_checks(
    backends: tuple[str, ...] = SUPPORTED_BACKENDS,
) -> DoctorReport:
    """Run image-health checks across *backends* and return a structured report.

    Image resolution follows ``get_default_image`` — local build first, then
    versioned GHCR tag. Unreachable images (docker not installed, no such tag,
    inspect timeout) become ``UNREACHABLE`` rows rather than blowing up.
    """
    host_fp = compute_expconf_fingerprint()
    skip_active = skip_check_enabled()
    results: list[BackendDoctorResult] = []

    for backend in backends:
        image = get_default_image(backend)
        stamp = inspect_image_stamp(image)

        if stamp.expconf_fingerprint is None and stamp.pkg_version is None:
            results.append(
                BackendDoctorResult(
                    backend=backend,
                    image=image,
                    pkg_version=None,
                    image_fingerprint=None,
                    status=DoctorStatus.UNREACHABLE,
                    detail="no labels (image missing or built pre-handshake)",
                )
            )
            continue

        if stamp.expconf_fingerprint is None:
            results.append(
                BackendDoctorResult(
                    backend=backend,
                    image=image,
                    pkg_version=stamp.pkg_version,
                    image_fingerprint=None,
                    status=DoctorStatus.UNVERIFIED,
                    detail="image predates schema-fingerprint label — rebuild to verify",
                )
            )
            continue

        if stamp.expconf_fingerprint == host_fp:
            results.append(
                BackendDoctorResult(
                    backend=backend,
                    image=image,
                    pkg_version=stamp.pkg_version,
                    image_fingerprint=stamp.expconf_fingerprint,
                    status=DoctorStatus.OK,
                )
            )
        else:
            results.append(
                BackendDoctorResult(
                    backend=backend,
                    image=image,
                    pkg_version=stamp.pkg_version,
                    image_fingerprint=stamp.expconf_fingerprint,
                    status=DoctorStatus.MISMATCH,
                    detail=f"rebuild: make docker-build-{backend}",
                )
            )

    return DoctorReport(
        host_pkg_version=__version__,
        host_fingerprint=host_fp,
        skip_check_active=skip_active,
        results=results,
    )
