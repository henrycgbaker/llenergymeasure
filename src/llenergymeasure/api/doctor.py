"""Image-health doctor checks for the ``llem doctor`` command."""

from __future__ import annotations

from dataclasses import dataclass

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import BACKEND_PYTORCH, BACKEND_TENSORRT, BACKEND_VLLM
from llenergymeasure.infra.image_registry import get_default_image
from llenergymeasure.infra.version_handshake import (
    ImageStamp,
    compute_expconf_fingerprint,
    inspect_image_stamp,
    rebuild_hint,
    skip_check_enabled,
)
from llenergymeasure.utils.compat import StrEnum

__all__ = [
    "BackendDoctorResult",
    "DoctorReport",
    "DoctorStatus",
    "run_doctor_checks",
]

SUPPORTED_BACKENDS: tuple[str, ...] = (BACKEND_PYTORCH, BACKEND_VLLM, BACKEND_TENSORRT)


class DoctorStatus(StrEnum):
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


def _classify(backend: str, stamp: ImageStamp, host_fp: str) -> tuple[DoctorStatus, str]:
    if stamp.expconf_fingerprint is None and stamp.pkg_version is None:
        return DoctorStatus.UNREACHABLE, "no labels (image missing or built pre-handshake)"
    if stamp.expconf_fingerprint is None:
        return (
            DoctorStatus.UNVERIFIED,
            "image predates schema-fingerprint label — rebuild to verify",
        )
    if stamp.expconf_fingerprint == host_fp:
        return DoctorStatus.OK, ""
    return DoctorStatus.MISMATCH, f"rebuild: {rebuild_hint(backend)}"


def run_doctor_checks(
    backends: tuple[str, ...] = SUPPORTED_BACKENDS,
) -> DoctorReport:
    """Run image-health checks across *backends* and return a structured report.

    Image resolution follows ``get_default_image`` — local build first, then
    versioned GHCR tag. Unreachable images (docker not installed, no such tag,
    inspect timeout) become ``UNREACHABLE`` rows rather than blowing up.
    """
    host_fp = compute_expconf_fingerprint()
    results: list[BackendDoctorResult] = []

    for backend in backends:
        image = get_default_image(backend)
        stamp = inspect_image_stamp(image)
        status, detail = _classify(backend, stamp, host_fp)
        results.append(
            BackendDoctorResult(
                backend=backend,
                image=image,
                pkg_version=stamp.pkg_version,
                image_fingerprint=stamp.expconf_fingerprint,
                status=status,
                detail=detail,
            )
        )

    return DoctorReport(
        host_pkg_version=__version__,
        host_fingerprint=host_fp,
        skip_check_active=skip_check_enabled(),
        results=results,
    )
