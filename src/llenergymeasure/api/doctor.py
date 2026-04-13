"""Image-health doctor checks for the ``llem doctor`` command."""

from __future__ import annotations

from dataclasses import dataclass

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import ENGINE_PYTORCH, ENGINE_TENSORRT, ENGINE_VLLM
from llenergymeasure.infra.image_registry import get_default_image
from llenergymeasure.infra.version_handshake import (
    SchemaStatus,
    classify_stamp,
    compute_expconf_fingerprint,
    inspect_image_stamp,
    rebuild_hint,
    skip_check_enabled,
)

__all__ = [
    "DoctorReport",
    "EngineDoctorResult",
    "SchemaStatus",
    "run_doctor_checks",
]

SUPPORTED_ENGINES: tuple[str, ...] = (ENGINE_PYTORCH, ENGINE_VLLM, ENGINE_TENSORRT)

_DETAIL_FOR_STATUS: dict[SchemaStatus, str] = {
    SchemaStatus.OK: "",
    SchemaStatus.UNVERIFIED: "image predates schema-fingerprint label — rebuild to verify",
    SchemaStatus.UNREACHABLE: "no labels (image missing or built pre-handshake)",
}


@dataclass(frozen=True)
class EngineDoctorResult:
    """One row of the doctor table."""

    engine: str
    image: str
    pkg_version: str | None
    image_fingerprint: str | None
    status: SchemaStatus
    detail: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """Full doctor output: per-engine rows plus host-side context."""

    host_pkg_version: str
    host_fingerprint: str
    skip_check_active: bool
    results: list[EngineDoctorResult]

    @property
    def any_mismatch(self) -> bool:
        return any(r.status is SchemaStatus.MISMATCH for r in self.results)


def _detail_for(engine: str, status: SchemaStatus) -> str:
    if status is SchemaStatus.MISMATCH:
        return f"rebuild: {rebuild_hint(engine)}"
    return _DETAIL_FOR_STATUS.get(status, "")


def run_doctor_checks(
    engines: tuple[str, ...] = SUPPORTED_ENGINES,
) -> DoctorReport:
    """Run image-health checks across *engines* and return a structured report.

    Image resolution follows ``get_default_image`` — local build first, then
    versioned GHCR tag. Unreachable images (docker not installed, no such tag,
    inspect timeout) become ``UNREACHABLE`` rows rather than blowing up.
    """
    host_fp = compute_expconf_fingerprint()
    results: list[EngineDoctorResult] = []

    for engine in engines:
        image = get_default_image(engine)
        stamp = inspect_image_stamp(image)
        status = classify_stamp(stamp, host_fp)
        results.append(
            EngineDoctorResult(
                engine=engine,
                image=image,
                pkg_version=stamp.pkg_version,
                image_fingerprint=stamp.expconf_fingerprint,
                status=status,
                detail=_detail_for(engine, status),
            )
        )

    return DoctorReport(
        host_pkg_version=__version__,
        host_fingerprint=host_fp,
        skip_check_active=skip_check_enabled(),
        results=results,
    )
