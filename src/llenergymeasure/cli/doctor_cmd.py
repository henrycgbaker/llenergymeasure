"""``llem doctor`` — verify that Docker images match the host schema.

Focused purely on image health (host/container schema-fingerprint handshake).
The existing ``llem config`` command still covers env / GPU / backend probes;
``doctor`` is the pre-flight diagnostic for the new fingerprint handshake and
exits non-zero on mismatch so CI and scripts can gate on it.
"""

from __future__ import annotations

import typer

__all__ = ["doctor_command"]


def doctor_command() -> None:
    """Report per-backend image schema status and exit non-zero on mismatch."""
    from llenergymeasure.api.doctor import run_doctor_checks

    report = run_doctor_checks()

    header = (
        f"{'Backend':<10s}  {'Image':<50s}  "
        f"{'Pkg ver':<10s}  {'Img SHA-256':<14s}  {'Host SHA-256':<14s}  {'Status':<12s}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    host_fp_short = report.host_fingerprint[:12]
    for row in report.results:
        img = row.image if len(row.image) <= 50 else "…" + row.image[-49:]
        pkg = (row.pkg_version or "-")[:10]
        img_fp = (row.image_fingerprint or "-")[:12]
        status_text = row.status.value
        line = (
            f"{row.backend:<10s}  {img:<50s}  "
            f"{pkg:<10s}  {img_fp:<14s}  {host_fp_short:<14s}  {status_text:<12s}"
        )
        typer.echo(line)
        if row.detail:
            typer.echo(f"{'':<10s}  └─ {row.detail}")

    typer.echo("")
    typer.echo(f"Host llenergymeasure version: {report.host_pkg_version}")
    typer.echo(f"Host ExperimentConfig SHA-256: {report.host_fingerprint}")
    if report.skip_check_active:
        typer.echo(
            "WARNING: LLEM_SKIP_IMAGE_CHECK=1 is active — runtime schema handshake is bypassed."
        )

    if report.any_mismatch:
        raise typer.Exit(code=1)
