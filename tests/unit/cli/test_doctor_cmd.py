"""Tests for ``llem doctor`` CLI command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from llenergymeasure.api.doctor import (
    BackendDoctorResult,
    DoctorReport,
    SchemaStatus,
)
from llenergymeasure.cli import app

runner = CliRunner()


def _report(
    results: list[BackendDoctorResult],
    *,
    host_fp: str = "a" * 64,
    host_pkg: str = "0.9.0",
    skip_active: bool = False,
) -> DoctorReport:
    return DoctorReport(
        host_pkg_version=host_pkg,
        host_fingerprint=host_fp,
        skip_check_active=skip_active,
        results=results,
    )


def test_all_ok_exits_zero() -> None:
    report = _report(
        [
            BackendDoctorResult(
                backend="pytorch",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "pytorch" in result.output
    assert "OK" in result.output


def test_mismatch_exits_nonzero() -> None:
    report = _report(
        [
            BackendDoctorResult(
                backend="pytorch",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="b" * 64,
                status=SchemaStatus.MISMATCH,
                detail="rebuild: make docker-build-pytorch",
            ),
            BackendDoctorResult(
                backend="vllm",
                image="llenergymeasure:vllm",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            ),
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "MISMATCH" in result.output
    assert "rebuild" in result.output


def test_unreachable_is_not_mismatch() -> None:
    report = _report(
        [
            BackendDoctorResult(
                backend="pytorch",
                image="llenergymeasure:pytorch",
                pkg_version=None,
                image_fingerprint=None,
                status=SchemaStatus.UNREACHABLE,
                detail="no labels",
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "UNREACHABLE" in result.output


def test_skip_check_warning_rendered() -> None:
    report = _report(
        [
            BackendDoctorResult(
                backend="pytorch",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ],
        skip_active=True,
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "LLEM_SKIP_IMAGE_CHECK" in result.output


def test_host_footer_rendered() -> None:
    report = _report(
        [
            BackendDoctorResult(
                backend="pytorch",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert "Host llenergymeasure version: 0.9.0" in result.output
    assert "Host ExperimentConfig SHA-256:" in result.output
