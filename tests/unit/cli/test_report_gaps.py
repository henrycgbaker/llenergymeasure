"""Unit tests for cli/report_gaps.py — Typer smoke tests.

All tests use ``typer.testing.CliRunner``. Fixture study dirs are built
through the same helpers as ``tests/unit/api/test_report_gaps.py`` — we
stay deliberately close to the real JSONL shape so a future schema bump
surfaces here too.
"""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from llenergymeasure.cli import app
from tests.helpers.runtime_obs import (
    fake_hash as _fake_hash,
)
from tests.helpers.runtime_obs import (
    write_jsonl_record as _write_jsonl_record,
)
from tests.helpers.runtime_obs import (
    write_resolution as _write_resolution,
)

runner = CliRunner()

# Typer/Rich emits ANSI escape codes that break substring matches on CI (no
# NO_COLOR default). Match the pattern used by tests/unit/cli/test_cli_run.py.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Help + dispatch
# ---------------------------------------------------------------------------


def test_report_gaps_help() -> None:
    result = runner.invoke(app, ["report-gaps", "--help"])
    assert result.exit_code == 0
    joined = " ".join(_strip(result.output).split())
    for flag in ("--source", "--study-dir", "--engine", "--out", "--include-exceptions"):
        assert flag in joined


def test_report_gaps_writes_yaml(tmp_path: Path) -> None:
    """Happy path: CLI writes YAML fragments when a gap is discovered."""
    study = tmp_path / "study-writes"
    study.mkdir()

    h_a = _fake_hash("cli-fire-a")
    h_b = _fake_hash("cli-fire-b")
    h_c = _fake_hash("cli-notfire")
    _write_resolution(study, 1, 1, "transformers", h_a, {"do_sample": False})
    _write_resolution(study, 2, 1, "transformers", h_b, {"do_sample": False})
    _write_resolution(study, 3, 1, "transformers", h_c, {"do_sample": True})

    _write_jsonl_record(study, config_hash=h_a, warnings_emitted=["CLI smoke warning"])
    _write_jsonl_record(study, config_hash=h_b, warnings_emitted=["CLI smoke warning"])
    _write_jsonl_record(study, config_hash=h_c)

    out = tmp_path / "proposals.yaml"
    result = runner.invoke(
        app,
        [
            "report-gaps",
            "--source",
            "runtime-warnings",
            "--study-dir",
            str(study),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "CLI smoke warning" in body
    assert "added_by: runtime_warning" in body
    # Summary line printed.
    assert "1 gap" in result.output


def test_report_gaps_empty_emits_no_file(tmp_path: Path) -> None:
    """When no gaps are found the CLI does not write the out file and exits 0."""
    study = tmp_path / "empty-study"
    study.mkdir()
    # Create the JSONL file but with no emissions — no templates → no gaps.
    _write_jsonl_record(study, config_hash=_fake_hash("empty"), warnings_emitted=[])

    out = tmp_path / "proposals.yaml"
    result = runner.invoke(
        app,
        [
            "report-gaps",
            "--source",
            "runtime-warnings",
            "--study-dir",
            str(study),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert not out.exists()
    assert "No gaps found" in result.output


def test_report_gaps_missing_study_dir_errors(tmp_path: Path) -> None:
    """Omitting --study-dir exits with a clear Typer BadParameter."""
    out = tmp_path / "proposals.yaml"
    result = runner.invoke(
        app,
        ["report-gaps", "--source", "runtime-warnings", "--out", str(out)],
    )
    assert result.exit_code != 0
    stripped = _strip(result.output).lower()
    assert "study-dir" in stripped or "study_dir" in stripped


def test_report_gaps_missing_out_errors(tmp_path: Path) -> None:
    """Omitting --out exits with a clear Typer BadParameter."""
    study = tmp_path / "s"
    study.mkdir()
    result = runner.invoke(
        app,
        [
            "report-gaps",
            "--source",
            "runtime-warnings",
            "--study-dir",
            str(study),
        ],
    )
    assert result.exit_code != 0
    stripped = _strip(result.output).lower()
    assert "--out" in stripped or "out" in stripped


def test_report_gaps_unsupported_source_errors(tmp_path: Path) -> None:
    """--source h3-collisions (etc.) is not wired in this release."""
    study = tmp_path / "s"
    study.mkdir()
    out = tmp_path / "out.yaml"
    result = runner.invoke(
        app,
        [
            "report-gaps",
            "--source",
            "h3-collisions",
            "--study-dir",
            str(study),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "runtime-warnings" in _strip(result.output)
