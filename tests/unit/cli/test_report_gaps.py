"""Unit tests for cli/report_gaps.py — Typer smoke tests.

All tests use ``typer.testing.CliRunner``. Fixture study dirs are built
through the same helpers as ``tests/unit/api/test_report_gaps.py`` — we
stay deliberately close to the real JSONL shape so a future schema bump
surfaces here too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from llenergymeasure.cli import app

runner = CliRunner()


def _fake_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def _write_resolution(
    study_dir: Path,
    index: int,
    cycle: int,
    engine: str,
    full_hash: str,
    overrides: dict[str, Any],
) -> None:
    subdir = study_dir / f"{index:03d}_c{cycle}_gpt2-{engine}_{full_hash[:8]}"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "_resolution.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "overrides": {k: {"effective": v, "source": "sweep"} for k, v in overrides.items()},
            }
        )
    )


def _write_jsonl_record(
    study_dir: Path,
    *,
    config_hash: str,
    engine: str = "transformers",
    library_version: str = "4.56.0",
    cycle: int = 1,
    outcome: str = "success",
    warnings_emitted: list[str] | None = None,
) -> None:
    from llenergymeasure.study.message_normalise import normalise

    rec = {
        "schema_version": 1,
        "observed_at": "2026-04-24T04:00:00Z",
        "study_run_id": "fixture-run-id",
        "config_hash": config_hash,
        "cycle": cycle,
        "engine": engine,
        "library_version": library_version,
        "outcome": outcome,
        "warnings": [
            {
                "category": "UserWarning",
                "message": w,
                "message_template": normalise(w).template,
                "filename": "fixture.py",
                "lineno": 1,
            }
            for w in (warnings_emitted or [])
        ],
        "logger_records": [],
        "exception": None,
        "exit_reason": None,
        "exit_code": None,
    }
    with open(study_dir / "runtime_observations.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Help + dispatch
# ---------------------------------------------------------------------------


def test_report_gaps_help() -> None:
    result = runner.invoke(app, ["report-gaps", "--help"])
    assert result.exit_code == 0
    # Typer's help formatter may wrap at narrow widths on small TTYs; strip
    # whitespace/newlines before looking up flag names.
    joined = " ".join(result.output.split())
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
    assert "study-dir" in result.output.lower() or "study_dir" in result.output.lower()


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
    assert "--out" in result.output or "out" in result.output.lower()


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
    assert "runtime-warnings" in result.output
