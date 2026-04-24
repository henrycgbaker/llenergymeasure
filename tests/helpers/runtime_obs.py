"""Shared fixture helpers for ``report-gaps`` tests.

Used by both :mod:`tests.unit.api.test_report_gaps` and
:mod:`tests.unit.cli.test_report_gaps` so the JSONL + sidecar shapes stay
in lock-step with :mod:`llenergymeasure.study.runtime_observations` and
:mod:`llenergymeasure.study.manifest`.

These are plain module-level functions (not pytest fixtures) because each
call takes per-emission parameters like ``index`` and ``full_hash``; the
fixture form would require one fixture per call pattern.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def fake_hash(*parts: Any) -> str:
    """Build a deterministic 64-char hex hash keyed by ``parts``."""
    raw = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(raw).hexdigest()


def write_resolution(
    study_dir: Path,
    index: int,
    cycle: int,
    engine: str,
    full_hash: str,
    overrides: dict[str, Any],
) -> Path:
    """Create the experiment subdir + ``_resolution.json`` sidecar.

    Returns the experiment subdirectory path so callers can write a
    matching ``manifest.json`` entry pointing at it.
    """
    dir_name = f"{index:03d}_c{cycle}_gpt2-{engine}_{full_hash[:8]}"
    subdir = study_dir / dir_name
    subdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "overrides": {k: {"effective": v, "source": "sweep"} for k, v in overrides.items()},
    }
    (subdir / "_resolution.json").write_text(json.dumps(payload))
    return subdir


def write_jsonl_record(
    study_dir: Path,
    *,
    config_hash: str,
    engine: str = "transformers",
    library_version: str = "4.56.0",
    cycle: int = 1,
    outcome: str = "success",
    warnings_emitted: list[str] | None = None,
    logger_emitted: list[tuple[str, str]] | None = None,
    exception: dict[str, Any] | None = None,
    exit_reason: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Append one canonical JSONL record to ``runtime_observations.jsonl``.

    Warnings/log records are passed as raw messages; this helper normalises
    them to templates so fixtures stay readable. JSONL schema matches
    :mod:`llenergymeasure.study.runtime_observations`.
    """
    from llenergymeasure.study.message_normalise import normalise

    rec: dict[str, Any] = {
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
        "logger_records": [
            {
                "level": "WARNING",
                "logger": lname,
                "message": msg,
                "message_template": normalise(msg).template,
                "filename": "fixture.py",
                "lineno": 1,
            }
            for (lname, msg) in (logger_emitted or [])
        ],
        "exception": exception,
        "exit_reason": exit_reason,
        "exit_code": exit_code,
    }
    path = study_dir / "runtime_observations.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
