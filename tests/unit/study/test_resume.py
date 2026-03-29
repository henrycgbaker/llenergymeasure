"""Unit tests for study resume capability.

Tests:
- find_resumable_study: most-recent interrupted study, completed skipped, empty dir
- load_resume_state: skip-set construction, missing manifest raises StudyError
- validate_config_drift: matching hash passes, mismatch raises with clear message
- prepare_resume_manifest: resets non-completed to pending, writes to disk
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llenergymeasure.study.resume import (
    find_resumable_study,
    load_resume_state,
    prepare_resume_manifest,
    validate_config_drift,
)
from llenergymeasure.utils.exceptions import StudyError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(study_dir: Path, status: str, started_at: str, experiments: list[dict]) -> None:
    """Write a minimal manifest.json to study_dir."""
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2.0",
        "study_name": "test-study",
        "study_design_hash": "abc123",
        "llenergymeasure_version": "0.9.0",
        "started_at": started_at,
        "completed_at": None,
        "status": status,
        "total_experiments": len(experiments),
        "completed": sum(1 for e in experiments if e.get("status") == "completed"),
        "failed": sum(1 for e in experiments if e.get("status") == "failed"),
        "pending": sum(1 for e in experiments if e.get("status") in ("pending", "running")),
        "skipped": sum(1 for e in experiments if e.get("status") == "skipped"),
        "interrupted": sum(1 for e in experiments if e.get("status") == "interrupted"),
        "experiments": experiments,
    }
    (study_dir / "manifest.json").write_text(json.dumps(manifest))


def _make_experiment_entry(config_hash: str, cycle: int, status: str) -> dict:
    return {
        "config_hash": config_hash,
        "config_summary": f"model/{config_hash[:4]}",
        "cycle": cycle,
        "status": status,
        "result_file": None,
        "log_file": None,
        "error_type": None,
        "error_message": None,
        "started_at": None,
        "completed_at": None,
    }


# ---------------------------------------------------------------------------
# find_resumable_study
# ---------------------------------------------------------------------------


def test_find_resumable_study_returns_most_recent(tmp_path: Path) -> None:
    """Returns the interrupted study with the most recent started_at."""
    output_dir = tmp_path / "results"

    older_dir = output_dir / "study_2026-01-01T00-00-00"
    _write_manifest(
        older_dir,
        status="interrupted",
        started_at="2026-01-01T00:00:00+00:00",
        experiments=[],
    )

    newer_dir = output_dir / "study_2026-03-01T00-00-00"
    _write_manifest(
        newer_dir,
        status="interrupted",
        started_at="2026-03-01T00:00:00+00:00",
        experiments=[],
    )

    result = find_resumable_study(output_dir)
    assert result == newer_dir


def test_find_resumable_study_returns_none_when_no_resumable(tmp_path: Path) -> None:
    """Returns None when all studies are completed."""
    output_dir = tmp_path / "results"

    completed_dir = output_dir / "study_2026-01-01T00-00-00"
    _write_manifest(
        completed_dir,
        status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        experiments=[],
    )

    result = find_resumable_study(output_dir)
    assert result is None


def test_find_resumable_study_returns_none_for_empty_dir(tmp_path: Path) -> None:
    """Returns None for an output dir with no study subdirectories."""
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    result = find_resumable_study(output_dir)
    assert result is None


def test_find_resumable_study_returns_none_for_nonexistent_dir(tmp_path: Path) -> None:
    """Returns None when output_dir does not exist."""
    result = find_resumable_study(tmp_path / "nonexistent")
    assert result is None


def test_find_resumable_study_skips_completed_studies(tmp_path: Path) -> None:
    """find_resumable_study skips completed studies and finds interrupted ones."""
    output_dir = tmp_path / "results"

    completed_dir = output_dir / "study_completed"
    _write_manifest(
        completed_dir,
        status="completed",
        started_at="2026-03-29T10:00:00+00:00",
        experiments=[],
    )

    interrupted_dir = output_dir / "study_interrupted"
    _write_manifest(
        interrupted_dir,
        status="interrupted",
        started_at="2026-03-29T09:00:00+00:00",
        experiments=[],
    )

    result = find_resumable_study(output_dir)
    assert result == interrupted_dir


def test_find_resumable_study_handles_all_resumable_statuses(tmp_path: Path) -> None:
    """find_resumable_study finds circuit_breaker and timed_out studies too."""
    output_dir = tmp_path / "results"

    for status in ("circuit_breaker", "timed_out", "failed"):
        study_dir = output_dir / f"study_{status}"
        _write_manifest(
            study_dir,
            status=status,
            started_at="2026-03-01T00:00:00+00:00",
            experiments=[],
        )

    # All are resumable — should find one
    result = find_resumable_study(output_dir)
    assert result is not None


def test_find_resumable_study_skips_invalid_manifests(tmp_path: Path) -> None:
    """Directories with corrupt manifest.json are silently skipped."""
    output_dir = tmp_path / "results"

    bad_dir = output_dir / "study_bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "manifest.json").write_text("not json {{{{")

    good_dir = output_dir / "study_good"
    _write_manifest(
        good_dir,
        status="interrupted",
        started_at="2026-03-01T00:00:00+00:00",
        experiments=[],
    )

    result = find_resumable_study(output_dir)
    assert result == good_dir


# ---------------------------------------------------------------------------
# load_resume_state
# ---------------------------------------------------------------------------


def test_load_resume_state_builds_correct_skip_set(tmp_path: Path) -> None:
    """load_resume_state returns skip-set containing only completed experiments."""
    study_dir = tmp_path / "study_001"
    experiments = [
        _make_experiment_entry("hash_a", 1, "completed"),
        _make_experiment_entry("hash_a", 2, "completed"),
        _make_experiment_entry("hash_b", 1, "failed"),
        _make_experiment_entry("hash_b", 2, "interrupted"),
        _make_experiment_entry("hash_c", 1, "pending"),
    ]
    _write_manifest(study_dir, "interrupted", "2026-03-01T00:00:00+00:00", experiments)

    manifest, skip_set = load_resume_state(study_dir)

    assert ("hash_a", 1) in skip_set
    assert ("hash_a", 2) in skip_set
    assert ("hash_b", 1) not in skip_set
    assert ("hash_b", 2) not in skip_set
    assert ("hash_c", 1) not in skip_set
    assert len(skip_set) == 2


def test_load_resume_state_empty_skip_set_when_no_completed(tmp_path: Path) -> None:
    """load_resume_state returns empty skip-set when no experiments are completed."""
    study_dir = tmp_path / "study_001"
    experiments = [
        _make_experiment_entry("hash_a", 1, "failed"),
        _make_experiment_entry("hash_b", 1, "pending"),
    ]
    _write_manifest(study_dir, "failed", "2026-03-01T00:00:00+00:00", experiments)

    _, skip_set = load_resume_state(study_dir)
    assert skip_set == set()


def test_load_resume_state_raises_on_missing_manifest(tmp_path: Path) -> None:
    """load_resume_state raises StudyError when manifest.json does not exist."""
    study_dir = tmp_path / "study_no_manifest"
    study_dir.mkdir()

    with pytest.raises(StudyError, match="manifest.json not found"):
        load_resume_state(study_dir)


def test_load_resume_state_raises_on_corrupt_manifest(tmp_path: Path) -> None:
    """load_resume_state raises StudyError when manifest.json is unparseable."""
    study_dir = tmp_path / "study_bad"
    study_dir.mkdir()
    (study_dir / "manifest.json").write_text("{invalid json}")

    with pytest.raises(StudyError, match="failed to parse manifest.json"):
        load_resume_state(study_dir)


# ---------------------------------------------------------------------------
# validate_config_drift
# ---------------------------------------------------------------------------


def _make_manifest_with_hash(config_hash: str):
    """Return a minimal StudyManifest with the given study_design_hash."""
    from llenergymeasure.study.manifest import StudyManifest

    return StudyManifest(
        study_name="test",
        study_design_hash=config_hash,
        llenergymeasure_version="0.9.0",
        started_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        total_experiments=0,
        pending=0,
        experiments=[],
    )


def test_validate_config_drift_passes_on_matching_hash() -> None:
    """validate_config_drift does not raise when hashes match."""
    manifest = _make_manifest_with_hash("abc123")

    study = MagicMock()
    study.study_design_hash = "abc123"

    validate_config_drift(manifest, study)  # Must not raise


def test_validate_config_drift_raises_on_hash_mismatch() -> None:
    """validate_config_drift raises StudyError with clear hash diff on mismatch."""
    manifest = _make_manifest_with_hash("old_hash_abc")

    study = MagicMock()
    study.study_design_hash = "new_hash_xyz"

    with pytest.raises(StudyError) as exc_info:
        validate_config_drift(manifest, study)

    msg = str(exc_info.value)
    assert "Config drift detected" in msg
    assert "old_hash_abc" in msg
    assert "new_hash_xyz" in msg
    assert "Cannot resume with a different config" in msg


# ---------------------------------------------------------------------------
# prepare_resume_manifest
# ---------------------------------------------------------------------------


def test_prepare_resume_manifest_resets_non_completed_to_pending(tmp_path: Path) -> None:
    """prepare_resume_manifest resets failed/interrupted/skipped entries to pending."""
    study_dir = tmp_path / "study_001"
    experiments = [
        _make_experiment_entry("hash_a", 1, "completed"),
        _make_experiment_entry("hash_b", 1, "failed"),
        _make_experiment_entry("hash_c", 1, "interrupted"),
        _make_experiment_entry("hash_d", 1, "skipped"),
        _make_experiment_entry("hash_e", 1, "pending"),
    ]
    _write_manifest(study_dir, "interrupted", "2026-03-01T00:00:00+00:00", experiments)

    manifest, _ = load_resume_state(study_dir)
    writer = prepare_resume_manifest(study_dir, manifest)

    # Completed entry remains completed
    completed_entry = next(e for e in writer.manifest.experiments if e.config_hash == "hash_a")
    assert completed_entry.status == "completed"

    # All others reset to pending
    for h in ("hash_b", "hash_c", "hash_d", "hash_e"):
        entry = next(e for e in writer.manifest.experiments if e.config_hash == h)
        assert entry.status == "pending", f"{h} should be pending, got {entry.status}"


def test_prepare_resume_manifest_sets_status_to_running(tmp_path: Path) -> None:
    """prepare_resume_manifest sets manifest status to 'running'."""
    study_dir = tmp_path / "study_001"
    _write_manifest(
        study_dir,
        "interrupted",
        "2026-03-01T00:00:00+00:00",
        [_make_experiment_entry("hash_a", 1, "interrupted")],
    )
    manifest, _ = load_resume_state(study_dir)
    writer = prepare_resume_manifest(study_dir, manifest)

    assert writer.manifest.status == "running"


def test_prepare_resume_manifest_writes_to_disk(tmp_path: Path) -> None:
    """prepare_resume_manifest writes the updated manifest to manifest.json."""
    study_dir = tmp_path / "study_001"
    _write_manifest(
        study_dir,
        "circuit_breaker",
        "2026-03-01T00:00:00+00:00",
        [_make_experiment_entry("hash_a", 1, "failed")],
    )
    manifest, _ = load_resume_state(study_dir)
    prepare_resume_manifest(study_dir, manifest)

    # Re-read from disk
    data = json.loads((study_dir / "manifest.json").read_text())
    assert data["status"] == "running"
    assert data["experiments"][0]["status"] == "pending"
