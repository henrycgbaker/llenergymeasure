"""TDD tests for study manifest models and writer.

Tests for:
- ExperimentManifestEntry model
- StudyManifest model (distinct from StudyResult)
- ManifestWriter with atomic writes
- create_study_dir output directory helper
- experiment_result_filename flat-file naming
- build_config_summary helper
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.study.manifest import (
    ExperimentManifestEntry,
    ManifestWriter,
    StudyManifest,
    build_config_summary,
    create_study_dir,
    experiment_result_filename,
)
from tests.conftest import TEST_CONFIG_HASH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_experiment(
    model: str = "meta-llama/Llama-3.1-8B", engine: str = "transformers"
) -> ExperimentConfig:
    return ExperimentConfig(task={"model": model}, engine=engine, dtype="bfloat16")


def _make_study(n_experiments: int = 2, n_cycles: int = 2) -> StudyConfig:
    experiments = [_make_experiment(f"model-{i}") for i in range(n_experiments)]
    return StudyConfig(
        study_name="test-study",
        experiments=experiments,
        study_execution=ExecutionConfig(n_cycles=n_cycles),
        study_design_hash=TEST_CONFIG_HASH,
    )


# ---------------------------------------------------------------------------
# ExperimentManifestEntry tests
# ---------------------------------------------------------------------------


def test_experiment_manifest_entry_defaults() -> None:
    entry = ExperimentManifestEntry(
        config_hash="abc123",
        config_summary="pytorch / llama-3.1-8b / bf16",
        cycle=1,
        status="pending",
    )
    assert entry.config_hash == "abc123"
    assert entry.config_summary == "pytorch / llama-3.1-8b / bf16"
    assert entry.cycle == 1
    assert entry.status == "pending"
    assert entry.result_file is None
    assert entry.error_type is None
    assert entry.error_message is None
    assert entry.started_at is None
    assert entry.completed_at is None


def test_experiment_manifest_entry_completed() -> None:
    now = datetime.now(timezone.utc)
    entry = ExperimentManifestEntry(
        config_hash="abc123",
        config_summary="pytorch / llama-3.1-8b / bf16",
        cycle=2,
        status="completed",
        result_file="results/abc123.json",
        started_at=now,
        completed_at=now,
    )
    assert entry.status == "completed"
    assert entry.result_file == "results/abc123.json"
    assert entry.started_at == now
    assert entry.completed_at == now


def test_experiment_manifest_entry_failed() -> None:
    entry = ExperimentManifestEntry(
        config_hash="deadbeef",
        config_summary="vllm / llama-3.1-8b / fp16",
        cycle=1,
        status="failed",
        error_type="RuntimeError",
        error_message="CUDA out of memory",
    )
    assert entry.status == "failed"
    assert entry.error_type == "RuntimeError"
    assert entry.error_message == "CUDA out of memory"


# ---------------------------------------------------------------------------
# StudyManifest tests
# ---------------------------------------------------------------------------


def test_study_manifest_schema_version() -> None:
    manifest = StudyManifest(
        study_name="test-study",
        study_design_hash="abc123",
        llenergymeasure_version="1.17.0",
        started_at=datetime.now(timezone.utc),
        total_experiments=0,
        pending=0,
        experiments=[],
    )
    assert manifest.schema_version == "2.0"


def test_study_manifest_aggregate_counters() -> None:
    now = datetime.now(timezone.utc)
    entries = [
        ExperimentManifestEntry(
            config_hash="aaa",
            config_summary="pytorch / model-a / bf16",
            cycle=1,
            status="completed",
        ),
        ExperimentManifestEntry(
            config_hash="bbb",
            config_summary="pytorch / model-b / bf16",
            cycle=1,
            status="pending",
        ),
        ExperimentManifestEntry(
            config_hash="ccc",
            config_summary="pytorch / model-c / bf16",
            cycle=1,
            status="failed",
        ),
    ]
    manifest = StudyManifest(
        study_name="test",
        study_design_hash="hash123",
        llenergymeasure_version="1.17.0",
        started_at=now,
        total_experiments=3,
        completed=1,
        failed=1,
        pending=1,
        experiments=entries,
    )
    assert manifest.total_experiments == 3
    assert manifest.completed == 1
    assert manifest.failed == 1
    assert manifest.pending == 1


def test_study_manifest_roundtrip_json() -> None:
    now = datetime.now(timezone.utc)
    manifest = StudyManifest(
        study_name="roundtrip-study",
        study_design_hash="deadbeef",
        llenergymeasure_version="1.17.0",
        started_at=now,
        total_experiments=1,
        pending=1,
        experiments=[
            ExperimentManifestEntry(
                config_hash="h1",
                config_summary="pytorch / llama / bf16",
                cycle=1,
                status="pending",
            )
        ],
    )
    json_str = manifest.model_dump_json()
    restored = StudyManifest.model_validate_json(json_str)
    assert restored == manifest


def test_study_manifest_distinct_from_study_result() -> None:
    from llenergymeasure import StudyResult

    assert StudyManifest is not StudyResult
    assert not issubclass(StudyManifest, StudyResult)
    assert not issubclass(StudyResult, StudyManifest)


# ---------------------------------------------------------------------------
# ManifestWriter tests
# ---------------------------------------------------------------------------


def test_manifest_writer_creates_initial_manifest(tmp_path: Path) -> None:
    study = _make_study(n_experiments=2, n_cycles=2)
    ManifestWriter(study=study, study_dir=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    # 2 experiments x 2 cycles = 4 entries
    assert len(data["experiments"]) == 4
    for entry in data["experiments"]:
        assert entry["status"] == "pending"


def test_manifest_writer_mark_running(tmp_path: Path) -> None:
    study = _make_study(n_experiments=2, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    # Get first config hash from the manifest
    data = json.loads((tmp_path / "manifest.json").read_text())
    first_hash = data["experiments"][0]["config_hash"]

    writer.mark_running(first_hash, cycle=1)

    data = json.loads((tmp_path / "manifest.json").read_text())
    entry = next(
        e for e in data["experiments"] if e["config_hash"] == first_hash and e["cycle"] == 1
    )
    assert entry["status"] == "running"
    assert entry["started_at"] is not None


def test_manifest_writer_mark_completed(tmp_path: Path) -> None:
    study = _make_study(n_experiments=2, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    data = json.loads((tmp_path / "manifest.json").read_text())
    first_hash = data["experiments"][0]["config_hash"]

    writer.mark_running(first_hash, cycle=1)
    writer.mark_completed(first_hash, cycle=1, result_file="result.json")

    data = json.loads((tmp_path / "manifest.json").read_text())
    entry = next(
        e for e in data["experiments"] if e["config_hash"] == first_hash and e["cycle"] == 1
    )
    assert entry["status"] == "completed"
    assert entry["result_file"] == "result.json"
    assert entry["completed_at"] is not None


def test_manifest_writer_mark_failed(tmp_path: Path) -> None:
    study = _make_study(n_experiments=2, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    data = json.loads((tmp_path / "manifest.json").read_text())
    first_hash = data["experiments"][0]["config_hash"]

    writer.mark_running(first_hash, cycle=1)
    writer.mark_failed(first_hash, cycle=1, error_type="RuntimeError", error_message="boom")

    data = json.loads((tmp_path / "manifest.json").read_text())
    entry = next(
        e for e in data["experiments"] if e["config_hash"] == first_hash and e["cycle"] == 1
    )
    assert entry["status"] == "failed"
    assert entry["error_type"] == "RuntimeError"
    assert entry["error_message"] == "boom"
    assert entry["completed_at"] is not None


def test_manifest_writer_uses_atomic_write(tmp_path: Path) -> None:
    study = _make_study(n_experiments=1, n_cycles=1)

    with patch("llenergymeasure.study.manifest._atomic_write") as mock_write:
        ManifestWriter(study=study, study_dir=tmp_path)
        assert mock_write.called, "_atomic_write should be called during __init__"


def test_manifest_writer_write_failure_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    data = json.loads((tmp_path / "manifest.json").read_text())
    config_hash = data["experiments"][0]["config_hash"]

    import logging

    with (
        patch("llenergymeasure.study.manifest._atomic_write", side_effect=OSError("disk full")),
        caplog.at_level(logging.WARNING, logger="llenergymeasure.study.manifest"),
    ):
        # Should not raise
        writer.mark_running(config_hash, cycle=1)

    assert any("disk full" in r.message or "manifest" in r.message.lower() for r in caplog.records)


def test_manifest_writer_find_raises_for_unknown(tmp_path: Path) -> None:
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    with pytest.raises(KeyError):
        writer.mark_running("nonexistent-hash", cycle=1)


# ---------------------------------------------------------------------------
# create_study_dir tests
# ---------------------------------------------------------------------------


def test_create_study_dir_layout(tmp_path: Path) -> None:
    result = create_study_dir(name="batch-sweep", output_dir=tmp_path)
    assert result.exists()
    assert result.is_dir()
    # Must match {name}_{timestamp} pattern
    pattern = re.compile(r"^batch-sweep_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$")
    assert pattern.match(result.name), f"Name {result.name!r} does not match expected pattern"
    assert result.parent == tmp_path


def test_create_study_dir_raises_study_error_on_failure(tmp_path: Path) -> None:
    from llenergymeasure.utils.exceptions import StudyError

    with (
        patch.object(Path, "mkdir", side_effect=PermissionError("no write")),
        pytest.raises(StudyError),
    ):
        create_study_dir(name="sweep", output_dir=tmp_path)


# ---------------------------------------------------------------------------
# experiment_result_filename tests
# ---------------------------------------------------------------------------


def test_experiment_result_filename() -> None:
    result = experiment_result_filename(
        model="meta-llama/Llama-3.1-8B",
        engine="transformers",
        config_hash="abcdef1234567890",
    )
    assert result == "Llama-3.1-8B-transformers_abcdef12.json"


def test_experiment_result_filename_parquet() -> None:
    result = experiment_result_filename(
        model="meta-llama/Llama-3.1-8B",
        engine="transformers",
        config_hash="abcdef1234567890",
        extension=".parquet",
    )
    assert result == "Llama-3.1-8B-transformers_abcdef12.parquet"


# ---------------------------------------------------------------------------
# build_config_summary tests
# ---------------------------------------------------------------------------


def test_config_summary_from_experiment() -> None:
    config = ExperimentConfig(
        task={"model": "meta-llama/Llama-3.1-8B"},
        engine="transformers",
        dtype="bfloat16",
    )
    summary = build_config_summary(config)
    # Uses format_experiment_header: "Llama-3.1-8B / pytorch"
    assert "Llama-3.1-8B" in summary
    assert "transformers" in summary


# ---------------------------------------------------------------------------
# ManifestWriter status / interrupted tests
# ---------------------------------------------------------------------------


def test_manifest_initial_status_is_running(tmp_path: Path) -> None:
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)
    assert writer.manifest.status == "running"


def test_mark_interrupted_sets_status(tmp_path: Path) -> None:
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    writer.mark_interrupted()

    # In-memory check
    assert writer.manifest.status == "interrupted"

    # On-disk check
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["status"] == "interrupted"


# ---------------------------------------------------------------------------
# Bug fix tests: _build_entries deduplication and mark_study_completed (STU-09)
# ---------------------------------------------------------------------------


def test_build_entries_deduplicates_cycled_experiments(tmp_path: Path) -> None:
    """_build_entries() produces exactly n_unique * n_cycles entries, not len(experiments) * n_cycles.

    study.experiments is the pre-cycled list from apply_cycles() (6 entries for
    2 configs x 3 cycles). Without deduplication, _build_entries would loop over
    those 6 entries and multiply by n_cycles=3, producing 18 entries. With
    deduplication by config_hash it should produce exactly 6.
    """
    from collections import defaultdict

    from llenergymeasure.config.grid import ExperimentOrder, apply_cycles
    from llenergymeasure.config.models import DatasetConfig

    exp_a = ExperimentConfig(
        task={
            "model": "model-a",
            "dataset": DatasetConfig(n_prompts=10),
        },
        engine="transformers",
    )
    exp_b = ExperimentConfig(
        task={
            "model": "model-b",
            "dataset": DatasetConfig(n_prompts=10),
        },
        engine="transformers",
    )
    ordered = apply_cycles([exp_a, exp_b], 3, ExperimentOrder.INTERLEAVE, "aabb0011", None)
    assert len(ordered) == 6, "sanity: apply_cycles should produce 6 entries"

    study = StudyConfig(
        experiments=ordered,
        study_name="dedup-test",
        study_execution=ExecutionConfig(n_cycles=3, experiment_order="interleave"),
        study_design_hash="aabb0011",
    )
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    # Should be 6 entries (2 unique configs x 3 cycles), NOT 18
    assert len(writer.manifest.experiments) == 6, (
        f"Expected 6 entries, got {len(writer.manifest.experiments)}"
    )

    # Each unique config_hash should have exactly cycles [1, 2, 3]
    cycles_by_hash: dict[str, list[int]] = defaultdict(list)
    for entry in writer.manifest.experiments:
        cycles_by_hash[entry.config_hash].append(entry.cycle)

    assert len(cycles_by_hash) == 2, f"Expected 2 unique config hashes, got {len(cycles_by_hash)}"
    for h, cycles in cycles_by_hash.items():
        assert sorted(cycles) == [1, 2, 3], (
            f"config_hash {h!r}: expected cycles [1, 2, 3], got {sorted(cycles)}"
        )


def test_mark_study_completed(tmp_path: Path) -> None:
    """mark_study_completed() sets status='completed' and records completed_at."""
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    writer.mark_study_completed()

    # In-memory check
    assert writer.manifest.status == "completed"
    assert writer.manifest.completed_at is not None

    # On-disk check
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["status"] == "completed"
    assert data["completed_at"] is not None


def test_manifest_status_after_all_experiments_complete(tmp_path: Path) -> None:
    """Full lifecycle: running → completed per entry, then mark_study_completed → 'completed'."""
    study = _make_study(n_experiments=2, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    # Drive each entry through pending → running → completed
    for entry in writer.manifest.experiments:
        writer.mark_running(entry.config_hash, entry.cycle)
        writer.mark_completed(entry.config_hash, entry.cycle, result_file="result.json")

    # All experiments done — pending should be 0
    assert writer.manifest.pending == 0

    # Mark study as a whole completed
    writer.mark_study_completed()

    assert writer.manifest.status == "completed"
    assert writer.manifest.completed_at is not None

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["status"] == "completed"
    assert data["pending"] == 0


# ---------------------------------------------------------------------------
# New status literal tests (skipped, interrupted, circuit_breaker, timed_out)
# ---------------------------------------------------------------------------


def test_experiment_entry_accepts_skipped_status() -> None:
    """ExperimentManifestEntry validates 'skipped' status."""
    entry = ExperimentManifestEntry(
        config_hash="abc123",
        config_summary="pytorch / llama / bf16",
        cycle=1,
        status="skipped",
    )
    assert entry.status == "skipped"


def test_experiment_entry_accepts_interrupted_status() -> None:
    """ExperimentManifestEntry validates 'interrupted' status."""
    entry = ExperimentManifestEntry(
        config_hash="abc123",
        config_summary="pytorch / llama / bf16",
        cycle=1,
        status="interrupted",
    )
    assert entry.status == "interrupted"


def test_study_manifest_accepts_circuit_breaker_status() -> None:
    """StudyManifest validates 'circuit_breaker' status."""
    manifest = StudyManifest(
        study_name="test",
        study_design_hash="abc123",
        llenergymeasure_version="0.9.0",
        started_at=datetime.now(timezone.utc),
        total_experiments=3,
        pending=0,
        status="circuit_breaker",
        experiments=[],
    )
    assert manifest.status == "circuit_breaker"


def test_study_manifest_accepts_timed_out_status() -> None:
    """StudyManifest validates 'timed_out' status."""
    manifest = StudyManifest(
        study_name="test",
        study_design_hash="abc123",
        llenergymeasure_version="0.9.0",
        started_at=datetime.now(timezone.utc),
        total_experiments=3,
        pending=0,
        status="timed_out",
        experiments=[],
    )
    assert manifest.status == "timed_out"


def test_mark_interrupted_downgrades_running_entries(tmp_path: Path) -> None:
    """mark_interrupted() downgrades 'running' entries to 'interrupted', leaves others unchanged."""
    study = _make_study(n_experiments=3, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    entries = writer.manifest.experiments
    first_hash = entries[0].config_hash
    second_hash = entries[1].config_hash

    # Put first entry in running, complete second, leave third pending
    writer.mark_running(first_hash, cycle=1)
    writer.mark_running(second_hash, cycle=1)
    writer.mark_completed(second_hash, cycle=1, result_file="result.json")

    writer.mark_interrupted()

    # Study status set
    assert writer.manifest.status == "interrupted"

    # Running entry downgraded to interrupted
    first = next(e for e in writer.manifest.experiments if e.config_hash == first_hash)
    assert first.status == "interrupted"

    # Completed entry left unchanged
    second = next(e for e in writer.manifest.experiments if e.config_hash == second_hash)
    assert second.status == "completed"

    # Third entry stays pending
    third_hash = entries[2].config_hash
    third = next(e for e in writer.manifest.experiments if e.config_hash == third_hash)
    assert third.status == "pending"

    # interrupted counter incremented
    assert writer.manifest.interrupted == 1


def test_mark_skipped_sets_entry_status(tmp_path: Path) -> None:
    """mark_skipped() sets entry status to 'skipped' with reason."""
    study = _make_study(n_experiments=2, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    first_hash = writer.manifest.experiments[0].config_hash
    writer.mark_skipped(first_hash, cycle=1, reason="circuit breaker tripped")

    entry = writer._find(first_hash, cycle=1)
    assert entry.status == "skipped"
    assert entry.error_message == "circuit breaker tripped"
    assert entry.completed_at is not None

    # Counter incremented
    assert writer.manifest.skipped == 1


def test_mark_skipped_default_reason(tmp_path: Path) -> None:
    """mark_skipped() uses default reason when none provided."""
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    first_hash = writer.manifest.experiments[0].config_hash
    writer.mark_skipped(first_hash, cycle=1)

    entry = writer._find(first_hash, cycle=1)
    assert entry.error_message == "skipped by circuit breaker or timeout"


def test_mark_study_circuit_breaker(tmp_path: Path) -> None:
    """mark_study_circuit_breaker() sets status and completed_at."""
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    writer.mark_study_circuit_breaker()

    assert writer.manifest.status == "circuit_breaker"
    assert writer.manifest.completed_at is not None

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["status"] == "circuit_breaker"
    assert data["completed_at"] is not None


def test_mark_study_timed_out(tmp_path: Path) -> None:
    """mark_study_timed_out() sets status and completed_at."""
    study = _make_study(n_experiments=1, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    writer.mark_study_timed_out()

    assert writer.manifest.status == "timed_out"
    assert writer.manifest.completed_at is not None

    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["status"] == "timed_out"
    assert data["completed_at"] is not None


def test_recount_counts_skipped_and_interrupted(tmp_path: Path) -> None:
    """_recount() correctly tallies skipped and interrupted entries."""
    study = _make_study(n_experiments=3, n_cycles=1)
    writer = ManifestWriter(study=study, study_dir=tmp_path)

    entries = writer.manifest.experiments
    hashes = [e.config_hash for e in entries]

    writer.mark_running(hashes[0], cycle=1)
    writer.mark_completed(hashes[0], cycle=1, result_file="result.json")
    writer.mark_skipped(hashes[1], cycle=1, reason="timeout")
    writer.mark_running(hashes[2], cycle=1)
    # Simulate interrupted via direct status change + _recount
    for entry in writer.manifest.experiments:
        if entry.config_hash == hashes[2]:
            entry.status = "interrupted"
    writer._recount()

    assert writer.manifest.completed == 1
    assert writer.manifest.skipped == 1
    assert writer.manifest.interrupted == 1
    assert writer.manifest.failed == 0
    assert writer.manifest.pending == 0
