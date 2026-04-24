"""Unit tests for _save_and_record timeseries sidecar handling.

Tests cover:
- Parquet sidecar is copied into experiment result subdirectory when present
- Stale source file is cleaned up after copy
- No regression when timeseries is None (no sidecar, no crash)
- Graceful handling when source parquet file is missing from disk
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.study.runner import _save_and_record

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    with_timeseries: bool = True,
) -> ExperimentResult:
    """Construct a minimal ExperimentResult for testing _save_and_record."""
    return ExperimentResult(
        experiment_id="test-save-record-001",
        measurement_config_hash="aabb1122ccdd3344",
        measurement_methodology="total",
        model_name="gpt2",
        total_tokens=256,
        total_energy_j=10.0,
        total_inference_time_sec=2.0,
        avg_tokens_per_second=128.0,
        avg_energy_per_token_j=0.039,
        total_flops=5e10,
        timeseries="timeseries.parquet" if with_timeseries else None,
        start_time=datetime(2026, 3, 25, 10, 0, 0),
        end_time=datetime(2026, 3, 25, 10, 0, 2),
    )


def _create_parquet(path: Path) -> None:
    """Write a minimal parquet file at the given path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"timestamp_s": [0.0, 1.0], "gpu_power_w": [100.0, 105.0]})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_and_record_copies_timeseries_sidecar(tmp_path: Path) -> None:
    """Parquet sidecar is copied into the experiment result subdirectory.

    The stale flat file written by MeasurementHarness in output_dir is also
    removed after the copy.
    """
    # Create a real parquet file at the location MeasurementHarness would write it
    source_parquet = tmp_path / "timeseries.parquet"
    _create_parquet(source_parquet)
    assert source_parquet.exists(), "Pre-condition: source parquet must exist"

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=True)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result, study_dir, manifest, "aabb1122", 1, result_files, ts_source_dir=tmp_path
    )

    # result.json should have been saved into a subdirectory under study_dir
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()
    assert result_json_path.name == "result.json"

    # timeseries.parquet should be next to result.json in the subdirectory
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert sidecar_dest.exists(), "Parquet sidecar must be copied into experiment subdirectory"

    # Stale flat file must be cleaned up
    assert not source_parquet.exists(), "Stale source parquet must be removed after copy"

    # Manifest must be updated with a non-empty result path
    manifest.mark_completed.assert_called_once()
    call_args = manifest.mark_completed.call_args
    rel_path = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("result_file", "")
    assert rel_path, "mark_completed must be called with a non-empty result_file"


def test_save_and_record_no_timeseries(tmp_path: Path) -> None:
    """When result.timeseries is None, no sidecar is copied and no crash occurs."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    assert result.timeseries is None

    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(result, study_dir, manifest, "ccdd5566", 1, result_files)

    # result.json still saved
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()

    # No timeseries sidecar in the subdirectory
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert not sidecar_dest.exists(), "No sidecar should be created when timeseries is None"

    # Manifest updated normally
    manifest.mark_completed.assert_called_once()


def test_save_and_record_missing_source_file(tmp_path: Path) -> None:
    """When timeseries field is set but the parquet file is not on disk, no crash.

    result.json is still saved and manifest.mark_completed is still called.
    The missing file is simply skipped (save_result handles this with a warning log).
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Result claims timeseries exists but we deliberately do NOT create the file
    result = _make_result(with_timeseries=True)
    assert result.timeseries == "timeseries.parquet"
    source_parquet = tmp_path / "timeseries.parquet"
    assert not source_parquet.exists(), "Pre-condition: source file must NOT exist"

    manifest = MagicMock()
    result_files: list[str] = []

    # Must not raise
    _save_and_record(
        result, study_dir, manifest, "eeff7788", 1, result_files, ts_source_dir=tmp_path
    )

    # result.json still saved
    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    assert result_json_path.exists()

    # No timeseries.parquet in the subdirectory (nothing to copy)
    sidecar_dest = result_json_path.parent / "timeseries.parquet"
    assert not sidecar_dest.exists()

    # Manifest updated normally (non-empty path)
    manifest.mark_completed.assert_called_once()
    call_args = manifest.mark_completed.call_args
    rel_path = call_args[0][2] if len(call_args[0]) >= 3 else call_args[1].get("result_file", "")
    assert rel_path, "mark_completed must be called with a non-empty result_file"


def test_save_and_record_writes_resolved_config_hash(tmp_path: Path) -> None:
    """resolved_config_hash must be written into config.json sidecar when provided.

    Regression test for Bug 2: _save_and_record had a resolved_config_hash
    parameter that was never passed from the call site, leaving the sidecar
    branch unreachable.  This test verifies the end-to-end write-and-read path.
    """
    import json

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    # Write a minimal config.json in the ts_source_dir (simulates harness output)
    config_sidecar_src = tmp_path / "config.json"
    config_sidecar_src.write_text(
        json.dumps(
            {
                "experiment_id": "test-resolved-001",
                "config_hash": "aabb1122ccdd3344",
                "engine": "transformers",
                "library_version": "4.50.0",
                "observed_config_hash": "sha256_h3_stub",
            }
        )
    )

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    _save_and_record(
        result,
        study_dir,
        manifest,
        "aabb1122",
        1,
        result_files,
        ts_source_dir=tmp_path,
        resolved_config_hash="resolved_sha256_h1_value",
    )

    assert len(result_files) == 1
    result_json_path = Path(result_files[0])
    dest_config = result_json_path.parent / "config.json"
    assert dest_config.exists(), "config.json sidecar must be moved to result dir"

    payload = json.loads(dest_config.read_text())
    assert payload.get("resolved_config_hash") == "resolved_sha256_h1_value", (
        "resolved_config_hash must be patched into config.json by _save_and_record"
    )
    # Source sidecar must be cleaned up
    assert not config_sidecar_src.exists()


def test_save_and_record_calls_mark_failed_on_exception(tmp_path: Path) -> None:
    """When save_result raises, manifest.mark_failed is called (not mark_completed).

    Previously the except clause called mark_completed with result_file="" which
    silently recorded a failure as a success with no result path. This test
    verifies the corrected behaviour: mark_failed is called with a meaningful
    error type and message.
    """
    from unittest.mock import patch

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    result = _make_result(with_timeseries=False)
    manifest = MagicMock()
    result_files: list[str] = []

    with patch(
        "llenergymeasure.results.persistence.save_result",
        side_effect=OSError("disk full"),
    ):
        _save_and_record(result, study_dir, manifest, "aabb1122", 1, result_files)

    # mark_failed must be called — NOT mark_completed
    manifest.mark_failed.assert_called_once()
    call_args = manifest.mark_failed.call_args[0]
    assert call_args[0] == "aabb1122"  # config_hash
    assert call_args[1] == 1  # cycle
    assert "OSError" in call_args[2]  # error_type
    assert "disk full" in call_args[3]  # error_message

    manifest.mark_completed.assert_not_called()
