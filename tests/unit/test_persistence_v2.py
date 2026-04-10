"""Unit tests for v2.0 results persistence API.

Tests cover:
- Subdirectory creation and naming
- Atomic writes
- Collision handling
- Round-trip fidelity (datetime, tuple, nested models)
- Timeseries sidecar management
- Graceful degradation on missing sidecar
- Never-overwrite guarantee
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.results.persistence import load_result, save_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_result() -> ExperimentResult:
    """Minimal valid ExperimentResult with pytorch backend and known model."""
    return ExperimentResult(
        experiment_id="persist-test-001",
        measurement_config_hash="abcdef0123456789",
        measurement_methodology="total",
        total_tokens=512,
        total_energy_j=25.6,
        total_inference_time_sec=5.0,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=1e11,
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 5),
        model_name="gpt2",
    )


@pytest.fixture()
def hf_model_result() -> ExperimentResult:
    """ExperimentResult with a HuggingFace-style model path containing /."""
    return ExperimentResult(
        experiment_id="persist-test-hf",
        measurement_config_hash="fedcba9876543210",
        measurement_methodology="steady_state",
        steady_state_window=(1.0, 4.0),
        total_tokens=1024,
        total_energy_j=51.2,
        total_inference_time_sec=10.0,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=2e11,
        start_time=datetime(2026, 2, 26, 14, 0, 0),
        end_time=datetime(2026, 2, 26, 14, 0, 10),
        model_name="meta-llama/Llama-3.1-8B",
    )


@pytest.fixture()
def result_with_timeseries() -> ExperimentResult:
    """ExperimentResult that claims to have a timeseries sidecar."""
    return ExperimentResult(
        experiment_id="persist-test-ts",
        measurement_config_hash="1122334455667788",
        measurement_methodology="windowed",
        total_tokens=256,
        total_energy_j=12.8,
        total_inference_time_sec=2.5,
        avg_tokens_per_second=102.4,
        avg_energy_per_token_j=0.05,
        total_flops=5e10,
        timeseries="timeseries.parquet",
        start_time=datetime(2026, 2, 26, 15, 0, 0),
        end_time=datetime(2026, 2, 26, 15, 0, 2, 500000),
        model_name="gpt2",
    )


def _create_temp_parquet(tmp_path: Path) -> Path:
    """Create a minimal parquet file for sidecar tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"timestamp_s": [0.0, 1.0], "gpu_power_w": [100.0, 105.0]})
    path = tmp_path / "temp_timeseries.parquet"
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# Directory creation and naming
# ---------------------------------------------------------------------------


def test_save_creates_subdirectory(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() creates a subdirectory containing result.json."""
    result_path = save_result(minimal_result, tmp_path)
    assert result_path.is_file()
    assert result_path.name == "result.json"
    # result.json is inside a subdirectory of tmp_path
    assert result_path.parent.parent == tmp_path


def test_save_directory_name_format(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Directory name matches c{cycle}_{model_short}-{backend}_{hash} pattern."""
    result_path = save_result(minimal_result, tmp_path)
    dir_name = result_path.parent.name
    # Pattern: c1_gpt2-pytorch_abcdef01 (c{cycle}_{model_short}-{backend}_{hash[:8]})
    pattern = re.compile(r"^c\d+_[\w\.\-]+-[a-z]+_[a-f0-9]{8,}(_\d+)?$")
    assert pattern.match(dir_name), f"Directory name '{dir_name}' does not match expected pattern"


def test_save_model_slug_normalisation(tmp_path: Path, hf_model_result: ExperimentResult) -> None:
    """HuggingFace model path strips org prefix, uses short model name."""
    result_path = save_result(hf_model_result, tmp_path)
    dir_name = result_path.parent.name
    # meta-llama/Llama-3.1-8B -> Llama-3.1-8B (short name, no org prefix)
    assert "Llama-3.1-8B-pytorch" in dir_name, f"Slug normalisation failed: {dir_name}"


def test_save_returns_result_json_path(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() return value ends with result.json."""
    result_path = save_result(minimal_result, tmp_path)
    assert result_path.name == "result.json"
    assert isinstance(result_path, Path)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_save_atomic_write(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """result.json is valid JSON after save_result()."""
    import json

    result_path = save_result(minimal_result, tmp_path)
    content = result_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert parsed["experiment_id"] == "persist-test-001"
    from tests.conftest import EXPERIMENT_SCHEMA_VERSION

    assert parsed["schema_version"] == EXPERIMENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------


def test_collision_suffix_applied(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Second save to same base dir gets _1 suffix."""
    first = save_result(minimal_result, tmp_path)
    second = save_result(minimal_result, tmp_path)
    assert first != second
    assert "_1" in second.parent.name


def test_collision_multiple_suffixes(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Third save gets _2 (or _3 if first was _1 and _2 already exists)."""
    first = save_result(minimal_result, tmp_path)
    second = save_result(minimal_result, tmp_path)
    third = save_result(minimal_result, tmp_path)
    dirs = {first.parent, second.parent, third.parent}
    assert len(dirs) == 3, "Three saves should produce 3 distinct directories"


def test_never_overwrites(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """5 saves produce 5 distinct directories (never-overwrite guarantee)."""
    paths = [save_result(minimal_result, tmp_path) for _ in range(5)]
    dirs = [p.parent for p in paths]
    assert len(set(dirs)) == 5, f"Expected 5 distinct dirs, got {len(set(dirs))}: {dirs}"


# ---------------------------------------------------------------------------
# Load / round-trip
# ---------------------------------------------------------------------------


def test_from_json_loads_correctly(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """load_result() returns ExperimentResult with same field values."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)

    assert loaded.experiment_id == minimal_result.experiment_id
    assert loaded.measurement_config_hash == minimal_result.measurement_config_hash
    from tests.conftest import EXPERIMENT_SCHEMA_VERSION

    assert loaded.schema_version == EXPERIMENT_SCHEMA_VERSION
    assert loaded.measurement_methodology == minimal_result.measurement_methodology
    assert loaded.total_tokens == minimal_result.total_tokens
    assert loaded.total_energy_j == pytest.approx(minimal_result.total_energy_j)
    assert loaded.backend == minimal_result.backend


def test_from_json_round_trip(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """model_dump_json() on loaded result matches original (datetime, tuple fidelity)."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)

    original_json = minimal_result.model_dump_json(indent=2)
    loaded_json = loaded.model_dump_json(indent=2)
    assert original_json == loaded_json


def test_model_name_round_trips(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """model_name field survives save/load unchanged."""
    result_path = save_result(minimal_result, tmp_path)
    loaded = load_result(result_path)
    assert loaded.model_name == minimal_result.model_name


def test_steady_state_window_round_trips(tmp_path: Path, hf_model_result: ExperimentResult) -> None:
    """steady_state_window tuple (float, float) survives JSON round-trip."""
    result_path = save_result(hf_model_result, tmp_path)
    loaded = load_result(result_path)

    assert loaded.steady_state_window is not None
    assert len(loaded.steady_state_window) == 2
    assert loaded.steady_state_window[0] == pytest.approx(1.0)
    assert loaded.steady_state_window[1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Timeseries sidecar
# ---------------------------------------------------------------------------


def test_save_with_timeseries_sidecar(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() with timeseries_source copies parquet file to output dir."""
    parquet_path = _create_temp_parquet(tmp_path)
    result_path = save_result(minimal_result, tmp_path, timeseries_source=parquet_path)
    sidecar = result_path.parent / "timeseries.parquet"
    assert sidecar.exists(), "timeseries.parquet should be copied to output dir"


def test_save_without_timeseries(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """save_result() with no timeseries_source produces no parquet file."""
    result_path = save_result(minimal_result, tmp_path)
    sidecar = result_path.parent / "timeseries.parquet"
    assert not sidecar.exists(), "No timeseries.parquet should exist when not provided"


def test_from_json_missing_sidecar_warns(
    tmp_path: Path,
    result_with_timeseries: ExperimentResult,
) -> None:
    """Loading result that claims sidecar but sidecar absent emits UserWarning."""
    # Save without providing the actual sidecar file
    result_path = save_result(result_with_timeseries, tmp_path)

    with pytest.warns(UserWarning, match="Timeseries sidecar missing"):
        loaded = load_result(result_path)

    # Result still loads
    assert loaded.experiment_id == result_with_timeseries.experiment_id


def test_from_json_missing_sidecar_loads_successfully(
    tmp_path: Path,
    result_with_timeseries: ExperimentResult,
) -> None:
    """Result loads with timeseries field preserved even when sidecar missing."""
    result_path = save_result(result_with_timeseries, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_result(result_path)

    # timeseries field is preserved (not set to None) — it's what was stored
    assert loaded.timeseries == "timeseries.parquet"
    # But the file itself is not there
    assert not (result_path.parent / "timeseries.parquet").exists()


# ---------------------------------------------------------------------------
# Cycle in directory name
# ---------------------------------------------------------------------------


def test_save_cycle_in_directory_name(tmp_path: Path, minimal_result: ExperimentResult) -> None:
    """Directory name includes cycle number."""
    result_path = save_result(minimal_result, tmp_path, cycle=2)
    dir_name = result_path.parent.name
    assert dir_name.startswith("c2_"), f"Expected 'c2_' prefix in '{dir_name}'"


def test_save_experiment_index_in_directory_name(
    tmp_path: Path, minimal_result: ExperimentResult
) -> None:
    """Directory name includes experiment index when provided."""
    result_path = save_result(minimal_result, tmp_path, experiment_index=5, cycle=3)
    dir_name = result_path.parent.name
    assert dir_name.startswith("005_c3_"), f"Expected '005_c3_' prefix in '{dir_name}'"
