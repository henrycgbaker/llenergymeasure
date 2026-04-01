"""Tests for StudyResult full schema (RES-13)."""

from llenergymeasure.domain.experiment import StudyResult


def test_study_result_has_full_schema():
    """StudyResult includes all RES-13 fields."""
    result = StudyResult(
        experiments=[],
        study_name="my-study",
        study_design_hash="abcdef0123456789",
        measurement_protocol={
            "n_cycles": 3,
            "experiment_order": "interleave",
            "experiment_gap_seconds": 30,
            "cycle_gap_seconds": 60,
        },
        result_files=["exp1/result.json", "exp2/result.json"],
        total_experiments=4,
        completed=3,
        failed=1,
        total_wall_time_s=120.5,
        total_energy_j=450.0,
        warnings=["1 experiment failed"],
    )
    assert result.study_design_hash == "abcdef0123456789"
    assert result.measurement_protocol["n_cycles"] == 3
    assert len(result.result_files) == 2
    assert result.total_experiments == 4
    assert result.failed == 1
    assert result.warnings == ["1 experiment failed"]


def test_study_result_backwards_compat():
    """StudyResult still works with M1-style minimal construction."""
    result = StudyResult(experiments=[], study_name="test")
    assert result.study_design_hash is None
    assert result.measurement_protocol == {}
    assert result.result_files == []
    assert result.total_experiments == 0
    assert result.warnings == []
    assert result.skipped_experiments == []


def test_study_result_summary_field_defaults():
    """StudyResult summary field defaults: completed=0, failed=0, energy=0, warnings=[]."""
    result = StudyResult(experiments=[], total_experiments=5)
    assert result.completed == 0
    assert result.failed == 0
    assert result.total_wall_time_s == 0.0
    assert result.total_energy_j == 0.0
    assert result.warnings == []


def test_study_result_skipped_experiments_default():
    """StudyResult.skipped_experiments defaults to empty list."""
    result = StudyResult(experiments=[])
    assert result.skipped_experiments == []


def test_study_result_skipped_experiments_populated():
    """StudyResult.skipped_experiments can be populated with skipped grid points."""
    result = StudyResult(
        experiments=[],
        skipped_experiments=[{"raw_config": {"model": "x"}, "reason": "invalid", "errors": []}],
    )
    assert len(result.skipped_experiments) == 1


def test_result_files_are_paths_not_embedded():
    """RES-15: result_files contains path strings, not ExperimentResult objects."""
    result = StudyResult(
        experiments=[],
        result_files=["study_2026/exp1/result.json", "study_2026/exp2/result.json"],
    )
    assert all(isinstance(f, str) for f in result.result_files)
    assert all(f.endswith("result.json") for f in result.result_files)
