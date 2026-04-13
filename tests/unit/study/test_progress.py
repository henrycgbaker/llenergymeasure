"""Unit tests for study/_progress.py — print_study_progress and helpers.

All tests run GPU-free. Output is captured via capsys (writes to stderr).
"""

from __future__ import annotations

from llenergymeasure.study._progress import (
    _format_duration,
    _sig3,
    print_study_progress,
)
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# _sig3 helper
# ---------------------------------------------------------------------------


def test_sig3_zero_returns_zero():
    """_sig3(0) returns '0'."""
    assert _sig3(0) == "0"


def test_sig3_small_integer():
    """_sig3(312.4) returns '312'."""
    result = _sig3(312.4)
    assert result == "312"


def test_sig3_decimal():
    """_sig3(3.12) returns '3.12'."""
    result = _sig3(3.12)
    assert result == "3.12"


def test_sig3_very_small():
    """_sig3(0.00312) returns '0.00312'."""
    result = _sig3(0.00312)
    assert result == "0.00312"


def test_sig3_large_rounds():
    """_sig3(1234) rounds to 3 sig figs."""
    result = _sig3(1234)
    assert result == "1230"


# ---------------------------------------------------------------------------
# _format_duration helper
# ---------------------------------------------------------------------------


def test_format_duration_sub_minute():
    """Values under 60s shown as X.Xs."""
    assert _format_duration(4.2) == "4.2s"


def test_format_duration_minutes():
    """272s formats as '4m 32s'."""
    assert _format_duration(272) == "4m 32s"


def test_format_duration_hours():
    """3900s formats as '1h 05m'."""
    assert _format_duration(3900) == "1h 05m"


def test_format_duration_exactly_60():
    """Exactly 60s formats as '1m 00s'."""
    assert _format_duration(60) == "1m 00s"


# ---------------------------------------------------------------------------
# print_study_progress — output format
# ---------------------------------------------------------------------------


def test_print_study_progress_running_status(capsys):
    """Running status outputs '...' icon."""
    config = make_config()
    print_study_progress(1, 5, config, status="running")
    captured = capsys.readouterr()
    assert "..." in captured.err
    assert "[1/5]" in captured.err


def test_print_study_progress_completed_status(capsys):
    """Completed status outputs 'OK' icon."""
    config = make_config()
    print_study_progress(3, 10, config, status="completed")
    captured = capsys.readouterr()
    assert "OK" in captured.err
    assert "[3/10]" in captured.err


def test_print_study_progress_failed_status(capsys):
    """Failed status outputs 'FAIL' icon."""
    config = make_config()
    print_study_progress(2, 4, config, status="failed")
    captured = capsys.readouterr()
    assert "FAIL" in captured.err


def test_print_study_progress_unknown_status(capsys):
    """Unknown status outputs '?' icon."""
    config = make_config()
    print_study_progress(1, 1, config, status="weirdstatus")
    captured = capsys.readouterr()
    assert "?" in captured.err


def test_print_study_progress_includes_model_and_engine(capsys):
    """Output line includes model name and engine."""
    config = make_config(model="meta-llama/Llama-3-8B", engine="vllm")
    print_study_progress(1, 1, config, status="running")
    captured = capsys.readouterr()
    assert "meta-llama/Llama-3-8B" in captured.err
    assert "vllm" in captured.err


def test_print_study_progress_with_elapsed(capsys):
    """Elapsed time is included when provided."""
    config = make_config()
    print_study_progress(1, 5, config, status="completed", elapsed=30.5)
    captured = capsys.readouterr()
    assert "30.5s" in captured.err
    assert "--" in captured.err


def test_print_study_progress_without_elapsed(capsys):
    """No '--' separator or elapsed time when elapsed is None."""
    config = make_config()
    print_study_progress(1, 5, config, status="running", elapsed=None)
    captured = capsys.readouterr()
    assert "--" not in captured.err


def test_print_study_progress_with_energy(capsys):
    """Energy in joules is included when provided."""
    config = make_config()
    print_study_progress(1, 5, config, status="completed", elapsed=10.0, energy=150.5)
    captured = capsys.readouterr()
    assert "J" in captured.err


def test_print_study_progress_without_energy(capsys):
    """No '(...)' energy term when energy is None."""
    config = make_config()
    print_study_progress(1, 5, config, status="running")
    captured = capsys.readouterr()
    # Should not contain any parenthesised energy value
    assert "(" not in captured.err


def test_print_study_progress_writes_to_stderr(capsys):
    """Output goes to stderr, not stdout."""
    config = make_config()
    print_study_progress(1, 3, config, status="running")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""
