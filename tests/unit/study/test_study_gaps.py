"""Tests for study gap countdown display and skip logic.

Tests cover:
- format_gap_duration formatting rules
- run_gap normal completion (0-second gap)
- run_gap with interrupt_event pre-set (immediate return)
- run_gap in non-TTY environment (no crash)
"""

from __future__ import annotations

import threading
from unittest.mock import patch

from llenergymeasure.study.gaps import format_gap_duration, run_gap

# ---------------------------------------------------------------------------
# format_gap_duration
# ---------------------------------------------------------------------------


def test_format_gap_duration_under_120s() -> None:
    result = format_gap_duration(47.9)
    assert result == "47s remaining"


def test_format_gap_duration_over_120s() -> None:
    result = format_gap_duration(272.0)  # 4m 32s
    assert result == "4m 32s remaining"


def test_format_gap_duration_exactly_120s() -> None:
    result = format_gap_duration(120.0)
    assert result == "2m 00s remaining"


def test_format_gap_duration_zero() -> None:
    result = format_gap_duration(0.0)
    assert result == "0s remaining"


def test_format_gap_duration_one_minute() -> None:
    result = format_gap_duration(60.0)
    # 60 < 120, so should use seconds form
    assert result == "60s remaining"


def test_format_gap_duration_large() -> None:
    result = format_gap_duration(3661.0)  # 61m 1s
    assert result == "61m 01s remaining"


# ---------------------------------------------------------------------------
# run_gap
# ---------------------------------------------------------------------------


def test_run_gap_completes_normally() -> None:
    """0-second gap returns immediately without exception."""
    event = threading.Event()
    # Should return instantly (seconds=0 path)
    run_gap(0, "Config gap", event)


def test_run_gap_negative_seconds() -> None:
    """Negative duration returns immediately without exception."""
    event = threading.Event()
    run_gap(-5, "Config gap", event)


def test_run_gap_interrupt_event_pre_set() -> None:
    """If interrupt_event is already set, run_gap returns without waiting."""
    event = threading.Event()
    event.set()

    returned_event = threading.Event()

    def _run_gap() -> None:
        run_gap(60, "Config gap", event)
        returned_event.set()

    t = threading.Thread(target=_run_gap, daemon=True)
    t.start()
    returned = returned_event.wait(timeout=2.0)
    assert returned, "run_gap did not return when interrupt_event was pre-set"
    t.join(timeout=1.0)


def test_run_gap_interrupt_event_set_during_gap() -> None:
    """Interrupt event set during a gap causes run_gap to return without waiting out the full gap."""
    interrupt_event = threading.Event()
    returned_event = threading.Event()

    def _run_gap() -> None:
        run_gap(10, "Config gap", interrupt_event)
        returned_event.set()

    t = threading.Thread(target=_run_gap, daemon=True)
    t.start()
    # Signal interrupt immediately after thread starts
    interrupt_event.set()
    # Safety net: wait up to 2s — the assertion is that the event was set, not how long it took
    returned = returned_event.wait(timeout=2.0)
    assert returned, "run_gap did not return after interrupt_event was set"
    t.join(timeout=1.0)


def test_run_gap_non_tty_no_crash() -> None:
    """Patching sys.stdin.isatty to return False: no crash, gap completes normally."""
    event = threading.Event()
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False
        # 0-second gap in non-TTY: should return without error
        run_gap(0, "Cycle gap", event)


def test_run_gap_import() -> None:
    """Public API exports are correct."""
    from llenergymeasure.study.gaps import format_gap_duration, run_gap  # noqa: F401
