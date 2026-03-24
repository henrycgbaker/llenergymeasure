"""Thermal gap countdown display with Enter-to-skip and SIGINT-safe abort.

Provides run_gap() for use by StudyRunner between experiments.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time

__all__ = ["format_gap_duration", "run_gap"]


def format_gap_duration(seconds: float) -> str:
    """Format gap duration for inline countdown display.

    Args:
        seconds: Remaining seconds (non-negative).

    Returns:
        Under 120s: "47s remaining"
        120s and over: "4m 32s remaining"
    """
    if seconds < 120:
        return f"{int(seconds)}s remaining"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s remaining"


def run_gap(seconds: float, label: str, interrupt_event: threading.Event) -> None:
    """Run a thermal gap with inline countdown display and Enter-to-skip.

    Displays an in-place countdown that overwrites itself each tick:
        "Config gap: 47s remaining (Enter to skip)"

    Enter-to-skip: only active in TTY environments. Degrades gracefully in
    non-TTY (e.g., CI, piped output) — gap runs for the full duration.

    Ctrl+C (SIGINT) during a gap: interrupt_event is set by the StudyRunner
    SIGINT handler. gap exits immediately without printing "done".

    Args:
        seconds: Gap duration in seconds.
        label: Display label — "Config gap" or "Cycle gap".
        interrupt_event: threading.Event set by SIGINT handler; if set on
                         entry or during the gap, abort immediately.

    When Rich Live is active, StudyRunner._run_gap() renders the countdown
    inside the live display instead of calling this function.
    """
    if seconds <= 0:
        return

    # Check interrupt before doing anything
    if interrupt_event.is_set():
        return

    skip_event = threading.Event()

    # Enter-to-skip: daemon readline thread (TTY only)
    if sys.stdin.isatty():

        def _wait_for_enter() -> None:
            with contextlib.suppress(Exception):
                sys.stdin.readline()
            skip_event.set()

        reader = threading.Thread(target=_wait_for_enter, daemon=True)
        reader.start()

    start = time.monotonic()
    end = start + seconds

    while True:
        now = time.monotonic()
        remaining = end - now

        # Hard stop: SIGINT received
        if interrupt_event.is_set():
            # Clear line without printing "done" — caller handles exit output
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
            return

        # Gap skipped by Enter keypress
        if skip_event.is_set():
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
            print(f"{label}: done")
            return

        if remaining <= 0:
            break

        # Print in-place countdown
        msg = f"\r{label}: {format_gap_duration(remaining)} (Enter to skip)  "
        sys.stdout.write(msg)
        sys.stdout.flush()

        time.sleep(1)

    # Normal completion: clear line and print "done"
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    print(f"{label}: done")
