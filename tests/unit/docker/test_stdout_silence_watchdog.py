"""Tests for the unified stdout-silence + wall-clock watchdog (issue #366).

The watchdog lives in ``DockerRunner._run_container_streaming`` and replaces
the previous blocking ``for line in proc.stdout`` shape that could hang
indefinitely on a stuck CUDA / NCCL / compile step.

Three load-bearing cases pin the contract:

1. **Stuck-stdout subprocess** — opens stdout then ``time.sleep(9999)``
   without writing. The watchdog must kill it within
   ``silence_timeout + epsilon`` and raise ``DockerStdoutSilenceError``.
2. **Wall-clock fires when stdout is active** — a chatty subprocess
   that prints continuously must still be killed when its total runtime
   exceeds ``timeout``, raising ``DockerTimeoutError``.
3. **Both budgets disabled** — the watchdog must not interfere with a
   normal short-lived subprocess.

We avoid spinning up real Docker containers by exercising the watchdog
loop against ``subprocess.Popen`` directly via a private helper. This is
the same shape the runner uses internally (the watchdog is independent
of whether the underlying process is ``docker run`` or any other
``Popen``).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator

import pytest

from llenergymeasure.infra.docker_errors import (
    DockerStdoutSilenceError,
    DockerTimeoutError,
)
from llenergymeasure.infra.docker_runner import DockerRunner


def _stuck_stdout_script() -> list[str]:
    """Subprocess command that writes one line, flushes, then sleeps forever.

    Models a hung container: stdout pipe is open, process is alive, but
    no further output ever arrives. The watchdog's silence timer must
    fire because the wall-clock isn't necessarily expired.
    """
    return [
        sys.executable,
        "-u",  # unbuffered so the first line lands immediately
        "-c",
        "import sys, time; sys.stdout.write('hello\\n'); sys.stdout.flush(); time.sleep(9999)",
    ]


def _chatty_script(duration_s: float = 30.0, interval_s: float = 0.05) -> list[str]:
    """Subprocess that prints a line every ``interval_s`` for ``duration_s``."""
    return [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys, time; "
            f"end = time.monotonic() + {duration_s}; "
            "i = 0\n"
            f"while time.monotonic() < end: "
            f"  sys.stdout.write(f'tick {{i}}\\n'); sys.stdout.flush(); "
            f"  i += 1; time.sleep({interval_s})"
        ),
    ]


def _quick_script() -> list[str]:
    """Subprocess that prints once and exits — fits well inside any budget."""
    return [sys.executable, "-u", "-c", "print('done')"]


@pytest.fixture
def runner_factory() -> Iterator[callable]:
    """Build a DockerRunner stub with watchdog params; yield a factory."""
    runners: list[DockerRunner] = []

    def _make(timeout: float | None = None, silence_timeout: float | None = None) -> DockerRunner:
        r = DockerRunner(image="not-used", timeout=timeout, silence_timeout=silence_timeout)
        runners.append(r)
        return r

    yield _make
    # No cleanup needed — DockerRunner is stateless beyond constructor args.


def _run_watchdog(runner: DockerRunner, cmd: list[str]) -> tuple[int, str]:
    """Drive the watchdog directly against a Popen of ``cmd``.

    Mirrors how ``run()`` invokes ``_run_container_streaming``, but
    skipping the docker-specific argv build. The watchdog logic is
    process-agnostic — it operates on the Popen's stdout pipe.
    """
    return runner._run_container_streaming(cmd)


# ---------------------------------------------------------------------------
# Case 1: stuck-stdout subprocess fires the silence watchdog
# ---------------------------------------------------------------------------


class TestStdoutSilenceFires:
    def test_stuck_subprocess_killed_within_silence_budget(self, runner_factory) -> None:
        runner = runner_factory(timeout=60.0, silence_timeout=2.0)
        start = time.monotonic()
        with pytest.raises(DockerStdoutSilenceError) as ei:
            _run_watchdog(runner, _stuck_stdout_script())
        elapsed = time.monotonic() - start
        # Generous epsilon for the queue-poll cadence (~0.5s) + process
        # cleanup. CI scheduling jitter justifies up to 4x the configured
        # ceiling — tighter assertions flake on shared runners.
        assert elapsed < 8.0, f"Watchdog took {elapsed:.1f}s for a 2s silence budget"
        assert "no stdout" in str(ei.value).lower()

    def test_silence_error_carries_fix_suggestion(self, runner_factory) -> None:
        runner = runner_factory(timeout=60.0, silence_timeout=1.0)
        with pytest.raises(DockerStdoutSilenceError) as ei:
            _run_watchdog(runner, _stuck_stdout_script())
        assert getattr(ei.value, "fix_suggestion", None)
        assert "stdout_silence_timeout_seconds" in ei.value.fix_suggestion


# ---------------------------------------------------------------------------
# Case 2: wall-clock still fires when stdout is active
# ---------------------------------------------------------------------------


class TestWallClockStillFires:
    def test_chatty_subprocess_killed_at_wall_clock(self, runner_factory) -> None:
        # Wall-clock 2s, silence 60s — wall-clock must win against a
        # chatty subprocess that would otherwise reset the silence timer.
        runner = runner_factory(timeout=2.0, silence_timeout=60.0)
        start = time.monotonic()
        with pytest.raises(DockerTimeoutError) as ei:
            _run_watchdog(runner, _chatty_script(duration_s=30.0, interval_s=0.05))
        elapsed = time.monotonic() - start
        assert elapsed < 8.0, f"Watchdog took {elapsed:.1f}s for a 2s wall-clock budget"
        assert "wall-clock" in str(ei.value).lower() or "timed out" in str(ei.value).lower()


# ---------------------------------------------------------------------------
# Case 3: disabled budgets don't interfere with normal completion
# ---------------------------------------------------------------------------


class TestDisabledBudgets:
    def test_quick_subprocess_completes_with_no_budgets(self, runner_factory) -> None:
        runner = runner_factory(timeout=None, silence_timeout=None)
        rc, _stderr = _run_watchdog(runner, _quick_script())
        assert rc == 0

    def test_silence_disabled_via_zero(self, runner_factory) -> None:
        # silence_timeout=0 maps to disabled per DockerRunner.__init__.
        runner = runner_factory(timeout=10.0, silence_timeout=0.0)
        assert runner.silence_timeout is None
        rc, _stderr = _run_watchdog(runner, _quick_script())
        assert rc == 0


# ---------------------------------------------------------------------------
# Constructor normalisation
# ---------------------------------------------------------------------------


class TestSilenceTimeoutNormalisation:
    @pytest.mark.parametrize("raw", [None, 0, 0.0, -1.0])
    def test_disabled_inputs_normalise_to_none(self, raw) -> None:
        runner = DockerRunner(image="x", silence_timeout=raw)
        assert runner.silence_timeout is None

    def test_positive_value_preserved(self) -> None:
        runner = DockerRunner(image="x", silence_timeout=120.0)
        assert runner.silence_timeout == 120.0
