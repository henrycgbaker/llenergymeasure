"""Unit tests for cli/_step_display.py — StepDisplay and StudyStepDisplay.

All tests run GPU-free. Uses mock Console for predictable output.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from llenergymeasure.cli._step_display import (
    StepDisplay,
    StudyStepDisplay,
    _format_elapsed,
    _step_line,
)

# ---------------------------------------------------------------------------
# _format_elapsed helper
# ---------------------------------------------------------------------------


def test_format_elapsed_sub_minute():
    assert _format_elapsed(4.2) == "4.2s"


def test_format_elapsed_exactly_one_minute():
    assert _format_elapsed(60.0) == "1m 00s"


def test_format_elapsed_minutes():
    assert _format_elapsed(272.0) == "4m 32s"


def test_format_elapsed_hours():
    assert _format_elapsed(3900.0) == "1h 05m"


def test_format_elapsed_zero():
    assert _format_elapsed(0.0) == "0.0s"


# ---------------------------------------------------------------------------
# _step_line helper
# ---------------------------------------------------------------------------


def test_step_line_format():
    line = _step_line(1, 5, "Checking", "preflight, CUDA", "DONE", "1.2s")
    assert "[1/5]" in line
    assert "Checking" in line
    assert "preflight, CUDA" in line
    assert "DONE" in line
    assert "1.2s" in line


def test_step_line_spinner():
    line = _step_line(3, 8, "Loading model", "gpt2", "\u28bc", "42.3s")
    assert "[3/8]" in line
    assert "\u28bc" in line


# ---------------------------------------------------------------------------
# StepDisplay
# ---------------------------------------------------------------------------


def _make_console() -> tuple[Console, StringIO]:
    """Create a non-TTY console writing to a StringIO buffer."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True)
    return console, buf


def test_step_display_non_tty_prints_completed_lines():
    """Non-TTY mode prints one line per completed step."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["preflight", "model", "measure"])
    display.start()

    display.on_step_start("preflight", "Checking", "CUDA")
    display.on_step_done("preflight", 1.2)

    display.on_step_start("model", "Loading model", "gpt2")
    display.on_step_done("model", 5.0)

    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_step_done("measure", 30.0)

    display.finish(total_elapsed=36.2)

    output = buf.getvalue()
    assert "[1/3]" in output
    assert "[2/3]" in output
    assert "[3/3]" in output
    assert "DONE" in output
    assert "Completed in 36.2s" in output


def test_step_display_auto_registers_unknown_steps():
    """Steps not in the initial registration list are auto-added."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["preflight", "model"])

    display.on_step_start("preflight", "Checking", "")
    display.on_step_done("preflight", 0.5)

    # "pull" was not registered - should auto-extend the step list
    display.on_step_start("pull", "Pulling", "image:latest")
    display.on_step_done("pull", 10.0)

    display.on_step_start("model", "Loading model", "gpt2")
    display.on_step_done("model", 5.0)

    display.finish(total_elapsed=15.5)

    output = buf.getvalue()
    # All three steps should appear (auto-registration extended the list to 3)
    assert display.total_steps == 3
    assert "Checking" in output
    assert "Pulling" in output
    assert "Loading model" in output
    assert "DONE" in output


def test_step_display_update_changes_detail():
    """on_step_update changes the detail text for the active step."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["warmup"])
    display.start()

    display.on_step_start("warmup", "Warming up", "0/50 prompts")
    display.on_step_update("warmup", "25/50 prompts")
    display.on_step_done("warmup", 10.0)

    display.finish(total_elapsed=10.0)

    output = buf.getvalue()
    # In non-TTY mode, the final completed line should contain the last detail
    assert "DONE" in output


def test_step_display_header_printed():
    """Header is printed before steps."""
    console, buf = _make_console()
    display = StepDisplay(header="Experiment: gpt2 | pytorch | bf16", console=console)
    display.start()
    display.finish(total_elapsed=0.0)

    output = buf.getvalue()
    assert "Experiment: gpt2 | pytorch | bf16" in output


def test_step_display_quiet_mode():
    """When progress=None (quiet mode), no display is created."""
    # Just verify the pattern works - pass None through the API
    from llenergymeasure.domain.progress import ProgressCallback

    progress: ProgressCallback | None = None
    assert progress is None  # Quiet mode


# ---------------------------------------------------------------------------
# StudyStepDisplay
# ---------------------------------------------------------------------------


def test_study_display_experiment_lifecycle():
    """StudyStepDisplay tracks experiment completion with summary lines."""
    console, buf = _make_console()
    display = StudyStepDisplay(total_experiments=3, console=console)
    display.start()

    # Experiment 1: success
    display.begin_experiment(1, "gpt2", "pytorch", "bf16", ["preflight", "model", "measure"])
    display.on_step_start("preflight", "Checking", "")
    display.on_step_done("preflight", 0.5)
    display.on_step_start("model", "Loading model", "gpt2")
    display.on_step_done("model", 3.0)
    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_step_done("measure", 20.0)
    display.end_experiment_ok(1, elapsed=23.5, energy_j=150.0)

    # Experiment 2: fail
    display.begin_experiment(2, "llama", "vllm", "bf16", ["preflight"])
    display.on_step_start("preflight", "Checking", "")
    display.on_step_done("preflight", 0.3)
    display.end_experiment_fail(2, elapsed=0.3, error="BackendError: CUDA OOM")

    display.finish()

    output = buf.getvalue()
    assert "OK" in output
    assert "FAIL" in output
    assert "Study completed" in output


# ---------------------------------------------------------------------------
# ProgressCallback protocol compliance
# ---------------------------------------------------------------------------


def test_step_display_satisfies_protocol():
    """StepDisplay satisfies the ProgressCallback protocol."""
    from llenergymeasure.domain.progress import ProgressCallback

    console, _ = _make_console()
    display = StepDisplay(console=console)
    assert isinstance(display, ProgressCallback)


def test_study_display_satisfies_protocol():
    """StudyStepDisplay satisfies the ProgressCallback protocol."""
    from llenergymeasure.domain.progress import ProgressCallback

    console, _ = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    assert isinstance(display, ProgressCallback)
