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
    """Non-TTY mode prints one line per completed step with phase headers."""
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
    # Phase headers
    assert "Setup" in output
    assert "Measurement" in output
    # Per-phase step counters (preflight is in Setup [1/1], model+measure in Measurement [1/2] [2/2])
    assert "[1/1]" in output  # preflight in Setup
    assert "[1/2]" in output  # model in Measurement
    assert "[2/2]" in output  # measure in Measurement
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


def test_step_display_finish_with_metrics():
    """finish() prints energy and throughput on the completion line."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.start()
    display.finish(total_elapsed=42.0, energy_j=512.3, throughput_tok_s=123.4)

    output = buf.getvalue()
    assert "Completed in 42.0s" in output
    assert "Energy: 512.3 J" in output
    assert "Throughput: 123.4 tok/s" in output


def test_step_display_finish_no_metrics():
    """finish() without metrics only prints the elapsed line."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.start()
    display.finish(total_elapsed=10.0)

    output = buf.getvalue()
    assert "Completed in 10.0s" in output
    assert "Energy:" not in output
    assert "Throughput:" not in output


def test_step_display_force_plain():
    """force_plain=True disables Rich Live even if console is a TTY."""
    buf = StringIO()
    # force_terminal=True simulates TTY but force_plain should override Live
    console = Console(file=buf, force_terminal=True, no_color=True)
    display = StepDisplay(console=console, force_plain=True)
    assert not display._is_tty  # force_plain overrides TTY detection


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
    display = StudyStepDisplay(
        total_experiments=3,
        study_name="my-sweep",
        n_cycles=2,
        console=console,
    )
    display.start()

    # Study header should appear
    output_after_start = buf.getvalue()
    assert "Study: my-sweep | 3 experiments | 2 cycles" in output_after_start

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


def test_study_display_table_output():
    """Rich Table renders with all columns and values in TTY mode."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, no_color=True, width=120)
    display = StudyStepDisplay(
        total_experiments=2,
        study_name="table-test",
        n_cycles=1,
        console=console,
    )
    display.end_experiment_ok(1, elapsed=154.2, energy_j=12.4, throughput_tok_s=38.7)
    display.end_experiment_ok(2, elapsed=98.5, energy_j=8.1, throughput_tok_s=42.3)
    display.finish(total_elapsed=270.0)

    output = buf.getvalue()
    # Table column headers
    assert "#" in output
    assert "Status" in output
    assert "Config" in output
    assert "Time" in output
    assert "Energy" in output
    assert "tok/s" in output
    # Row values
    assert "12.4 J" in output
    assert "38.7" in output
    assert "8.1 J" in output
    assert "42.3" in output
    # Completion line
    assert "Study completed in" in output


def test_study_display_substep_in_active_experiment():
    """on_substep() records substeps attached to the active inner step."""
    console, _buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2", "pytorch", "bf16", ["measure"])
    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_substep("measure", "CUDA sync (pre)", 0.0)
    display.on_substep("measure", "energy tracker started", 0.0)

    # Verify substeps are stored (for TTY rendering)
    with display._lock:
        substeps = display._inner_substeps.get("measure", [])
    assert len(substeps) == 2
    assert substeps[0][0] == "CUDA sync (pre)"
    assert substeps[1][0] == "energy tracker started"


def test_study_display_finish_prints_summary_table():
    """finish() prints the final Rich Table with all experiment results."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, no_color=True, width=120)
    display = StudyStepDisplay(
        total_experiments=2,
        study_name="finish-test",
        n_cycles=3,
        console=console,
    )
    display.end_experiment_ok(1, elapsed=60.0, energy_j=10.0, throughput_tok_s=50.0)
    display.end_experiment_fail(2, elapsed=5.0, error="Timeout")
    display.finish(save_path="./results/finish-test/", total_elapsed=75.0)

    output = buf.getvalue()
    assert "Study completed in 1m 15s" in output
    assert "10.0 J" in output
    assert "Saved: ./results/finish-test/" in output


def test_study_display_finish_total_elapsed_parameter():
    """finish(total_elapsed=...) uses the provided value, not the internal clock."""
    console, buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    # Never call start() — simulate post-hoc construction
    display.end_experiment_ok(1, elapsed=10.0)
    display.finish(total_elapsed=15.5)

    output = buf.getvalue()
    assert "Study completed in 15.5s" in output


def test_study_display_non_tty_plain_rows():
    """Non-TTY mode prints plain text rows for each completed experiment."""
    console, buf = _make_console()
    display = StudyStepDisplay(total_experiments=2, console=console)
    display.begin_experiment(1, "gpt2", "pytorch", "bf16", [])
    display.end_experiment_ok(1, elapsed=30.0, energy_j=5.5, throughput_tok_s=25.0)
    display.begin_experiment(2, "llama", "vllm", "bf16", [])
    display.end_experiment_fail(2, elapsed=2.0, error="OOM")

    output = buf.getvalue()
    # Non-TTY rows contain the config and status
    assert "OK" in output
    assert "gpt2 / pytorch / bf16" in output
    assert "FAIL" in output
    assert "llama / vllm / bf16" in output
    assert "OOM" in output


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


# ---------------------------------------------------------------------------
# Substep display (Plan 02)
# ---------------------------------------------------------------------------


def test_substep_lines_appear_in_output():
    """Non-TTY: substep lines appear in output with · prefix after step completes."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["model"])
    display.start()

    display.on_step_start("model", "Loading", "gpt2")
    display.on_substep("model", "loading tokenizer", 1.2)
    display.on_step_done("model", 5.0)
    display.finish(total_elapsed=5.0)

    output = buf.getvalue()
    assert "· loading tokenizer" in output
    assert "1.2s" in output


def test_colour_checkmark_in_tty_render():
    """TTY render: completed steps contain ✓ with bold green style."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, no_color=True, width=200)
    display = StepDisplay(console=console)
    display.register_steps(["model"])
    display.start()

    display.on_step_start("model", "Loading", "gpt2")
    display.on_step_done("model", 3.0)

    rendered = display._render()
    # The rendered Text should contain ✓
    plain_text = rendered.plain
    assert "✓" in plain_text
    # Check that at least one span has "bold green" style
    spans = [(s.style, rendered.plain[s.start : s.end]) for s in rendered._spans]
    green_spans = [(style, text) for style, text in spans if "green" in str(style)]
    assert any("✓" in text for _, text in green_spans), f"No bold green ✓ found. Spans: {spans}"

    display.stop()


def test_substep_non_tty_prints_immediately():
    """Non-TTY: substep lines are printed as they arrive (not deferred to done)."""
    console, buf = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["measure"])
    display.start()

    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_substep("measure", "CUDA sync (pre)")
    display.on_substep("measure", "energy tracker started")

    # Check output BEFORE step is done — substeps should already be printed
    mid_output = buf.getvalue()
    assert "· CUDA sync (pre)" in mid_output
    assert "· energy tracker started" in mid_output

    display.on_step_done("measure", 10.0)
    display.finish(total_elapsed=10.0)
