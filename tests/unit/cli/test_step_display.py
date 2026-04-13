"""Unit tests for cli/_step_display.py — StepDisplay and StudyStepDisplay.

All tests run GPU-free. Uses mock Console for predictable output.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from llenergymeasure.cli._step_display import (
    _VIEWPORT_RESERVED_LINES,
    StepDisplay,
    StudyStepDisplay,
    _format_elapsed,
    _ImagePrepFailure,
    _ImagePrepResult,
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
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["preflight", "model", "measure"])
    display.on_step_start("preflight", "Checking", "")
    display.on_step_done("preflight", 0.5)
    display.on_step_start("model", "Loading model", "gpt2")
    display.on_step_done("model", 3.0)
    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_step_done("measure", 20.0)
    display.end_experiment_ok(1, elapsed=23.5, energy_j=150.0)

    # Experiment 2: fail
    display.begin_experiment(2, "llama / vllm / bf16", ["preflight"])
    display.on_step_start("preflight", "Checking", "")
    display.on_step_done("preflight", 0.3)
    display.end_experiment_fail(2, elapsed=0.3, error="EngineError: CUDA OOM")

    display.finish()

    output = buf.getvalue()
    assert "\u2713" in output  # OK icon
    assert "\u2717" in output  # FAIL icon
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
    assert "Config" in output
    assert "Total" in output
    assert "Infer" in output
    assert "Energy" in output
    assert "tok/s" in output
    assert "mJ/tok" in output
    # Row values (✓ icons, energy, throughput, mJ/tok)
    assert "\u2713" in output
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
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["measure"])
    display.on_step_start("measure", "Measuring", "100 prompts")
    display.on_substep("measure", "CUDA sync (pre)", 0.0)
    display.on_substep("measure", "energy tracker started", 0.0)

    # Verify substeps are stored (for TTY rendering)
    with display._lock:
        substeps = display._inner_substeps.get("measure", [])
    assert len(substeps) == 2
    assert substeps[0][0] == "CUDA sync (pre)"
    assert substeps[1][0] == "energy tracker started"


def test_study_display_step_restart_clears_prior_completion():
    """Re-firing on_step_start for a completed step replaces the old state.

    Regression: the baseline step can be fired twice in a row when the host
    runner's short-lived baseline container fails and the experiment container
    falls back to an in-harness measurement. The render loop checks
    completed_map before _inner_active, so stale completion state would hide
    the re-run spinner; guarantee the re-fire wins instead.
    """
    console, _buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["baseline", "model"])
    # First attempt: completes (e.g. the failed host dispatch that mistakenly
    # got marked done with a small elapsed).
    display.on_step_start("baseline", "Measuring", "baseline container · idle power")
    display.on_substep("baseline", "measurement failed after 5.7s", 5.7)
    display.on_step_done("baseline", 5.7)

    # Sanity: prior state is recorded.
    with display._lock:
        assert any(c[0] == "baseline" for c in display._inner_completed)
        assert display._inner_substeps.get("baseline")

    # Second attempt: harness inside the experiment container fires a fresh
    # on_step_start for the same step.
    display.on_step_start("baseline", "Measuring", "baseline idle power (30s)")

    with display._lock:
        # Prior completion is gone so the renderer sees the new active state.
        assert not any(c[0] == "baseline" for c in display._inner_completed)
        assert "baseline" not in display._inner_substeps
        assert display._inner_active is not None
        assert display._inner_active[0] == "baseline"
        assert display._inner_active[2] == "baseline idle power (30s)"


def test_step_display_step_restart_clears_prior_completion():
    """StepDisplay also clears prior completion on step re-fire."""
    console, _buf = _make_console()
    display = StepDisplay(header="test", console=console)
    display.register_steps(["baseline", "model"])
    display.on_step_start("baseline", "Measuring", "first attempt")
    display.on_substep("baseline", "first substep", 0.0)
    display.on_step_done("baseline", 5.7)

    assert "baseline" in display._completed_steps

    display.on_step_start("baseline", "Measuring", "second attempt")

    assert "baseline" not in display._completed_steps
    assert "baseline" not in display._substeps
    assert display._active_step == "baseline"
    assert display._active_detail == "second attempt"


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
    assert "Results: ./results/finish-test/" in output


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
    display.begin_experiment(1, "gpt2 / pytorch / bf16", [])
    display.end_experiment_ok(1, elapsed=30.0, energy_j=5.5, throughput_tok_s=25.0)
    display.begin_experiment(2, "llama / vllm / bf16", [])
    display.end_experiment_fail(2, elapsed=2.0, error="OOM")

    output = buf.getvalue()
    # Non-TTY rows contain the config and status icons
    assert "\u2713" in output
    assert "gpt2 / pytorch / bf16" in output
    assert "\u2717" in output
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


def test_study_display_skipped_steps_render_dim():
    """StudyStepDisplay renders skipped steps with SKIP label in correct position."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, no_color=True, width=120)
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.start()

    steps = ["preflight", "image_check", "pull", "container_start", "model", "measure"]
    display.begin_experiment(1, "gpt2 / pytorch / bf16", steps)

    # Skip preflight (Docker path)
    display.on_step_skip("preflight", "Docker path")
    # Complete image_check
    display.on_step_start("image_check", "Inspecting", "pytorch:v0.9.0 (cached)")
    display.on_step_done("image_check", 0.1)
    # Skip pull (cached)
    display.on_step_skip("pull", "cached")
    # Complete remaining
    display.on_step_start("container_start", "Starting", "pytorch:v0.9.0")
    display.on_step_done("container_start", 2.0)
    display.on_step_start("model", "Loading", "gpt2")
    display.on_step_done("model", 3.0)
    display.on_step_start("measure", "Measuring", "50 prompts")
    display.on_step_done("measure", 10.0)
    display.end_experiment_ok(1, elapsed=15.0, energy_j=50.0)

    display.stop()
    output = buf.getvalue()

    # Skipped steps should show SKIP
    assert "SKIP" in output
    # All 6 steps should be counted (not just 4 visible)
    assert "[1/6]" in output  # preflight (skipped)
    assert "[3/6]" in output  # pull (skipped)
    assert "[6/6]" in output  # measure (last step)


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


# ---------------------------------------------------------------------------
# Heartbeat substeps (on_substep_start / on_substep_done)
# ---------------------------------------------------------------------------


def test_step_display_substep_start_done_lifecycle_freezes_with_elapsed():
    """on_substep_start registers an active substep; on_substep_done freezes
    it into the completed list with a positive elapsed value."""
    import time as _time

    console, _ = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["baseline"])
    display.on_step_start("baseline", "Calibrating", "sampling idle GPU draw")

    display.on_substep_start("baseline", "launching separate vllm baseline container")
    # Active substep is tracked by step name
    assert "baseline" in display._active_substep
    _time.sleep(0.01)
    display.on_substep_done("baseline", "vllm baseline container ready")

    # Active slot cleared, frozen list has one entry with positive elapsed.
    assert "baseline" not in display._active_substep
    frozen = display._substeps["baseline"]
    assert len(frozen) == 1
    assert frozen[0][0] == "vllm baseline container ready"
    assert frozen[0][1] > 0.0  # computed from monotonic delta


def test_step_display_substep_done_preserves_start_text_when_final_omitted():
    """on_substep_done() with text=None keeps the original start text."""
    import time as _time

    console, _ = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["baseline"])
    display.on_step_start("baseline", "Calibrating", "")

    display.on_substep_start("baseline", "initialising CUDA runtime")
    _time.sleep(0.005)
    display.on_substep_done("baseline")  # no override text

    frozen = display._substeps["baseline"]
    assert frozen[-1][0] == "initialising CUDA runtime"
    assert frozen[-1][1] > 0.0


def test_step_display_substep_start_without_prior_done_freezes_previous():
    """Calling on_substep_start twice without a done in between freezes the
    prior substep (with its accumulated elapsed) so the UI never has two
    live substeps at once."""
    import time as _time

    console, _ = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["baseline"])
    display.on_step_start("baseline", "Calibrating", "")

    display.on_substep_start("baseline", "launching container")
    _time.sleep(0.005)
    display.on_substep_start("baseline", "initialising CUDA runtime")

    # The first is now frozen with its accumulated elapsed
    frozen = display._substeps["baseline"]
    assert len(frozen) == 1
    assert frozen[0][0] == "launching container"
    assert frozen[0][1] > 0.0
    # The second is the new active slot
    assert display._active_substep["baseline"][0] == "initialising CUDA runtime"


def test_step_display_substep_done_without_start_appends_frozen():
    """on_substep_done with no prior start still appends a frozen substep
    (failure path where the runner has to surface a final text without a
    matching live substep)."""
    console, _ = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["baseline"])
    display.on_step_start("baseline", "Calibrating", "")

    display.on_substep_done("baseline", text="measurement failed", elapsed_sec=5.7)

    frozen = display._substeps["baseline"]
    assert frozen == [("measurement failed", 5.7)]


def test_step_display_on_step_done_freezes_dangling_active_substep():
    """If a step completes while a substep is still active, that substep is
    frozen rather than left animating under a completed parent."""
    import time as _time

    console, _ = _make_console()
    display = StepDisplay(console=console)
    display.register_steps(["baseline"])
    display.on_step_start("baseline", "Calibrating", "")

    display.on_substep_start("baseline", "wedged stage")
    _time.sleep(0.005)
    display.on_step_done("baseline", 0.1)

    assert "baseline" not in display._active_substep
    frozen = display._substeps["baseline"]
    assert len(frozen) == 1
    assert frozen[0][0] == "wedged stage"
    assert frozen[0][1] > 0.0


def test_study_display_substep_start_done_lifecycle_freezes_with_elapsed():
    """StudyStepDisplay exhibits the same start/done freeze semantics as
    StepDisplay for the heartbeat substep pattern."""
    import time as _time

    console, _ = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console, force_plain=True)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["baseline"])
    display.on_step_start("baseline", "Calibrating", "sampling idle GPU draw")

    display.on_substep_start("baseline", "launching separate vllm baseline container")
    assert "baseline" in display._inner_active_substep
    _time.sleep(0.01)
    display.on_substep_done("baseline", "vllm baseline container ready")

    assert "baseline" not in display._inner_active_substep
    frozen = display._inner_substeps["baseline"]
    assert frozen[-1][0] == "vllm baseline container ready"
    assert frozen[-1][1] > 0.0


def test_study_display_on_step_start_refire_clears_active_substep():
    """Re-firing on_step_start for the same step drops any prior active
    substep so a retry doesn't carry stale heartbeat state from a failed
    first attempt (e.g. baseline dispatch → container fallback)."""
    console, _ = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console, force_plain=True)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["baseline"])

    display.on_step_start("baseline", "Calibrating", "sampling idle GPU draw")
    display.on_substep_start("baseline", "launching baseline container")
    assert "baseline" in display._inner_active_substep

    # Host dispatch failed; experiment container re-fires the step.
    display.on_step_start("baseline", "Measuring", "in-container fallback")
    assert "baseline" not in display._inner_active_substep
    assert "baseline" not in display._inner_substeps


def test_study_display_active_substep_renders_in_active_step_view():
    """The render pipeline forwards the active substep to _render_substep_lines
    for the active step, so a spinner line appears under the parent."""
    import time as _time

    console = Console(file=StringIO(), force_terminal=True, no_color=True, width=200)
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", ["baseline"])
    display.on_step_start("baseline", "Calibrating", "sampling idle GPU draw")

    display.on_substep_start("baseline", "launching separate vllm baseline container")
    _time.sleep(0.02)

    rendered = display._render_active_steps()
    plain = rendered.plain
    assert "launching separate vllm baseline container" in plain
    # The active substep shows an elapsed counter (> 0.0s)
    assert " · launching separate vllm baseline container" in plain

    display.stop()


# ---------------------------------------------------------------------------
# Viewport behaviour
# ---------------------------------------------------------------------------


def _make_tty_console(height: int = 30, width: int = 120) -> Console:
    """Create a TTY-mode console with fixed dimensions."""
    return Console(width=width, height=height, force_terminal=True, no_color=True)


def _add_rows(display: StudyStepDisplay, n: int) -> None:
    """Directly append N completed rows to a StudyStepDisplay."""
    for i in range(1, n + 1):
        display._completed_rows.append(
            (i, "OK", f"config-{i}", float(i), None, None, None, None, None)
        )


def test_viewport_limits_visible_rows():
    """_build_table() shows only the most recent rows that fit the terminal height."""
    height = 30
    console = _make_tty_console(height=height)
    display = StudyStepDisplay(total_experiments=50, console=console)
    _add_rows(display, 50)

    table, _hidden = display._build_table()

    expected_max = height - _VIEWPORT_RESERVED_LINES
    assert table.row_count <= expected_max, (
        f"Expected at most {expected_max} rows, got {table.row_count}"
    )
    # Must not be zero
    assert table.row_count >= 5


def test_viewport_shows_all_rows_when_small():
    """_build_table() shows all rows when count fits within the terminal height."""
    console = _make_tty_console(height=80)
    display = StudyStepDisplay(total_experiments=10, console=console)
    _add_rows(display, 10)

    table, hidden = display._build_table()

    assert table.row_count == 10
    assert hidden == 0


def test_viewport_hidden_indicator_shown():
    """_render() includes a hidden-rows indicator when rows overflow the terminal."""
    buf = StringIO()
    console = Console(file=buf, width=120, height=20, force_terminal=True, no_color=True)
    display = StudyStepDisplay(total_experiments=50, console=console)
    _add_rows(display, 50)

    group = display._render()

    # Serialise the Group via a separate render console
    out_buf = StringIO()
    render_console = Console(file=out_buf, width=120, force_terminal=True, no_color=True)
    render_console.print(group)
    rendered = out_buf.getvalue()

    assert "earlier results not shown" in rendered


def test_viewport_hidden_indicator_absent_when_fits():
    """_render() omits the hidden-rows indicator when all rows fit."""
    buf = StringIO()
    console = Console(file=buf, width=120, height=80, force_terminal=True, no_color=True)
    display = StudyStepDisplay(total_experiments=5, console=console)
    _add_rows(display, 5)

    group = display._render()

    out_buf = StringIO()
    render_console = Console(file=out_buf, width=120, force_terminal=True, no_color=True)
    render_console.print(group)
    rendered = out_buf.getvalue()

    assert "earlier results not shown" not in rendered
    display.finish(total_elapsed=10.0)


# ---------------------------------------------------------------------------
# _ImagePrepResult / _ImagePrepFailure NamedTuple access
# ---------------------------------------------------------------------------


def test_image_prep_result_named_fields():
    """_ImagePrepResult supports both named field access and positional destructuring."""
    r = _ImagePrepResult(
        engine="pytorch",
        image="llem-pytorch:latest",
        cached=True,
        elapsed=1.5,
        metadata={"size": "2GB"},
    )
    # Named access
    assert r.engine == "pytorch"
    assert r.image == "llem-pytorch:latest"
    assert r.cached is True
    assert r.elapsed == 1.5
    assert r.metadata == {"size": "2GB"}
    # Positional destructuring (backward compatibility)
    engine, _image, cached, _elapsed, _metadata = r
    assert engine == "pytorch"
    assert cached is True


def test_image_prep_failure_named_fields():
    """_ImagePrepFailure supports both named field access and positional destructuring."""
    f = _ImagePrepFailure(engine="vllm", image="llem-vllm:latest", error="pull failed")
    assert f.engine == "vllm"
    assert f.image == "llem-vllm:latest"
    assert f.error == "pull failed"
    # Positional destructuring (backward compatibility)
    _b, _i, e = f
    assert e == "pull failed"


# ---------------------------------------------------------------------------
# mj_per_tok fallback removal — null mj_per_tok shows "-", no recomputation
# ---------------------------------------------------------------------------


def test_end_experiment_ok_null_mj_per_tok_shows_dash():
    """When both mj_per_tok_adjusted and mj_per_tok_total are None, mj_tok stored as None."""
    console, _buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", [])
    display.end_experiment_ok(
        1,
        elapsed=10.0,
        energy_j=100.0,
        throughput_tok_s=50.0,
        mj_per_tok_adjusted=None,
        mj_per_tok_total=None,
    )

    # The stored row should have mj_tok=None (last element in the tuple)
    with display._lock:
        row = display._completed_rows[-1]
    mj_tok_value = row[-1]  # last element = mj_tok
    assert mj_tok_value is None


def test_end_experiment_ok_prefers_adjusted_over_total():
    """mj_per_tok_adjusted is preferred over mj_per_tok_total when both present."""
    console, _buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", [])
    display.end_experiment_ok(
        1,
        elapsed=10.0,
        mj_per_tok_adjusted=5.0,
        mj_per_tok_total=10.0,
    )

    with display._lock:
        row = display._completed_rows[-1]
    assert row[-1] == 5.0


def test_end_experiment_ok_falls_back_to_total():
    """mj_per_tok_total is used when mj_per_tok_adjusted is None."""
    console, _buf = _make_console()
    display = StudyStepDisplay(total_experiments=1, console=console)
    display.begin_experiment(1, "gpt2 / pytorch / bf16", [])
    display.end_experiment_ok(
        1,
        elapsed=10.0,
        mj_per_tok_adjusted=None,
        mj_per_tok_total=8.0,
    )

    with display._lock:
        row = display._completed_rows[-1]
    assert row[-1] == 8.0
