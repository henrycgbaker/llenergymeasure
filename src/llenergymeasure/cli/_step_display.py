"""Rich-based step display for CLI progress output.

Implements the ProgressCallback protocol from domain/progress.py.
Renders Docker-BuildKit-style hierarchical output with phase headers
and numbered sub-steps. Uses fixed step counts with SKIP for
inapplicable steps.

TTY mode:    Rich Live for flicker-free in-place updates + spinner.
Non-TTY:     Phase headers + one line per completed/skipped step +
             heartbeat every 10s for long-running steps.
"""

from __future__ import annotations

import contextlib
import threading
import time

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from llenergymeasure.domain.progress import PHASE_MEASUREMENT, STEP_LABELS, STEP_PHASES
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed
from llenergymeasure.utils.formatting import truncate_detail as _truncate_detail

# Braille spinner frames (same as Docker BuildKit / ora)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Heartbeat interval for non-TTY mode (seconds)
_HEARTBEAT_INTERVAL = 5.0

# Minimum step duration before heartbeat kicks in (seconds)
_HEARTBEAT_THRESHOLD = 3.0


class _DynamicRenderable:
    """Proxy that calls a render function on every Rich auto-refresh.

    Without this, ``Live`` holds a static ``Text`` snapshot and the
    spinner / elapsed counters freeze between callback events.
    """

    def __init__(self, render_fn: object) -> None:
        self._render_fn = render_fn

    def __rich_console__(self, console: Console, options: object) -> object:
        yield self._render_fn()


def _step_line(
    idx: int,
    total: int | None,
    label: str,
    detail: str,
    status: str,
    elapsed_str: str,
) -> str:
    """Format a single step line.

    When total is None, the counter shows just ``[x]`` (no denominator).
    Detail is truncated to 34 chars to prevent line wrapping on 80-col terminals.
    """
    counter = f"[{idx}/{total}]" if total is not None else f"[{idx}]"
    detail = _truncate_detail(detail)
    return f"   {counter:>7s}  {label:<16s} {detail:<34s} {status:>4s}  {elapsed_str}"


def _step_line_prefix(
    idx: int,
    total: int | None,
    label: str,
    detail: str,
) -> str:
    """Format the counter/label/detail portion of a step line (without status/elapsed).

    Used by _render() to build styled output where status and elapsed are
    appended separately with colour styles. Non-TTY mode continues to use
    _step_line() which includes status and elapsed in the same string.
    """
    counter = f"[{idx}/{total}]" if total is not None else f"[{idx}]"
    detail = _truncate_detail(detail)
    return f"   {counter:>7s}  {label:<16s} {detail:<34s}"


def _phase_for_step(step: str) -> str:
    """Look up the phase for a step name, defaulting to Measurement."""
    return STEP_PHASES.get(step, PHASE_MEASUREMENT)


class StepDisplay:
    """Rich-based step display with hierarchical phase grouping.

    Steps are automatically grouped into phases (Setup, Measurement)
    based on the STEP_PHASES mapping. Phase headers are rendered before
    their first sub-step.

    When steps are pre-registered via register_steps(), a fixed [x/y]
    counter is shown. Steps that don't apply are shown as SKIP.

    Thread-safe: harness calls on_step_start/update/done from a worker
    thread while Rich Live refreshes from its own thread.

    Usage::

        display = StepDisplay(header="Experiment: gpt2 | pytorch | bf16")
        display.register_steps(STEPS_DOCKER)
        display.start()
        # ... pass display as ProgressCallback to harness ...
        display.finish()
    """

    def __init__(
        self, header: str = "", console: Console | None = None, force_plain: bool = False
    ) -> None:
        self._console = console or Console(stderr=True)
        self._header = header
        self._lock = threading.Lock()

        # Phase tracking: ordered list of phases seen, steps per phase
        self._phases: list[str] = []
        self._phase_steps: dict[str, list[str]] = {}
        self._explicitly_registered: bool = False

        # Step state: done, skipped, or active
        self._step_data: dict[str, tuple[str, str, float]] = {}  # step -> (label, detail, elapsed)
        self._completed_steps: set[str] = set()
        self._skipped_steps: set[str] = set()

        # Substeps per step: list of (text, elapsed_sec) tuples
        self._substeps: dict[str, list[tuple[str, float]]] = {}

        # Active step
        self._active_step: str | None = None
        self._active_label: str = ""
        self._active_detail: str = ""
        self._active_start: float = 0.0

        # Rich Live (TTY only); force_plain disables Live mode even in a TTY
        self._live: Live | None = None
        self._is_tty = self._console.is_terminal and not force_plain
        self._total_start: float = 0.0

        # Non-TTY: track which phases have been printed
        self._printed_phases: set[str] = set()

        # Heartbeat thread (non-TTY only)
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

    @property
    def total_steps(self) -> int:
        return sum(len(steps) for steps in self._phase_steps.values())

    def _ensure_step_registered(self, step: str) -> str:
        """Register a step into its phase if not already present. Returns phase name.

        Must be called with self._lock held.
        """
        phase = _phase_for_step(step)
        if phase not in self._phases:
            self._phases.append(phase)
        if phase not in self._phase_steps:
            self._phase_steps[phase] = []
        if step not in self._phase_steps[phase]:
            self._phase_steps[phase].append(step)
        return phase

    def register_steps(self, steps: list[str]) -> None:
        """Pre-register steps for fixed [x/y] counting.

        When registered, the counter denominator is fixed from the start.
        Steps not started by the end are shown as SKIP.
        """
        with self._lock:
            self._explicitly_registered = True
            for step in steps:
                self._ensure_step_registered(step)

    def start(self) -> None:
        """Begin the display. Prints header and starts Rich Live if TTY."""
        self._total_start = time.monotonic()
        if self._header:
            self._console.print(self._header, highlight=False)
        if self._is_tty:
            self._live = Live(
                _DynamicRenderable(self._render),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        else:
            # Start heartbeat thread for non-TTY mode
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop Rich Live and heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(
        self,
        total_elapsed: float | None = None,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
    ) -> None:
        """Print completion footer with optional key metrics."""
        self.stop()
        if total_elapsed is None:
            total_elapsed = time.monotonic() - self._total_start
        self._console.print(f"\nCompleted in {_format_elapsed(total_elapsed)}", highlight=False)
        # Key metrics line (Energy + Throughput)
        metrics_parts = []
        if energy_j is not None and energy_j > 0:
            metrics_parts.append(f"Energy: {energy_j:.1f} J")
        if throughput_tok_s is not None and throughput_tok_s > 0:
            metrics_parts.append(f"Throughput: {throughput_tok_s:.1f} tok/s")
        if metrics_parts:
            self._console.print("  ".join(metrics_parts), highlight=False)

    # -- ProgressCallback implementation --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            phase = self._ensure_step_registered(step)
            self._active_step = step
            self._active_label = description or STEP_LABELS.get(step, step)
            self._active_detail = detail
            self._active_start = time.monotonic()

        if not self._is_tty:
            self._print_phase_header_if_new(phase)
            self._print_started_step(step, description or STEP_LABELS.get(step, step), detail)
        self._refresh()

    def on_step_update(self, step: str, detail: str) -> None:
        elapsed = 0.0
        with self._lock:
            if self._active_step == step:
                self._active_detail = detail
                elapsed = time.monotonic() - self._active_start
        if not self._is_tty and elapsed >= 1.0:
            # Only print updates in non-TTY for steps running > 1s
            # (avoids noisy duplicate lines for fast sub-second steps)
            self._print_update_line(step, detail)
        self._refresh()

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        with self._lock:
            label = self._active_label if self._active_step == step else STEP_LABELS.get(step, step)
            detail = self._active_detail if self._active_step == step else ""
            self._step_data[step] = (label, detail, elapsed_sec)
            self._completed_steps.add(step)
            if self._active_step == step:
                self._active_step = None
        if not self._is_tty:
            self._print_completed_step(step, label, detail, elapsed_sec)
        self._refresh()

    def on_step_skip(self, step: str, reason: str = "") -> None:
        with self._lock:
            phase = self._ensure_step_registered(step)
            label = STEP_LABELS.get(step, step)
            # Store reason as detail, keep label as the verb
            self._step_data[step] = (label, reason or "-", 0.0)
            self._skipped_steps.add(step)

        if not self._is_tty:
            self._print_phase_header_if_new(phase)
            self._print_skipped_step(step, label, reason)
        self._refresh()

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Record a completed sub-operation within the active step.

        In TTY mode: stored and rendered as indented · lines below the parent step.
        In non-TTY mode: also printed immediately as they arrive.
        """
        with self._lock:
            if step not in self._substeps:
                self._substeps[step] = []
            self._substeps[step].append((text, elapsed_sec))
        if not self._is_tty:
            self._print_substep_line(step, text, elapsed_sec)
        self._refresh()

    # -- Heartbeat (non-TTY only) --

    def _heartbeat_loop(self) -> None:
        """Print periodic status for long-running steps (non-TTY mode)."""
        while not self._heartbeat_stop.wait(timeout=_HEARTBEAT_INTERVAL):
            with self._lock:
                if self._active_step is not None:
                    elapsed = time.monotonic() - self._active_start
                    if elapsed >= _HEARTBEAT_THRESHOLD:
                        phase = _phase_for_step(self._active_step)
                        idx = self._step_index_in_phase(self._active_step, phase)
                        total = self._phase_total(phase)
                        line = _step_line(
                            idx,
                            total,
                            self._active_label,
                            self._active_detail,
                            " ...",
                            _format_elapsed(elapsed),
                        )
                        self._console.print(line, highlight=False)

    # -- Rendering --

    def _step_index_in_phase(self, step: str, phase: str) -> int:
        """1-based index of step within its phase."""
        steps = self._phase_steps.get(phase, [])
        try:
            return steps.index(step) + 1
        except ValueError:
            return len(steps)

    def _phase_total(self, phase: str) -> int | None:
        """Total step count for a phase.

        Returns None in non-TTY mode when steps are auto-registered,
        to avoid misleading [1/1] -> [2/2] jitter as new steps arrive.
        TTY mode always returns a count (Live re-renders all lines).
        """
        if self._explicitly_registered or self._is_tty:
            return len(self._phase_steps.get(phase, []))
        return None

    def _render(self) -> Text:
        """Build current display state as a Rich Text renderable.

        Docker BuildKit-style: only show steps that have started, completed,
        or been skipped. Pending steps are NOT shown — they appear progressively
        as the harness reaches them. This gives a growing output that shows
        exactly where execution is.

        Colour coding:
        - Phase headers: bold white
        - Completed steps: green ✓ checkmark
        - Active steps: yellow braille spinner
        - Skipped steps: all dim grey
        - Substep lines: dim grey, indented with · prefix
        """
        lines = Text()
        with self._lock:
            for phase in self._phases:
                steps = self._phase_steps.get(phase, [])
                if not steps:
                    continue

                phase_total = len(steps)

                # Only show phase header if at least one step in this phase
                # has been started, completed, or skipped
                has_visible = any(
                    step in self._completed_steps
                    or step in self._skipped_steps
                    or step == self._active_step
                    for step in steps
                )
                if not has_visible:
                    continue

                lines.append(f"\n  {phase}\n", style="bold white")

                for step in steps:
                    idx = self._step_index_in_phase(step, phase)

                    if step in self._completed_steps:
                        label, detail, elapsed = self._step_data[step]
                        prefix = _step_line_prefix(idx, phase_total, label, detail)
                        lines.append(prefix)
                        lines.append("  ✓", style="bold green")
                        lines.append(f"  {_format_elapsed(elapsed)}\n")
                        # Substep lines below completed step
                        for sub_text, sub_elapsed in self._substeps.get(step, []):
                            lines.append(f"              · {sub_text}", style="dim")
                            if sub_elapsed > 0:
                                lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
                            lines.append("\n")
                    elif step in self._skipped_steps:
                        label, reason, _ = self._step_data[step]
                        prefix = _step_line_prefix(idx, phase_total, label, reason)
                        lines.append(prefix, style="dim")
                        lines.append("  SKIP", style="dim")
                        lines.append("\n")
                    elif step == self._active_step:
                        elapsed = time.monotonic() - self._active_start
                        frame_idx = int(elapsed * 8) % len(_SPINNER_FRAMES)
                        spinner = _SPINNER_FRAMES[frame_idx]
                        prefix = _step_line_prefix(
                            idx, phase_total, self._active_label, self._active_detail
                        )
                        lines.append(prefix)
                        lines.append(f"  {spinner}", style="yellow")
                        lines.append(f"  {_format_elapsed(elapsed)}\n")
                        # Substep lines so far for the active step
                        for sub_text, sub_elapsed in self._substeps.get(step, []):
                            lines.append(f"              · {sub_text}", style="dim")
                            if sub_elapsed > 0:
                                lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
                            lines.append("\n")
                    # Pending steps: NOT shown (Docker BuildKit-style progressive output)

        return lines

    def _refresh(self) -> None:
        """Trigger immediate Live repaint (auto-refresh handles animation)."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.refresh()

    def _print_phase_header_if_new(self, phase: str) -> None:
        """Print phase header in non-TTY mode (once per phase)."""
        if phase not in self._printed_phases:
            self._printed_phases.add(phase)
            self._console.print(f"\n  {phase}", highlight=False)

    def _print_update_line(self, step: str, detail: str) -> None:
        """Print a step_update line in non-TTY mode (sub-step detail)."""
        phase = _phase_for_step(step)
        with self._lock:
            idx = self._step_index_in_phase(step, phase)
            total = self._phase_total(phase)
            label = self._active_label if self._active_step == step else STEP_LABELS.get(step, step)
            elapsed = time.monotonic() - self._active_start if self._active_step == step else 0.0
        line = _step_line(idx, total, label, detail, " ...", _format_elapsed(elapsed))
        self._console.print(line, highlight=False)

    def _print_started_step(self, step: str, label: str, detail: str) -> None:
        """Print a step start line (non-TTY mode) for immediate feedback."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, detail, " ...", "")
        self._console.print(line, highlight=False)

    def _print_completed_step(self, step: str, label: str, detail: str, elapsed_sec: float) -> None:
        """Print a single completed step line (non-TTY mode)."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, detail, "DONE", _format_elapsed(elapsed_sec))
        self._console.print(line, highlight=False)

    def _print_skipped_step(self, step: str, label: str, reason: str) -> None:
        """Print a skipped step line (non-TTY mode)."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, reason or "-", "SKIP", "")
        self._console.print(line, highlight=False)

    def _print_substep_line(self, step: str, text: str, elapsed_sec: float) -> None:
        """Print a substep line in non-TTY mode (indented with · prefix)."""
        elapsed_str = f"  {_format_elapsed(elapsed_sec)}" if elapsed_sec > 0 else ""
        self._console.print(f"              · {text}{elapsed_str}", highlight=False)


class StudyStepDisplay:
    """Step display for study mode using a Rich Table for completed experiments.

    Completed experiments appear as table rows with Config, Time, Energy, tok/s columns.
    The active experiment shows nested step progress below the table (deferred for
    multi-process study runs — see module docstring).

    Thread-safe: event methods may be called from worker threads.
    """

    def __init__(
        self,
        total_experiments: int,
        study_name: str = "",
        n_cycles: int = 1,
        console: Console | None = None,
        force_plain: bool = False,
    ) -> None:
        self._console = console or Console(stderr=True)
        self._total = total_experiments
        self._study_name = study_name
        self._n_cycles = n_cycles
        self._is_tty = self._console.is_terminal and not force_plain
        self._lock = threading.Lock()

        # Completed experiment rows: (index, status, config, elapsed, energy_j, throughput)
        self._completed_rows: list[tuple[int, str, str, float, float | None, float | None]] = []

        # Active experiment state
        self._active_index: int = 0
        self._active_header: str = ""
        self._inner_completed: list[tuple[str, str, str, float]] = []
        self._inner_active: tuple[str, str, str, float] | None = None  # step, label, detail, start
        self._inner_steps: list[str] = []
        self._inner_substeps: dict[str, list[tuple[str, float]]] = {}

        self._live: Live | None = None
        self._total_start: float = 0.0

    def start(self) -> None:
        """Begin the display. Prints study header and starts Rich Live if TTY."""
        self._total_start = time.monotonic()
        # Print study header per CONTEXT.md format
        header = f"Study: {self._study_name}" if self._study_name else "Study"
        header += f" | {self._total} experiments | {self._n_cycles} cycles"
        self._console.print(header, highlight=False)
        if self._is_tty:
            self._live = Live(
                _DynamicRenderable(self._render),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()

    def stop(self) -> None:
        """Stop Rich Live."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def begin_experiment(
        self, index: int, model: str, backend: str, precision: str, steps: list[str]
    ) -> None:
        """Start tracking a new experiment within the study."""
        with self._lock:
            self._active_index = index
            self._active_header = f"{model} / {backend} / {precision}"
            self._inner_completed = []
            self._inner_active = None
            self._inner_steps = steps
            self._inner_substeps = {}
        self._refresh()

    def end_experiment_ok(
        self,
        index: int,
        elapsed: float,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
    ) -> None:
        """Mark experiment as successfully completed."""
        with self._lock:
            self._inner_active = None
            self._completed_rows.append(
                (index, "OK", self._active_header, elapsed, energy_j, throughput_tok_s)
            )
        if not self._is_tty:
            self._print_completed_row(
                index, "OK", self._active_header, elapsed, energy_j, throughput_tok_s
            )
        self._refresh()

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Mark experiment as failed."""
        with self._lock:
            self._inner_active = None
            self._completed_rows.append((index, "FAIL", self._active_header, elapsed, None, None))
        if not self._is_tty:
            self._print_completed_row(index, "FAIL", self._active_header, elapsed, None, None)
            if error:
                self._console.print(f"         {error}", highlight=False)
        self._refresh()

    def finish(
        self,
        save_path: str | None = None,
        total_elapsed: float | None = None,
    ) -> None:
        """Print study completion footer with final results table.

        Args:
            save_path: Optional path to saved results directory.
            total_elapsed: Total elapsed time in seconds. If None, falls back to
                monotonic clock delta from start() (which may be wrong if start()
                was never called — callers constructing post-hoc should always pass this).
        """
        self.stop()
        if total_elapsed is not None:
            total = total_elapsed
        elif self._total_start > 0:
            total = time.monotonic() - self._total_start
        else:
            total = 0.0
        self._console.print(f"\nStudy completed in {_format_elapsed(total)}", highlight=False)

        # Print final summary table with all experiments
        if self._completed_rows:
            table = self._build_table()
            self._console.print(table)

        if save_path:
            self._console.print(f"\nSaved: {save_path}", highlight=False)

    # -- ProgressCallback for inner steps --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            label = description or STEP_LABELS.get(step, step)
            self._inner_active = (step, label, detail, time.monotonic())
        self._refresh()

    def on_step_update(self, step: str, detail: str) -> None:
        with self._lock:
            if self._inner_active and self._inner_active[0] == step:
                self._inner_active = (step, self._inner_active[1], detail, self._inner_active[3])
        self._refresh()

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        with self._lock:
            if self._inner_active and self._inner_active[0] == step:
                label = self._inner_active[1]
                detail = self._inner_active[2]
                self._inner_completed.append((step, label, detail, elapsed_sec))
                self._inner_active = None
        self._refresh()

    def on_step_skip(self, step: str, reason: str = "") -> None:
        """No-op for study display (inner steps don't show SKIP)."""

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Record a completed sub-operation within the active step.

        Implemented for protocol compliance and single-process study paths.
        In multi-process study runs, the progress_queue only carries
        started/completed/failed events, so this will not receive events.
        """
        with self._lock:
            if step not in self._inner_substeps:
                self._inner_substeps[step] = []
            self._inner_substeps[step].append((text, elapsed_sec))
        self._refresh()

    # -- Rendering --

    def _build_table(self) -> Table:
        """Build the Rich Table of completed experiments."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", width=3, justify="right")
        table.add_column("Status", width=6)
        table.add_column("Config")
        table.add_column("Time", justify="right")
        table.add_column("Energy", justify="right")
        table.add_column("tok/s", justify="right")
        for idx, status, config, elapsed, energy, throughput in self._completed_rows:
            status_text = (
                Text("\u2713", style="bold green")
                if status == "OK"
                else Text("\u2717", style="bold red")
            )
            energy_str = f"{energy:.1f} J" if energy is not None else "-"
            throughput_str = f"{throughput:.1f}" if throughput is not None else "-"
            table.add_row(
                str(idx),
                status_text,
                config,
                _format_elapsed(elapsed),
                energy_str,
                throughput_str,
            )
        return table

    def _render_active_steps(self) -> Text:
        """Render the active experiment's step progress."""
        lines = Text()
        if not self._active_header:
            return lines

        lines.append(f"\n  [{self._active_index}/{self._total}] {self._active_header}\n")

        inner_total = len(self._inner_steps) or (
            len(self._inner_completed) + (1 if self._inner_active else 0)
        )

        for i, (_step, label, detail, elapsed) in enumerate(self._inner_completed):
            idx = i + 1
            counter = f"[{idx}/{inner_total}]"
            trunc_detail = _truncate_detail(detail)
            lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
            lines.append("  \u2713", style="bold green")
            lines.append(f"  {_format_elapsed(elapsed)}\n")
            for sub_text, sub_elapsed in self._inner_substeps.get(_step, []):
                lines.append(f"                    \u00b7 {sub_text}", style="dim")
                if sub_elapsed > 0:
                    lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
                lines.append("\n")

        if self._inner_active:
            _step, label, detail, start = self._inner_active
            idx = len(self._inner_completed) + 1
            elapsed = time.monotonic() - start
            frame_idx = int(elapsed * 8) % len(_SPINNER_FRAMES)
            spinner = _SPINNER_FRAMES[frame_idx]
            counter = f"[{idx}/{inner_total}]"
            trunc_detail = _truncate_detail(detail)
            lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
            lines.append(f"  {spinner}", style="yellow")
            lines.append(f"  {_format_elapsed(elapsed)}\n")
            for sub_text, sub_elapsed in self._inner_substeps.get(_step, []):
                lines.append(f"                    \u00b7 {sub_text}", style="dim")
                if sub_elapsed > 0:
                    lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
                lines.append("\n")

        return lines

    def _render(self) -> object:
        """Render completed experiments table + active experiment step display."""
        with self._lock:
            table = self._build_table()
            step_text = self._render_active_steps()
        return Group(table, step_text)

    def _refresh(self) -> None:
        """Trigger immediate Live repaint (auto-refresh handles animation)."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.refresh()

    def _print_completed_row(
        self,
        index: int,
        status: str,
        config: str,
        elapsed: float,
        energy: float | None,
        throughput: float | None,
    ) -> None:
        """Print a completed experiment row in non-TTY mode."""
        status_str = "OK" if status == "OK" else "FAIL"
        energy_str = f"  {energy:.1f} J" if energy is not None else ""
        throughput_str = f"  {throughput:.1f} tok/s" if throughput is not None else ""
        line = (
            f" [{index:>2d}/{self._total}]  {status_str:<4s}  {config:<42s}"
            f" {_format_elapsed(elapsed):>8s}{energy_str}{throughput_str}"
        )
        self._console.print(line, highlight=False)
