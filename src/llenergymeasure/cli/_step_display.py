"""Rich-based step display for CLI progress output.

Implements the ProgressCallback protocol from domain/progress.py.
Renders Docker-build-style numbered step output with per-step timing
and a spinner heartbeat for long operations.

TTY mode:    Rich Live for flicker-free in-place updates.
Non-TTY:     One line per completed step, no spinner or animation.
"""

from __future__ import annotations

import contextlib
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

from llenergymeasure.domain.progress import STEP_LABELS

# Braille spinner frames (same as Docker BuildKit / ora)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _format_elapsed(seconds: float) -> str:
    """Format seconds as human-readable elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


def _step_line(
    idx: int,
    total: int,
    label: str,
    detail: str,
    status: str,
    elapsed_str: str,
) -> str:
    """Format a single step line."""
    counter = f"[{idx}/{total}]"
    return f" {counter:>7s}  {label:<18s} {detail:<38s} {status:>4s}  {elapsed_str}"


class StepDisplay:
    """Rich-based step display for single experiment progress.

    Thread-safe: harness calls on_step_start/update/done from a worker
    thread while Rich Live refreshes from its own thread.

    Usage::

        display = StepDisplay(header="Experiment: gpt2 | pytorch | bf16")
        display.register_steps(["preflight", "model", "warmup", "measure", "save"])
        display.start()
        # ... pass display as ProgressCallback to harness ...
        display.finish()
    """

    def __init__(self, header: str = "", console: Console | None = None) -> None:
        self._console = console or Console(stderr=True)
        self._header = header
        self._lock = threading.Lock()

        # Step registration
        self._steps: list[str] = []

        # State
        self._completed: list[tuple[str, str, str, float]] = []  # (step, label, detail, elapsed)
        self._active_step: str | None = None
        self._active_label: str = ""
        self._active_detail: str = ""
        self._active_start: float = 0.0

        # Rich Live (TTY only)
        self._live: Live | None = None
        self._is_tty = self._console.is_terminal
        self._total_start: float = 0.0

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    def register_steps(self, steps: list[str]) -> None:
        """Set the ordered list of step names for [x/y] counting."""
        with self._lock:
            self._steps = list(steps)

    def start(self) -> None:
        """Begin the display. Prints header and starts Rich Live if TTY."""
        self._total_start = time.monotonic()
        if self._header:
            self._console.print(self._header, highlight=False)
            self._console.print()
        if self._is_tty:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()

    def stop(self) -> None:
        """Stop Rich Live if running."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(self, total_elapsed: float | None = None) -> None:
        """Print completion footer and stop the live display."""
        self.stop()
        if total_elapsed is None:
            total_elapsed = time.monotonic() - self._total_start
        self._console.print(f"\nCompleted in {_format_elapsed(total_elapsed)}", highlight=False)

    # -- ProgressCallback implementation --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            # Auto-register unknown steps so the counter stays accurate
            if step not in self._steps:
                self._steps.append(step)
            self._active_step = step
            self._active_label = description or STEP_LABELS.get(step, step)
            self._active_detail = detail
            self._active_start = time.monotonic()
        self._refresh()

    def on_step_update(self, step: str, detail: str) -> None:
        with self._lock:
            if self._active_step == step:
                self._active_detail = detail
        self._refresh()

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        with self._lock:
            self._completed.append((step, self._active_label, self._active_detail, elapsed_sec))
            if self._active_step == step:
                self._active_step = None
        if not self._is_tty:
            # Non-TTY: print completed line immediately
            self._print_completed_line(step, elapsed_sec)
        self._refresh()

    # -- Rendering --

    def _step_index(self, step: str) -> int:
        """1-based index of step in registered list, or append position."""
        try:
            return self._steps.index(step) + 1
        except ValueError:
            return len(self._completed) + (1 if self._active_step == step else 0)

    def _render(self) -> Text:
        """Build current display state as a Rich Text renderable."""
        lines = Text()
        with self._lock:
            total = self.total_steps or (len(self._completed) + (1 if self._active_step else 0))
            for step, label, detail, elapsed in self._completed:
                idx = self._step_index(step)
                line = _step_line(idx, total, label, detail, "DONE", _format_elapsed(elapsed))
                lines.append(line + "\n")

            if self._active_step is not None:
                idx = self._step_index(self._active_step)
                elapsed = time.monotonic() - self._active_start
                frame_idx = int(elapsed * 8) % len(_SPINNER_FRAMES)
                spinner = _SPINNER_FRAMES[frame_idx]
                line = _step_line(
                    idx,
                    total,
                    self._active_label,
                    self._active_detail,
                    spinner,
                    _format_elapsed(elapsed),
                )
                lines.append(line)
        return lines

    def _refresh(self) -> None:
        """Push updated renderable to Live display."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.update(self._render())

    def _print_completed_line(self, step: str, elapsed_sec: float) -> None:
        """Print a single completed step line (non-TTY mode)."""
        with self._lock:
            total = self.total_steps or len(self._completed)
            idx = self._step_index(step)
            label = STEP_LABELS.get(step, step)
            detail_text = ""
            for s, _lbl, det, _el in self._completed:
                if s == step:
                    detail_text = det
                    break
        line = _step_line(idx, total, label, detail_text, "DONE", _format_elapsed(elapsed_sec))
        self._console.print(line, highlight=False)


class StudyStepDisplay:
    """Step display for study mode with outer experiment counter and nested inner steps.

    Shows [x/y] outer counter per experiment. The active experiment shows
    nested inner step progress. Completed experiments collapse to a summary line.
    """

    def __init__(self, total_experiments: int, console: Console | None = None) -> None:
        self._console = console or Console(stderr=True)
        self._total = total_experiments
        self._is_tty = self._console.is_terminal
        self._lock = threading.Lock()

        # Completed experiment summaries
        self._completed_lines: list[str] = []

        # Active inner display
        self._active_index: int = 0
        self._active_header: str = ""
        self._inner_completed: list[tuple[str, str, str, float]] = []
        self._inner_active: tuple[str, str, str, float] | None = None  # step, label, detail, start
        self._inner_steps: list[str] = []

        self._live: Live | None = None
        self._total_start: float = 0.0

    def start(self) -> None:
        self._total_start = time.monotonic()
        if self._is_tty:
            self._live = Live(
                Text(""),
                console=self._console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()

    def stop(self) -> None:
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
        self._refresh()

    def end_experiment_ok(self, index: int, elapsed: float, energy_j: float | None = None) -> None:
        """Mark experiment as successfully completed."""
        with self._lock:
            self._inner_active = None
            energy_str = f"  {energy_j:.0f} J" if energy_j is not None else ""
            line = (
                f" [{index:>2d}/{self._total}]  OK    {self._active_header:<42s}"
                f" {_format_elapsed(elapsed):>8s}{energy_str}"
            )
            self._completed_lines.append(line)
        if not self._is_tty:
            self._console.print(line, highlight=False)
        self._refresh()

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Mark experiment as failed."""
        with self._lock:
            self._inner_active = None
            line = f" [{index:>2d}/{self._total}]  FAIL  {self._active_header:<42s} {_format_elapsed(elapsed):>8s}"
            self._completed_lines.append(line)
            if error:
                self._completed_lines.append(f"         {error}")
        if not self._is_tty:
            self._console.print(
                f" [{index:>2d}/{self._total}]  FAIL  {self._active_header}  {_format_elapsed(elapsed)}",
                highlight=False,
            )
            if error:
                self._console.print(f"         {error}", highlight=False)
        self._refresh()

    def finish(self) -> None:
        """Print study completion footer."""
        self.stop()
        total = time.monotonic() - self._total_start
        self._console.print(f"\nStudy completed in {_format_elapsed(total)}", highlight=False)

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
        if not self._is_tty:
            self._print_inner_line(step, elapsed_sec)
        self._refresh()

    # -- Rendering --

    def _render(self) -> Text:
        lines = Text()
        with self._lock:
            # Completed experiments
            for cline in self._completed_lines:
                lines.append(cline + "\n")

            # Active experiment header
            if self._active_header:
                lines.append(f" [{self._active_index:>2d}/{self._total}]  {self._active_header}\n")

                inner_total = len(self._inner_steps) or (
                    len(self._inner_completed) + (1 if self._inner_active else 0)
                )

                # Inner completed steps
                for i, (_step, label, detail, elapsed) in enumerate(self._inner_completed):
                    idx = i + 1
                    inner_line = _step_line(
                        idx, inner_total, label, detail, "DONE", _format_elapsed(elapsed)
                    )
                    lines.append(f"         {inner_line}\n")

                # Inner active step
                if self._inner_active:
                    _step, label, detail, start = self._inner_active
                    idx = len(self._inner_completed) + 1
                    elapsed = time.monotonic() - start
                    frame_idx = int(elapsed * 8) % len(_SPINNER_FRAMES)
                    spinner = _SPINNER_FRAMES[frame_idx]
                    inner_line = _step_line(
                        idx, inner_total, label, detail, spinner, _format_elapsed(elapsed)
                    )
                    lines.append(f"         {inner_line}")

        return lines

    def _refresh(self) -> None:
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.update(self._render())

    def _print_inner_line(self, step: str, elapsed_sec: float) -> None:
        """Print inner step line in non-TTY mode."""
        with self._lock:
            inner_total = len(self._inner_steps) or len(self._inner_completed)
            idx = len(self._inner_completed)
            label = STEP_LABELS.get(step, step)
            detail = ""
            for s, _l, d, _e in self._inner_completed:
                if s == step:
                    detail = d
                    break
        line = _step_line(idx, inner_total, label, detail, "DONE", _format_elapsed(elapsed_sec))
        self._console.print(f"         {line}", highlight=False)
