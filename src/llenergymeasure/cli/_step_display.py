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
from collections.abc import Callable
from typing import NamedTuple

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.table import Table
from rich.text import Text

from llenergymeasure.config.ssot import ENV_TABLE_ROWS
from llenergymeasure.domain.progress import PHASE_MEASUREMENT, STEP_LABELS, STEP_PHASES
from llenergymeasure.utils.compat import StrEnum
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed
from llenergymeasure.utils.formatting import short_name as _short_image
from llenergymeasure.utils.formatting import truncate_detail as _truncate_detail

# Braille spinner frames (same as Docker BuildKit / ora)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_FPS = 8

# Heartbeat interval for non-TTY mode (seconds)
_HEARTBEAT_INTERVAL = 5.0

# Minimum step duration before heartbeat kicks in (seconds)
_HEARTBEAT_THRESHOLD = 3.0

# Lines to reserve when computing viewport height for the completed-rows table.
# Accounts for: study header (1), image prep block (~4), hidden indicator (1),
# table header (1), active experiment block (~8), gap (1), completion line (1).
_VIEWPORT_RESERVED_LINES = 12


class _ImagePrepResult(NamedTuple):
    """Result of a successfully prepared Docker image."""

    engine: str
    image: str
    cached: bool
    elapsed: float
    metadata: dict[str, str] | None


class _ImagePrepFailure(NamedTuple):
    """Result of a failed Docker image preparation."""

    engine: str
    image: str
    error: str


class _DynamicRenderable:
    """Proxy that calls a render function on every Rich auto-refresh.

    Without this, ``Live`` holds a static ``Text`` snapshot and the
    spinner / elapsed counters freeze between callback events.
    """

    def __init__(self, render_fn: Callable[[], Text | Group | Table]) -> None:
        self._render_fn = render_fn

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
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


def _render_substep_lines(
    lines: Text,
    substeps: list[tuple[str, float]],
    indent: str = "              ",
    active: tuple[str, float] | None = None,
) -> None:
    """Append substep lines (dim · prefix) to a Rich Text renderable.

    Shared between StepDisplay and StudyStepDisplay to avoid duplication.
    Frozen substeps render as dim ``· text ✓ elapsed``; the optional
    ``active`` substep (``(text, start_monotonic)``) renders with a dim
    spinner and rising elapsed counter so Rich Live animates it each frame.
    """
    for sub_text, sub_elapsed in substeps:
        lines.append(f"{indent}· {sub_text}", style="dim")
        if sub_elapsed > 0:
            lines.append("  \u2713", style="dim")
            lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
        lines.append("\n")
    if active is not None:
        sub_text, start_ts = active
        elapsed = time.monotonic() - start_ts
        frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
        spinner = _SPINNER_FRAMES[frame_idx]
        lines.append(f"{indent}· {sub_text}", style="dim")
        lines.append(f"  {spinner}", style="dim")
        lines.append(f"  {_format_elapsed(elapsed)}", style="dim")
        lines.append("\n")


class ImageSource(StrEnum):
    """Provenance values for Docker image selection."""

    LOCAL_BUILD = "local_build"
    REGISTRY = "registry"
    REGISTRY_CACHED = "registry_cached"
    ENV = "env"
    YAML = "yaml"
    RUNNER_OVERRIDE = "runner_override"
    USER_CONFIG = "user_config"


_IMAGE_SOURCE_LABELS: dict[ImageSource, str] = {
    ImageSource.LOCAL_BUILD: "LOCAL BUILD — current source tree (via docker compose build)",
    ImageSource.REGISTRY: "REGISTRY — versioned release image",
    ImageSource.REGISTRY_CACHED: "REGISTRY — cached locally from prior pull",
    ImageSource.ENV: "OVERRIDE — image set via environment variable",
    ImageSource.YAML: "OVERRIDE — image set in study YAML images: section",
    ImageSource.RUNNER_OVERRIDE: "OVERRIDE — image set via docker:<image> in runners:",
    ImageSource.USER_CONFIG: "OVERRIDE — image set in user config (~/.config/llenergymeasure/config.yaml)",
}

_RUNNER_SOURCE_LABELS: dict[str, str] = {
    "env": "env var",
    "yaml": "study YAML",
    "user_config": "user config",
    "auto_detected": "auto-detected",
    "default": "default",
    "multi_engine_elevation": "multi-engine auto-elevation",
}


def _render_runner_info(lines: Text, info: dict[str, str | None]) -> None:
    """Render runner/image provenance lines below the experiment header."""
    mode = info.get("mode", "unknown")
    source = info.get("source", "")
    image = info.get("image")
    image_source = info.get("image_source")

    source_label = _RUNNER_SOURCE_LABELS.get(source or "", source or "")

    if mode == "local":
        lines.append(f"       mode:    local ({source_label})\n", style="dim")
        lines.append(
            "               no container isolation — running directly on host\n", style="dim"
        )
    elif mode == "docker" and image:
        lines.append(f"       mode:    docker ({source_label})\n", style="dim")
        lines.append(f"       image:   {image}\n", style="dim")
        if image_source:
            try:
                key = ImageSource(image_source)
                detail = _IMAGE_SOURCE_LABELS.get(key, image_source)
            except ValueError:
                detail = image_source
            lines.append(f"               {detail}\n", style="dim")


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
        display.register_steps(
            docker_steps(images_prepared=False, host_baseline=True)
        )
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
        # Active substep per step: (text, start_monotonic) — drives the
        # heartbeat spinner for long-running sub-operations (e.g. CUDA init
        # inside the baseline container) so Rich Live animates them.
        self._active_substep: dict[str, tuple[str, float]] = {}

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
            # Clear any prior completion/skip/substeps for this step so a
            # re-fire (e.g. host dispatch failed and the experiment container
            # fell back to in-harness measurement) shows as an active spinner
            # instead of being masked by stale completed state.
            self._completed_steps.discard(step)
            self._skipped_steps.discard(step)
            self._step_data.pop(step, None)
            self._substeps.pop(step, None)
            self._active_substep.pop(step, None)
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
            # Freeze any dangling active substep so it doesn't keep animating
            # under a completed step. Uses its accumulated elapsed.
            dangling = self._active_substep.pop(step, None)
            if dangling is not None:
                d_text, d_start = dangling
                self._substeps.setdefault(step, []).append(
                    (d_text, max(0.0, time.monotonic() - d_start))
                )
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

    def on_substep_start(self, step: str, text: str) -> None:
        """Begin a live sub-operation rendered with a dim spinner + counter."""
        with self._lock:
            # If a prior active substep for this step never got a matching
            # done, freeze it so the new active substep doesn't overwrite
            # silently. Uses its accumulated elapsed.
            prior = self._active_substep.pop(step, None)
            if prior is not None:
                prior_text, prior_start = prior
                self._substeps.setdefault(step, []).append(
                    (prior_text, max(0.0, time.monotonic() - prior_start))
                )
            self._active_substep[step] = (text, time.monotonic())
        if not self._is_tty:
            self._print_substep_line(step, text, 0.0)
        self._refresh()

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Freeze the currently-active substep with final text + elapsed."""
        with self._lock:
            active = self._active_substep.pop(step, None)
            if active is None:
                # No matching start — fall through as a regular completed substep.
                final_text = text or ""
                final_elapsed = elapsed_sec if elapsed_sec is not None else 0.0
            else:
                start_text, start_ts = active
                final_text = text if text is not None else start_text
                final_elapsed = (
                    elapsed_sec
                    if elapsed_sec is not None
                    else max(0.0, time.monotonic() - start_ts)
                )
            if final_text:
                self._substeps.setdefault(step, []).append((final_text, final_elapsed))
        if not self._is_tty and final_text:
            self._print_substep_line(step, final_text, final_elapsed)
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
                        _render_substep_lines(
                            lines,
                            self._substeps.get(step, []),
                            active=self._active_substep.get(step),
                        )
                    elif step in self._skipped_steps:
                        label, reason, _ = self._step_data[step]
                        prefix = _step_line_prefix(idx, phase_total, label, reason)
                        lines.append(prefix, style="dim")
                        lines.append("  SKIP", style="dim")
                        lines.append("\n")
                    elif step == self._active_step:
                        elapsed = time.monotonic() - self._active_start
                        frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
                        spinner = _SPINNER_FRAMES[frame_idx]
                        prefix = _step_line_prefix(
                            idx, phase_total, self._active_label, self._active_detail
                        )
                        lines.append(prefix)
                        lines.append(f"  {spinner}", style="yellow")
                        lines.append(f"  {_format_elapsed(elapsed)}\n")
                        _render_substep_lines(
                            lines,
                            self._substeps.get(step, []),
                            active=self._active_substep.get(step),
                        )
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

        # Completed experiment rows:
        # (index, status, config, elapsed, inference_sec, energy_j, adj_energy_j, throughput, mj_per_tok)
        self._completed_rows: list[
            tuple[
                int,
                str,
                str,
                float,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ] = []

        # Active experiment state
        self._active_index: int = 0
        self._active_header: str = ""
        self._inner_completed: list[tuple[str, str, str, float]] = []
        self._inner_active: tuple[str, str, str, float] | None = None  # step, label, detail, start
        self._inner_steps: list[str] = []
        self._inner_skipped: dict[str, str] = {}  # step -> reason
        self._inner_substeps: dict[str, list[tuple[str, float]]] = {}
        # Active substep per step: (text, start_monotonic) — same spinner
        # heartbeat treatment as StepDisplay, used for live baseline stages.
        self._inner_active_substep: dict[str, tuple[str, float]] = {}
        self._runner_info: dict[str, str | None] | None = None

        # Per-experiment save paths: (index, host_path, container_path | None)
        self._saved_paths: list[tuple[int, str, str | None]] = []

        self._live: Live | None = None
        self._total_start: float = 0.0
        self._gap_text: str = ""

        # Image prep state (study-level Docker image preparation)
        self._image_prep_active: bool = False
        self._image_prep_total: int = 0
        self._image_prep_done: list[_ImagePrepResult] = []
        self._image_prep_failed: _ImagePrepFailure | None = None

    def start(self, *, print_header: bool = True) -> None:
        """Begin the display. Optionally prints study header and starts Rich Live if TTY.

        Args:
            print_header: When False, suppresses the header line (caller prints it
                separately, e.g. with a preflight summary in between).
        """
        self._total_start = time.monotonic()
        if print_header:
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

    def add_historical_rows(
        self,
        rows: list[
            tuple[
                int,
                str,
                str,
                float,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ],
    ) -> None:
        """Pre-populate the completed table with rows from a previous run.

        Used by --resume to show previously completed experiments in the same
        table format as live results. Must be called before start().

        Historical rows use status "PREV_OK" / "PREV_FAIL" to render dimmed
        with a distinct marker, visually separating them from live results.

        Args:
            rows: List of (index, status, config_summary, elapsed_seconds,
                  inference_sec, energy_j, adj_energy_j, throughput, mj_tok).
                  status is "OK" or "FAIL".
        """
        with self._lock:
            for idx, status, config, elapsed, infer, energy, adj_e, tput, mj in rows:
                hist_status = f"PREV_{status}"
                self._completed_rows.append(
                    (idx, hist_status, config, elapsed, infer, energy, adj_e, tput, mj)
                )

    def stop(self) -> None:
        """Stop Rich Live."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def begin_experiment(
        self,
        index: int,
        header: str,
        steps: list[str],
        runner_info: dict[str, str | None] | None = None,
    ) -> None:
        """Start tracking a new experiment within the study."""
        with self._lock:
            self._active_index = index
            self._active_header = header
            self._inner_completed = []
            self._inner_active = None
            self._inner_steps = steps
            self._inner_skipped = {}
            self._inner_substeps = {}
            self._inner_active_substep = {}
            self._runner_info = runner_info
        self._refresh()

    def end_experiment_ok(
        self,
        index: int,
        elapsed: float,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
        inference_time_sec: float | None = None,
        adj_energy_j: float | None = None,
        mj_per_tok_adjusted: float | None = None,
        mj_per_tok_total: float | None = None,
    ) -> None:
        """Mark experiment as successfully completed."""
        # Prefer mj_per_tok_adjusted (baseline-subtracted) when available,
        # fall back to mj_per_tok_total. No recomputation — show "-" if both null.
        mj_tok: float | None
        if mj_per_tok_adjusted is not None:
            mj_tok = mj_per_tok_adjusted
        elif mj_per_tok_total is not None:
            mj_tok = mj_per_tok_total
        else:
            mj_tok = None

        with self._lock:
            self._inner_active = None
            self._completed_rows.append(
                (
                    index,
                    "OK",
                    self._active_header,
                    elapsed,
                    inference_time_sec,
                    energy_j,
                    adj_energy_j,
                    throughput_tok_s,
                    mj_tok,
                )
            )
        if not self._is_tty:
            self._print_completed_row(
                index,
                "OK",
                self._active_header,
                elapsed,
                inference_time_sec,
                energy_j,
                adj_energy_j,
                throughput_tok_s,
                mj_tok,
            )
        self._refresh()

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Mark experiment as failed."""
        with self._lock:
            self._inner_active = None
            self._completed_rows.append(
                (index, "FAIL", self._active_header, elapsed, None, None, None, None, None)
            )
        if not self._is_tty:
            self._print_completed_row(
                index, "FAIL", self._active_header, elapsed, None, None, None, None, None
            )
            if error:
                self._console.print(f"         {error}", highlight=False)
        self._refresh()

    def finish(
        self,
        save_path: str | None = None,
        total_elapsed: float | None = None,
    ) -> None:
        """Print study completion footer with final results table.

        When Live was active (TTY mode), the table is already on screen from the
        live display, so only the completion line and save path are printed.
        When used post-hoc (no Live), prints the full table.

        Args:
            save_path: Optional path to saved results directory.
            total_elapsed: Total elapsed time in seconds. If None, falls back to
                monotonic clock delta from start() (which may be wrong if start()
                was never called — callers constructing post-hoc should always pass this).
        """
        was_live = self._live is not None
        self.stop()
        if total_elapsed is not None:
            total = total_elapsed
        elif self._total_start > 0:
            total = time.monotonic() - self._total_start
        else:
            total = 0.0
        self._console.print(f"\nStudy completed in {_format_elapsed(total)}", highlight=False)

        # Only print table in post-hoc mode — Live already shows it on screen
        if not was_live and self._completed_rows:
            table, _hidden = self._build_table()
            self._console.print(table)

        # Print study results directory
        if save_path:
            self._console.print(f"\n  Results: {save_path}", style="dim", highlight=False)

        # Print per-experiment save paths (only in TTY mode — non-TTY prints inline)
        if was_live and self._saved_paths:
            for idx, host_path, container_path in self._saved_paths:
                if container_path:
                    self._console.print(
                        f"  [{idx}] container: {container_path}", style="dim", highlight=False
                    )
                self._console.print(
                    f"  [{idx}] host:      {host_path}", style="dim", highlight=False
                )

    # -- ProgressCallback for inner steps --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            # Clear any prior completion/skip/substeps for this step. A re-fire
            # means "this step is running again" (e.g. host dispatch failed and
            # the experiment container fell back to in-harness measurement), so
            # stale state from a prior attempt must not mask the new active
            # spinner — the renderer checks completed_map before _inner_active
            # and would otherwise hide the re-run.
            self._inner_completed = [c for c in self._inner_completed if c[0] != step]
            self._inner_skipped.pop(step, None)
            self._inner_substeps.pop(step, None)
            self._inner_active_substep.pop(step, None)
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
            # Freeze any dangling active substep so it doesn't keep animating
            # under a completed step.
            dangling = self._inner_active_substep.pop(step, None)
            if dangling is not None:
                d_text, d_start = dangling
                self._inner_substeps.setdefault(step, []).append(
                    (d_text, max(0.0, time.monotonic() - d_start))
                )
        self._refresh()

    def on_step_skip(self, step: str, reason: str = "") -> None:
        """Record a skipped step (rendered dim grey in the step list)."""
        with self._lock:
            self._inner_skipped[step] = reason or "-"
        self._refresh()

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Record a completed sub-operation within the active step."""
        with self._lock:
            if step not in self._inner_substeps:
                self._inner_substeps[step] = []
            self._inner_substeps[step].append((text, elapsed_sec))
        self._refresh()

    def on_substep_start(self, step: str, text: str) -> None:
        """Begin a live sub-operation rendered with a dim spinner + counter."""
        with self._lock:
            prior = self._inner_active_substep.pop(step, None)
            if prior is not None:
                prior_text, prior_start = prior
                self._inner_substeps.setdefault(step, []).append(
                    (prior_text, max(0.0, time.monotonic() - prior_start))
                )
            self._inner_active_substep[step] = (text, time.monotonic())
        self._refresh()

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Freeze the currently-active substep with final text + elapsed."""
        with self._lock:
            active = self._inner_active_substep.pop(step, None)
            if active is None:
                final_text = text or ""
                final_elapsed = elapsed_sec if elapsed_sec is not None else 0.0
            else:
                start_text, start_ts = active
                final_text = text if text is not None else start_text
                final_elapsed = (
                    elapsed_sec
                    if elapsed_sec is not None
                    else max(0.0, time.monotonic() - start_ts)
                )
            if final_text:
                self._inner_substeps.setdefault(step, []).append((final_text, final_elapsed))
        self._refresh()

    def on_experiment_saved(
        self, index: int, host_path: str, container_path: str | None = None
    ) -> None:
        """Display save path info after experiment results are written to disk."""
        with self._lock:
            self._saved_paths.append((index, host_path, container_path))
        if not self._is_tty:
            if container_path:
                self._console.print(f"         \u00b7 container: {container_path}", highlight=False)
            self._console.print(f"         \u00b7 host:      {host_path}", highlight=False)

    # -- Gap display --

    def show_gap(self, text: str) -> None:
        """Show a gap countdown line below the table (e.g. 'Experiment gap: 7s')."""
        with self._lock:
            self._gap_text = text
        self._refresh()

    def clear_gap(self) -> None:
        """Clear the gap countdown line."""
        with self._lock:
            self._gap_text = ""
        self._refresh()

    # -- Image prep (study-level Docker image preparation) --

    def begin_image_prep(self, engines: list[str]) -> None:
        """Signal the start of study-level Docker image preparation."""
        with self._lock:
            self._image_prep_active = True
            self._image_prep_total = len(engines)
        if not self._is_tty:
            self._console.print("\n  Preparing Docker images", highlight=False)
        self._refresh()

    def image_ready(
        self,
        engine: str,
        image: str,
        cached: bool,
        elapsed: float,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Signal that a Docker image is ready."""
        with self._lock:
            self._image_prep_done.append(_ImagePrepResult(engine, image, cached, elapsed, metadata))
        if not self._is_tty:
            idx = len(self._image_prep_done)
            total = self._image_prep_total
            status = "cached" if cached else "pulled"
            short_img = _short_image(image)
            line = f"      [{idx}/{total}]  {engine:<16s}{short_img} ({status})"
            line += f"  \u2713  {_format_elapsed(elapsed)}"
            self._console.print(line, highlight=False)
            if metadata:
                parts = [f"{k}: {v}" for k, v in metadata.items()]
                meta_text = " \u00b7 ".join(parts)
                self._console.print(
                    f"                    \u00b7 {meta_text}",
                    style="dim",
                    highlight=False,
                )
        self._refresh()

    def image_failed(self, engine: str, image: str, error: str) -> None:
        """Signal that a Docker image could not be prepared."""
        with self._lock:
            self._image_prep_failed = _ImagePrepFailure(engine, image, error)
        if not self._is_tty:
            idx = len(self._image_prep_done) + 1
            total = self._image_prep_total
            short_img = _short_image(image)
            self._console.print(
                f"      [{idx}/{total}]  {engine:<16s}{short_img}  \u2717",
                highlight=False,
            )
            self._console.print(
                f"                    \u00b7 {error}",
                style="dim",
                highlight=False,
            )
        self._refresh()

    def end_image_prep(self) -> None:
        """Signal the end of study-level Docker image preparation."""
        with self._lock:
            self._image_prep_active = False
        self._refresh()

    def _render_image_prep(self) -> Text:
        """Render the Docker image preparation section."""
        lines = Text()
        if not self._image_prep_done and self._image_prep_failed is None:
            if self._image_prep_active:
                lines.append("\n  Preparing Docker images\n", style="bold")
            return lines

        lines.append("\n  Preparing Docker images\n")
        total = self._image_prep_total

        for idx, (engine, image, cached, elapsed, metadata) in enumerate(self._image_prep_done, 1):
            status = "cached" if cached else "pulled"
            short_img = _short_image(image)
            counter = f"[{idx}/{total}]"
            lines.append(f"      {counter:>7s}  {engine:<16s}{short_img} ({status})")
            lines.append("  \u2713", style="bold green")
            lines.append(f"  {_format_elapsed(elapsed)}\n")
            if metadata:
                parts = [f"{k}: {v}" for k, v in metadata.items()]
                meta_text = " \u00b7 ".join(parts)
                lines.append(f"                    \u00b7 {meta_text}\n", style="dim")

        if self._image_prep_failed:
            fail_engine, fail_image, fail_error = self._image_prep_failed
            idx = len(self._image_prep_done) + 1
            short_img = _short_image(fail_image)
            counter = f"[{idx}/{total}]"
            lines.append(f"      {counter:>7s}  {fail_engine:<16s}{short_img}")
            lines.append("  \u2717", style="bold red")
            lines.append("\n")
            lines.append(f"                    \u00b7 {fail_error}\n", style="dim")

        return lines

    # -- Rendering --

    def _viewport_size(self) -> int:
        """Maximum number of completed rows visible in the terminal.

        Respects LLEM_TABLE_ROWS env var if set (overrides terminal height calc).
        """
        import os

        env_rows = os.environ.get(ENV_TABLE_ROWS)
        if env_rows:
            try:
                return max(3, int(env_rows))
            except ValueError:
                pass
        return max(5, self._console.size.height - _VIEWPORT_RESERVED_LINES)

    def _build_table(self) -> tuple[Table, int]:
        """Build the Rich Table of completed experiments with viewport limiting.

        Returns (table, hidden_count).
        """
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", width=3, justify="right")
        table.add_column("", width=2)
        table.add_column("Config", max_width=45, overflow="ellipsis", no_wrap=True)
        table.add_column("Total", justify="right")
        table.add_column("Infer", justify="right")
        table.add_column("Energy", justify="right")
        table.add_column("Adj. E", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("mJ/tok", justify="right")
        rows = self._completed_rows
        available = self._viewport_size()
        hidden = max(0, len(rows) - available)
        visible = rows[max(0, len(rows) - available) :]
        for (
            idx,
            status,
            config,
            elapsed,
            infer_sec,
            energy,
            adj_energy,
            throughput,
            mj_tok,
        ) in visible:
            is_historical = status.startswith("PREV_")
            if status == "OK":
                status_text = Text("\u2713", style="bold green")
            elif status == "PREV_OK":
                status_text = Text("\u2713", style="dim green")
            elif status == "PREV_FAIL":
                status_text = Text("\u2717", style="dim red")
            else:
                status_text = Text("\u2717", style="bold red")
            row_style = "dim" if is_historical else None
            infer_str = _format_elapsed(infer_sec) if infer_sec is not None else "-"
            energy_str = f"{energy:.1f} J" if energy is not None else "-"
            adj_energy_str = f"{adj_energy:.1f} J" if adj_energy is not None else "-"
            throughput_str = f"{throughput:.1f}" if throughput is not None else "-"
            mj_str = f"{mj_tok:.1f}" if mj_tok is not None else "-"
            table.add_row(
                str(idx),
                status_text,
                config,
                _format_elapsed(elapsed),
                infer_str,
                energy_str,
                adj_energy_str,
                throughput_str,
                mj_str,
                style=row_style,
            )
        return table, hidden

    def _render_active_steps(self) -> Text:
        """Render the active experiment's step progress.

        Iterates registered steps in order so completed, skipped, and active
        steps all appear at their correct [x/N] position. Skipped steps
        render dim grey with SKIP label (mirrors StepDisplay behaviour).
        Pending steps are not shown (Docker BuildKit-style progressive output).
        """
        lines = Text()
        if not self._active_header:
            return lines

        lines.append(f"\n  [{self._active_index}/{self._total}] {self._active_header}\n")

        inner_total = len(self._inner_steps) or (
            len(self._inner_completed) + len(self._inner_skipped) + (1 if self._inner_active else 0)
        )

        # Index completed steps by name for O(1) lookup while preserving order
        completed_map: dict[str, tuple[str, str, float]] = {}
        for step, label, detail, elapsed in self._inner_completed:
            completed_map[step] = (label, detail, elapsed)

        idx = 0
        for step in self._inner_steps:
            if step in completed_map:
                idx += 1
                label, detail, elapsed = completed_map[step]
                counter = f"[{idx}/{inner_total}]"
                trunc_detail = _truncate_detail(detail)
                lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
                lines.append("  \u2713", style="bold green")
                lines.append(f"  {_format_elapsed(elapsed)}\n")
                _render_substep_lines(
                    lines,
                    self._inner_substeps.get(step, []),
                    indent="                    ",
                    active=self._inner_active_substep.get(step),
                )
            elif step in self._inner_skipped:
                idx += 1
                label = STEP_LABELS.get(step, step)
                reason = self._inner_skipped[step]
                counter = f"[{idx}/{inner_total}]"
                trunc_reason = _truncate_detail(reason)
                lines.append(
                    f"      {counter:>7s}  {label:<16s} {trunc_reason:<34s}  SKIP\n",
                    style="dim",
                )
            elif self._inner_active and self._inner_active[0] == step:
                idx += 1
                _step, label, detail, start = self._inner_active
                elapsed = time.monotonic() - start
                frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
                spinner = _SPINNER_FRAMES[frame_idx]
                counter = f"[{idx}/{inner_total}]"
                trunc_detail = _truncate_detail(detail)
                lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
                lines.append(f"  {spinner}", style="yellow")
                lines.append(f"  {_format_elapsed(elapsed)}\n")
                _render_substep_lines(
                    lines,
                    self._inner_substeps.get(step, []),
                    indent="                    ",
                    active=self._inner_active_substep.get(step),
                )
            # Pending steps: not shown (Docker BuildKit-style progressive output)

        return lines

    def _render(self) -> Group:
        """Render image prep + hidden-row indicator + completed experiments table + active steps + gap."""
        with self._lock:
            image_prep = self._render_image_prep()
            table, hidden = self._build_table()
            step_text = self._render_active_steps()
            gap = Text(f"\n  {self._gap_text}", style="dim") if self._gap_text else Text("")
            if hidden > 0:
                indicator = Text(f"  ({hidden} earlier results not shown)\n", style="dim")
            else:
                indicator = Text("")
        return Group(image_prep, indicator, table, step_text, gap)

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
        inference_sec: float | None,
        energy: float | None,
        adj_energy: float | None,
        throughput: float | None,
        mj_tok: float | None = None,
    ) -> None:
        """Print a completed experiment row in non-TTY mode."""
        status_icon = "\u2713" if status == "OK" else "\u2717"
        infer_str = f"  infer={_format_elapsed(inference_sec)}" if inference_sec is not None else ""
        energy_str = f"  {energy:.1f} J" if energy is not None else ""
        adj_energy_str = f"  adj={adj_energy:.1f} J" if adj_energy is not None else ""
        throughput_str = f"  {throughput:.1f} tok/s" if throughput is not None else ""
        mj_str = f"  {mj_tok:.1f} mJ/tok" if mj_tok is not None else ""
        line = (
            f" [{index:>2d}/{self._total}]  {status_icon}  {config:<42s}"
            f" {_format_elapsed(elapsed):>8s}{infer_str}{energy_str}{adj_energy_str}"
            f"{throughput_str}{mj_str}"
        )
        self._console.print(line, highlight=False)
