"""Progress callback protocol for step-by-step measurement reporting.

Lives in domain/ (Layer 0) so every layer can import it without
violating the architectural layering rules.

Steps are grouped into phases for hierarchical display (Docker BuildKit
style). The display renders phase headers with indented sub-steps.
Steps that don't apply are shown as SKIP with a fixed total count.

The protocol also supports on_substep() for fine-grained sub-operation
reporting within an active step (e.g. CUDA check, model access check).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Callback protocol for reporting measurement step progress.

    Step names use a fixed vocabulary grouped into two phases::

        Setup:       preflight, image_check, pull, container_start, container_preflight
        Measurement: baseline, model, prompts, warmup, thermal_floor,
                     energy_select, measure, flops, save

    Steps that don't apply in a given run are reported via on_step_skip()
    and rendered as SKIP in the display. This keeps a fixed [x/y] counter.

    Sub-step granularity: on_substep() reports completed sub-operations
    within an active step (e.g. "CUDA available", "model accessible").

    Implementors:
        - StepDisplay (cli/_step_display.py) -- Rich-based TTY rendering
        - StreamProgressCallback (entrypoints/container.py) -- JSON lines to stdout
        - _QueueProgressAdapter (study/runner.py) -- multiprocessing.Queue bridge
    """

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        """Signal that a named step has begun.

        Args:
            step: Step identifier from the fixed vocabulary.
            description: Human-readable verb label (e.g. "Loading model").
            detail: Additional context (e.g. model name, image tag).
        """
        ...

    def on_step_update(self, step: str, detail: str) -> None:
        """Update the detail text of the currently active step.

        Used for live progress within a step (e.g. warmup iteration count,
        CV convergence progress).

        Args:
            step: Step identifier (must match most recent on_step_start).
            detail: Updated detail text.
        """
        ...

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        """Signal that a step has completed.

        Args:
            step: Step identifier (must match most recent on_step_start).
            elapsed_sec: Wall-clock time for this step in seconds.
        """
        ...

    def on_step_skip(self, step: str, reason: str = "") -> None:
        """Signal that a step was skipped (not applicable to this run).

        Args:
            step: Step identifier from the fixed vocabulary.
            reason: Optional human-readable reason (e.g. "disabled", "cached").
        """
        ...

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Signal a completed sub-operation within the active step.

        Args:
            step: Parent step identifier (must match most recent on_step_start).
            text: Human-readable substep description (e.g. "CUDA available").
            elapsed_sec: Wall-clock time for this substep (0.0 = instantaneous).
        """
        ...


@runtime_checkable
class StudyProgressCallback(ProgressCallback, Protocol):
    """Extended callback for study-level experiment tracking + per-step progress.

    Adds begin/end experiment methods on top of ProgressCallback's step events.
    Used by StudyStepDisplay (cli/_step_display.py) and consumed by
    StudyRunner (study/runner.py).

    Implementors:
        - StudyStepDisplay (cli/_step_display.py)
    """

    def begin_experiment(
        self, index: int, model: str, backend: str, precision: str, steps: list[str]
    ) -> None:
        """Signal that a new experiment is starting within the study.

        Args:
            index: 1-based experiment position in the study.
            model: Model name (e.g. "gpt2").
            backend: Backend name (e.g. "pytorch").
            precision: Precision string (e.g. "bf16").
            steps: Ordered step names for this experiment's [x/y] counter.
        """
        ...

    def end_experiment_ok(
        self,
        index: int,
        elapsed: float,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
    ) -> None:
        """Signal that an experiment completed successfully.

        Args:
            index: 1-based experiment position.
            elapsed: Total wall-clock time in seconds.
            energy_j: Total energy in joules (None if unavailable).
            throughput_tok_s: Throughput in tokens/second (None if unavailable).
        """
        ...

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Signal that an experiment failed.

        Args:
            index: 1-based experiment position.
            elapsed: Wall-clock time until failure in seconds.
            error: Human-readable error message.
        """
        ...


# Phase names -- top-level groups in the hierarchical display.
PHASE_SETUP = "Setup"
PHASE_MEASUREMENT = "Measurement"

# Step vocabulary -- canonical names used across all layers.
# Setup phase
STEP_PREFLIGHT = "preflight"
STEP_IMAGE_CHECK = "image_check"
STEP_PULL = "pull"
STEP_CONTAINER_START = "container_start"
STEP_CONTAINER_PREFLIGHT = "container_preflight"
STEP_CONTAINER = "container"  # kept for backward compat (study runner)

# Measurement phase
STEP_BASELINE = "baseline"
STEP_MODEL = "model"
STEP_PROMPTS = "prompts"
STEP_WARMUP = "warmup"
STEP_THERMAL_FLOOR = "thermal_floor"
STEP_ENERGY_SELECT = "energy_select"
STEP_MEASURE = "measure"
STEP_FLOPS = "flops"
STEP_SAVE = "save"

STEP_LABELS: dict[str, str] = {
    STEP_PREFLIGHT: "Checking",
    STEP_IMAGE_CHECK: "Inspecting",
    STEP_PULL: "Pulling",
    STEP_CONTAINER_START: "Starting",
    STEP_CONTAINER_PREFLIGHT: "Checking",
    STEP_CONTAINER: "Starting",
    STEP_BASELINE: "Measuring",
    STEP_MODEL: "Loading",
    STEP_PROMPTS: "Loading",
    STEP_WARMUP: "Warming up",
    STEP_THERMAL_FLOOR: "Waiting",
    STEP_ENERGY_SELECT: "Selecting",
    STEP_MEASURE: "Measuring",
    STEP_FLOPS: "Estimating",
    STEP_SAVE: "Saving",
}

# Step-to-phase mapping -- determines which phase header each step appears under.
# Unknown steps default to PHASE_MEASUREMENT.
STEP_PHASES: dict[str, str] = {
    STEP_PREFLIGHT: PHASE_SETUP,
    STEP_IMAGE_CHECK: PHASE_SETUP,
    STEP_PULL: PHASE_SETUP,
    STEP_CONTAINER_START: PHASE_SETUP,
    STEP_CONTAINER_PREFLIGHT: PHASE_SETUP,
    STEP_CONTAINER: PHASE_SETUP,
    STEP_BASELINE: PHASE_MEASUREMENT,
    STEP_MODEL: PHASE_MEASUREMENT,
    STEP_PROMPTS: PHASE_MEASUREMENT,
    STEP_WARMUP: PHASE_MEASUREMENT,
    STEP_THERMAL_FLOOR: PHASE_MEASUREMENT,
    STEP_ENERGY_SELECT: PHASE_MEASUREMENT,
    STEP_MEASURE: PHASE_MEASUREMENT,
    STEP_FLOPS: PHASE_MEASUREMENT,
    STEP_SAVE: PHASE_MEASUREMENT,
}

# Ordered step lists for pre-registration (fixed [x/y] counters).
# Docker path: host setup + container measurement forwarded as top-level steps.
STEPS_DOCKER = [
    STEP_PREFLIGHT,
    STEP_IMAGE_CHECK,
    STEP_PULL,
    STEP_CONTAINER_START,
    STEP_CONTAINER_PREFLIGHT,
    STEP_BASELINE,
    STEP_MODEL,
    STEP_PROMPTS,
    STEP_WARMUP,
    STEP_THERMAL_FLOOR,
    STEP_ENERGY_SELECT,
    STEP_MEASURE,
    STEP_FLOPS,
    STEP_SAVE,
]

# Local path: no Docker steps, direct harness measurement.
STEPS_LOCAL = [
    STEP_PREFLIGHT,
    STEP_CONTAINER_PREFLIGHT,
    STEP_BASELINE,
    STEP_MODEL,
    STEP_PROMPTS,
    STEP_WARMUP,
    STEP_THERMAL_FLOOR,
    STEP_ENERGY_SELECT,
    STEP_MEASURE,
    STEP_FLOPS,
    STEP_SAVE,
]
