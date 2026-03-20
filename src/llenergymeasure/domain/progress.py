"""Progress callback protocol for step-by-step measurement reporting.

Lives in domain/ (Layer 0) so every layer can import it without
violating the architectural layering rules.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Callback protocol for reporting measurement step progress.

    Step names use a fixed vocabulary::

        preflight, pull, container, baseline, model, warmup, measure, save

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


# Step vocabulary -- canonical names used across all layers.
STEP_PREFLIGHT = "preflight"
STEP_PULL = "pull"
STEP_CONTAINER = "container"
STEP_BASELINE = "baseline"
STEP_MODEL = "model"
STEP_WARMUP = "warmup"
STEP_MEASURE = "measure"
STEP_SAVE = "save"

STEP_LABELS: dict[str, str] = {
    STEP_PREFLIGHT: "Checking",
    STEP_PULL: "Pulling",
    STEP_CONTAINER: "Starting",
    STEP_BASELINE: "Measuring baseline",
    STEP_MODEL: "Loading model",
    STEP_WARMUP: "Warming up",
    STEP_MEASURE: "Measuring",
    STEP_SAVE: "Saving",
}
