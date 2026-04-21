"""Per-experiment progress display for study runs.

Moved from cli/_display.py to break the study -> cli layering violation.
The study layer must not import from cli.
"""

from __future__ import annotations

import sys

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.utils.formatting import format_elapsed as _format_duration
from llenergymeasure.utils.formatting import sig3 as _sig3


def print_study_progress(
    index: int,
    total: int,
    config: ExperimentConfig,
    status: str = "running",
    elapsed: float | None = None,
    energy: float | None = None,
) -> None:
    """Print a per-experiment progress line to stderr.

    Format: [3/12] <icon> model engine dtype -- elapsed (energy)
    Icons: completed=OK, failed=FAIL, running=...

    Args:
        index: 1-based experiment index.
        total: Total experiments in study.
        config: ExperimentConfig for this experiment.
        status: "running", "completed", or "failed".
        elapsed: Elapsed time in seconds (None if not yet available).
        energy: Energy in joules (None if not yet available).
    """
    icons = {"running": "...", "completed": "OK", "failed": "FAIL"}
    icon = icons.get(status, "?")

    parts = [f"[{index}/{total}]", icon, config.task.model, config.engine, config.dtype]

    if elapsed is not None:
        parts.append("--")
        parts.append(_format_duration(elapsed))
    if energy is not None:
        parts.append(f"({_sig3(energy)} J)")

    line = " ".join(parts)
    print(line, file=sys.stderr)
