"""Per-experiment progress display for study runs.

Moved from cli/_display.py to break the study -> cli layering violation.
The study layer must not import from cli.
"""

from __future__ import annotations

import math
import sys

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.utils.formatting import format_elapsed as _format_duration


def _sig3(value: float) -> str:
    """Format a float to 3 significant figures.

    Examples:
        312.4  -> "312"
        3.12   -> "3.12"
        0.00312 -> "0.00312"
        847.0  -> "847"
        0      -> "0"
        1234   -> "1230"
    """
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    # Number of decimal places needed for 3 sig figs
    decimal_places = max(0, 2 - magnitude)
    rounded = round(value, decimal_places - int(magnitude >= 3) * 0)
    # Recompute for clarity: round to 3 sig figs
    factor = 10 ** (magnitude - 2)
    rounded = round(value / factor) * factor
    # Format without trailing zeros
    if decimal_places <= 0:
        return str(int(rounded))
    formatted = f"{rounded:.{decimal_places}f}"
    # Strip trailing zeros after decimal point
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def print_study_progress(
    index: int,
    total: int,
    config: ExperimentConfig,
    status: str = "running",
    elapsed: float | None = None,
    energy: float | None = None,
) -> None:
    """Print a per-experiment progress line to stderr.

    Format: [3/12] <icon> model backend precision -- elapsed (energy)
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

    parts = [f"[{index}/{total}]", icon, config.model, config.backend, config.precision]

    if elapsed is not None:
        parts.append("--")
        parts.append(_format_duration(elapsed))
    if energy is not None:
        parts.append(f"({_sig3(energy)} J)")

    line = " ".join(parts)
    print(line, file=sys.stderr)
