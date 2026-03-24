"""Shared formatting utilities (Layer 0 — importable from any layer)."""

from __future__ import annotations

import math


def format_elapsed(seconds: float) -> str:
    """Format seconds as human-readable elapsed time.

    Examples:
        4.2  -> "4.2s"
        272  -> "4m 32s"
        3900 -> "1h 05m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


_DETAIL_MAX_LEN = 34
"""Maximum detail string length before truncation (80-col terminal safe)."""


def truncate_detail(detail: str) -> str:
    """Truncate detail text to fit 80-column terminals."""
    if len(detail) > _DETAIL_MAX_LEN:
        return detail[: _DETAIL_MAX_LEN - 3] + "..."
    return detail


def sig3(value: float) -> str:
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
