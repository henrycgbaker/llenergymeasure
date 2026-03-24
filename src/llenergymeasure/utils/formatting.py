"""Shared formatting utilities (Layer 0 — importable from any layer)."""

from __future__ import annotations


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
