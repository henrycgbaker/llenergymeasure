"""Shared dictionary utilities: unflatten dotted keys, deep merge.

Canonical home for these utilities, imported by config/loader.py
and config/grid.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dicts. Non-dotted keys pass through.

    Example:
        {"engine.block_size": 16} -> {"engine": {"block_size": 16}}
        {"batch_size": 4}         -> {"batch_size": 4}
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        if "." in key:
            parts = key.split(".", 1)
            if parts[0] not in result:
                result[parts[0]] = {}
            if isinstance(result[parts[0]], dict):
                result[parts[0]][parts[1]] = value
        else:
            result[key] = value
    return result


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. overlay values take precedence over base values."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
