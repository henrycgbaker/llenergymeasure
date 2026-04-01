"""Config resolution log — tracks provenance of each non-default experiment field.

Produces a per-experiment ``_resolution.json`` sidecar that records which config
values were overridden relative to Pydantic defaults, and *why* (CLI flag, sweep
expansion, or YAML file).

Usage::

    log = build_resolution_log(config, cli_overrides={"dtype": "float16"}, swept_fields={"model"})
    # -> {"schema_version": "1.0", "overrides": {"dtype": {...}, "model": {...}}}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_resolution_log(
    config_dict: dict[str, Any],
    *,
    cli_overrides: dict[str, Any] | None = None,
    swept_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Build a resolution log for a single experiment config.

    Compares the effective config against Pydantic defaults and labels each
    non-default field with its source: ``cli_flag``, ``sweep``, or ``yaml``.

    Args:
        config_dict: Fully resolved experiment config as a dict (from model_dump()).
        cli_overrides: Flat CLI override keys (e.g. {"model": "gpt2", "dataset.n_prompts": 100}).
            None values are ignored.
        swept_fields: Dotted field paths that vary across experiments (from get_swept_field_paths).

    Returns:
        Resolution log dict with ``schema_version`` and ``overrides``.
    """
    from llenergymeasure.config.models import ExperimentConfig

    flat_effective = _flatten_dict(config_dict)
    flat_defaults = _get_defaults_flat(ExperimentConfig)
    cli_keys = _normalise_cli_keys(cli_overrides)
    swept = swept_fields or set()

    overrides: dict[str, dict[str, Any]] = {}
    for key, value in sorted(flat_effective.items()):
        # Skip None values (unset optional sub-configs)
        if value is None:
            continue

        # Skip if matches Pydantic default
        if key in flat_defaults and _values_equal(value, flat_defaults[key]):
            continue

        # Determine source (priority: cli > sweep > yaml)
        if key in cli_keys:
            source = "cli_flag"
        elif key in swept:
            source = "sweep"
        else:
            source = "yaml"

        entry: dict[str, Any] = {"effective": value, "source": source}
        if key in flat_defaults:
            entry["default"] = flat_defaults[key]

        overrides[key] = entry

    return {"schema_version": "1.0", "overrides": overrides}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted keys, skipping None sub-dicts."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{key}."))
        elif isinstance(v, list):
            # Store lists as-is (e.g. gpu_indices, measurement_warnings)
            items[key] = v
        else:
            items[key] = v
    return items


def _get_defaults_flat(model_cls: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    """Extract flattened Pydantic defaults from a model, recursing into sub-models."""
    defaults: dict[str, Any] = {}
    for name, field_info in model_cls.model_fields.items():
        key = f"{prefix}{name}" if prefix else name

        # Get default value
        if field_info.default is not PydanticUndefined:
            val = field_info.default
        elif field_info.default_factory is not None:
            val = field_info.default_factory()
        else:
            continue  # Required field (no default) — any value is explicitly set

        # Recurse into BaseModel sub-config defaults
        if isinstance(val, BaseModel):
            defaults.update(_get_defaults_flat(type(val), f"{key}."))
        else:
            defaults[key] = val

    return defaults


def _normalise_cli_keys(cli_overrides: dict[str, Any] | None) -> set[str]:
    """Extract non-None CLI override keys as a flat set."""
    if not cli_overrides:
        return set()
    return {k for k, v in cli_overrides.items() if v is not None}


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values for equality, handling common edge cases."""
    # Handle list comparison (order-sensitive)
    if isinstance(a, list) and isinstance(b, list):
        return a == b
    return a == b
