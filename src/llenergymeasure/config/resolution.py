"""Config resolution log -- tracks provenance, semantic context, and effective defaults.

Produces a per-experiment ``_resolution.json`` sidecar that records:

1. **overrides** -- which config values differ from Pydantic defaults and why
   (CLI flag, sweep expansion, or YAML file).
2. **semantic_context** -- which fields are active vs dormant based on controlling
   field values (e.g. convergence_detection controls warmup mode).
3. **defaults_in_effect** -- significant default values that actively control behaviour.

Usage::

    log = build_resolution_log(config, cli_overrides={"dtype": "float16"}, swept_fields={"model"})
    # -> {"schema_version": "1.0", "overrides": {...}, "semantic_context": {...}, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Semantic rule registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticRule:
    """A rule mapping a controlling field to its active/dormant dependents.

    Each rule defines one or more conditions on a controlling field. When a
    condition matches, the rule specifies which fields become active and which
    become dormant. The ``note`` gives a human-readable explanation.

    For simple boolean/value-based rules, use ``conditions`` -- a list of
    ``SemanticCondition`` objects, each describing one mode.
    """

    name: str
    controlling_field: str
    conditions: list[SemanticCondition]


@dataclass(frozen=True)
class SemanticCondition:
    """One mode within a semantic rule."""

    mode: str
    match: _ConditionMatch
    active_fields: list[str]
    dormant_fields: list[str]
    note: str


# Condition matchers
@dataclass(frozen=True)
class _ValueEquals:
    """Match when the controlling field equals a specific value."""

    value: Any


@dataclass(frozen=True)
class _ValueGreaterThan:
    """Match when the controlling field is greater than a threshold."""

    threshold: int | float


@dataclass(frozen=True)
class _ValueIsSet:
    """Match when the controlling field is not None."""


@dataclass(frozen=True)
class _ValueIsNotSet:
    """Match when the controlling field is None or not present."""


# Union type for condition matching
_ConditionMatch = _ValueEquals | _ValueGreaterThan | _ValueIsSet | _ValueIsNotSet


def _check_condition(match: _ConditionMatch, value: Any) -> bool:
    """Evaluate whether a condition matches the given value."""
    if isinstance(match, _ValueEquals):
        return bool(value == match.value)
    if isinstance(match, _ValueGreaterThan):
        return value is not None and bool(value > match.threshold)
    if isinstance(match, _ValueIsSet):
        return value is not None
    if isinstance(match, _ValueIsNotSet):
        return value is None
    return False  # pragma: no cover


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

SEMANTIC_RULES: list[SemanticRule] = [
    # 1. Warmup mode
    SemanticRule(
        name="warmup_mode",
        controlling_field="warmup.convergence_detection",
        conditions=[
            SemanticCondition(
                mode="fixed",
                match=_ValueEquals(value=False),
                active_fields=["warmup.n_warmup"],
                dormant_fields=[
                    "warmup.cv_threshold",
                    "warmup.max_prompts",
                    "warmup.window_size",
                    "warmup.min_prompts",
                ],
                note="Fixed warmup: runs exactly n_warmup iterations. CV computed for info only.",
            ),
            SemanticCondition(
                mode="convergence",
                match=_ValueEquals(value=True),
                active_fields=[
                    "warmup.cv_threshold",
                    "warmup.max_prompts",
                    "warmup.window_size",
                    "warmup.min_prompts",
                ],
                dormant_fields=[],
                note=(
                    "CV-based warmup: runs until convergence or max_prompts. "
                    "n_warmup is minimum before CV checking begins."
                ),
            ),
        ],
    ),
    # 2. Sampling mode
    SemanticRule(
        name="sampling_mode",
        controlling_field="decoder.do_sample",
        conditions=[
            SemanticCondition(
                mode="greedy",
                match=_ValueEquals(value=False),
                active_fields=[],
                dormant_fields=[
                    "decoder.temperature",
                    "decoder.top_k",
                    "decoder.top_p",
                    "decoder.min_p",
                ],
                note="Greedy/beam decoding: sampling parameters have no effect.",
            ),
            SemanticCondition(
                mode="sampling",
                match=_ValueEquals(value=True),
                active_fields=[
                    "decoder.temperature",
                    "decoder.top_k",
                    "decoder.top_p",
                    "decoder.min_p",
                ],
                dormant_fields=[],
                note="Stochastic sampling active: temperature/top_k/top_p/min_p control distribution.",
            ),
        ],
    ),
    # 3. PyTorch 4-bit quantisation
    SemanticRule(
        name="quantization_mode_4bit",
        controlling_field="pytorch.load_in_4bit",
        conditions=[
            SemanticCondition(
                mode="4bit",
                match=_ValueEquals(value=True),
                active_fields=[
                    "pytorch.bnb_4bit_compute_dtype",
                    "pytorch.bnb_4bit_quant_type",
                    "pytorch.bnb_4bit_use_double_quant",
                ],
                dormant_fields=["pytorch.load_in_8bit"],
                note="4-bit quantisation active; BNB sub-params are effective.",
            ),
        ],
    ),
    # 3b. PyTorch 8-bit quantisation
    SemanticRule(
        name="quantization_mode_8bit",
        controlling_field="pytorch.load_in_8bit",
        conditions=[
            SemanticCondition(
                mode="8bit",
                match=_ValueEquals(value=True),
                active_fields=[],
                dormant_fields=[
                    "pytorch.load_in_4bit",
                    "pytorch.bnb_4bit_compute_dtype",
                    "pytorch.bnb_4bit_quant_type",
                    "pytorch.bnb_4bit_use_double_quant",
                ],
                note="8-bit quantisation active; 4-bit params are dormant.",
            ),
        ],
    ),
    # 4. torch.compile
    SemanticRule(
        name="torch_compile_mode",
        controlling_field="pytorch.torch_compile",
        conditions=[
            SemanticCondition(
                mode="disabled",
                match=_ValueEquals(value=False),
                active_fields=[],
                dormant_fields=[
                    "pytorch.torch_compile_backend",
                    "pytorch.torch_compile_mode",
                ],
                note="torch.compile disabled; compile backend/mode have no effect.",
            ),
            SemanticCondition(
                mode="enabled",
                match=_ValueEquals(value=True),
                active_fields=[
                    "pytorch.torch_compile_backend",
                    "pytorch.torch_compile_mode",
                ],
                dormant_fields=[],
                note="torch.compile enabled; backend and mode control compilation.",
            ),
        ],
    ),
    # 5. KV cache
    SemanticRule(
        name="kv_cache_mode",
        controlling_field="pytorch.use_cache",
        conditions=[
            SemanticCondition(
                mode="disabled",
                match=_ValueEquals(value=False),
                active_fields=[],
                dormant_fields=["pytorch.cache_implementation"],
                note="KV cache disabled; cache_implementation has no effect.",
            ),
            SemanticCondition(
                mode="enabled",
                match=_ValueEquals(value=True),
                active_fields=["pytorch.cache_implementation"],
                dormant_fields=[],
                note="KV cache enabled; cache_implementation controls cache strategy.",
            ),
        ],
    ),
    # 6. Beam search (PyTorch)
    SemanticRule(
        name="beam_search_mode",
        controlling_field="pytorch.num_beams",
        conditions=[
            SemanticCondition(
                mode="beam_search",
                match=_ValueGreaterThan(threshold=1),
                active_fields=[
                    "pytorch.early_stopping",
                    "pytorch.length_penalty",
                    "pytorch.no_repeat_ngram_size",
                ],
                dormant_fields=[],
                note="Beam search active; beam-specific parameters control search behaviour.",
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Significant defaults -- fields whose default value actively controls behaviour
# ---------------------------------------------------------------------------

#: Mapping of field path -> significance description for important defaults.
SIGNIFICANT_DEFAULTS: dict[str, str] = {
    "warmup.convergence_detection": (
        "Controls warmup mode: false=fixed (n_warmup), true=CV-based (max_prompts)"
    ),
    "decoder.do_sample": ("Enables stochastic sampling; temperature/top_k/top_p are active"),
    "warmup.enabled": "Controls whether warmup phase runs at all",
    "baseline.enabled": "Controls whether baseline power measurement runs",
    "dataset.order": ("Prompt ordering strategy: interleaved (round-robin), grouped, or shuffled"),
}


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
    Also evaluates semantic rules and identifies significant defaults.

    Args:
        config_dict: Fully resolved experiment config as a dict (from model_dump()).
        cli_overrides: Flat CLI override keys (e.g. {"model": "gpt2", "dataset.n_prompts": 100}).
            None values are ignored.
        swept_fields: Dotted field paths that vary across experiments (from get_swept_field_paths).

    Returns:
        Resolution log dict with ``schema_version``, ``overrides``,
        ``semantic_context``, and ``defaults_in_effect``.
    """
    from llenergymeasure.config.models import ExperimentConfig

    flat_effective = _flatten_dict(config_dict)
    flat_defaults = _get_defaults_flat(ExperimentConfig)
    cli_keys = _normalise_cli_keys(cli_overrides)
    swept = swept_fields or set()

    overrides = _build_overrides(flat_effective, flat_defaults, cli_keys, swept)
    semantic_context = _build_semantic_context(flat_effective)
    defaults_in_effect = _build_defaults_in_effect(flat_effective, flat_defaults)

    return {
        "schema_version": "1.0",
        "overrides": overrides,
        "semantic_context": semantic_context,
        "defaults_in_effect": defaults_in_effect,
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_overrides(
    flat_effective: dict[str, Any],
    flat_defaults: dict[str, Any],
    cli_keys: set[str],
    swept: set[str],
) -> dict[str, dict[str, Any]]:
    """Build the overrides section (unchanged from v1.0 logic)."""
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

    return overrides


def _build_semantic_context(
    flat_effective: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate semantic rules against the effective config.

    Returns a dict keyed by rule name, containing mode, controlling field info,
    active/dormant field lists, and a human-readable note.
    """
    context: dict[str, dict[str, Any]] = {}

    for rule in SEMANTIC_RULES:
        controlling_value = flat_effective.get(rule.controlling_field)

        for condition in rule.conditions:
            if _check_condition(condition.match, controlling_value):
                context[rule.name] = {
                    "mode": condition.mode,
                    "controlling_field": rule.controlling_field,
                    "controlling_value": controlling_value,
                    "active_fields": condition.active_fields,
                    "dormant_fields": condition.dormant_fields,
                    "note": condition.note,
                }
                break  # First matching condition wins

    return context


def _build_defaults_in_effect(
    flat_effective: dict[str, Any],
    flat_defaults: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Identify significant default values that are actively controlling behaviour.

    Only includes fields that (a) are at their Pydantic default and (b) are in the
    SIGNIFICANT_DEFAULTS registry.
    """
    defaults_section: dict[str, dict[str, Any]] = {}

    for field_path, significance in SIGNIFICANT_DEFAULTS.items():
        if field_path not in flat_effective:
            continue
        effective_value = flat_effective[field_path]

        # Only include if the field is at its default value
        if field_path in flat_defaults and _values_equal(
            effective_value, flat_defaults[field_path]
        ):
            defaults_section[field_path] = {
                "value": effective_value,
                "significance": significance,
            }

    return defaults_section


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
            val = field_info.default_factory()  # type: ignore[call-arg]
        else:
            continue  # Required field (no default) -- any value is explicitly set

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
    return bool(a == b)
