"""Pydantic v2 model → ``RuleCandidate[]`` lift.

Sub-library type-system lift consumed by per-engine miners. Walks
:class:`pydantic.BaseModel` subclasses (or ``pydantic.dataclasses.dataclass``-decorated
classes) and emits one rule candidate per ``annotated-types`` constraint or
``Literal[...]`` allowlist found on a field.

This is the Tier-1 adoption from the locked design's §5: ``model_json_schema()``
and ``FieldInfo.metadata`` are well-defined Pydantic v2 surfaces, so the lift
is deterministic, has no probing, and emits a predictable rule shape per
field constraint.

Used by:

- vLLM ``vllm.config.*`` (~27 pydantic-dataclasses with rich constraint metadata).
- TensorRT-LLM ``TrtLlmArgs`` (Pydantic-v2 root, schema-poor — most bounds live
  in validator AST, but Literal-typed enum fields are picked up here).

Operator vocabulary
-------------------
The lift maps ``annotated-types`` predicate types directly to
:class:`RuleCandidate.match_fields` operator strings, aligning the corpus with
the standard library:

- ``Gt``  → ``">"``
- ``Ge``  → ``">="``
- ``Lt``  → ``"<"``
- ``Le``  → ``"<="``
- ``MultipleOf`` → ``"multiple_of"``
- ``MinLen`` → ``"min_len"``
- ``MaxLen`` → ``"max_len"``

For ``Literal[a, b, c]`` annotations the lift emits a value-allowlist rule
with ``match_fields[<field>] == {"in": [a, b, c]}``.

Determinism
-----------
No randomness, no library probing, no time-based seeds — output is a pure
function of (target type, namespace, today, source path). See
``feedback_miner_pipeline_deterministic.md``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Literal, get_args, get_origin

import annotated_types as at

from scripts.miners._base import MinerSource, RuleCandidate

# ---------------------------------------------------------------------------
# Operator mapping (annotated-types -> match_fields key)
# ---------------------------------------------------------------------------


_NUMERIC_OPS: tuple[tuple[type, str, str], ...] = (
    (at.Gt, ">", "gt"),
    (at.Ge, ">=", "ge"),
    (at.Lt, "<", "lt"),
    (at.Le, "<=", "le"),
    (at.MultipleOf, "multiple_of", "multiple_of"),
)

_LENGTH_OPS: tuple[tuple[type, str, str], ...] = (
    (at.MinLen, "min_len", "min_length"),
    (at.MaxLen, "max_len", "max_length"),
)


def _violates_numeric(op_key: str, threshold: Any) -> Any:
    """Return a value that violates the numeric predicate ``op_key threshold``.

    Used to populate ``kwargs_positive`` (the kwargs that *should* trip the rule).
    The complement value populates ``kwargs_negative``.
    """
    if op_key == ">":
        return threshold  # x > T fails when x == T
    if op_key == ">=":
        return threshold - 1 if isinstance(threshold, int) else threshold - 1.0
    if op_key == "<":
        return threshold  # x < T fails when x == T
    if op_key == "<=":
        return threshold + 1 if isinstance(threshold, int) else threshold + 1.0
    if op_key == "multiple_of":
        return threshold + 1 if isinstance(threshold, int) else threshold + 0.5
    raise ValueError(f"Unknown numeric op {op_key!r}")


def _satisfies_numeric(op_key: str, threshold: Any) -> Any:
    """Return a value that satisfies the numeric predicate ``op_key threshold``."""
    if op_key == ">":
        return threshold + 1 if isinstance(threshold, int) else threshold + 1.0
    if op_key == ">=":
        return threshold
    if op_key == "<":
        return threshold - 1 if isinstance(threshold, int) else threshold - 1.0
    if op_key == "<=":
        return threshold
    if op_key == "multiple_of":
        return threshold * 2 if isinstance(threshold, (int, float)) else threshold
    raise ValueError(f"Unknown numeric op {op_key!r}")


# ---------------------------------------------------------------------------
# Field iteration
# ---------------------------------------------------------------------------


def _iter_model_fields(target_type: type) -> Iterable[tuple[str, Any]]:
    """Yield ``(field_name, FieldInfo)`` pairs for a Pydantic model.

    Supports both ``pydantic.BaseModel`` subclasses and
    ``pydantic.dataclasses.dataclass``-decorated classes (which expose the
    same ``__pydantic_fields__`` attribute under v2).
    """
    fields = getattr(target_type, "model_fields", None)
    if fields is None:
        # pydantic.dataclasses surface (v2)
        fields = getattr(target_type, "__pydantic_fields__", None)
    if fields is None:
        return
    yield from fields.items()


def _extract_literal_values(annotation: Any) -> tuple[Any, ...] | None:
    """Return Literal values if ``annotation`` is a (possibly nested) Literal.

    Handles bare ``Literal[...]``, ``Optional[Literal[...]]``, and
    ``Annotated[Literal[...], ...]`` shapes. Returns ``None`` when no Literal
    is present.
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    # Look one level into Annotated / Union / Optional
    if origin is not None:
        for arg in get_args(annotation):
            inner = _extract_literal_values(arg)
            if inner is not None:
                return inner
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lift(
    target_type: type,
    *,
    namespace: str,
    today: str,
    source_path: str,
) -> list[RuleCandidate]:
    """Extract validation-rule candidates from ``target_type`` via Pydantic v2 introspection.

    Parameters
    ----------
    target_type:
        A ``pydantic.BaseModel`` subclass or ``pydantic.dataclasses.dataclass``
        class. Other types yield an empty list.
    namespace:
        Engine field-namespace prefix used in ``match_fields``
        (e.g. ``"vllm.cache"`` → emits rules keyed
        ``"vllm.cache.<field>"``).
    today:
        ISO-8601 date string used for the rule's ``added_at`` field.
    source_path:
        Source path recorded on each rule's :class:`MinerSource`. Typically
        the relative path within ``site-packages/`` for reproducibility.

    Returns
    -------
    list[RuleCandidate]
        One candidate per ``annotated-types`` constraint or ``Literal[...]``
        allowlist found on a field. Empty if ``target_type`` exposes no
        Pydantic field metadata.
    """
    candidates: list[RuleCandidate] = []
    library = target_type.__module__.split(".", 1)[0]
    type_name = target_type.__name__
    method = "<pydantic_lift>"

    try:
        line = inspect.getsourcelines(target_type)[1]
    except (OSError, TypeError):
        line = 0

    for field_name, info in _iter_model_fields(target_type):
        for meta in getattr(info, "metadata", ()) or ():
            cand = _from_numeric(meta, field_name, target_type, namespace)
            if cand is not None:
                candidates.append(
                    _build(cand, library, type_name, method, line, source_path, today)
                )
                continue
            cand = _from_length(meta, field_name, target_type, namespace)
            if cand is not None:
                candidates.append(
                    _build(cand, library, type_name, method, line, source_path, today)
                )

        literal_values = _extract_literal_values(info.annotation)
        if literal_values:
            cand = _from_literal(literal_values, field_name, target_type, namespace)
            candidates.append(_build(cand, library, type_name, method, line, source_path, today))

    return candidates


# ---------------------------------------------------------------------------
# Per-shape helpers
# ---------------------------------------------------------------------------


def _from_numeric(
    meta: Any, field_name: str, target_type: type, namespace: str
) -> dict[str, Any] | None:
    """Build a partial rule dict from an ``annotated-types`` numeric constraint."""
    for cls, op_key, attr in _NUMERIC_OPS:
        if isinstance(meta, cls):
            threshold = getattr(meta, attr)
            return {
                "id_suffix": f"{field_name}_{attr}_{_slug(threshold)}",
                "rule_under_test": (
                    f"{target_type.__name__}.{field_name} requires {op_key} {threshold!r}"
                ),
                "match_fields": {f"{namespace}.{field_name}": {op_key: threshold}},
                "kwargs_positive": {field_name: _violates_numeric(op_key, threshold)},
                "kwargs_negative": {field_name: _satisfies_numeric(op_key, threshold)},
                "message_template": (f"`{field_name}` must satisfy {op_key} {threshold!r}"),
            }
    return None


def _from_length(
    meta: Any, field_name: str, target_type: type, namespace: str
) -> dict[str, Any] | None:
    """Build a partial rule dict from a length constraint (``MinLen``/``MaxLen``)."""
    for cls, op_key, attr in _LENGTH_OPS:
        if isinstance(meta, cls):
            threshold = getattr(meta, attr)
            is_min = op_key == "min_len"
            # ``kwargs_positive`` is the value that *violates* the constraint.
            # For min_len that's an empty string; for max_len that's a string
            # one longer than the threshold.
            bad = "" if is_min else "x" * (threshold + 1)
            good = "x" * threshold
            return {
                "id_suffix": f"{field_name}_{op_key}_{threshold}",
                "rule_under_test": (
                    f"{target_type.__name__}.{field_name} requires {op_key} {threshold}"
                ),
                "match_fields": {f"{namespace}.{field_name}": {op_key: threshold}},
                "kwargs_positive": {field_name: bad},
                "kwargs_negative": {field_name: good},
                "message_template": (f"`{field_name}` length must satisfy {op_key} {threshold}"),
            }
    return None


def _from_literal(
    values: tuple[Any, ...], field_name: str, target_type: type, namespace: str
) -> dict[str, Any]:
    """Build a partial rule dict from a ``Literal[...]`` annotation."""
    sample_invalid = "<invalid_pydantic_lift_probe>"
    sample_valid = values[0]
    return {
        "id_suffix": f"{field_name}_in_{len(values)}_values",
        "rule_under_test": (f"{target_type.__name__}.{field_name} must be one of {list(values)!r}"),
        "match_fields": {f"{namespace}.{field_name}": {"in": list(values)}},
        "kwargs_positive": {field_name: sample_invalid},
        "kwargs_negative": {field_name: sample_valid},
        "message_template": (f"`{field_name}` must be one of {list(values)!r}"),
    }


def _build(
    partial: dict[str, Any],
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    today: str,
) -> RuleCandidate:
    """Wrap a partial-dict from one of the ``_from_*`` helpers in a RuleCandidate."""
    rid = f"{library}_{type_name.lower()}_{partial['id_suffix']}"
    return RuleCandidate(
        id=rid,
        engine=library,
        library=library,
        rule_under_test=partial["rule_under_test"],
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields=partial["match_fields"],
        kwargs_positive=partial["kwargs_positive"],
        kwargs_negative=partial["kwargs_negative"],
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=partial["message_template"],
        references=[f"{library}.{type_name} — Pydantic FieldInfo metadata"],
        added_by="pydantic_lift",
        added_at=today,
    )


def _slug(value: Any) -> str:
    """Stable, filesystem-safe slug for embedding numeric thresholds in rule ids."""
    return str(value).replace("-", "neg").replace(".", "p").replace(" ", "_").replace("/", "_")


__all__ = ["lift"]
