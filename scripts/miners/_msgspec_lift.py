"""msgspec ``Struct`` → ``RuleCandidate[]`` lift.

Sub-library type-system lift consumed by per-engine miners. Walks
:class:`msgspec.Struct` subclasses via :func:`msgspec.inspect.type_info` and
emits one rule candidate per ``Meta(ge=, le=, ...)`` constraint or
``Literal[...]`` allowlist.

Used by:

- vLLM ``SamplingParams`` (msgspec.Struct). Per
  ``research-vllm-extractor.md``, the live class ships **zero**
  ``Meta`` annotations — that's expected; this lift returns ``[]`` and the
  static miner picks up the slack via AST parsing of ``_verify_args``.
  The lift exists so the day vLLM (or another msgspec user) starts
  annotating ``Meta(ge=...)`` we capture it for free.

Determinism
-----------
No randomness, no library probing, no time-based seeds. Per
``feedback_miner_pipeline_deterministic.md``.
"""

from __future__ import annotations

import inspect
from typing import Any

import msgspec

from scripts.miners._base import MinerSource, RuleCandidate

# msgspec maps annotated-types-style constraints onto these attribute names
# on its ``Constraints`` info object. We mirror the pydantic lift's op-key
# vocabulary so the corpus stays uniform across lifts.
_NUMERIC_OPS: tuple[tuple[str, str], ...] = (
    ("gt", ">"),
    ("ge", ">="),
    ("lt", "<"),
    ("le", "<="),
    ("multiple_of", "multiple_of"),
)

_LENGTH_OPS: tuple[tuple[str, str], ...] = (
    ("min_length", "min_len"),
    ("max_length", "max_len"),
)


def _violates_numeric(op_key: str, threshold: Any) -> Any:
    if op_key == ">":
        return threshold
    if op_key == ">=":
        return threshold - 1 if isinstance(threshold, int) else threshold - 1.0
    if op_key == "<":
        return threshold
    if op_key == "<=":
        return threshold + 1 if isinstance(threshold, int) else threshold + 1.0
    if op_key == "multiple_of":
        return threshold + 1 if isinstance(threshold, int) else threshold + 0.5
    raise ValueError(f"Unknown numeric op {op_key!r}")


def _satisfies_numeric(op_key: str, threshold: Any) -> Any:
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


def lift(
    target_type: type,
    *,
    namespace: str,
    today: str,
    source_path: str,
) -> list[RuleCandidate]:
    """Extract validation-rule candidates from a ``msgspec.Struct`` subclass.

    Parameters
    ----------
    target_type:
        A subclass of :class:`msgspec.Struct`. Other types yield an empty list.
    namespace:
        Engine field-namespace prefix used in ``match_fields``.
    today:
        ISO-8601 date string for ``added_at``.
    source_path:
        Source-file path recorded on each rule's :class:`MinerSource`.

    Returns
    -------
    list[RuleCandidate]
        One candidate per ``Meta(...)`` numeric/length constraint or per
        ``Literal[...]`` allowlist. Empty if the struct has no annotations.
    """
    if not (isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)):
        return []

    library = target_type.__module__.split(".", 1)[0]
    type_name = target_type.__name__
    method = "<msgspec_lift>"
    try:
        line = inspect.getsourcelines(target_type)[1]
    except (OSError, TypeError):
        line = 0

    info = msgspec.inspect.type_info(target_type)
    if not isinstance(info, msgspec.inspect.StructType):
        return []

    candidates: list[RuleCandidate] = []
    for field in info.fields:
        candidates.extend(
            _candidates_for_field(
                field,
                library=library,
                type_name=type_name,
                method=method,
                line=line,
                source_path=source_path,
                namespace=namespace,
                today=today,
            )
        )
    return candidates


def _candidates_for_field(
    field: msgspec.inspect.Field,
    *,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> list[RuleCandidate]:
    """Emit candidates for one ``msgspec`` struct field's type info."""
    out: list[RuleCandidate] = []
    field_type = field.type
    field_name = field.name

    # Numeric / length constraints sit on IntType, FloatType, StrType, etc.
    for attr, op_key in _NUMERIC_OPS:
        threshold = getattr(field_type, attr, None)
        if threshold is None:
            continue
        out.append(
            _build_numeric(
                field_name,
                op_key,
                threshold,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    for attr, op_key in _LENGTH_OPS:
        threshold = getattr(field_type, attr, None)
        if threshold is None:
            continue
        out.append(
            _build_length(
                field_name,
                op_key,
                threshold,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    # Literal allowlists — msgspec exposes these as LiteralType with .values.
    if isinstance(field_type, msgspec.inspect.LiteralType):
        out.append(
            _build_literal(
                field_name,
                field_type.values,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    return out


def _build_numeric(
    field_name: str,
    op_key: str,
    threshold: Any,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> RuleCandidate:
    """Materialise a numeric-constraint rule candidate."""
    return RuleCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_{op_key.replace('>', 'gt').replace('<', 'lt').replace('=', 'e')}_{_slug(threshold)}",
        engine=library,
        library=library,
        rule_under_test=(f"{type_name}.{field_name} requires {op_key} {threshold!r}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {op_key: threshold}},
        kwargs_positive={field_name: _violates_numeric(op_key, threshold)},
        kwargs_negative={field_name: _satisfies_numeric(op_key, threshold)},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` must satisfy {op_key} {threshold!r}",
        references=[f"{library}.{type_name} — msgspec.Meta constraint"],
        added_by="msgspec_lift",
        added_at=today,
    )


def _build_length(
    field_name: str,
    op_key: str,
    threshold: int,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> RuleCandidate:
    """Materialise a length-constraint rule candidate."""
    bad = "" if op_key == "min_len" else "x" * (threshold + 1)
    good = "x" * (threshold if op_key == "min_len" else threshold)
    return RuleCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_{op_key}_{threshold}",
        engine=library,
        library=library,
        rule_under_test=(f"{type_name}.{field_name} length must satisfy {op_key} {threshold}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {op_key: threshold}},
        kwargs_positive={field_name: bad},
        kwargs_negative={field_name: good},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` length must satisfy {op_key} {threshold}",
        references=[f"{library}.{type_name} — msgspec.Meta length constraint"],
        added_by="msgspec_lift",
        added_at=today,
    )


def _build_literal(
    field_name: str,
    values: tuple[Any, ...],
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> RuleCandidate:
    """Materialise a Literal-allowlist rule candidate."""
    return RuleCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_in_{len(values)}_values",
        engine=library,
        library=library,
        rule_under_test=(f"{type_name}.{field_name} must be one of {list(values)!r}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {"in": list(values)}},
        kwargs_positive={field_name: "<invalid_msgspec_lift_probe>"},
        kwargs_negative={field_name: values[0]},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` must be one of {list(values)!r}",
        references=[f"{library}.{type_name} — msgspec Literal annotation"],
        added_by="msgspec_lift",
        added_at=today,
    )


def _slug(value: Any) -> str:
    return str(value).replace("-", "neg").replace(".", "p").replace(" ", "_").replace("/", "_")


__all__ = ["lift"]
