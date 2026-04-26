"""``@dataclasses.dataclass`` → ``RuleCandidate[]`` lift.

Sub-library type-system lift consumed by per-engine miners. Walks
:func:`dataclasses.fields` and emits one rule candidate per
``Literal[...]`` allowlist found on a field's type annotation.

Used by:

- transformers ``GenerationConfig`` (when applicable) + ``BitsAndBytesConfig``
- vLLM ``EngineArgs`` (~175-field stdlib dataclass)
- TensorRT-LLM ``BuildConfig`` + ``QuantConfig``

Coverage scope
--------------
Plain ``@dataclasses.dataclass`` carries **no numeric-bound metadata** by
default — bounds aren't part of the dataclass spec. The only structural axis
the lift can derive from a stdlib dataclass is:

- ``Literal[a, b, c]`` annotations → value-allowlist rules.
- The annotated type itself (``int``, ``str``, ``Path``, …) → no rule
  emitted in this lift; type-check rules require library-side runtime checks
  the lift cannot mechanically derive (and stdlib dataclasses don't enforce
  type annotations at construction time).

If a target dataclass mixes in ``annotated-types`` constraints via
``Annotated[int, Gt(0)]`` (rare but valid), this lift treats them as opaque
type metadata — those constraints belong in the ``_pydantic_lift`` /
``_msgspec_lift`` paths if the library uses one of those frameworks.

Determinism
-----------
No randomness, no probing, no time-based seeds. Per
``feedback_miner_pipeline_deterministic.md``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Literal, get_args, get_origin, get_type_hints

from scripts.miners._base import MinerSource, RuleCandidate


def _extract_literal_values(annotation: Any) -> tuple[Any, ...] | None:
    """Return Literal values if ``annotation`` is a (possibly nested) Literal.

    Handles ``Literal[...]`` directly plus one level of containment
    (``Optional[Literal[...]]``, ``Annotated[Literal[...], ...]``,
    ``Union[Literal[...], None]``).
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            inner = _extract_literal_values(arg)
            if inner is not None:
                return inner
    return None


def lift(
    target_type: type,
    *,
    namespace: str,
    today: str,
    source_path: str,
) -> list[RuleCandidate]:
    """Extract validation-rule candidates from a ``@dataclasses.dataclass`` class.

    Parameters
    ----------
    target_type:
        A class decorated with ``@dataclasses.dataclass``. Other types yield
        an empty list.
    namespace:
        Engine field-namespace prefix used in ``match_fields``.
    today:
        ISO-8601 date string for ``added_at``.
    source_path:
        Source-file path recorded on each rule's :class:`MinerSource`.

    Returns
    -------
    list[RuleCandidate]
        One candidate per field whose type annotation contains a
        ``Literal[...]``. Empty if ``target_type`` is not a dataclass or has
        no Literal-typed fields.
    """
    if not dataclasses.is_dataclass(target_type):
        return []

    library = target_type.__module__.split(".", 1)[0]
    type_name = target_type.__name__
    method = "<dataclass_lift>"
    try:
        line = inspect.getsourcelines(target_type)[1]
    except (OSError, TypeError):
        line = 0

    # Use ``get_type_hints`` so string annotations from
    # ``from __future__ import annotations`` resolve to real types. ``field.type``
    # alone is a string under PEP 563 — ``Literal[...]`` would not parse.
    try:
        hints = get_type_hints(target_type, include_extras=True)
    except Exception:
        # Resolution can fail when forward references reference types only
        # available in TYPE_CHECKING blocks. Fall back to the raw string and
        # let ``_extract_literal_values`` no-op on it.
        hints = {f.name: f.type for f in dataclasses.fields(target_type)}

    candidates: list[RuleCandidate] = []
    for field in dataclasses.fields(target_type):
        annotation = hints.get(field.name, field.type)
        values = _extract_literal_values(annotation)
        if not values:
            continue
        candidates.append(
            _build_literal(
                field.name,
                values,
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


def _build_literal(
    field_name: str,
    values: tuple[Any, ...],
    *,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> RuleCandidate:
    """Materialise a Literal-allowlist rule candidate from a dataclass field."""
    return RuleCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_in_{len(values)}_values",
        engine=library,
        library=library,
        rule_under_test=(f"{type_name}.{field_name} must be one of {list(values)!r}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {"in": list(values)}},
        kwargs_positive={field_name: "<invalid_dataclass_lift_probe>"},
        kwargs_negative={field_name: values[0]},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` must be one of {list(values)!r}",
        references=[f"{library}.{type_name} — dataclass Literal annotation"],
        added_by="dataclass_lift",
        added_at=today,
    )


__all__ = ["lift"]
