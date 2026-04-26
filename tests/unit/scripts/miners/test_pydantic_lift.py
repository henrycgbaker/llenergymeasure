"""Tests for :mod:`scripts.miners._pydantic_lift`.

Covers happy-path constraint extraction (numeric ge/le, multiple_of, Literal)
plus the empty-on-non-Pydantic edge case. Uses ad-hoc Pydantic models so
the tests don't depend on any vendored library version.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners._pydantic_lift import lift  # noqa: E402


class _Bounds(BaseModel):
    """Mini fixture model exercising every supported annotated-types op."""

    temperature: float = Field(ge=0.0, le=2.0)
    block_size: int = Field(multiple_of=8)
    name: str = Field(min_length=1, max_length=64)
    mode: Literal["greedy", "beam", "sample"] = "greedy"


def _by_op(rules: list, op: str) -> list:
    return [
        r
        for r in rules
        if any(op in (m if isinstance(m, dict) else {}) for m in r.match_fields.values())
    ]


def test_lift_numeric_bounds() -> None:
    rules = lift(_Bounds, namespace="t.bounds", today="2026-04-26", source_path="x.py")
    ge_rules = _by_op(rules, ">=")
    le_rules = _by_op(rules, "<=")
    assert len(ge_rules) == 1
    assert len(le_rules) == 1
    ge_rule = ge_rules[0]
    assert ge_rule.added_by == "pydantic_lift"
    assert ge_rule.match_fields == {"t.bounds.temperature": {">=": 0.0}}
    # kwargs_positive must violate, kwargs_negative must satisfy.
    assert ge_rule.kwargs_positive == {"temperature": -1.0}
    assert ge_rule.kwargs_negative == {"temperature": 0.0}


def test_lift_multiple_of_and_lengths() -> None:
    rules = lift(_Bounds, namespace="t.bounds", today="2026-04-26", source_path="x.py")
    mo = _by_op(rules, "multiple_of")
    assert len(mo) == 1
    assert mo[0].match_fields == {"t.bounds.block_size": {"multiple_of": 8}}

    min_len = _by_op(rules, "min_len")
    max_len = _by_op(rules, "max_len")
    assert len(min_len) == 1
    assert len(max_len) == 1
    assert min_len[0].kwargs_positive == {"name": ""}
    assert max_len[0].kwargs_positive == {"name": "x" * 65}


def test_lift_literal_allowlist() -> None:
    rules = lift(_Bounds, namespace="t.bounds", today="2026-04-26", source_path="x.py")
    in_rules = _by_op(rules, "in")
    assert len(in_rules) == 1
    rule = in_rules[0]
    assert rule.match_fields == {"t.bounds.mode": {"in": ["greedy", "beam", "sample"]}}
    # negative case picks the first valid value
    assert rule.kwargs_negative == {"mode": "greedy"}


def test_lift_returns_empty_on_non_pydantic_type() -> None:
    """Edge case: non-Pydantic types yield ``[]`` rather than raising."""

    class _Plain:
        x: int

    assert lift(_Plain, namespace="t", today="2026-04-26", source_path="x.py") == []


def test_lift_handles_optional_literal() -> None:
    """Edge case: ``Optional[Literal[...]]`` should still surface the Literal."""

    class _OptionalLiteral(BaseModel):
        choice: Literal["a", "b"] | None = None

    rules = lift(_OptionalLiteral, namespace="t", today="2026-04-26", source_path="x.py")
    in_rules = _by_op(rules, "in")
    assert len(in_rules) == 1
    assert in_rules[0].match_fields == {"t.choice": {"in": ["a", "b"]}}
