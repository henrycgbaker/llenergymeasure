"""Tests for :mod:`scripts.miners._msgspec_lift`.

Covers happy-path numeric-bound + Literal extraction on a synthetic
``msgspec.Struct`` plus the empty-on-non-Struct edge case and the
zero-Meta-annotation edge case (which mirrors live vLLM ``SamplingParams``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import msgspec

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners._msgspec_lift import lift  # noqa: E402


class _BoundedStruct(msgspec.Struct):
    """Fixture struct with one numeric Meta and one Literal field."""

    temperature: Annotated[float, msgspec.Meta(ge=0.0, le=2.0)] = 1.0
    mode: Literal["greedy", "beam"] = "greedy"


class _PlainStruct(msgspec.Struct):
    """Fixture mirroring vLLM ``SamplingParams``: no ``Meta`` annotations."""

    seed: int = 0
    n: int = 1


def _by_op(rules: list, op: str) -> list:
    return [
        r
        for r in rules
        if any(op in (m if isinstance(m, dict) else {}) for m in r.match_fields.values())
    ]


def test_lift_numeric_meta_and_literal() -> None:
    rules = lift(_BoundedStruct, namespace="m.s", today="2026-04-26", source_path="x.py")
    ge_rules = _by_op(rules, ">=")
    le_rules = _by_op(rules, "<=")
    in_rules = _by_op(rules, "in")
    assert len(ge_rules) == 1
    assert len(le_rules) == 1
    assert len(in_rules) == 1

    ge_rule = ge_rules[0]
    assert ge_rule.added_by == "msgspec_lift"
    assert ge_rule.match_fields == {"m.s.temperature": {">=": 0.0}}
    assert ge_rule.kwargs_positive == {"temperature": -1.0}
    assert ge_rule.kwargs_negative == {"temperature": 0.0}

    in_rule = in_rules[0]
    # msgspec sorts Literal values when building its type info; we mirror
    # that order rather than the source-declaration order.
    assert in_rule.match_fields == {"m.s.mode": {"in": ["beam", "greedy"]}}


def test_lift_returns_empty_for_struct_without_meta() -> None:
    """Edge case: vLLM ``SamplingParams`` ships zero ``Meta`` — must yield ``[]``.

    Per locked design §1, a Meta-less struct returning ``[]`` is the expected
    contract; the static miner picks up the slack via AST.
    """
    assert lift(_PlainStruct, namespace="m.p", today="2026-04-26", source_path="x.py") == []


def test_lift_returns_empty_on_non_struct_type() -> None:
    class _NotStruct:
        pass

    assert lift(_NotStruct, namespace="m.n", today="2026-04-26", source_path="x.py") == []
