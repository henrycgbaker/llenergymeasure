"""Tests for :mod:`scripts.miners._dataclass_lift`.

Covers happy-path Literal extraction on a synthetic stdlib dataclass plus
the two edge cases the lift handles structurally:

- non-dataclass target → ``[]`` (matches transformers ``GenerationConfig``,
  which is not a dataclass on transformers 4.x).
- dataclass with no Literal-annotated fields → ``[]``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners._dataclass_lift import lift  # noqa: E402


@dataclasses.dataclass
class _LiteralDataclass:
    quant_method: Literal["fp4", "nf4", "int8"] = "nf4"
    name: str = "x"
    optional_choice: Literal["a", "b"] | None = None


@dataclasses.dataclass
class _NoLiterals:
    x: int = 0
    y: str = "y"


def _by_op(rules: list, op: str) -> list:
    return [
        r
        for r in rules
        if any(op in (m if isinstance(m, dict) else {}) for m in r.match_fields.values())
    ]


def test_lift_extracts_literal_allowlists() -> None:
    rules = lift(_LiteralDataclass, namespace="d.lit", today="2026-04-26", source_path="x.py")
    in_rules = _by_op(rules, "in")
    # Two Literal fields: quant_method and optional_choice (one nested in Optional).
    assert len(in_rules) == 2

    primary = next(r for r in in_rules if "quant_method" in next(iter(r.match_fields)))
    assert primary.added_by == "dataclass_lift"
    assert primary.match_fields == {"d.lit.quant_method": {"in": ["fp4", "nf4", "int8"]}}
    assert primary.kwargs_negative == {"quant_method": "fp4"}


def test_lift_returns_empty_when_no_literals() -> None:
    assert lift(_NoLiterals, namespace="d.n", today="2026-04-26", source_path="x.py") == []


def test_lift_returns_empty_on_non_dataclass() -> None:
    """Edge case: non-dataclass yields ``[]``.

    This is the contract that lets ``transformers_dynamic_miner.py`` call
    ``_dataclass_lift(GenerationConfig)`` without raising — the lift just
    no-ops on non-dataclass targets.
    """

    class _Plain:
        x: int

    assert lift(_Plain, namespace="d.p", today="2026-04-26", source_path="x.py") == []
