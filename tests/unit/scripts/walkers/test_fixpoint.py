"""Tests for :mod:`scripts.walkers._fixpoint_test`.

Covers:
- Convergence on idempotent rules.
- Cycle detection (two rules that flip a field forever).
- Order-dependence detection (two non-commuting rules).
- Corpus-level integration against the seeded transformers corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers._fixpoint_test import (  # noqa: E402
    CanonicaliserCycleError,
    NonIdempotentRuleError,
    OrderDependentRuleError,
    _ProjectedRule,
    apply_to_fixpoint,
    assert_idempotent,
    assert_shuffle_stable,
    construct_seed_states,
    fixpoint_test_corpus,
    load_dormant_rules,
)


def _rule(
    rule_id: str, match_fields: dict[str, Any], normalised: tuple[str, ...]
) -> _ProjectedRule:
    return _ProjectedRule(
        id=rule_id,
        severity="dormant",
        match_fields=match_fields,
        normalised_fields=normalised,
    )


# ---------------------------------------------------------------------------
# apply_to_fixpoint
# ---------------------------------------------------------------------------


class TestApplyToFixpoint:
    def test_no_applicable_rules(self) -> None:
        rules = [_rule("r1", {"a": 1}, ("b",))]
        seed = {"a": 2}
        state, applied = apply_to_fixpoint(seed, rules)
        assert state == seed
        assert applied == []

    def test_single_rule_converges(self) -> None:
        rules = [_rule("strip_temp", {"do_sample": False}, ("temperature",))]
        seed = {"do_sample": False, "temperature": 0.9}
        state, applied = apply_to_fixpoint(seed, rules)
        assert state["temperature"] is None
        assert applied == ["strip_temp"]

    def test_two_independent_rules(self) -> None:
        rules = [
            _rule("strip_a", {"mode": "greedy"}, ("a",)),
            _rule("strip_b", {"mode": "greedy"}, ("b",)),
        ]
        seed = {"mode": "greedy", "a": 1, "b": 2}
        state, applied = apply_to_fixpoint(seed, rules)
        assert state["a"] is None
        assert state["b"] is None
        assert sorted(applied) == ["strip_a", "strip_b"]


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_idempotent_rule_passes(self) -> None:
        rule = _rule("strip_x", {"do_sample": False}, ("x",))
        seed = {"do_sample": False, "x": 1}
        assert_idempotent(rule, seed)  # should not raise

    def test_non_idempotent_rule_raises(self) -> None:
        class TogglingRule(_ProjectedRule):
            def apply(self, state: dict[str, Any]) -> dict[str, Any]:
                next_state = dict(state)
                next_state["x"] = next_state.get("x", 0) + 1
                return next_state

        rule = TogglingRule(
            id="toggle",
            severity="dormant",
            match_fields={"do_sample": False},
            normalised_fields=("x",),
        )
        seed = {"do_sample": False, "x": 1}
        with pytest.raises(NonIdempotentRuleError):
            assert_idempotent(rule, seed)


# ---------------------------------------------------------------------------
# Shuffle stability
# ---------------------------------------------------------------------------


class TestShuffleStability:
    def test_commuting_rules_stable(self) -> None:
        rules = [
            _rule("r1", {"greedy": True}, ("a",)),
            _rule("r2", {"greedy": True}, ("b",)),
            _rule("r3", {"greedy": True}, ("c",)),
        ]
        seed = {"greedy": True, "a": 1, "b": 2, "c": 3}
        final = assert_shuffle_stable(rules, seed, shuffle_count=5)
        assert final["a"] is None
        assert final["b"] is None
        assert final["c"] is None

    def test_cycle_detected(self) -> None:
        # Two rules where r_a sets x=1, r_b sets x=2, both always applicable.
        class SetValue(_ProjectedRule):
            def __init__(self, rule_id: str, value: int) -> None:
                super().__init__(
                    id=rule_id,
                    severity="dormant",
                    match_fields={},
                    normalised_fields=("x",),
                )
                object.__setattr__(self, "_value", value)

            def applies(self, state: dict[str, Any]) -> bool:
                return state.get("x") != self._value

            def apply(self, state: dict[str, Any]) -> dict[str, Any]:
                next_state = dict(state)
                next_state["x"] = self._value
                return next_state

        rules = [SetValue("ra", 1), SetValue("rb", 2)]
        seed = {"x": 0}
        with pytest.raises(CanonicaliserCycleError):
            apply_to_fixpoint(seed, rules)

    def test_order_dependent_detected(self) -> None:
        # r1: if y is None, set x=0. r2: always set y=None.
        # Order A: r1 then r2 -> x=0, y=None (seed has y=1, r1 applies-no, r2
        # applies-yes, second pass r1 applies).
        # Order B: r2 then r1 -> r2 sets y=None, then r1 applies and sets x=0.
        # Make them non-commuting:
        # r1: if y=1 set x=99 (forever).  r2: if x is None set y=99 (forever).
        class R1(_ProjectedRule):
            def applies(self, state: dict[str, Any]) -> bool:
                return state.get("y") == 1 and state.get("x") != 99

            def apply(self, state: dict[str, Any]) -> dict[str, Any]:
                next_state = dict(state)
                next_state["x"] = 99
                return next_state

        class R2(_ProjectedRule):
            def applies(self, state: dict[str, Any]) -> bool:
                return state.get("x") is None and state.get("y") != 99

            def apply(self, state: dict[str, Any]) -> dict[str, Any]:
                next_state = dict(state)
                next_state["y"] = 99
                return next_state

        r1 = R1(id="r1", severity="dormant", match_fields={}, normalised_fields=())
        r2 = R2(id="r2", severity="dormant", match_fields={}, normalised_fields=())

        seed = {"x": None, "y": 1}

        # Order [r1, r2]: r1 applies (y=1) -> x=99. r2 applies-no (x=99). Done.
        #     final: {x: 99, y: 1}
        # Order [r2, r1]: r2 applies (x=None) -> y=99. r1 applies-no (y=99). Done.
        #     final: {x: None, y: 99}
        with pytest.raises(OrderDependentRuleError):
            assert_shuffle_stable([r1, r2], seed, shuffle_count=10, seed_rng=0)


# ---------------------------------------------------------------------------
# Corpus integration
# ---------------------------------------------------------------------------


class TestCorpusIntegration:
    def test_seeded_transformers_corpus_converges(self) -> None:
        import yaml

        corpus_path = _PROJECT_ROOT / "configs" / "validation_rules" / "transformers.yaml"
        corpus = yaml.safe_load(corpus_path.read_text())
        # Should not raise.
        fixpoint_test_corpus(corpus)

    def test_load_dormant_rules_filters(self) -> None:
        corpus = {
            "rules": [
                {
                    "id": "r_error",
                    "severity": "error",
                    "match": {"fields": {"x": 1}},
                    "expected_outcome": {"normalised_fields": []},
                },
                {
                    "id": "r_dormant",
                    "severity": "dormant",
                    "match": {"fields": {"x": 1}},
                    "expected_outcome": {"normalised_fields": ["x"]},
                },
            ]
        }
        rules = load_dormant_rules(corpus)
        assert len(rules) == 1
        assert rules[0].id == "r_dormant"

    def test_construct_seed_states_satisfies_matches(self) -> None:
        rules = [
            _rule(
                "r1",
                {"do_sample": False, "temperature": {"present": True, "not_equal": 1.0}},
                ("temperature",),
            )
        ]
        seeds = construct_seed_states(rules)
        assert len(seeds) == 1
        assert seeds[0]["do_sample"] is False
        # The generated sentinel must not equal 1.0 (so the predicate fires).
        assert seeds[0]["temperature"] != 1.0
