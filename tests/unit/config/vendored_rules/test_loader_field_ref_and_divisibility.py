"""Tests for cross-field ``@field_path`` references and divisibility operators.

Covers the loader-grammar extensions that close gaps surfaced by PR #387:

- ``@field_path`` substitution on the right-hand side of any operator,
  with sibling and dotted-from-root resolution semantics.
- ``divisible_by`` / ``not_divisible_by`` operators with strict
  non-bool integer operands and zero-divisor guards.
- Spec walking through nested lists / dicts so refs anywhere in the
  predicate tree get resolved before evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenergymeasure.config.vendored_rules import (
    Rule,
    evaluate_predicate,
)
from llenergymeasure.config.vendored_rules.loader import (
    _is_int_pair,
    _resolve_field_refs_in_spec,
)

# ---------------------------------------------------------------------------
# Config stubs (mirror test_rule_matching.py shape)
# ---------------------------------------------------------------------------


@dataclass
class _Sampling:
    num_beams: int | None = None
    num_beam_groups: int | None = None
    num_return_sequences: int | None = None
    diversity_penalty: float | None = None


@dataclass
class _Transformers:
    sampling: _Sampling


@dataclass
class _Config:
    transformers: _Transformers


def _make_rule(*, match_fields: dict[str, Any]) -> Rule:
    return Rule(
        id="rule_x",
        engine="transformers",
        library="transformers",
        rule_under_test="test",
        severity="error",
        native_type="transformers.GenerationConfig",
        match_engine="transformers",
        match_fields=match_fields,
        kwargs_positive={},
        kwargs_negative={},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "runtime_exception",
            "normalised_fields": [],
        },
        message_template="msg {declared_value}",
        miner_source={},
        references=(),
        added_by="manual_seed",
        added_at="2026-04-25",
    )


# ---------------------------------------------------------------------------
# @field_ref resolution — sibling
# ---------------------------------------------------------------------------


def test_field_ref_sibling_substitutes_value() -> None:
    config = {"a": {"b": {"x": 5, "y": 3}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": 3}


def test_field_ref_sibling_resolves_to_none_when_missing() -> None:
    config = {"a": {"b": {"x": 5}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": None}


@pytest.mark.parametrize(
    "x_value, y_value, expected_fires",
    [
        (5, 3, True),  # x > y → fires
        (3, 3, False),  # x == y → does not fire
        (2, 3, False),  # x < y → does not fire
        (5, None, False),  # y missing → comparison is None-safe, does not fire
    ],
)
def test_field_ref_sibling_via_evaluate_predicate(
    x_value: int, y_value: int | None, expected_fires: bool
) -> None:
    config = {"a": {"b": {"x": x_value, "y": y_value}}}
    spec = {">": "@y"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert evaluate_predicate(x_value, resolved) is expected_fires


# ---------------------------------------------------------------------------
# @field_ref resolution — dotted from root
# ---------------------------------------------------------------------------


def test_field_ref_dotted_resolves_from_root() -> None:
    config = {"deep": {"nested": {"path": 42}}, "a": {"b": {"x": 1}}}
    spec = {">": "@deep.nested.path"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {">": 42}


def test_field_ref_dotted_resolves_through_attribute_chains() -> None:
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=4, num_return_sequences=6))
    )
    spec = {">": "@transformers.sampling.num_beams"}
    resolved = _resolve_field_refs_in_spec(
        spec, config, "transformers.sampling.num_return_sequences"
    )
    assert resolved == {">": 4}


# ---------------------------------------------------------------------------
# Spec walk — recursion through lists and nested dicts
# ---------------------------------------------------------------------------


def test_spec_walk_resolves_refs_inside_list() -> None:
    config = {"a": {"b": {"x": 1, "y": 7, "z": 9}}}
    spec = {"in": ["@y", "@z"]}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.b.x")
    assert resolved == {"in": [7, 9]}


def test_spec_walk_leaves_non_ref_strings_alone() -> None:
    config = {"a": {"x": 1, "name": "foo"}}
    spec = {"==": "literal_value"}
    resolved = _resolve_field_refs_in_spec(spec, config, "a.x")
    assert resolved == {"==": "literal_value"}


def test_spec_walk_passes_through_bare_value() -> None:
    # Bare-value spec (equality) is returned unchanged when not a ref.
    assert _resolve_field_refs_in_spec(0.5, {"a": 1}, "a") == 0.5


# ---------------------------------------------------------------------------
# divisible_by / not_divisible_by operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, actual, expected",
    [
        # not_divisible_by: positive case (rule fires)
        ({"not_divisible_by": 4}, 6, True),  # 6 % 4 == 2 → fires
        ({"not_divisible_by": 3}, 6, False),  # 6 % 3 == 0 → does not fire
        # divisible_by: positive case (rule fires)
        ({"divisible_by": 3}, 6, True),  # 6 % 3 == 0 → fires
        ({"divisible_by": 4}, 6, False),  # 6 % 4 == 2 → does not fire
    ],
)
def test_divisibility_basic(spec: dict[str, Any], actual: int, expected: bool) -> None:
    assert evaluate_predicate(actual, spec) is expected


def test_divisibility_zero_divisor_does_not_fire() -> None:
    # b == 0: never fires (avoids ZeroDivisionError, no rule should match).
    assert evaluate_predicate(6, {"not_divisible_by": 0}) is False
    assert evaluate_predicate(6, {"divisible_by": 0}) is False


@pytest.mark.parametrize(
    "actual, divisor",
    [
        (None, 3),  # missing field — predicate must not fire
        (6.0, 3),  # float operand — strict int-only
        ("6", 3),  # str operand — strict int-only
        (6, 3.0),  # float divisor — strict int-only
    ],
)
def test_divisibility_rejects_non_int_operands(actual: Any, divisor: Any) -> None:
    assert evaluate_predicate(actual, {"not_divisible_by": divisor}) is False
    assert evaluate_predicate(actual, {"divisible_by": divisor}) is False


@pytest.mark.parametrize(
    "actual, divisor",
    [
        (True, 1),  # bool actual: would otherwise pass via bool < int
        (False, 1),
        (6, True),  # bool divisor
        (True, True),
    ],
)
def test_divisibility_rejects_bool_operands(actual: Any, divisor: Any) -> None:
    assert evaluate_predicate(actual, {"not_divisible_by": divisor}) is False
    assert evaluate_predicate(actual, {"divisible_by": divisor}) is False


def test_is_int_pair_helper() -> None:
    # Direct unit on the helper for a tight regression guard.
    assert _is_int_pair(6, 3) is True
    assert _is_int_pair(0, 1) is True
    assert _is_int_pair(True, 1) is False
    assert _is_int_pair(1, False) is False
    assert _is_int_pair(6.0, 3) is False
    assert _is_int_pair(None, 3) is False


# ---------------------------------------------------------------------------
# End-to-end via Rule.try_match — corpus-shape predicate
# ---------------------------------------------------------------------------


def test_try_match_with_field_ref_fires_when_left_exceeds_right() -> None:
    # Mirrors the rewritten transformers_num_return_sequences_exceeds_num_beams
    # rule: fires when num_return_sequences > num_beams.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_return_sequences": {">": "@num_beams"},
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=2, num_return_sequences=4))
    )
    match = rule.try_match(config)
    assert match is not None
    assert match.declared_value == 4


def test_try_match_with_field_ref_does_not_fire_when_left_le_right() -> None:
    # The valid case (num_return_sequences=2, num_beams=4) — rule must not fire.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_return_sequences": {">": "@num_beams"},
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(num_beams=4, num_return_sequences=2))
    )
    assert rule.try_match(config) is None


def test_try_match_with_not_divisible_by_field_ref_fires() -> None:
    # Mirrors the new transformers_num_beams_not_divisible_by_groups rule.
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_beam_groups": {">": 1},
            "transformers.sampling.num_beams": {"not_divisible_by": "@num_beam_groups"},
        }
    )
    config = _Config(transformers=_Transformers(sampling=_Sampling(num_beams=6, num_beam_groups=4)))
    match = rule.try_match(config)
    assert match is not None
    # Last predicate's field is the subject (num_beams).
    assert match.declared_value == 6


def test_try_match_with_not_divisible_by_field_ref_does_not_fire_on_valid() -> None:
    rule = _make_rule(
        match_fields={
            "transformers.sampling.num_beam_groups": {">": 1},
            "transformers.sampling.num_beams": {"not_divisible_by": "@num_beam_groups"},
        }
    )
    # 6 % 3 == 0 → divisible → rule does not fire.
    config = _Config(transformers=_Transformers(sampling=_Sampling(num_beams=6, num_beam_groups=3)))
    assert rule.try_match(config) is None
