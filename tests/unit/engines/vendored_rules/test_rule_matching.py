"""Tests for :class:`Rule.try_match`, the predicate engine, and message rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llenergymeasure.engines.vendored_rules import (
    Rule,
    evaluate_predicate,
    resolve_field_path,
)

# ---------------------------------------------------------------------------
# Config stubs
# ---------------------------------------------------------------------------


@dataclass
class _Sampling:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool | None = None
    num_beams: int | None = None


@dataclass
class _Transformers:
    sampling: _Sampling


@dataclass
class _Config:
    transformers: _Transformers


def _make_rule(
    *,
    id: str = "rule_x",
    match_fields: dict[str, Any] | None = None,
    severity: str = "dormant",
    message: str | None = "Declared {declared_value}",
) -> Rule:
    return Rule(
        id=id,
        engine="transformers",
        library="transformers",
        rule_under_test="test",
        severity=severity,
        native_type="transformers.GenerationConfig",
        match_engine="transformers",
        match_fields=match_fields or {},
        kwargs_positive={},
        kwargs_negative={},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "minor_issues_dict",
            "normalised_fields": [],
        },
        message_template=message,
        walker_source={},
        references=(),
        added_by="ast_walker",
        added_at="2026-04-23",
    )


# ---------------------------------------------------------------------------
# Predicate operator coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, actual, expected",
    [
        (0.5, 0.5, True),
        (0.5, 0.6, False),
        ({"==": 1}, 1, True),
        ({"==": 1}, 2, False),
        ({"!=": 1.0}, 0.9, True),
        ({"!=": 1.0}, 1.0, False),
        ({"<": 1}, 0, True),
        ({"<": 1}, 1, False),
        ({"<=": 1}, 1, True),
        ({">": 1}, 2, True),
        ({">": 1}, 1, False),
        ({">=": 1}, 1, True),
        ({"in": ["a", "b"]}, "a", True),
        ({"in": ["a", "b"]}, "c", False),
        ({"not_in": ["a", "b"]}, "c", True),
        ({"not_in": ["a", "b"]}, "a", False),
        ({"present": True}, 0.5, True),
        ({"present": True}, None, False),
        ({"absent": True}, None, True),
        ({"absent": True}, 0.5, False),
        ({"equals": 1}, 1, True),
        ({"not_equal": 1}, 2, True),
        ({"not_equal": 1}, None, False),  # None is never not-equal (None-safe)
    ],
)
def test_evaluate_predicate_operators(spec: Any, actual: Any, expected: bool) -> None:
    assert evaluate_predicate(actual, spec) is expected


def test_evaluate_predicate_multi_key_all_must_pass() -> None:
    # AND-combine multi-key predicates.
    assert evaluate_predicate(0.9, {"present": True, "not_equal": 1.0}) is True
    assert evaluate_predicate(1.0, {"present": True, "not_equal": 1.0}) is False
    assert evaluate_predicate(None, {"present": True, "not_equal": 1.0}) is False


def test_evaluate_predicate_unknown_operator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown match operator"):
        evaluate_predicate(1, {"matches_regex": ".*"})


def test_evaluate_predicate_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="Empty match"):
        evaluate_predicate(1, {})


def test_comparison_operators_are_none_safe() -> None:
    # None on either side of a numeric comparator should return False, not
    # raise a TypeError.
    assert evaluate_predicate(None, {"<": 1}) is False
    assert evaluate_predicate(None, {">": 0}) is False


# ---------------------------------------------------------------------------
# Type predicates (type_is / type_is_not)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, actual, expected",
    [
        # single-name spec
        ({"type_is": "bool"}, True, True),
        ({"type_is": "bool"}, False, True),
        ({"type_is": "bool"}, 1, False),
        ({"type_is": "bool"}, "yes", False),
        ({"type_is": "int"}, 1, True),
        ({"type_is": "int"}, True, False),  # strict: bool is not int
        ({"type_is": "str"}, "foo", True),
        ({"type_is": "list"}, ["a"], True),
        ({"type_is": "dict"}, {"a": 1}, True),
        # any-of (list-of-names) spec
        ({"type_is": ["int", "float"]}, 1, True),
        ({"type_is": ["int", "float"]}, 1.0, True),
        ({"type_is": ["int", "float"]}, "1", False),
        # complement
        ({"type_is_not": "bool"}, "yes", True),
        ({"type_is_not": "bool"}, True, False),
        ({"type_is_not": "str"}, 1, True),
        # None is never typed as anything — predicate does not fire on absent fields
        ({"type_is": "bool"}, None, False),
        ({"type_is_not": "bool"}, None, False),
    ],
)
def test_type_predicates(spec: Any, actual: Any, expected: bool) -> None:
    assert evaluate_predicate(actual, spec) is expected


def test_type_predicate_with_custom_class_name() -> None:
    # Works for library-defined types via their __name__.
    class WatermarkingConfig:
        pass

    wc = WatermarkingConfig()
    assert evaluate_predicate(wc, {"type_is": "WatermarkingConfig"}) is True
    assert evaluate_predicate({"a": 1}, {"type_is_not": "WatermarkingConfig"}) is True


def test_type_predicate_multi_key_composes_with_present() -> None:
    # BNB-style predicate: field must be set AND have the wrong type.
    spec = {"present": True, "type_is_not": "bool"}
    assert evaluate_predicate("yes", spec) is True
    assert evaluate_predicate(True, spec) is False
    assert evaluate_predicate(None, spec) is False


# ---------------------------------------------------------------------------
# Field-path resolution
# ---------------------------------------------------------------------------


def test_resolve_field_path_nested_attrs() -> None:
    config = _Config(transformers=_Transformers(sampling=_Sampling(temperature=0.9)))
    assert resolve_field_path(config, "transformers.sampling.temperature") == 0.9


def test_resolve_field_path_missing_returns_none() -> None:
    config = _Config(transformers=_Transformers(sampling=_Sampling()))
    assert resolve_field_path(config, "transformers.sampling.missing") is None
    assert resolve_field_path(config, "vllm.engine.foo") is None


def test_resolve_field_path_dict_fallback() -> None:
    config = {"transformers": {"sampling": {"temperature": 0.9}}}
    assert resolve_field_path(config, "transformers.sampling.temperature") == 0.9


def test_resolve_field_path_mixed_dict_and_attr() -> None:
    config = _Config(transformers=_Transformers(sampling=_Sampling(temperature=0.5)))
    # Works even when the entry point is an attribute chain into a nested dataclass.
    assert resolve_field_path(config, "transformers.sampling.temperature") == 0.5


# ---------------------------------------------------------------------------
# Rule.try_match
# ---------------------------------------------------------------------------


def test_try_match_returns_none_when_no_predicate() -> None:
    rule = _make_rule(match_fields={"transformers.sampling.temperature": {"present": True}})
    config = _Config(transformers=_Transformers(sampling=_Sampling()))
    assert rule.try_match(config) is None


def test_try_match_returns_match_object_when_all_predicates_hold() -> None:
    rule = _make_rule(
        match_fields={
            "transformers.sampling.do_sample": False,
            "transformers.sampling.temperature": {
                "present": True,
                "not_equal": 1.0,
            },
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(do_sample=False, temperature=0.9))
    )
    match = rule.try_match(config)
    assert match is not None
    # declared_value is the last field's value — the subject the user cares about.
    assert match.declared_value == 0.9
    assert match.matched_fields == {
        "transformers.sampling.do_sample": False,
        "transformers.sampling.temperature": 0.9,
    }


def test_try_match_stops_at_first_failing_predicate() -> None:
    rule = _make_rule(
        match_fields={
            "transformers.sampling.do_sample": True,  # won't match
            "transformers.sampling.temperature": {"present": True},
        }
    )
    config = _Config(
        transformers=_Transformers(sampling=_Sampling(do_sample=False, temperature=0.9))
    )
    assert rule.try_match(config) is None


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def test_render_message_substitutes_declared_value() -> None:
    rule = _make_rule(
        match_fields={"transformers.sampling.temperature": {"present": True}},
        message="Dormant: temperature={declared_value} rule={rule_id}",
    )
    config = _Config(transformers=_Transformers(sampling=_Sampling(temperature=0.7)))
    match = rule.try_match(config)
    assert match is not None
    msg = rule.render_message(match)
    assert "0.7" in msg
    assert "rule_x" in msg


def test_render_message_missing_template_returns_fallback() -> None:
    rule = _make_rule(
        match_fields={"transformers.sampling.temperature": {"present": True}},
        message=None,
    )
    config = _Config(transformers=_Transformers(sampling=_Sampling(temperature=0.7)))
    match = rule.try_match(config)
    assert match is not None
    assert "rule_x" in rule.render_message(match)


def test_render_message_with_missing_format_key_falls_back_gracefully() -> None:
    # Template refers to a field not populated in the match.
    rule = _make_rule(
        match_fields={"transformers.sampling.temperature": {"present": True}},
        message="Field {not_in_match_or_declared}",
    )
    config = _Config(transformers=_Transformers(sampling=_Sampling(temperature=0.7)))
    match = rule.try_match(config)
    assert match is not None
    # Should not raise; should fall back to rule-id + raw template.
    out = rule.render_message(match)
    assert "rule_x" in out
