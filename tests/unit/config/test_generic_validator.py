"""Tests for the generic ``_apply_vendored_rules`` model validator.

Covers the three severity paths (error / warn / dormant) plus the
no-match / missing-corpus fallbacks. Rule loading is exercised by
``tests/unit/config/vendored_rules/test_loader.py``; this module focuses on
the validator's dispatch and the ``_dormant_observations`` contract.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic import ValidationError

from llenergymeasure.config.engine_configs import (
    TransformersConfig,
    TransformersSamplingConfig,
)
from llenergymeasure.config.models import (
    ExperimentConfig,
    _reset_rules_loader_cache,
)
from llenergymeasure.config.probe import DormantField
from llenergymeasure.config.vendored_rules.loader import Rule, VendoredRules
from llenergymeasure.config.warnings import ConfigValidationWarning


def _make_rule(
    rule_id: str,
    severity: str,
    match_fields: dict[str, Any],
    message_template: str | None = "test rule fired ({declared_value})",
    outcome: str = "error",
) -> Rule:
    """Build a minimal Rule for validator-path testing."""
    return Rule(
        id=rule_id,
        engine="transformers",
        library="transformers",
        rule_under_test="unit test rule",
        severity=severity,
        native_type="transformers.GenerationConfig",
        match_engine="transformers",
        match_fields=match_fields,
        kwargs_positive={},
        kwargs_negative={},
        expected_outcome={"outcome": outcome, "emission_channel": "none"},
        message_template=message_template,
        walker_source={},
        references=(),
        added_by="manual_seed",
        added_at="2026-04-23",
    )


class _StubLoader:
    def __init__(self, rules: list[Rule]) -> None:
        self._rules = tuple(rules)

    def load_rules(self, engine: str) -> VendoredRules:
        return VendoredRules(
            engine=engine,
            schema_version="1.0.0",
            engine_version="test",
            rules=self._rules,
        )


class _NoCorpusLoader:
    def load_rules(self, engine: str) -> VendoredRules:
        raise FileNotFoundError(f"no corpus for {engine}")


def _install_test_rules(monkeypatch: pytest.MonkeyPatch, rules: list[Rule]) -> None:
    """Substitute the module's loader accessor with a stub returning *rules*."""
    from llenergymeasure.config import models as models_mod

    stub = _StubLoader(rules)
    monkeypatch.setattr(models_mod, "_get_rules_loader", lambda: stub)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Each test starts with a fresh loader cache so real-corpus tests stay hermetic."""
    _reset_rules_loader_cache()


# ---------------------------------------------------------------------------
# Severity dispatch — error
# ---------------------------------------------------------------------------


def test_error_severity_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "test_error_rule",
        "error",
        {"transformers.attn_implementation": "sdpa"},
    )
    _install_test_rules(monkeypatch, [rule])

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="sdpa"),
        )
    assert "test_error_rule" in str(exc_info.value)


def test_error_severity_no_raise_when_match_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "test_error_rule",
        "error",
        {"transformers.attn_implementation": "flash_attention_2"},
    )
    _install_test_rules(monkeypatch, [rule])

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(attn_implementation="sdpa"),
    )
    assert cfg._dormant_observations == {}


# ---------------------------------------------------------------------------
# Severity dispatch — warn
# ---------------------------------------------------------------------------


def test_warn_severity_emits_config_validation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "test_warn_rule",
        "warn",
        {"transformers.attn_implementation": "eager"},
    )
    _install_test_rules(monkeypatch, [rule])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="eager"),
        )

    matched = [w for w in caught if issubclass(w.category, ConfigValidationWarning)]
    assert matched, "expected a ConfigValidationWarning"
    assert "test_warn_rule" in str(matched[0].message)


def test_warn_severity_not_fatal_under_simplefilter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``simplefilter('error', ConfigValidationWarning)`` escalates warn to raise."""
    rule = _make_rule(
        "test_warn_rule",
        "warn",
        {"transformers.attn_implementation": "eager"},
    )
    _install_test_rules(monkeypatch, [rule])

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigValidationWarning)
        with pytest.raises((ValidationError, ConfigValidationWarning)):
            ExperimentConfig(
                task={"model": "gpt2"},
                engine="transformers",
                transformers=TransformersConfig(attn_implementation="eager"),
            )


# ---------------------------------------------------------------------------
# Severity dispatch — dormant
# ---------------------------------------------------------------------------


def test_dormant_severity_populates_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "test_dormant_rule",
        "dormant",
        {"transformers.sampling.temperature": {"present": True, "not_equal": 1.0}},
        outcome="dormant_announced",
    )
    _install_test_rules(monkeypatch, [rule])

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(temperature=0.9),
        ),
    )

    observations = cfg._dormant_observations
    assert len(observations) == 1
    obs = observations["test_dormant_rule"]
    assert isinstance(obs, DormantField)
    assert obs.declared_value == 0.9
    assert "test_dormant_rule" in (obs.reason or "")


# ---------------------------------------------------------------------------
# Fallbacks — missing corpus / empty rule set
# ---------------------------------------------------------------------------


def test_missing_corpus_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """FileNotFoundError from the loader is swallowed and logged at debug."""
    from llenergymeasure.config import models as models_mod

    monkeypatch.setattr(models_mod, "_get_rules_loader", lambda: _NoCorpusLoader())

    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert cfg._dormant_observations == {}


def test_empty_rule_set_populates_empty_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_test_rules(monkeypatch, [])
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert cfg._dormant_observations == {}


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_multiple_dormant_rules_all_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    rule_a = _make_rule("a", "dormant", {"transformers.sampling.temperature": {"present": True}})
    rule_b = _make_rule("b", "dormant", {"transformers.sampling.top_p": {"present": True}})
    _install_test_rules(monkeypatch, [rule_a, rule_b])

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(temperature=0.9, top_p=0.95),
        ),
    )
    assert len(cfg._dormant_observations) == 2


def test_error_rule_shortcircuits_later_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    err = _make_rule("err", "error", {"transformers.attn_implementation": "eager"})
    dormant = _make_rule(
        "dormant_after_error",
        "dormant",
        {"transformers.sampling.temperature": {"present": True}},
    )
    _install_test_rules(monkeypatch, [err, dormant])

    with pytest.raises(ValidationError):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(
                attn_implementation="eager",
                sampling=TransformersSamplingConfig(temperature=0.9),
            ),
        )


# ---------------------------------------------------------------------------
# Unknown severity — fail-safe
# ---------------------------------------------------------------------------


def test_unknown_severity_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "weird_severity",
        "not_a_real_severity",
        {"transformers.attn_implementation": "sdpa"},
    )
    _install_test_rules(monkeypatch, [rule])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="sdpa"),
        )

    matched = [w for w in caught if issubclass(w.category, ConfigValidationWarning)]
    assert matched
    assert "weird_severity" in str(matched[0].message)


# ---------------------------------------------------------------------------
# Integration — real corpus on disk exercises every engine path without
# raising (catches predicate-operator / field-path regressions end-to-end)
# ---------------------------------------------------------------------------


def test_real_corpus_loads_and_default_config_passes() -> None:
    """Default transformers config doesn't trigger any error-severity rule.

    Sanity check against the actual packaged corpus — catches cases where a
    rule's match predicate fires on defaults (e.g. the early-stopping
    false-positive fixed in #375).
    """
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert cfg._dormant_observations == {}
