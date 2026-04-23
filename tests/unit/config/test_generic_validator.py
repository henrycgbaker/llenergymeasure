"""Tests for the generic ``_apply_vendored_rules`` model validator.

Covers the three severity paths (error / warn / dormant) plus the
no-match / missing-corpus fallbacks. Rule-loading is exercised by
``tests/unit/engines/vendored_rules/test_loader.py``; this module focuses on
the validator's dispatch and the ``_dormant_observations`` contract.
"""

from __future__ import annotations

import warnings
from pathlib import Path
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
from llenergymeasure.config.warnings import ConfigValidationWarning
from llenergymeasure.engines.protocol import DormantField
from llenergymeasure.engines.vendored_rules.loader import Rule, VendoredRules

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


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
        added_by="unit_test",
        added_at="2026-04-23",
    )


def _install_test_rules(monkeypatch: pytest.MonkeyPatch, rules: list[Rule]) -> None:
    """Replace the module-level rules loader with a stub returning *rules*.

    Swaps ``_RULES_LOADER`` in ``llenergymeasure.config.models`` — same
    surface the validator consumes — so the test controls exactly which
    rules fire.
    """
    from llenergymeasure.config import models as models_mod

    _reset_rules_loader_cache()

    class _StubLoader:
        def load_rules(self, engine: str) -> VendoredRules:
            return VendoredRules(
                engine=engine,
                schema_version="1.0.0",
                engine_version="test",
                rules=tuple(rules),
            )

    monkeypatch.setattr(models_mod, "_RULES_LOADER", _StubLoader())


# ---------------------------------------------------------------------------
# Severity dispatch — error path
# ---------------------------------------------------------------------------


def test_error_severity_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error-severity rule that matches raises ValidationError with the rule id."""
    rule = _make_rule(
        "test_error_rule",
        "error",
        {"transformers.attn_implementation": "flash_attention_2"},
    )
    _install_test_rules(monkeypatch, [rule])

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="flash_attention_2"),
        )

    assert "test_error_rule" in str(exc_info.value)


def test_error_severity_does_not_raise_when_match_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error-severity rule whose predicate doesn't match does not raise."""
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
    assert getattr(cfg, "_dormant_observations", None) == []


# ---------------------------------------------------------------------------
# Severity dispatch — warn path
# ---------------------------------------------------------------------------


def test_warn_severity_emits_config_validation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warn-severity rule emits ConfigValidationWarning via warnings.warn."""
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
    assert matched, "expected ConfigValidationWarning"
    assert "test_warn_rule" in str(matched[0].message)


# ---------------------------------------------------------------------------
# Severity dispatch — dormant path
# ---------------------------------------------------------------------------


def test_dormant_severity_populates_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dormant rule appends to _dormant_observations rather than raising."""
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
    obs = observations[0]
    assert isinstance(obs, DormantField)
    assert obs.declared_value == 0.9
    assert "test_dormant_rule" in (obs.reason or "")


def test_dormant_silent_populates_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """dormant_silent severity also lands in _dormant_observations (same bucket)."""
    rule = _make_rule(
        "test_dormant_silent",
        "dormant_silent",
        {"transformers.sampling.temperature": {"present": True}},
        outcome="dormant_silent",
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
    assert observations[0].declared_value == 0.9


# ---------------------------------------------------------------------------
# No rules loaded / missing corpus
# ---------------------------------------------------------------------------


def test_missing_corpus_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """FileNotFoundError from the loader is logged and swallowed."""
    from llenergymeasure.config import models as models_mod

    _reset_rules_loader_cache()

    class _NoCorpus:
        def load_rules(self, engine: str) -> VendoredRules:
            raise FileNotFoundError(f"no corpus for {engine}")

    monkeypatch.setattr(models_mod, "_RULES_LOADER", _NoCorpus())

    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert cfg._dormant_observations == []


def test_empty_rule_set_populates_empty_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine with zero rules still initialises ``_dormant_observations``."""
    _install_test_rules(monkeypatch, [])

    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    assert cfg._dormant_observations == []


# ---------------------------------------------------------------------------
# Multiple rules — dispatch composes
# ---------------------------------------------------------------------------


def test_multiple_dormant_rules_all_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    rule_a = _make_rule(
        "a",
        "dormant",
        {"transformers.sampling.temperature": {"present": True}},
    )
    rule_b = _make_rule(
        "b",
        "dormant",
        {"transformers.sampling.top_p": {"present": True}},
    )
    _install_test_rules(monkeypatch, [rule_a, rule_b])

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(temperature=0.9, top_p=0.95),
        ),
    )
    observations = cfg._dormant_observations
    assert len(observations) == 2


def test_error_rule_shortcircuits_later_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once an error is raised, remaining rules don't run (natural Python behaviour)."""
    err = _make_rule(
        "err",
        "error",
        {"transformers.attn_implementation": "flash_attention_2"},
    )
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
                attn_implementation="flash_attention_2",
                sampling=TransformersSamplingConfig(temperature=0.9),
            ),
        )


# ---------------------------------------------------------------------------
# Migration parity — flash_attn_requires_half_precision
# ---------------------------------------------------------------------------


def test_flash_attn_float32_rejected_via_corpus() -> None:
    """The removed hand-written validator's behaviour is now handled by the corpus."""
    _reset_rules_loader_cache()
    with pytest.raises(ValidationError, match=r"flash_attention_2.*requires.*float16"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(
                dtype="float32", attn_implementation="flash_attention_2"
            ),
        )


def test_flash_attn_bfloat16_accepted_via_corpus() -> None:
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="bfloat16", attn_implementation="flash_attention_2"),
    )
    assert cfg.transformers.dtype == "bfloat16"


def test_flash_attn_float16_accepted_via_corpus() -> None:
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float16", attn_implementation="flash_attention_3"),
    )
    assert cfg.transformers.dtype == "float16"


def test_eager_float32_still_accepted_via_corpus() -> None:
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float32", attn_implementation="eager"),
    )
    assert cfg.transformers.dtype == "float32"


def test_no_attn_impl_float32_still_accepted_via_corpus() -> None:
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(dtype="float32"),
    )
    assert cfg.transformers.dtype == "float32"


# ---------------------------------------------------------------------------
# Unknown severity — warn as a fail-safe (don't silently drop)
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
    assert any("unknown severity" in str(w.message) for w in matched)


# ---------------------------------------------------------------------------
# Corpus on disk — real load fallback
# ---------------------------------------------------------------------------


def test_full_corpus_load_emits_no_warnings_on_minimal_config() -> None:
    """The shipped corpus does not spuriously fire on a bare default config."""
    _reset_rules_loader_cache()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        ExperimentConfig(task={"model": "gpt2"}, engine="transformers")

    config_warnings = [w for w in caught if issubclass(w.category, ConfigValidationWarning)]
    assert config_warnings == []


def test_full_corpus_load_records_greedy_dormant(tmp_path: Path) -> None:
    """``do_sample=False`` + ``temperature=0.9`` triggers a corpus dormant rule."""
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(do_sample=False, temperature=0.9),
        ),
    )
    observations = cfg._dormant_observations
    assert any(
        "greedy" in (obs.reason or "").lower() or "temperature" in (obs.reason or "").lower()
        for obs in observations
    )


# ---------------------------------------------------------------------------
# Loader cache contract — tests can freely reset
# ---------------------------------------------------------------------------


def test_reset_rules_loader_cache_picks_up_corpus_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``_reset_rules_loader_cache``, the next config load re-parses."""
    from llenergymeasure.config import models as models_mod

    _install_test_rules(
        monkeypatch,
        [
            _make_rule(
                "stub_err",
                "error",
                {"transformers.attn_implementation": "flash_attention_2"},
            )
        ],
    )
    with pytest.raises(ValidationError, match="stub_err"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="flash_attention_2"),
        )

    # Reset + swap to a loader that returns zero rules; config should now load.
    _reset_rules_loader_cache()
    monkeypatch.setattr(
        models_mod,
        "_RULES_LOADER",
        type(
            "Z",
            (),
            {
                "load_rules": lambda self, engine: VendoredRules(
                    engine=engine, schema_version="1.0.0", engine_version="test", rules=()
                )
            },
        )(),
    )

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(attn_implementation="flash_attention_2"),
    )
    assert cfg.transformers.attn_implementation == "flash_attention_2"


# ---------------------------------------------------------------------------
# Rule message rendering — declared_value substitution
# ---------------------------------------------------------------------------


def test_dormant_message_substitutes_declared_value(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = _make_rule(
        "test_msg",
        "dormant",
        {"transformers.sampling.temperature": {"present": True}},
        message_template="temperature={declared_value} is dormant under greedy",
    )
    _install_test_rules(monkeypatch, [rule])

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(temperature=0.75),
        ),
    )
    observations = cfg._dormant_observations
    assert observations
    assert "0.75" in (observations[0].reason or "")


# ---------------------------------------------------------------------------
# Replace-style immutability — tests shouldn't pollute each other's loaders
# ---------------------------------------------------------------------------


def test_test_isolation_after_monkeypatch_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installing a stub loader is effective within the test."""
    rule = _make_rule(
        "iso",
        "error",
        {"transformers.attn_implementation": "sdpa"},
    )
    _install_test_rules(monkeypatch, [rule])
    with pytest.raises(ValidationError, match="iso"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation="sdpa"),
        )


# ---------------------------------------------------------------------------
# Predicate-spec coverage — exercise the operator engine indirectly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "match_fields,attn_impl,dtype,should_fire",
    [
        ({"transformers.attn_implementation": "sdpa"}, "sdpa", None, True),
        ({"transformers.attn_implementation": "sdpa"}, "eager", None, False),
        (
            {"transformers.attn_implementation": {"in": ["sdpa", "eager"]}},
            "eager",
            None,
            True,
        ),
        (
            {"transformers.attn_implementation": {"not_in": ["sdpa", "eager"]}},
            "eager",
            None,
            False,
        ),
        (
            {"transformers.attn_implementation": {"present": True}},
            None,
            None,
            False,
        ),
        (
            {"transformers.attn_implementation": {"present": True}},
            "sdpa",
            None,
            True,
        ),
    ],
)
def test_predicate_operators_via_error_path(
    monkeypatch: pytest.MonkeyPatch,
    match_fields: dict[str, Any],
    attn_impl: str | None,
    dtype: str | None,
    should_fire: bool,
) -> None:
    rule = _make_rule("op_test", "error", match_fields)
    _install_test_rules(monkeypatch, [rule])

    def _build() -> ExperimentConfig:
        return ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(attn_implementation=attn_impl, dtype=dtype),
        )

    if should_fire:
        with pytest.raises(ValidationError, match="op_test"):
            _build()
    else:
        _build()
