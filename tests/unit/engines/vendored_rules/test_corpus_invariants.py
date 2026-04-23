"""Invariants enforced on the seeded validation-rules corpus.

Phase 50.2b's vendor CI gate will run a richer equivalent of these checks
(also verifying each rule's kwargs_positive actually triggers in the real
library, and kwargs_negative doesn't). For 50.2a, the schema-level
invariants are the backstop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.engines.vendored_rules import VendoredRulesLoader

_VALID_SEVERITIES = {"error", "warn", "dormant"}
_VALID_OUTCOMES = {
    "pass",
    "warn",
    "error",
    "dormant_silent",
    "dormant_announced",
}
_VALID_EMISSION_CHANNELS = {
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "runtime_exception",
    "none",
}
_VALID_CONFIDENCE = {"high", "medium", "low"}


@pytest.fixture(scope="module")
def transformers_corpus():
    loader = VendoredRulesLoader()
    return loader.load_rules("transformers")


def test_corpus_has_minimum_rule_count(transformers_corpus) -> None:
    # Acceptance criteria #6: >= 30 entries after walker runs on pinned version.
    assert len(transformers_corpus.rules) >= 30


def test_corpus_schema_version_is_current(transformers_corpus) -> None:
    assert transformers_corpus.schema_version.startswith("1.")


def test_corpus_ids_unique(transformers_corpus) -> None:
    ids = [rule.id for rule in transformers_corpus.rules]
    assert len(ids) == len(set(ids))


def test_corpus_match_fields_non_empty(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        assert rule.match_fields, f"Rule {rule.id} has empty match.fields"


def test_corpus_kwargs_positive_and_negative_populated(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        assert rule.kwargs_positive, f"Rule {rule.id} has empty kwargs_positive"
        assert rule.kwargs_negative, f"Rule {rule.id} has empty kwargs_negative"


def test_corpus_severity_values_are_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        assert rule.severity in _VALID_SEVERITIES, rule.id


def test_corpus_outcome_values_are_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        outcome = rule.expected_outcome.get("outcome")
        assert outcome in _VALID_OUTCOMES, f"{rule.id}: bad outcome {outcome!r}"


def test_corpus_emission_channels_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        channel = rule.expected_outcome.get("emission_channel")
        assert channel in _VALID_EMISSION_CHANNELS, f"{rule.id}: {channel!r}"


def test_corpus_severity_outcome_consistency(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        outcome = rule.expected_outcome["outcome"]
        if rule.severity == "error":
            assert outcome == "error", f"{rule.id}: severity=error but outcome={outcome}"
        elif rule.severity == "warn":
            assert outcome == "warn", f"{rule.id}: severity=warn but outcome={outcome}"
        elif rule.severity == "dormant":
            assert outcome in {"dormant_silent", "dormant_announced"}, (
                f"{rule.id}: severity=dormant but outcome={outcome}"
            )


def test_corpus_dormant_silent_has_normalised_fields(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        if rule.expected_outcome.get("outcome") == "dormant_silent":
            normalised = rule.expected_outcome.get("normalised_fields")
            assert normalised, f"{rule.id}: dormant_silent but no normalised_fields"


def test_corpus_walker_confidence_values_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        confidence = rule.walker_source.get("walker_confidence")
        if confidence is not None:
            assert confidence in _VALID_CONFIDENCE, f"{rule.id}: {confidence!r}"


def test_corpus_added_by_values_valid(transformers_corpus) -> None:
    valid = {"ast_walker", "manual_seed", "runtime_warning_pr", "h3_collision_pr"}
    for rule in transformers_corpus.rules:
        assert rule.added_by in valid, f"{rule.id}: added_by={rule.added_by!r}"


def test_corpus_file_is_valid_yaml() -> None:
    # Sanity: the on-disk file is parseable YAML (redundant with loader, but
    # guards against accidental corruption by direct edits).
    import yaml

    path = (
        Path(__file__).resolve().parents[4] / "configs" / "validation_rules" / "transformers.yaml"
    )
    assert path.exists()
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict)
    assert "rules" in doc
