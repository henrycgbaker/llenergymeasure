"""Invariants enforced on the seeded validation-rules corpus.

The vendor CI pipeline (a separate follow-up PR) runs a richer equivalent
of these checks — empirically verifying each rule's ``kwargs_positive``
actually fires in the real library and ``kwargs_negative`` doesn't. These
schema-level invariants are the offline backstop.

Most enum coverage is redundant with the loader's own ``UnknownEnumValueError``
family (severity / outcome / emission_channel / added_by are all validated at
corpus parse time). These tests re-iterate for defence-in-depth and as a
human-readable catalogue of what the corpus must look like.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.config.vendored_rules import (
    VALID_ADDED_BY,
    VALID_EMISSION_CHANNEL,
    VALID_OUTCOME,
    VALID_SEVERITY,
    VendoredRulesLoader,
)

_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
# SSOT: scripts/walkers/_base.py::Confidence / VALID_CONFIDENCE. Duplicated
# here to avoid a cross-layer (engines → scripts) test import.


@pytest.fixture(scope="module")
def transformers_corpus():
    loader = VendoredRulesLoader()
    return loader.load_rules("transformers")


def test_corpus_has_expected_rule_ids(transformers_corpus) -> None:
    # Exact set — catches accidental removal or silent addition of a rule.
    # When the walker gets a new rule, this test intentionally fails and the
    # corpus-authoring PR must include the id here.
    expected_ids = {
        # Greedy dormancy (do_sample=False strips sampling params)
        "transformers_greedy_strips_temperature",
        "transformers_greedy_strips_top_p",
        "transformers_greedy_strips_top_k",
        "transformers_greedy_strips_min_p",
        "transformers_greedy_strips_typical_p",
        "transformers_greedy_strips_epsilon_cutoff",
        "transformers_greedy_strips_eta_cutoff",
        # Single-beam dormancy (num_beams=1 strips beam params)
        "transformers_single_beam_strips_early_stopping",
        "transformers_single_beam_strips_num_beam_groups",
        "transformers_single_beam_strips_diversity_penalty",
        "transformers_single_beam_strips_length_penalty",
        "transformers_single_beam_strips_constraints",
        # No-return-dict dormancy (return_dict_in_generate=False strips
        # scalar-output-only fields). Surfaced by introspection auto-discovery;
        # not present in the prior hand-curated corpus.
        "transformers_no_return_dict_strips_output_scores",
        "transformers_no_return_dict_strips_output_attentions",
        "transformers_no_return_dict_strips_output_hidden_states",
        # GenerationConfig.validate() — raises + announced-dormant mixture
        "transformers_negative_max_new_tokens",
        "transformers_invalid_cache_implementation",
        "transformers_invalid_early_stopping",
        "transformers_num_return_sequences_exceeds_num_beams",
        "transformers_greedy_rejects_num_return_sequences",
        "transformers_negative_pad_token_id",
        "transformers_compile_config_type",
        # BitsAndBytesConfig.post_init() type-check errors
        "transformers_bnb_load_in_4bit_type",
        "transformers_bnb_load_in_8bit_type",
        "transformers_bnb_llm_int8_threshold_type",
        "transformers_bnb_llm_int8_skip_modules_type",
        "transformers_bnb_llm_int8_enable_fp32_cpu_offload_type",
        "transformers_bnb_llm_int8_has_fp16_weight_type",
        "transformers_bnb_bnb_4bit_compute_dtype_type",
        "transformers_bnb_bnb_4bit_quant_type_type",
        "transformers_bnb_bnb_4bit_use_double_quant_type",
    }
    actual_ids = {rule.id for rule in transformers_corpus.rules}
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    assert not missing, f"missing rule ids: {sorted(missing)}"
    assert not extra, f"unexpected rule ids (add to expected set or remove): {sorted(extra)}"


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
        assert rule.severity in VALID_SEVERITY, rule.id


def test_corpus_outcome_values_are_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        outcome = rule.expected_outcome.get("outcome")
        assert outcome in VALID_OUTCOME, f"{rule.id}: bad outcome {outcome!r}"


def test_corpus_emission_channels_valid(transformers_corpus) -> None:
    for rule in transformers_corpus.rules:
        channel = rule.expected_outcome.get("emission_channel")
        assert channel in VALID_EMISSION_CHANNEL, f"{rule.id}: {channel!r}"


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
    # Defers to the loader's single source of truth (VALID_ADDED_BY) — the
    # loader's UnknownAddedByError would have rejected the rule at parse
    # time if the value were outside the enum. Kept here for human-readable
    # catalogue + defence-in-depth.
    for rule in transformers_corpus.rules:
        assert rule.added_by in VALID_ADDED_BY, f"{rule.id}: added_by={rule.added_by!r}"


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
