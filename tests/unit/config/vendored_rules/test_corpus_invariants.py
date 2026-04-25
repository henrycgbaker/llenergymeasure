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


def test_corpus_covers_required_invariants(transformers_corpus) -> None:
    """Coverage-by-invariant: every required field surface has at least one rule.

    Pins SEMANTIC coverage, not specific rule IDs. Extractor refinements
    that rename rules don't break this test; extractor regressions that
    drop coverage of a real invariant do. See
    ``.product/designs/config-deduplication-dormancy/runtime-config-validation.md``
    decision-log entry of 2026-04-25 (corpus-as-measurement principle).
    """
    rules = transformers_corpus.rules

    def covers_field(field_path: str) -> bool:
        return any(field_path in rule.match_fields for rule in rules)

    # Single-field invariants — at least one rule must touch each path.
    # Scope reflects the regenerated canonical corpus produced by
    # ``scripts/walkers/build_corpus.py`` with the vendor-validation gate.
    # Adding a path here means: a rule for that field must survive vendor
    # validation against the pinned engine library (see ``engine_version``
    # in the corpus envelope). If a path drops out, investigate WHY (real
    # extractor regression, real library change, or vendor-harness gap)
    # before weakening this list.
    required_fields = (
        # Greedy dormancy: do_sample=False / num_beams=1 strip these.
        "transformers.sampling.temperature",
        "transformers.sampling.top_p",
        "transformers.sampling.top_k",
        "transformers.sampling.min_p",
        "transformers.sampling.typical_p",
        "transformers.sampling.epsilon_cutoff",
        "transformers.sampling.eta_cutoff",
        # Single-beam dormancy.
        "transformers.sampling.early_stopping",
        "transformers.sampling.num_beam_groups",
        "transformers.sampling.diversity_penalty",
        "transformers.sampling.length_penalty",
        # No-return-dict dormancy.
        "transformers.sampling.output_scores",
        "transformers.sampling.output_attentions",
        "transformers.sampling.output_hidden_states",
        # GenerationConfig.validate() error rules.
        "transformers.sampling.max_new_tokens",
        "transformers.sampling.cache_implementation",
        "transformers.sampling.num_return_sequences",
        "transformers.sampling.pad_token_id",
        "transformers.sampling.compile_config",
        # PR #387 cross-field invariant gates.
        "transformers.sampling.num_beams",
        # Watermarking + BNB type-check paths landed in PR 5 (vendor-
        # validation gate). The BNB rules use the field paths the real
        # ExperimentConfig schema exposes (``transformers.load_in_4bit``
        # etc.) rather than the old ``transformers.quant.<field>`` paths
        # which never resolved at runtime.
        "transformers.sampling.watermarking_config",
        "transformers.load_in_4bit",
        "transformers.load_in_8bit",
        "transformers.llm_int8_threshold",
        "transformers.llm_int8_skip_modules",
        "transformers.llm_int8_enable_fp32_cpu_offload",
        "transformers.llm_int8_has_fp16_weight",
        "transformers.bnb_4bit_compute_dtype",
        "transformers.bnb_4bit_quant_type",
        "transformers.bnb_4bit_use_double_quant",
    )
    missing = [path for path in required_fields if not covers_field(path)]
    assert not missing, f"corpus is missing rules for {len(missing)} required invariants: {missing}"

    # Cross-field invariants — at least one rule must AND-combine the listed
    # fields. Catches regressions where the extractor lost the cross-field
    # predicate machinery. PR 5 added the (num_beam_groups,
    # diversity_penalty) pair (the AST walker's beam-search divisibility
    # invariant) and the (num_beams, num_return_sequences) pair (the
    # @field_ref-tightened greedy-rejects predicate).
    cross_field_pairs = (
        ("transformers.sampling.num_beams", "transformers.sampling.num_beam_groups"),
        ("transformers.sampling.num_beam_groups", "transformers.sampling.diversity_penalty"),
        ("transformers.sampling.num_beams", "transformers.sampling.num_return_sequences"),
    )
    missing_pairs = [
        pair
        for pair in cross_field_pairs
        if not any(all(p in rule.match_fields for p in pair) for rule in rules)
    ]
    assert not missing_pairs, (
        f"corpus missing cross-field rules for {len(missing_pairs)} invariants: {missing_pairs}"
    )

    # Corpus-as-measurement: the regenerated corpus is machine-extracted by
    # construction, so no rule should carry ``added_by: manual_seed``. Any
    # manual entry indicates a hand-edit of the canonical YAML that bypasses
    # the build_corpus.py + vendor-validation gate and would silently drift
    # on the next library bump. This PR's regeneration drops the legacy
    # hand-curated BNB type-check entries; the AST walker now emits them
    # under ``added_by: ast_walker``.
    manual = [rule.id for rule in rules if rule.added_by == "manual_seed"]
    assert not manual, (
        f"corpus contains {len(manual)} hand-seeded rules; corpus must be "
        f"machine-extracted (run scripts/walkers/build_corpus.py): {manual}"
    )


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
