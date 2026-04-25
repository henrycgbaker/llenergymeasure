"""Unit tests for the canonical corpus merger (``scripts/walkers/build_corpus.py``).

The merger orchestrates the per-engine staging extractors, dedups by
fingerprint with cross-validation provenance, and emits the canonical
:file:`configs/validation_rules/{engine}.yaml`. These tests exercise each
contract behaviour in isolation against synthetic staging files — no live
extractors, no real library dependencies.

Coverage:

- Fingerprint stability across float-precision jitter and dict-key ordering.
- Cross-validation: shared fingerprint -> single rule with both sources cited.
- Different fingerprints kept as separate rules.
- Per-field precedence: AST-walker wins predicates / kwargs; introspection
  wins message_template.
- Stability: byte-identical YAML on re-runs.
- ``--check`` mode: drift surfaces as exit 1 with a diff.
- Empty staging: error gracefully (no canonical write).
- ``cross_validated_by`` parses round-trip via the loader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Make scripts/ importable for direct module access in tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.config.vendored_rules import VendoredRulesLoader  # noqa: E402
from scripts.walkers import build_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: minimal staging-file rule shapes
# ---------------------------------------------------------------------------


def _ast_rule(
    *,
    rule_id: str = "transformers_negative_max_new_tokens",
    severity: str = "error",
    fields: dict[str, Any] | None = None,
    message: str = "max_new_tokens must be > 0 (AST-derived).",
    confidence: str = "high",
    line: int = 352,
    kwargs_positive: dict[str, Any] | None = None,
    kwargs_negative: dict[str, Any] | None = None,
    references: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "engine": "transformers",
        "library": "transformers",
        "rule_under_test": "max_new_tokens > 0",
        "severity": severity,
        "native_type": "transformers.GenerationConfig",
        "walker_source": {
            "path": "transformers/generation/configuration_utils.py",
            "method": "validate",
            "line_at_scan": line,
            "walker_confidence": confidence,
        },
        "match": {
            "engine": "transformers",
            "fields": fields or {"transformers.sampling.max_new_tokens": {"<=": 0}},
        },
        "kwargs_positive": kwargs_positive or {"max_new_tokens": -1},
        "kwargs_negative": kwargs_negative or {"max_new_tokens": 16},
        "expected_outcome": {
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        "message_template": message,
        "references": references or ["AST-walker reference"],
        "added_by": "ast_walker",
        "added_at": "2026-04-25",
    }


def _introspection_rule(
    *,
    rule_id: str = "transformers_negative_max_new_tokens",
    severity: str = "error",
    fields: dict[str, Any] | None = None,
    message: str = "`max_new_tokens` must be greater than 0, but is -1.",
    confidence: str = "high",
    observed_messages: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "engine": "transformers",
        "library": "transformers",
        "rule_under_test": "max_new_tokens > 0",
        "severity": severity,
        "native_type": "transformers.GenerationConfig",
        "walker_source": {
            "path": "transformers/generation/configuration_utils.py",
            "method": "validate",
            "line_at_scan": 0,  # introspection doesn't carry line numbers
            "walker_confidence": confidence,
        },
        "match": {
            "engine": "transformers",
            "fields": fields or {"transformers.sampling.max_new_tokens": {"<=": 0}},
        },
        "kwargs_positive": {"max_new_tokens": -1},
        "kwargs_negative": {"max_new_tokens": 16},
        "expected_outcome": {
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
            **({"observed_messages": observed_messages} if observed_messages else {}),
        },
        "message_template": message,
        "references": ["transformers.GenerationConfig — observed via construction-time ValueError"],
        "added_by": "introspection",
        "added_at": "2026-04-25",
    }


def _envelope(rules: list[dict[str, Any]], engine_version: str = "4.56.0") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": engine_version,
        "walker_pinned_range": ">=4.56,<4.57",
        "walked_at": "2026-04-25T00:00:00Z",
        "rules": rules,
    }


def _write_staging(staging_dir: Path, basename: str, envelope: dict[str, Any]) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / basename
    path.write_text(yaml.safe_dump(envelope, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_identical_rules_have_identical_fingerprints(self) -> None:
        rule_a = _ast_rule()
        rule_b = _ast_rule()
        assert build_corpus.fingerprint_rule(rule_a) == build_corpus.fingerprint_rule(rule_b)

    def test_fingerprint_ignores_dict_key_order(self) -> None:
        # canonical_serialise sorts keys; the merger inherits that.
        rule_a = _ast_rule(
            fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.num_beams": 1,
            },
        )
        rule_b = _ast_rule(
            fields={
                "transformers.sampling.num_beams": 1,
                "transformers.sampling.do_sample": False,
            },
        )
        assert build_corpus.fingerprint_rule(rule_a) == build_corpus.fingerprint_rule(rule_b)

    def test_fingerprint_stable_across_float_jitter(self) -> None:
        # Floats round to 12 sig digits via canonical_serialise; bit-level
        # jitter in the last place must not change the fingerprint.
        rule_a = _ast_rule(fields={"transformers.sampling.temperature": 0.7})
        rule_b = _ast_rule(fields={"transformers.sampling.temperature": 0.7000000000001})
        assert build_corpus.fingerprint_rule(rule_a) == build_corpus.fingerprint_rule(rule_b)

    def test_fingerprint_excludes_id_and_message(self) -> None:
        # Two rules with the same constraint but different ids / messages
        # still bucket together — the corpus is about the constraint.
        rule_a = _ast_rule(rule_id="foo", message="msg A")
        rule_b = _ast_rule(rule_id="bar", message="msg B")
        assert build_corpus.fingerprint_rule(rule_a) == build_corpus.fingerprint_rule(rule_b)

    def test_fingerprint_distinguishes_severity(self) -> None:
        rule_a = _ast_rule(severity="error")
        rule_b = _ast_rule(severity="warn")
        assert build_corpus.fingerprint_rule(rule_a) != build_corpus.fingerprint_rule(rule_b)


# ---------------------------------------------------------------------------
# Merge: cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_two_sources_one_fingerprint_merged_to_one_rule(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_introspection.yaml", _envelope([_introspection_rule()])
        )

        rules, _envelope_out = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(rules) == 1
        merged = rules[0]
        assert merged["added_by"] == "ast_walker"
        assert merged["cross_validated_by"] == ["introspection"]

    def test_introspection_message_overrides_ast_message(self, tmp_path: Path) -> None:
        # Per the precedence table, introspection's message_template wins
        # because it's the real library text.
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope([_ast_rule(message="AST-derived placeholder.")]),
        )
        _write_staging(
            staging,
            "transformers_introspection.yaml",
            _envelope([_introspection_rule(message="`max_new_tokens` must be > 0, but is -1.")]),
        )

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(rules) == 1
        assert rules[0]["message_template"] == "`max_new_tokens` must be > 0, but is -1."
        # Conflict surfaced for review (since the messages differed).
        assert "conflict_note" in rules[0]
        assert "message_template" in rules[0]["conflict_note"]

    def test_ast_kwargs_positive_overrides_introspection(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope(
                [
                    _ast_rule(
                        kwargs_positive={"max_new_tokens": -42},
                        kwargs_negative={"max_new_tokens": 99},
                    )
                ]
            ),
        )
        intro = _introspection_rule()
        intro["kwargs_positive"] = {"max_new_tokens": -1}
        intro["kwargs_negative"] = {"max_new_tokens": 16}
        _write_staging(staging, "transformers_introspection.yaml", _envelope([intro]))

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(rules) == 1
        assert rules[0]["kwargs_positive"] == {"max_new_tokens": -42}
        assert rules[0]["kwargs_negative"] == {"max_new_tokens": 99}

    def test_observed_messages_carry_over_from_introspection(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging,
            "transformers_introspection.yaml",
            _envelope(
                [
                    _introspection_rule(
                        observed_messages=["`max_new_tokens` must be greater than 0, but is -1."]
                    )
                ]
            ),
        )

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(rules) == 1
        observed = rules[0]["expected_outcome"].get("observed_messages")
        assert observed == ["`max_new_tokens` must be greater than 0, but is -1."]

    def test_walker_confidence_takes_min_across_sources(self, tmp_path: Path) -> None:
        # Even if both sources fingerprint the same, a "low" flag from any
        # source must surface — cross-validation does not auto-promote.
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope([_ast_rule(confidence="medium")]),
        )
        _write_staging(
            staging,
            "transformers_introspection.yaml",
            _envelope([_introspection_rule(confidence="low")]),
        )

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert rules[0]["walker_source"]["walker_confidence"] == "low"

    def test_references_unioned_across_sources(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope([_ast_rule(references=["AST ref 1"])]),
        )
        intro = _introspection_rule()
        intro["references"] = ["Introspection ref 2"]
        _write_staging(staging, "transformers_introspection.yaml", _envelope([intro]))

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        refs = rules[0]["references"]
        assert "AST ref 1" in refs
        assert "Introspection ref 2" in refs


# ---------------------------------------------------------------------------
# Merge: distinct fingerprints stay separate
# ---------------------------------------------------------------------------


class TestDistinctFingerprints:
    def test_different_match_fields_kept_as_two_rules(self, tmp_path: Path) -> None:
        # Same id, different match.fields -> two separate rules. Vendor CI
        # will prove which fires correctly on the live library.
        staging = tmp_path / "_staging"
        ast = _ast_rule(
            fields={"transformers.sampling.max_new_tokens": {"<=": 0}},
        )
        intro = _introspection_rule(
            fields={"transformers.sampling.max_new_tokens": {"<": 0}},
        )
        _write_staging(staging, "transformers_ast.yaml", _envelope([ast]))
        _write_staging(staging, "transformers_introspection.yaml", _envelope([intro]))

        rules, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(rules) == 2
        # Both keep their original added_by; neither has cross_validated_by.
        sources = {r["added_by"] for r in rules}
        assert sources == {"ast_walker", "introspection"}
        for r in rules:
            assert "cross_validated_by" not in r


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_repeated_runs_produce_identical_yaml(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope(
                [
                    _ast_rule(rule_id="rule_b", fields={"transformers.sampling.top_k": 51}),
                    _ast_rule(rule_id="rule_a"),
                ]
            ),
        )
        _write_staging(
            staging, "transformers_introspection.yaml", _envelope([_introspection_rule()])
        )

        first = build_corpus.build_corpus_text("transformers", tmp_path)
        second = build_corpus.build_corpus_text("transformers", tmp_path)
        assert first == second

    def test_rules_sorted_alphabetically_by_id(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope(
                [
                    _ast_rule(rule_id="zzz_late", fields={"f1": 1}),
                    _ast_rule(rule_id="aaa_early", fields={"f2": 2}),
                ]
            ),
        )
        text = build_corpus.build_corpus_text("transformers", tmp_path)
        doc = yaml.safe_load(text)
        ids = [r["id"] for r in doc["rules"]]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


class TestCheckMode:
    def test_check_passes_when_corpus_matches_staging(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_introspection.yaml", _envelope([_introspection_rule()])
        )

        # Build then immediately check -> should pass.
        build_corpus.write_corpus("transformers", tmp_path)
        code, _ = build_corpus.check_drift("transformers", tmp_path)
        assert code == 0

    def test_check_fails_with_diff_on_drift(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))

        build_corpus.write_corpus("transformers", tmp_path)

        # Mutate staging to introduce drift.
        _write_staging(
            staging,
            "transformers_ast.yaml",
            _envelope([_ast_rule(message="Different message — drift!")]),
        )
        code, diff = build_corpus.check_drift("transformers", tmp_path)
        assert code == 1
        assert "Different message" in diff

    def test_check_returns_2_when_canonical_corpus_missing(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))
        # No write_corpus call — canonical YAML missing.
        code, msg = build_corpus.check_drift("transformers", tmp_path)
        assert code == 2
        assert "not found" in msg


# ---------------------------------------------------------------------------
# Empty staging
# ---------------------------------------------------------------------------


class TestEmptyStaging:
    def test_no_staging_files_raises_filenotfounderror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No staging files"):
            build_corpus.build_corpus_text("transformers", tmp_path)

    def test_no_staging_does_not_touch_existing_corpus(self, tmp_path: Path) -> None:
        # A pre-existing corpus must NOT be wiped if the merger fails to
        # find staging — the canonical file stays untouched.
        canonical = tmp_path / "transformers.yaml"
        canonical.write_text("schema_version: 1.0.0\nengine: transformers\nrules: []\n")

        with pytest.raises(FileNotFoundError):
            build_corpus.build_corpus_text("transformers", tmp_path)

        assert canonical.read_text() == ("schema_version: 1.0.0\nengine: transformers\nrules: []\n")


# ---------------------------------------------------------------------------
# Loader round-trip — cross_validated_by parses correctly
# ---------------------------------------------------------------------------


class TestLoaderRoundTrip:
    def test_merger_output_loads_via_vendoredrulesloader(self, tmp_path: Path) -> None:
        staging = tmp_path / "_staging"
        _write_staging(staging, "transformers_ast.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_introspection.yaml", _envelope([_introspection_rule()])
        )

        build_corpus.write_corpus("transformers", tmp_path)

        loader = VendoredRulesLoader(corpus_root=tmp_path)
        parsed = loader.load_rules("transformers")
        assert len(parsed.rules) == 1
        rule = parsed.rules[0]
        assert rule.added_by == "ast_walker"
        assert rule.cross_validated_by == ("introspection",)

    def test_loader_rejects_unknown_cross_validated_by_value(self, tmp_path: Path) -> None:
        # Bypass the merger's single-source normalisation by writing a
        # corpus YAML directly with a bad cross_validated_by entry — the
        # loader must reject it, since the closed-enum guard is the
        # whole point of validating cross-validation provenance.
        from llenergymeasure.config.vendored_rules import UnknownAddedByError

        rule = _ast_rule()
        rule["cross_validated_by"] = ["NOT_A_REAL_PROVENANCE"]
        canonical = tmp_path / "transformers.yaml"
        canonical.write_text(yaml.safe_dump(_envelope([rule]), sort_keys=False))

        loader = VendoredRulesLoader(corpus_root=tmp_path)
        with pytest.raises(UnknownAddedByError):
            loader.load_rules("transformers")
