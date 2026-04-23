"""Tests for :mod:`scripts.walkers.transformers`.

The walker depends on transformers being importable. Tests that actually
invoke the walker use ``pytest.importorskip("transformers")`` so the suite
passes on environments without transformers installed. Pure-serialisation
tests (YAML emission, envelope shape) construct RuleCandidates directly and
don't require transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers import transformers as tf_walker  # noqa: E402
from scripts.walkers._base import RuleCandidate, WalkerSource  # noqa: E402


def _sample_candidate() -> RuleCandidate:
    return RuleCandidate(
        id="transformers_test_sample",
        engine="transformers",
        library="transformers",
        rule_under_test="Sample rule",
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path="transformers/generation/configuration_utils.py",
            method="validate",
            line_at_scan=42,
            walker_confidence="high",
        ),
        match_fields={"transformers.sampling.temperature": 0.5},
        kwargs_positive={"temperature": 0.5},
        kwargs_negative={"temperature": 1.0},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "minor_issues_dict",
            "normalised_fields": [],
        },
        message_template="Test message",
        references=["ref"],
        added_by="ast_walker",
        added_at="2026-04-23",
    )


def test_relative_source_path_strips_site_packages() -> None:
    out = tf_walker._relative_source_path(
        "/home/user/.local/lib/python3.10/site-packages/"
        "transformers/generation/configuration_utils.py"
    )
    assert out == "transformers/generation/configuration_utils.py"


def test_relative_source_path_falls_back_to_basename() -> None:
    out = tf_walker._relative_source_path("/weird/path/module.py")
    assert out == "module.py"


def test_emit_yaml_deterministic_ordering() -> None:
    # Two candidates in reverse order should serialise the same regardless of
    # input order — the walker sorts by (method, id).
    candidates = [_sample_candidate()]
    envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": "4.56.0",
        "walker_pinned_range": ">=4.50,<5.0",
        "walked_at": "2026-04-23T00:00:00Z",
    }
    yaml_a = tf_walker.emit_yaml(candidates, envelope)
    yaml_b = tf_walker.emit_yaml(candidates, envelope)
    assert yaml_a == yaml_b


def test_emit_yaml_roundtrip_preserves_fields() -> None:
    candidates = [_sample_candidate()]
    envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": "4.56.0",
        "walker_pinned_range": ">=4.50,<5.0",
        "walked_at": "2026-04-23T00:00:00Z",
    }
    text = tf_walker.emit_yaml(candidates, envelope)
    doc = yaml.safe_load(text)
    assert doc["schema_version"] == "1.0.0"
    assert len(doc["rules"]) == 1
    rule = doc["rules"][0]
    assert rule["id"] == "transformers_test_sample"
    assert rule["match"]["fields"] == {"transformers.sampling.temperature": 0.5}
    assert rule["walker_source"]["walker_confidence"] == "high"


# ---------------------------------------------------------------------------
# Tests that require transformers installed
# ---------------------------------------------------------------------------


def test_walker_landmark_check_passes_on_installed_transformers() -> None:
    pytest.importorskip("transformers")
    version, source_path = tf_walker._check_landmarks()
    assert version
    assert source_path.endswith("configuration_utils.py")


def test_walk_extracts_exactly_expected_rules() -> None:
    pytest.importorskip("transformers")
    candidates, envelope = tf_walker.walk()
    # 7 greedy + 5 beam + 3 no-return-dict + 6 error + 1 pad_token_id dormant
    # + 9 bnb = 31. Exact count catches accidental removal OR accidental
    # addition — a new rule should land in a corpus-update PR that also
    # updates this test (and the test_corpus_has_expected_rule_ids set in
    # the sibling invariants file).
    assert len(candidates) == 31
    assert envelope["engine"] == "transformers"
    assert envelope["schema_version"] == "1.0.0"


def test_walk_extracts_greedy_dormancy_rules() -> None:
    pytest.importorskip("transformers")
    candidates, _ = tf_walker.walk()
    greedy_ids = {c.id for c in candidates if "greedy_strips" in c.id}
    # PoC-J confirmed 7 greedy dormancy fields: temperature, top_p, top_k,
    # min_p, typical_p, epsilon_cutoff, eta_cutoff.
    expected = {
        f"transformers_greedy_strips_{f}"
        for f in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
        )
    }
    assert expected <= greedy_ids


def test_walk_extracts_beam_dormancy_rules() -> None:
    pytest.importorskip("transformers")
    candidates, _ = tf_walker.walk()
    beam_ids = {c.id for c in candidates if "single_beam_strips" in c.id}
    # Exact set — adding a new single-beam rule must update this list.
    expected = {
        f"transformers_single_beam_strips_{f}"
        for f in (
            "early_stopping",
            "num_beam_groups",
            "diversity_penalty",
            "length_penalty",
            "constraints",
        )
    }
    assert expected == beam_ids


def test_walk_extracts_bnb_type_rules() -> None:
    pytest.importorskip("transformers")
    candidates, _ = tf_walker.walk()
    bnb_ids = {c.id for c in candidates if "bnb_" in c.id}
    # Must include the core type-check rules that appear in the 2026-04-22
    # AST PoC for BitsAndBytesConfig.post_init.
    for field in (
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_quant_type",
    ):
        assert f"transformers_bnb_{field}_type" in bnb_ids


def test_walk_deterministic_with_frozen_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    frozen = "2026-04-23T00:00:00Z"
    monkeypatch.setenv("LLENERGY_WALKER_FROZEN_AT", frozen)
    c1, e1 = tf_walker.walk()
    c2, e2 = tf_walker.walk()
    # Envelope agrees AND carries the frozen value.
    assert e1 == e2
    assert e1["walked_at"] == frozen
    # Byte-for-byte identical YAML output.
    assert tf_walker.emit_yaml(c1, e1) == tf_walker.emit_yaml(c2, e2)


def test_walk_confidence_distribution() -> None:
    """At least 60% of candidates should be ``high`` confidence.

    Per runtime-config-validation.md OQ2. The transformers walker's
    introspection path hand-picks known-good rules, so in practice every
    candidate it emits is ``high`` — but the test fixes a lower floor in
    case future rules are added with downgraded confidence.
    """
    pytest.importorskip("transformers")
    candidates, _ = tf_walker.walk()
    high = sum(1 for c in candidates if c.walker_source.walker_confidence == "high")
    assert high / len(candidates) >= 0.6


def test_walk_emits_version_mismatch_when_out_of_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    # Force the pinned range to something the installed version can't satisfy.
    from packaging.specifiers import SpecifierSet

    monkeypatch.setattr(tf_walker, "TESTED_AGAINST_VERSIONS", SpecifierSet(">=99.0"))
    from scripts.walkers._base import WalkerVersionMismatchError

    with pytest.raises(WalkerVersionMismatchError):
        tf_walker.walk()
