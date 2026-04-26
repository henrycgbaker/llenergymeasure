"""Tests for :mod:`scripts.extractors.tensorrt_introspection_extractor`.

Unit tests for TensorRT-LLM schema introspection extractor.
Validates RuleCandidate structure, field coverage, and confidence tagging.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.extractors import tensorrt_introspection as intro  # noqa: E402


class TestIntrospectionBasics:
    """Tier A: walker internal invariants."""

    def test_walker_is_deterministic(self) -> None:
        """Walking twice with same inputs produces identical output."""
        a = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        b = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        assert [(c.id, c.severity, c.message_template) for c in a] == [
            (c.id, c.severity, c.message_template) for c in b
        ]

    def test_rules_are_tagged_introspection(self) -> None:
        """All rules from introspection walker are tagged 'introspection'."""
        candidates = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        if candidates:  # May be empty if tensorrt_llm not installed
            tags = {c.added_by for c in candidates}
            assert tags <= {"introspection"}

    def test_rules_have_valid_severity(self) -> None:
        """All emitted rules have known severity."""
        candidates = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        severities = {c.severity for c in candidates}
        assert severities <= {"error", "warn", "dormant"}

    def test_rules_have_valid_confidence(self) -> None:
        """All rules carry valid walker_confidence."""
        candidates = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        confidences = {c.walker_source.walker_confidence for c in candidates}
        assert confidences <= {"high", "medium", "low"}

    def test_rule_structure_valid(self) -> None:
        """Each rule has required fields."""
        candidates = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        for c in candidates:
            # Required fields per RuleCandidate schema
            assert c.id
            assert c.engine == "tensorrt"
            assert c.library == "tensorrt_llm"
            assert c.rule_under_test
            assert c.native_type
            assert isinstance(c.match_fields, dict)
            assert isinstance(c.kwargs_positive, dict)
            assert isinstance(c.kwargs_negative, dict)
            assert isinstance(c.expected_outcome, dict)
            assert c.walker_source.method in {"__init__", "validate"}
            assert c.walker_source.walker_confidence in {"high", "medium", "low"}


class TestCorpusYAMLShape:
    """Tier B: corpus YAML round-trip."""

    def test_walker_main_produces_valid_yaml(self, tmp_path: Path) -> None:
        """main() writes a valid corpus YAML file."""
        out_path = tmp_path / "test_tensorrt_introspection.yaml"
        ret = intro.main(["--out", str(out_path)])
        assert ret == 0
        assert out_path.exists()

        # Parse YAML to check structure
        doc = yaml.safe_load(out_path.read_text())
        assert doc["schema_version"] == "1.0.0"
        assert doc["engine"] == "tensorrt"
        assert "walked_at" in doc
        assert "rules" in doc
        assert isinstance(doc["rules"], list)

    def test_each_rule_in_corpus_has_required_fields(self, tmp_path: Path) -> None:
        """Each rule entry in corpus YAML is well-formed."""
        out_path = tmp_path / "test_tensorrt_introspection.yaml"
        intro.main(["--out", str(out_path)])

        doc = yaml.safe_load(out_path.read_text())
        required_fields = {
            "id",
            "engine",
            "library",
            "rule_under_test",
            "severity",
            "native_type",
            "walker_source",
            "match",
            "kwargs_positive",
            "kwargs_negative",
            "expected_outcome",
            "message_template",
            "references",
            "added_by",
            "added_at",
        }
        for rule in doc["rules"]:
            assert isinstance(rule, dict)
            assert required_fields <= set(rule.keys()), (
                f"Missing fields: {required_fields - set(rule.keys())}"
            )
            # Validate nested structure
            assert isinstance(rule["walker_source"], dict)
            assert "path" in rule["walker_source"]
            assert "method" in rule["walker_source"]
            assert "line_at_scan" in rule["walker_source"]
            assert "walker_confidence" in rule["walker_source"]
            assert isinstance(rule["match"], dict)
            assert "engine" in rule["match"]
            assert "fields" in rule["match"]

    def test_corpus_rules_are_queryable_by_engine(self, tmp_path: Path) -> None:
        """All rules in corpus correctly specify 'tensorrt' engine."""
        out_path = tmp_path / "test_tensorrt_introspection.yaml"
        intro.main(["--out", str(out_path)])

        doc = yaml.safe_load(out_path.read_text())
        for rule in doc["rules"]:
            assert rule["engine"] == "tensorrt"
            assert rule["match"]["engine"] == "tensorrt"


class TestEdgeCases:
    """Tier C: graceful degradation and error handling."""

    def test_walker_handles_missing_library(self, tmp_path: Path, capsys: Any) -> None:
        """Walker degrades gracefully when tensorrt_llm is not installed."""
        # We're in an environment where tensorrt_llm is likely not installed.
        # The walker should handle this and emit 0 rules with a warning.
        out_path = tmp_path / "test_missing_lib.yaml"
        ret = intro.main(["--out", str(out_path)])
        assert ret == 0
        assert out_path.exists()

        doc = yaml.safe_load(out_path.read_text())
        # Should have basic structure even with 0 rules
        assert doc["engine"] == "tensorrt"
        assert isinstance(doc["rules"], list)

    def test_candidates_to_dict_roundtrip(self) -> None:
        """_candidate_to_dict produces valid rule dicts."""
        candidates = intro.walk_tensorrt_args_rules(
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        for c in candidates:
            rule_dict = intro._candidate_to_dict(c)
            assert rule_dict["id"] == c.id
            assert rule_dict["engine"] == c.engine
            assert rule_dict["severity"] == c.severity
            assert rule_dict["match"]["engine"] == c.engine
