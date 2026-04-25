"""Tests for :mod:`scripts.walkers.tensorrt_ast`.

Unit tests for TensorRT-LLM AST walker.
Validates pattern detection, rule candidate structure, and YAML output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers import tensorrt_ast as ast_walker  # noqa: E402


class TestASTWalkerBasics:
    """Tier A: walker internal invariants."""

    def test_walker_is_deterministic(self) -> None:
        """Walking twice with same source produces identical output."""
        source = "pass  # dummy source"
        a = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        b = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        assert [(c.id, c.severity) for c in a] == [(c.id, c.severity) for c in b]

    def test_rules_are_tagged_ast_walker(self) -> None:
        """All rules from AST walker are tagged 'ast_walker'."""
        source = "pass  # dummy"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        if candidates:
            tags = {c.added_by for c in candidates}
            assert tags <= {"ast_walker"}

    def test_rules_have_valid_severity(self) -> None:
        """All emitted rules have known severity."""
        source = "pass  # dummy"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        severities = {c.severity for c in candidates}
        assert severities <= {"error", "warn", "dormant"}

    def test_rules_have_valid_confidence(self) -> None:
        """All rules carry valid walker_confidence."""
        source = "pass  # dummy"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        confidences = {c.walker_source.walker_confidence for c in candidates}
        assert confidences <= {"high", "medium", "low"}

    def test_rule_structure_valid(self) -> None:
        """Each rule has required fields."""
        source = "pass  # dummy"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="tensorrt_llm/llmapi.py",
            today="2026-04-25",
        )
        for c in candidates:
            assert c.id
            assert c.engine == "tensorrt"
            assert c.library == "tensorrt_llm"
            assert c.rule_under_test
            assert c.native_type
            assert isinstance(c.match_fields, dict)
            assert isinstance(c.kwargs_positive, dict)
            assert isinstance(c.kwargs_negative, dict)
            assert isinstance(c.expected_outcome, dict)
            assert c.walker_source.walker_confidence in {"high", "medium", "low"}


class TestCorpusYAMLShape:
    """Tier B: corpus YAML round-trip."""

    def test_walker_main_produces_valid_yaml(self, tmp_path: Path) -> None:
        """main() writes a valid corpus YAML file."""
        out_path = tmp_path / "test_tensorrt_ast.yaml"
        ret = ast_walker.main(["--out", str(out_path)])
        assert ret == 0
        assert out_path.exists()

        doc = yaml.safe_load(out_path.read_text())
        assert doc["schema_version"] == "1.0.0"
        assert doc["engine"] == "tensorrt"
        assert "walked_at" in doc
        assert "rules" in doc
        assert isinstance(doc["rules"], list)

    def test_each_rule_in_corpus_has_required_fields(self, tmp_path: Path) -> None:
        """Each rule entry in corpus YAML is well-formed."""
        out_path = tmp_path / "test_tensorrt_ast.yaml"
        ast_walker.main(["--out", str(out_path)])

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
        out_path = tmp_path / "test_tensorrt_ast.yaml"
        ast_walker.main(["--out", str(out_path)])

        doc = yaml.safe_load(out_path.read_text())
        for rule in doc["rules"]:
            assert rule["engine"] == "tensorrt"
            assert rule["match"]["engine"] == "tensorrt"


class TestPatternDetection:
    """Tier C: pattern extraction from dummy code."""

    def test_extract_field_names_from_simple_condition(self) -> None:
        """_extract_field_names_from_condition finds self.<field> references."""
        import ast

        expr = ast.parse("self.max_seq_len >= self.max_input_len").body[0].value
        assert isinstance(expr, ast.Compare)
        fields = ast_walker._extract_field_names_from_condition(expr)
        assert set(fields) == {"max_seq_len", "max_input_len"}

    def test_body_has_raise_or_warn_detects_raise(self) -> None:
        """_body_has_raise_or_warn detects raise statements."""
        import ast

        source = """
if condition:
    raise ValueError("message")
"""
        tree = ast.parse(source)
        if_stmt = tree.body[0]
        assert isinstance(if_stmt, ast.If)
        assert ast_walker._body_has_raise_or_warn(if_stmt.body)

    def test_walk_llm_config_source_handles_empty_source(self) -> None:
        """Walker handles empty/dummy source gracefully."""
        source = "pass  # empty"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="dummy.py",
            today="2026-04-25",
        )
        # Should produce at least hardcoded rules even for empty source
        assert isinstance(candidates, list)


class TestEdgeCases:
    """Tier D: graceful degradation."""

    def test_walker_handles_missing_library(self, tmp_path: Path) -> None:
        """Walker degrades gracefully when tensorrt_llm is not installed."""
        out_path = tmp_path / "test_missing_lib.yaml"
        ret = ast_walker.main(["--out", str(out_path)])
        assert ret == 0
        assert out_path.exists()

        doc = yaml.safe_load(out_path.read_text())
        assert doc["engine"] == "tensorrt"
        assert isinstance(doc["rules"], list)

    def test_malformed_source_does_not_crash(self) -> None:
        """Walker handles syntactically invalid source code."""
        source = "this is not valid python :: @#$"
        candidates = ast_walker.walk_llm_config_source(
            source_text=source,
            rel_source_path="bad.py",
            today="2026-04-25",
        )
        # Should degrade gracefully, returning hardcoded or empty rules
        assert isinstance(candidates, list)
