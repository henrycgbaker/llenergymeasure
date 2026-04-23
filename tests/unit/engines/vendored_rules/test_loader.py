"""Tests for :class:`VendoredRulesLoader` and corpus envelope parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.engines.vendored_rules import (
    UnsupportedSchemaVersionError,
    VendoredRules,
    VendoredRulesLoader,
)

_CORPUS_MINIMAL = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
rules:
  - id: transformers_test_rule
    engine: transformers
    library: transformers
    rule_under_test: "Test rule"
    severity: dormant
    native_type: transformers.GenerationConfig
    walker_source:
      path: transformers/generation/configuration_utils.py
      method: validate
      line_at_scan: 42
      walker_confidence: high
    match:
      engine: transformers
      fields:
        transformers.sampling.temperature: {present: true}
    kwargs_positive:
      temperature: 0.5
    kwargs_negative:
      temperature: null
    expected_outcome:
      outcome: dormant_announced
      emission_channel: minor_issues_dict
      normalised_fields: []
    message_template: "Dormant {declared_value}"
    references:
      - "ref"
    added_by: ast_walker
    added_at: "2026-04-23"
"""


def _write_corpus(root: Path, engine: str, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{engine}.yaml").write_text(text)


def test_load_rules_returns_parsed_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    corpus = loader.load_rules("transformers")
    assert isinstance(corpus, VendoredRules)
    assert corpus.engine == "transformers"
    assert corpus.schema_version == "1.0.0"
    assert corpus.engine_version == "4.56.0"
    assert len(corpus.rules) == 1
    assert corpus.rules[0].id == "transformers_test_rule"


def test_load_rules_per_instance_cache(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    corpus1 = loader.load_rules("transformers")
    corpus2 = loader.load_rules("transformers")
    # Same identity: pulled from cache on second call.
    assert corpus1 is corpus2


def test_invalidate_clears_cache(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    first = loader.load_rules("transformers")
    loader.invalidate("transformers")
    second = loader.load_rules("transformers")
    # Different instances: cache was cleared.
    assert first is not second


def test_invalidate_all_clears_all(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    loader.load_rules("transformers")
    assert loader._cache
    loader.invalidate()
    assert not loader._cache


def test_missing_corpus_raises_file_not_found(tmp_path: Path) -> None:
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_rules("transformers")


def test_unsupported_major_version_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace('"1.0.0"', '"2.0.0"', 1)
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnsupportedSchemaVersionError):
        loader.load_rules("transformers")


def test_missing_schema_version_raises(tmp_path: Path) -> None:
    corpus = """\
engine: transformers
engine_version: "4.56.0"
rules: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnsupportedSchemaVersionError):
        loader.load_rules("transformers")


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", "- just a list")
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        loader.load_rules("transformers")


def test_empty_rules_list_is_valid(tmp_path: Path) -> None:
    corpus = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
rules: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    result = loader.load_rules("transformers")
    assert result.rules == ()


def test_default_corpus_root_resolves_to_configs(tmp_path: Path) -> None:
    # Constructing without corpus_root uses the repo's configs/validation_rules/.
    loader = VendoredRulesLoader()
    assert loader.corpus_root.name == "validation_rules"
    assert loader.corpus_root.parent.name == "configs"
