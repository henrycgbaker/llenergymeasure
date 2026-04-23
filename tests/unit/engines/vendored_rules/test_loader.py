"""Tests for :class:`VendoredRulesLoader` and corpus envelope parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.engines.vendored_rules import (
    VALID_ADDED_BY,
    VALID_EMISSION_CHANNEL,
    VALID_OUTCOME,
    VALID_SEVERITY,
    UnknownAddedByError,
    UnknownEmissionChannelError,
    UnknownEnumValueError,
    UnknownOutcomeError,
    UnknownSeverityError,
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


# ---------------------------------------------------------------------------
# AddedBy provenance enum
# ---------------------------------------------------------------------------


def test_valid_added_by_set_has_all_five_provenance_classes() -> None:
    assert (
        frozenset({"ast_walker", "introspection", "manual_seed", "runtime_warning", "h3_collision"})
        == VALID_ADDED_BY
    )


@pytest.mark.parametrize(
    "provenance",
    ["ast_walker", "introspection", "manual_seed", "runtime_warning", "h3_collision"],
)
def test_all_added_by_values_round_trip(tmp_path: Path, provenance: str) -> None:
    corpus = _CORPUS_MINIMAL.replace("added_by: ast_walker", f"added_by: {provenance}")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    rules = loader.load_rules("transformers").rules
    assert rules[0].added_by == provenance


def test_unknown_added_by_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("added_by: ast_walker", "added_by: chatgpt_hallucination")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownAddedByError, match="chatgpt_hallucination"):
        loader.load_rules("transformers")


def test_missing_added_by_defaults_to_manual_seed(tmp_path: Path) -> None:
    # Omitting added_by is not a corpus authoring error — unknown provenance
    # falls back to manual_seed (the conservative default).
    corpus = _CORPUS_MINIMAL.replace("    added_by: ast_walker\n", "")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    rules = loader.load_rules("transformers").rules
    assert rules[0].added_by == "manual_seed"


# ---------------------------------------------------------------------------
# Closed-enum validation — severity / outcome / emission_channel
# ---------------------------------------------------------------------------


def test_valid_severity_set_matches_design_spec() -> None:
    assert frozenset({"dormant", "warn", "error"}) == VALID_SEVERITY


def test_valid_outcome_set_matches_design_spec() -> None:
    assert (
        frozenset({"dormant_silent", "dormant_announced", "warn", "error", "pass"}) == VALID_OUTCOME
    )


def test_valid_emission_channel_set_matches_design_spec() -> None:
    assert (
        frozenset(
            {
                "warnings_warn",
                "logger_warning",
                "logger_warning_once",
                "minor_issues_dict",
                "none",
                "runtime_exception",
            }
        )
        == VALID_EMISSION_CHANNEL
    )


def test_unknown_severity_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("severity: dormant", "severity: kritical")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownSeverityError, match="kritical"):
        loader.load_rules("transformers")


def test_unknown_outcome_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("outcome: dormant_announced", "outcome: totally_made_up")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownOutcomeError, match="totally_made_up"):
        loader.load_rules("transformers")


def test_unknown_emission_channel_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace(
        "emission_channel: minor_issues_dict", "emission_channel: smoke_signals"
    )
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = VendoredRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownEmissionChannelError, match="smoke_signals"):
        loader.load_rules("transformers")


def test_enum_value_errors_share_common_base_class() -> None:
    # Callers that don't care which enum is wrong can catch UnknownEnumValueError.
    assert issubclass(UnknownAddedByError, UnknownEnumValueError)
    assert issubclass(UnknownSeverityError, UnknownEnumValueError)
    assert issubclass(UnknownOutcomeError, UnknownEnumValueError)
    assert issubclass(UnknownEmissionChannelError, UnknownEnumValueError)


# ---------------------------------------------------------------------------
# Vendored JSON overlay (config-rules-refresh CI)
# ---------------------------------------------------------------------------


def test_overlay_applied_when_vendored_json_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.engines.vendored_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)

    vendored_payload = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": "4.56.0",
        "cases": [
            {
                "id": "transformers_test_rule",
                "outcome": "dormant_announced",
                "emission_channel": "logger_warning",
                "observed_messages": ["library emitted this"],
            }
        ],
        "divergences": [],
    }

    monkeypatch.setattr(loader_mod, "_try_load_vendored_json", lambda _engine: vendored_payload)

    result = VendoredRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert len(result.rules) == 1
    expected = result.rules[0].expected_outcome
    assert expected["observed_outcome"] == "dormant_announced"
    assert expected["observed_emission_channel"] == "logger_warning"
    assert expected["observed_messages"] == ["library emitted this"]


def test_no_overlay_when_vendored_json_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.engines.vendored_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    monkeypatch.setattr(loader_mod, "_try_load_vendored_json", lambda _e: None)

    result = VendoredRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert len(result.rules) == 1
    expected = result.rules[0].expected_outcome
    # The corpus's declared fields are preserved; no observed_* keys appear.
    assert "observed_outcome" not in expected
    assert "observed_emission_channel" not in expected


def test_overlay_skips_rules_without_matching_vendor_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.engines.vendored_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    monkeypatch.setattr(
        loader_mod,
        "_try_load_vendored_json",
        lambda _e: {"cases": [{"id": "some_other_rule", "outcome": "error"}]},
    )
    result = VendoredRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    # No matching case -> rule is returned unchanged.
    assert "observed_outcome" not in result.rules[0].expected_outcome


def test_try_load_vendored_json_rejects_non_numeric_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A corrupt commit-back could write a non-numeric schema_version
    # (e.g. "dev"). The loader must return None rather than propagating
    # UnsupportedSchemaVersionError from _major() — the vendor CI job
    # resurfaces the issue separately.
    from llenergymeasure.engines.vendored_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)

    def fake_read(_self: object, _name: str) -> str:
        import json as _json

        return _json.dumps({"schema_version": "dev", "cases": []})

    class _FakeFiles:
        def __truediv__(self, name: str) -> _FakeEntry:
            return _FakeEntry()

    class _FakeEntry:
        def read_text(self) -> str:
            import json as _json

            return _json.dumps({"schema_version": "dev", "cases": []})

    monkeypatch.setattr(loader_mod.resources, "files", lambda _pkg: _FakeFiles())

    # Should not raise; should fall back to YAML-only (no observed_* keys).
    result = VendoredRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert "observed_outcome" not in result.rules[0].expected_outcome
