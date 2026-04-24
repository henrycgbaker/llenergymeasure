"""Unit tests for api/report_gaps.py — predicate inference, partitioning, YAML round-trip.

All tests synthesise fixture study directories under ``tmp_path``. No GPU, no
real JSONL from a live run. JSONL schema matches
:mod:`llenergymeasure.study.runtime_observations`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llenergymeasure.api.report_gaps import (
    _field_value_distribution,
    _infer_predicate,
    _template_matched_by_corpus,
    find_runtime_gaps,
    render_yaml_fragment,
)
from llenergymeasure.config.vendored_rules import VendoredRules
from llenergymeasure.config.vendored_rules.loader import _parse_envelope
from tests.helpers.runtime_obs import (
    fake_hash as _fake_hash,
)
from tests.helpers.runtime_obs import (
    write_jsonl_record as _write_jsonl_record,
)
from tests.helpers.runtime_obs import (
    write_resolution as _write_resolution,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_empty_corpus() -> dict[str, VendoredRules]:
    """Return an empty-rules corpus for transformers (so nothing suppresses gaps)."""
    envelope = "schema_version: 1.0.0\nengine: transformers\nrules: []\n"
    return {"transformers": _parse_envelope("transformers", envelope)}


def _build_corpus_with_rule_matching(template_regex: str) -> dict[str, VendoredRules]:
    """Corpus with one rule whose ``observed_messages_regex`` matches ``template_regex``.

    Uses the minimal set of required fields expected by ``_parse_rule``.
    """
    envelope = f"""
schema_version: 1.0.0
engine: transformers
rules:
- id: t_runtime_existing
  engine: transformers
  severity: warn
  native_type: transformers.Fixture
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample:
        equals: false
  kwargs_positive:
    do_sample: false
  kwargs_negative: {{}}
  expected_outcome:
    outcome: warn
    emission_channel: logger_warning
    observed_messages_regex:
    - {json.dumps(template_regex)}
  added_by: manual_seed
"""
    return {"transformers": _parse_envelope("transformers", envelope)}


# ---------------------------------------------------------------------------
# Predicate inference
# ---------------------------------------------------------------------------


def test_predicate_inference_equality() -> None:
    """Single-field equality is recovered cleanly."""
    fired = [
        {"do_sample": False, "temperature": 0.3, "top_p": 0.95},
        {"do_sample": False, "temperature": 0.7, "top_p": 1.0},
        {"do_sample": False, "temperature": 1.0, "top_p": 0.5},
    ]
    not_fired = [
        {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
        {"do_sample": True, "temperature": 0.3, "top_p": 1.0},
    ]
    assert _infer_predicate(fired, not_fired) == {"do_sample": False}


def test_predicate_inference_multi_field() -> None:
    """Arity-2 predicate is recovered when no single field distinguishes."""
    # temperature=0.0 fires only when do_sample=True (greedy-with-temp-0 rule).
    fired = [
        {"do_sample": True, "temperature": 0.0, "top_p": 0.95},
        {"do_sample": True, "temperature": 0.0, "top_p": 1.0},
    ]
    not_fired = [
        {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
        {"do_sample": False, "temperature": 0.0, "top_p": 0.95},
        {"do_sample": False, "temperature": 0.7, "top_p": 0.95},
    ]
    assert _infer_predicate(fired, not_fired) == {"do_sample": True, "temperature": 0.0}


def test_predicate_inference_range_fails_safely() -> None:
    """Range predicate (varying fired values) returns None, evidence intact."""
    fired = [
        {"temperature": 0.001, "top_p": 0.95},
        {"temperature": 0.005, "top_p": 1.0},
        {"temperature": 0.0001, "top_p": 0.5},
    ]
    not_fired = [
        {"temperature": 0.5, "top_p": 0.95},
        {"temperature": 1.0, "top_p": 1.0},
    ]
    # Every fired config has a distinct temperature, so no single tuple is
    # shared — arity-1, 2, 3 all fail. present:true fallback is also False
    # because top_p is set in both partitions.
    assert _infer_predicate(fired, not_fired) is None
    evidence = _field_value_distribution(fired, not_fired)
    # Sanity: evidence still lists the fired temperature values.
    assert set(evidence["temperature"]["fired"]) == {"0.001", "0.005", "0.0001"}


def test_present_true_fallback() -> None:
    """Fields-set-vs-null recovers via present:true when equality fails."""
    # quant config present in all fired; absent in all not_fired.
    fired = [
        {"quant": "bnb_4bit"},
        {"quant": "bnb_8bit"},
        {"quant": "awq"},
    ]
    not_fired = [
        {"quant": None},
        {"quant": None},
    ]
    got = _infer_predicate(fired, not_fired)
    assert got == {"quant": {"present": True}}


# ---------------------------------------------------------------------------
# Partition filtering
# ---------------------------------------------------------------------------


def test_sentinel_records_excluded_from_b(tmp_path: Path) -> None:
    """subprocess_died records don't pollute the B partition.

    Setup: 2 configs fire the template; 1 sentinel config has the same
    kwargs as the fired ones (would otherwise falsify the predicate). After
    exclusion, the predicate stays inferable.
    """
    study = tmp_path / "study-1"
    study.mkdir()

    # 2 fired configs + 1 not-fired, all on do_sample=False.
    # Sentinel config has do_sample=False too — would collide with fired if B included it.
    hashes = {
        "fire_a": _fake_hash("fire_a"),
        "fire_b": _fake_hash("fire_b"),
        "notfire": _fake_hash("notfire"),
        "sentinel": _fake_hash("sentinel"),
    }
    _write_resolution(
        study, 1, 1, "transformers", hashes["fire_a"], {"do_sample": False, "temperature": 0.5}
    )
    _write_resolution(
        study, 2, 1, "transformers", hashes["fire_b"], {"do_sample": False, "temperature": 0.7}
    )
    _write_resolution(
        study, 3, 1, "transformers", hashes["notfire"], {"do_sample": True, "temperature": 0.5}
    )
    _write_resolution(
        study, 4, 1, "transformers", hashes["sentinel"], {"do_sample": False, "temperature": 0.7}
    )

    _write_jsonl_record(
        study,
        config_hash=hashes["fire_a"],
        warnings_emitted=["temperature is ignored when do_sample is False"],
    )
    _write_jsonl_record(
        study,
        config_hash=hashes["fire_b"],
        warnings_emitted=["temperature is ignored when do_sample is False"],
    )
    _write_jsonl_record(
        study,
        config_hash=hashes["notfire"],
        outcome="success",
    )
    _write_jsonl_record(
        study,
        config_hash=hashes["sentinel"],
        outcome="subprocess_died",
        exit_reason="SIGKILL",
        exit_code=-9,
    )

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus())
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.fired_count == 2
    # B partition contains only the notfire config — the sentinel is excluded.
    assert gap.not_fired_count == 1
    assert gap.match_fields == {"do_sample": False}


def test_exception_records_excluded_by_default(tmp_path: Path) -> None:
    """Exception records are excluded from the B partition by default."""
    study = tmp_path / "study-2"
    study.mkdir()

    hashes = {k: _fake_hash(k) for k in ("a", "b", "c", "exc")}
    _write_resolution(study, 1, 1, "transformers", hashes["a"], {"do_sample": False})
    _write_resolution(study, 2, 1, "transformers", hashes["b"], {"do_sample": False})
    _write_resolution(study, 3, 1, "transformers", hashes["c"], {"do_sample": True})
    _write_resolution(study, 4, 1, "transformers", hashes["exc"], {"do_sample": False})

    _write_jsonl_record(study, config_hash=hashes["a"], warnings_emitted=["temperature ignored"])
    _write_jsonl_record(study, config_hash=hashes["b"], warnings_emitted=["temperature ignored"])
    _write_jsonl_record(study, config_hash=hashes["c"], outcome="success")
    _write_jsonl_record(
        study,
        config_hash=hashes["exc"],
        outcome="exception",
        exception={
            "type": "ValueError",
            "message": "boom",
            "message_template": "boom",
            "traceback_truncated": "",
        },
    )

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus())
    assert len(gaps) == 1
    # exc record is excluded from B (would have had do_sample=False, breaking predicate).
    # Predicate still resolves cleanly.
    assert gaps[0].match_fields == {"do_sample": False}
    assert gaps[0].fired_count == 2
    assert gaps[0].not_fired_count == 1


# ---------------------------------------------------------------------------
# Corpus suppression + engine filter
# ---------------------------------------------------------------------------


def test_existing_corpus_rule_suppresses_gap(tmp_path: Path) -> None:
    """Templates matched by corpus ``observed_messages_regex`` don't produce gaps."""
    study = tmp_path / "study-3"
    study.mkdir()

    h_a = _fake_hash("covered_a")
    h_b = _fake_hash("covered_b")
    _write_resolution(study, 1, 1, "transformers", h_a, {"do_sample": False})
    _write_resolution(study, 2, 1, "transformers", h_b, {"do_sample": True})
    _write_jsonl_record(
        study,
        config_hash=h_a,
        warnings_emitted=["temperature is ignored when do_sample is False"],
    )
    _write_jsonl_record(study, config_hash=h_b)

    # The normaliser strips numbers/paths but leaves text intact; this raw
    # message has no numerics so the template equals the literal string.
    corpus = _build_corpus_with_rule_matching(r"\Atemperature is ignored when do_sample is False\Z")
    gaps = find_runtime_gaps([study], rules_corpus=corpus)
    assert gaps == []


def test_engine_filter(tmp_path: Path) -> None:
    """--engine vllm only scans vllm records."""
    study = tmp_path / "study-4"
    study.mkdir()

    h_t = _fake_hash("transformers-one")
    h_v = _fake_hash("vllm-one")
    _write_resolution(study, 1, 1, "transformers", h_t, {"do_sample": False})
    _write_resolution(study, 2, 1, "vllm", h_v, {"temperature": 0.0})
    _write_jsonl_record(
        study,
        config_hash=h_t,
        engine="transformers",
        warnings_emitted=["transformers-only warning"],
    )
    _write_jsonl_record(
        study,
        config_hash=h_v,
        engine="vllm",
        library_version="0.7.0",
        warnings_emitted=["vllm-only warning"],
    )

    # No corpus for vllm — engines without a corpus still allow proposals
    # through (the loader returns no suppressing rule).
    gaps = find_runtime_gaps([study], rules_corpus={}, engine="vllm")
    assert len(gaps) == 1
    assert gaps[0].engine == "vllm"
    assert "vllm-only" in gaps[0].normalised_template


# ---------------------------------------------------------------------------
# Include-exceptions flag
# ---------------------------------------------------------------------------


def test_include_exceptions_emits_error_severity(tmp_path: Path) -> None:
    study = tmp_path / "study-5"
    study.mkdir()

    h_exc = _fake_hash("exc-config")
    h_ok = _fake_hash("ok-config")
    _write_resolution(study, 1, 1, "transformers", h_exc, {"quant": "bnb_4bit"})
    _write_resolution(study, 2, 1, "transformers", h_ok, {"quant": None})

    _write_jsonl_record(
        study,
        config_hash=h_exc,
        outcome="exception",
        exception={
            "type": "ValueError",
            "message": "bnb_4bit requires CUDA 12",
            "message_template": "bnb_<NUM>bit requires CUDA <NUM>",
            "traceback_truncated": "",
        },
    )
    _write_jsonl_record(study, config_hash=h_ok, outcome="success")

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus(), include_exceptions=True)
    assert len(gaps) == 1
    assert gaps[0].severity == "error"
    assert gaps[0].source_channel == "runtime_exception"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_study_dir_returns_no_gaps(tmp_path: Path) -> None:
    """Study dir with no JSONL returns [] with no exception."""
    study = tmp_path / "empty"
    study.mkdir()
    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus())
    assert gaps == []


def test_empty_study_dir_list_raises() -> None:
    """Zero study dirs raises a clear error."""
    from llenergymeasure.api.report_gaps import ReportGapsError

    with pytest.raises(ReportGapsError):
        find_runtime_gaps([], rules_corpus={})


# ---------------------------------------------------------------------------
# YAML round-trip through loader
# ---------------------------------------------------------------------------


def test_round_trip_through_loader(tmp_path: Path) -> None:
    """Emitted proposal parses through load_rules without error."""
    study = tmp_path / "rt-study"
    study.mkdir()

    h_a = _fake_hash("rt-a")
    h_b = _fake_hash("rt-b")
    h_c = _fake_hash("rt-c")
    _write_resolution(study, 1, 1, "transformers", h_a, {"do_sample": False})
    _write_resolution(study, 2, 1, "transformers", h_b, {"do_sample": False})
    _write_resolution(study, 3, 1, "transformers", h_c, {"do_sample": True})

    _write_jsonl_record(study, config_hash=h_a, warnings_emitted=["round-trip test warning"])
    _write_jsonl_record(study, config_hash=h_b, warnings_emitted=["round-trip test warning"])
    _write_jsonl_record(study, config_hash=h_c)

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus())
    assert len(gaps) == 1
    yaml_body = render_yaml_fragment(gaps[0])

    # Wrap the fragment in a minimum-valid corpus envelope so the loader can
    # parse it alongside its own required top-level keys.
    # Re-parse the rendered YAML so we can inject it as a list item cleanly.
    rule_doc = yaml.safe_load(yaml_body)
    corpus_yaml = yaml.safe_dump(
        {
            "schema_version": "1.0.0",
            "engine": "transformers",
            "rules": [rule_doc],
        },
        sort_keys=False,
    )
    parsed = _parse_envelope("transformers", corpus_yaml)
    assert len(parsed.rules) == 1
    rule = parsed.rules[0]
    assert rule.added_by == "runtime_warning"
    assert rule.severity == "warn"
    assert rule.expected_outcome["emission_channel"] == "warnings_warn"
    # Banner comment present at top of raw YAML fragment output.
    assert "Rule fragment proposed by 'llem report-gaps'" in yaml_body


def test_render_yaml_error_severity_roundtrip(tmp_path: Path) -> None:
    """Proposal with severity=error parses back through the loader."""
    study = tmp_path / "rt-err"
    study.mkdir()

    h_a = _fake_hash("rte-a")
    h_b = _fake_hash("rte-b")
    _write_resolution(study, 1, 1, "transformers", h_a, {"quant": "bnb_4bit"})
    _write_resolution(study, 2, 1, "transformers", h_b, {"quant": None})

    _write_jsonl_record(
        study,
        config_hash=h_a,
        outcome="exception",
        exception={
            "type": "ValueError",
            "message": "bad quant",
            "message_template": "bad quant",
            "traceback_truncated": "",
        },
    )
    _write_jsonl_record(study, config_hash=h_b)

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus(), include_exceptions=True)
    assert len(gaps) == 1
    body = render_yaml_fragment(gaps[0])
    rule_doc = yaml.safe_load(body)
    parsed = _parse_envelope(
        "transformers",
        yaml.safe_dump({"schema_version": "1.0.0", "engine": "transformers", "rules": [rule_doc]}),
    )
    assert parsed.rules[0].severity == "error"
    assert parsed.rules[0].expected_outcome["outcome"] == "error"


# ---------------------------------------------------------------------------
# Corpus matcher helper
# ---------------------------------------------------------------------------


def test_template_matched_by_corpus_via_observed_messages() -> None:
    """Rules using plain observed_messages (vendored JSON overlay) also suppress gaps."""
    envelope = """
schema_version: 1.0.0
engine: transformers
rules:
- id: t_fixture
  engine: transformers
  severity: warn
  native_type: transformers.Fixture
  match:
    engine: transformers
    fields:
      do_sample:
        equals: false
  kwargs_positive: {do_sample: false}
  kwargs_negative: {}
  expected_outcome:
    outcome: warn
    emission_channel: logger_warning
    observed_messages:
    - "You have set temperature=0.5 which is below the minimum"
  added_by: manual_seed
"""
    corpus = _parse_envelope("transformers", envelope)
    # A different-temperature raw message normalises to the same template.
    from llenergymeasure.api.report_gaps import (
        _build_observed_template_index,
        _build_regex_index,
    )
    from llenergymeasure.study.message_normalise import normalise

    other_template = normalise("You have set temperature=0.9 which is below the minimum").template
    corpus_map = {"transformers": corpus}
    observed_idx = _build_observed_template_index(corpus_map)
    regex_idx = _build_regex_index(corpus_map)
    assert (
        _template_matched_by_corpus(
            other_template,
            corpus,
            observed_idx["transformers"],
            regex_idx["transformers"],
        )
        is True
    )


# ---------------------------------------------------------------------------
# Multi-study dir aggregation
# ---------------------------------------------------------------------------


def test_manifest_path_resolves_full_hash(tmp_path: Path) -> None:
    """When manifest.json is present, kwargs lookup uses the full hash (no prefix)."""
    study = tmp_path / "study-manifest"
    study.mkdir()

    h_fire = _fake_hash("manifest-fire")
    h_nf = _fake_hash("manifest-notfire")

    # Create experiment subdirs + sidecars.
    fire_dir = _write_resolution(study, 1, 1, "transformers", h_fire, {"do_sample": False})
    nf_dir = _write_resolution(study, 2, 1, "transformers", h_nf, {"do_sample": True})

    # Build a manifest keyed by full hash with result_file relative paths.
    manifest = {
        "schema_version": "2.0",
        "experiments": [
            {
                "config_hash": h_fire,
                "cycle": 1,
                "status": "completed",
                "result_file": f"{fire_dir.name}/result.json",
            },
            {
                "config_hash": h_nf,
                "cycle": 1,
                "status": "completed",
                "result_file": f"{nf_dir.name}/result.json",
            },
        ],
    }
    (study / "manifest.json").write_text(json.dumps(manifest))

    _write_jsonl_record(study, config_hash=h_fire, warnings_emitted=["manifest-path warn"])
    _write_jsonl_record(study, config_hash=h_nf)

    gaps = find_runtime_gaps([study], rules_corpus=_build_empty_corpus())
    assert len(gaps) == 1
    assert gaps[0].fired_count == 1
    assert gaps[0].not_fired_count == 1
    assert gaps[0].match_fields == {"do_sample": False}


def test_multiple_study_dirs_aggregate(tmp_path: Path) -> None:
    """Records from two study dirs combine into one gap set."""
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s1.mkdir()
    s2.mkdir()

    h1 = _fake_hash("s1-fire")
    h2 = _fake_hash("s2-fire")
    h3 = _fake_hash("s2-notfire")

    _write_resolution(s1, 1, 1, "transformers", h1, {"do_sample": False})
    _write_resolution(s2, 1, 1, "transformers", h2, {"do_sample": False})
    _write_resolution(s2, 2, 1, "transformers", h3, {"do_sample": True})

    _write_jsonl_record(s1, config_hash=h1, warnings_emitted=["cross-study warn"])
    _write_jsonl_record(s2, config_hash=h2, warnings_emitted=["cross-study warn"])
    _write_jsonl_record(s2, config_hash=h3)

    gaps = find_runtime_gaps([s1, s2], rules_corpus=_build_empty_corpus())
    assert len(gaps) == 1
    assert gaps[0].fired_count == 2
    assert gaps[0].not_fired_count == 1
