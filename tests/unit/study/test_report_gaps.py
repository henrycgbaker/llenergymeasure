"""Tests for study.report_gaps core — scanning, synthesis, YAML rendering."""

from __future__ import annotations

import json
from pathlib import Path

from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    PreRunGroup,
    write_equivalence_groups,
)
from llenergymeasure.study.report_gaps import (
    generate_report,
    load_runtime_observations,
    load_sidecars,
    render_candidates_yaml,
    scan_h3_collisions,
    scan_runtime_warnings,
)


def _write_sidecar(
    path: Path,
    *,
    experiment_id: str,
    h1: str,
    h3: str,
    engine: str = "transformers",
    library_version: str = "4.56.0",
    declared_kwargs: dict | None = None,
    effective_sampling_params: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "experiment_id": experiment_id,
        "measurement_config_hash": f"hash_{experiment_id}",
        "engine": engine,
        "library_version": library_version,
        "h1_hash": h1,
        "h3_hash": h3,
    }
    if declared_kwargs is not None:
        payload["declared_kwargs"] = declared_kwargs
    if effective_sampling_params is not None:
        payload["effective_sampling_params"] = effective_sampling_params
    path.write_text(json.dumps(payload))


class TestLoadSidecars:
    def test_walks_recursive(self, tmp_path: Path):
        _write_sidecar(tmp_path / "study_a/run_1/config.json", experiment_id="a1", h1="ha", h3="x")
        _write_sidecar(tmp_path / "study_b/run_2/config.json", experiment_id="b2", h1="hb", h3="y")
        sidecars = load_sidecars(tmp_path)
        assert len(sidecars) == 2
        ids = {s["experiment_id"] for s in sidecars}
        assert ids == {"a1", "b2"}

    def test_skips_sidecars_without_h3(self, tmp_path: Path):
        p = tmp_path / "study/run/config.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"engine": "transformers", "h1_hash": "h1"}))  # no h3
        assert load_sidecars(tmp_path) == []

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        assert load_sidecars(tmp_path / "does-not-exist") == []


class TestLoadRuntimeObservations:
    def test_parses_valid_lines(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cache.write_text(
            '{"engine":"transformers","outcome":"success","warnings":[]}\n'
            '{"engine":"vllm","outcome":"success","warnings":[]}\n'
        )
        observations = load_runtime_observations(cache)
        assert len(observations) == 2
        assert observations[0]["engine"] == "transformers"

    def test_skips_malformed_lines(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cache.write_text(
            '{"engine":"vllm","outcome":"success"}\n'
            "this is not json\n"
            '{"engine":"transformers","outcome":"success"}\n'
        )
        observations = load_runtime_observations(cache)
        assert len(observations) == 2

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_runtime_observations(tmp_path / "absent.jsonl") == []


class TestScanH3Collisions:
    def test_distinct_h1_shared_h3_flags_gap(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "run_a/config.json",
            experiment_id="a",
            h1="h1_a",
            h3="h3_shared",
            effective_sampling_params={"temperature": 1.0},
            declared_kwargs={"transformers.sampling.temperature": 0.5},
        )
        _write_sidecar(
            tmp_path / "run_b/config.json",
            experiment_id="b",
            h1="h1_b",
            h3="h3_shared",
            effective_sampling_params={"temperature": 1.0},
            declared_kwargs={"transformers.sampling.temperature": 0.7},
        )
        sidecars = load_sidecars(tmp_path)
        candidates, groups, dedup_off = scan_h3_collisions(sidecars, [])
        assert len(groups) == 1
        assert groups[0].gap_detected is True
        assert len(candidates) == 1
        assert candidates[0].source == "h3-collisions"
        assert candidates[0].engine == "transformers"
        assert candidates[0].verified is True
        assert dedup_off == 0

    def test_dedup_off_manifest_flags_unverified(self, tmp_path: Path):
        _write_sidecar(tmp_path / "run_a/config.json", experiment_id="a", h1="h1_a", h3="h3_shared")
        _write_sidecar(tmp_path / "run_b/config.json", experiment_id="b", h1="h1_b", h3="h3_shared")
        write_equivalence_groups(
            EquivalenceGroups(study_id="study_x", dedup_mode="off"),
            tmp_path / "equivalence_groups.json",
        )
        sidecars = load_sidecars(tmp_path)
        # Load the manifest list — scan_h3_collisions walks filesystem for mode.
        from llenergymeasure.study.report_gaps import resolve_equivalence_groups

        manifests = resolve_equivalence_groups(tmp_path)
        candidates, _groups, dedup_off = scan_h3_collisions(sidecars, manifests)
        assert len(candidates) == 1
        assert candidates[0].verified is False
        assert dedup_off == 1

    def test_engine_filter_applies(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "run_a/config.json",
            experiment_id="a",
            h1="h1_a",
            h3="h3_x",
            engine="transformers",
        )
        _write_sidecar(
            tmp_path / "run_b/config.json",
            experiment_id="b",
            h1="h1_b",
            h3="h3_x",
            engine="vllm",
        )
        sidecars = load_sidecars(tmp_path)
        # No collision when filter keeps only one engine at a time.
        candidates, _, _ = scan_h3_collisions(sidecars, [], engine_filter="transformers")
        assert candidates == []

    def test_single_field_diff_scores_high(self, tmp_path: Path):
        _write_sidecar(
            tmp_path / "run_a/config.json",
            experiment_id="a",
            h1="h1_a",
            h3="h3_x",
            declared_kwargs={"temperature": 0.1},
            effective_sampling_params={"temperature": 1.0},
        )
        _write_sidecar(
            tmp_path / "run_b/config.json",
            experiment_id="b",
            h1="h1_b",
            h3="h3_x",
            declared_kwargs={"temperature": 0.5},
            effective_sampling_params={"temperature": 1.0},
        )
        sidecars = load_sidecars(tmp_path)
        candidates, _, _ = scan_h3_collisions(sidecars, [])
        assert len(candidates) == 1
        assert candidates[0].confidence == "high"


class TestScanRuntimeWarnings:
    def test_unknown_warning_emits_candidate(self):
        observations = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "outcome": "success",
                "warnings": [],
                "logger_records": [
                    {
                        "level": "WARNING",
                        "name": "transformers",
                        "message": "novel-behaviour for temperature=0.001",
                    }
                ],
                "exception": None,
                "config_hash": "abc",
            }
        ]
        candidates = scan_runtime_warnings(observations)
        assert len(candidates) == 1
        assert candidates[0].source == "runtime-warnings"
        assert candidates[0].severity == "warn"
        assert "<NUM>" in candidates[0].message_template

    def test_exception_classified_as_error(self):
        observations = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "outcome": "exception",
                "warnings": [],
                "logger_records": [],
                "exception": {
                    "type": "ValueError",
                    "message": "bad config for max_num_batched_tokens=100",
                },
                "config_hash": "abc",
            }
        ]
        candidates = scan_runtime_warnings(observations)
        assert len(candidates) == 1
        assert candidates[0].severity == "error"
        assert candidates[0].expected_outcome["emission_channel"] == "runtime_exception"

    def test_known_corpus_rule_suppresses_candidate(self, monkeypatch):
        # Stub loader to return a rule whose template normalises to the same key.
        from llenergymeasure.engines.vendored_rules.loader import Rule, VendoredRules

        def _fake_rules(self, engine):
            return VendoredRules(
                engine=engine,
                schema_version="1.0.0",
                engine_version="4.56.0",
                rules=(
                    Rule(
                        id="existing_rule",
                        engine=engine,
                        library=engine,
                        rule_under_test="",
                        severity="warn",
                        native_type="transformers.GenerationConfig",
                        match_engine=engine,
                        match_fields={},
                        kwargs_positive={},
                        kwargs_negative={},
                        expected_outcome={"observed_messages_regex": [r"already-known.*"]},
                        message_template="already-known warning 0.001",
                        walker_source={},
                        references=(),
                        added_by="test",
                        added_at="2026-04-23",
                    ),
                ),
            )

        from llenergymeasure.engines.vendored_rules.loader import VendoredRulesLoader

        monkeypatch.setattr(VendoredRulesLoader, "load_rules", _fake_rules)

        observations = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "outcome": "success",
                "warnings": [],
                "logger_records": [
                    {
                        "level": "WARNING",
                        "name": "transformers",
                        "message": "already-known warning 1.0",
                    }
                ],
                "exception": None,
                "config_hash": "x",
            }
        ]
        candidates = scan_runtime_warnings(observations)
        assert candidates == []


class TestGenerateReport:
    def test_both_sources_combines(self, tmp_path: Path):
        # Set up both an H3 collision and a runtime observation.
        _write_sidecar(tmp_path / "run_a/config.json", experiment_id="a", h1="h1_a", h3="h3_shared")
        _write_sidecar(tmp_path / "run_b/config.json", experiment_id="b", h1="h1_b", h3="h3_shared")
        cache = tmp_path / "runtime_observations.jsonl"
        cache.write_text(
            json.dumps(
                {
                    "engine": "transformers",
                    "library_version": "4.56.0",
                    "outcome": "success",
                    "warnings": ["unknown warning here"],
                    "logger_records": [],
                    "exception": None,
                    "config_hash": "x",
                }
            )
            + "\n"
        )
        report = generate_report(source="both", results_dir=tmp_path, cache_path=cache)
        sources = {c.source for c in report.candidates}
        assert sources == {"h3-collisions", "runtime-warnings"}
        assert report.scanned_sidecars == 2
        assert report.scanned_observations == 1

    def test_h3_only_ignores_runtime(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cache.write_text(
            json.dumps({"engine": "transformers", "warnings": ["x"], "outcome": "success"}) + "\n"
        )
        report = generate_report(source="h3-collisions", results_dir=tmp_path, cache_path=cache)
        assert report.scanned_observations == 0


class TestRenderCandidatesYaml:
    def test_empty_yields_empty_rules(self):
        assert render_candidates_yaml([]).strip() == "rules: []"

    def test_rendered_yaml_parses_back(self, tmp_path: Path):
        import yaml

        _write_sidecar(tmp_path / "run_a/config.json", experiment_id="a", h1="h1_a", h3="h3_shared")
        _write_sidecar(tmp_path / "run_b/config.json", experiment_id="b", h1="h1_b", h3="h3_shared")
        sidecars = load_sidecars(tmp_path)
        candidates, _, _ = scan_h3_collisions(sidecars, [])
        yaml_body = render_candidates_yaml(candidates)
        parsed = yaml.safe_load(yaml_body)
        assert "rules" in parsed
        assert len(parsed["rules"]) == 1
        entry = parsed["rules"][0]
        assert entry["engine"] == "transformers"
        assert entry["severity"] == "dormant_silent"
        assert "added_by" in entry


class TestDedupOffFallback:
    def test_pre_run_group_with_would_dedup_but_no_collapse(self, tmp_path: Path):
        # Dedup off, group has would_dedup=True but deduplicated=False.
        groups = EquivalenceGroups(
            study_id="study_x",
            dedup_mode="off",
            groups=[
                PreRunGroup(
                    h1_hash="sha:1",
                    canonical_config_excerpt={"engine": "transformers"},
                    member_experiment_ids=("e1", "e2"),
                    member_count=2,
                    representative_experiment_id="e1",
                    would_dedup=True,
                    deduplicated=False,
                )
            ],
        )
        path = tmp_path / "equivalence_groups.json"
        write_equivalence_groups(groups, path)
        from llenergymeasure.study.report_gaps import resolve_equivalence_groups

        loaded = resolve_equivalence_groups(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].dedup_mode == "off"
