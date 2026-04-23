"""Tests for the equivalence-groups sidecar writer + post-run H3 grouping."""

from __future__ import annotations

from pathlib import Path

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    PostRunH3Group,
    PreRunGroup,
    build_pre_run_groups,
    find_h3_groups,
    load_equivalence_groups,
    write_equivalence_groups,
)
from llenergymeasure.study.sweep_canonicalise import dedup_sweep


def _mk_config(**overrides):
    base = {"task": {"model": "gpt2"}, "engine": "transformers"}
    base.update(overrides)
    return ExperimentConfig(**base)


class TestRoundTripSerialisation:
    def test_write_then_load_preserves_fields(self, tmp_path: Path):
        groups = EquivalenceGroups(
            study_id="study_abc",
            dedup_mode="h1",
            vendored_rules_version="transformers:4.56.0@deadbee",
            groups=[
                PreRunGroup(
                    h1_hash="sha256:abc",
                    canonical_config_excerpt={"engine": "transformers"},
                    member_experiment_ids=("exp_0001", "exp_0002"),
                    member_count=2,
                    representative_experiment_id="exp_0001",
                    would_dedup=True,
                    deduplicated=True,
                )
            ],
            post_run_h3_groups=[
                PostRunH3Group(
                    h3_hash="sha256:def",
                    engine="transformers",
                    library_version="4.56.0",
                    member_h1_hashes=("sha256:abc", "sha256:xyz"),
                    member_experiment_ids=("exp_0001", "exp_0003"),
                    gap_detected=True,
                    proposed_rule_id="candidate_rule_1",
                )
            ],
        )
        path = tmp_path / "equivalence_groups.json"
        write_equivalence_groups(groups, path)
        loaded = load_equivalence_groups(path)

        assert loaded.study_id == "study_abc"
        assert loaded.dedup_mode == "h1"
        assert loaded.vendored_rules_version.startswith("transformers:")
        assert len(loaded.groups) == 1
        assert loaded.groups[0].member_count == 2
        assert loaded.groups[0].would_dedup is True
        assert len(loaded.post_run_h3_groups) == 1
        assert loaded.post_run_h3_groups[0].gap_detected is True


class TestBuildPreRunGroups:
    def test_binds_indices_to_experiment_ids(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.7}})
        result = dedup_sweep([cfg_a, cfg_b])
        # Both collapse via the real corpus' greedy rules.
        pre = build_pre_run_groups(result, experiment_ids=["exp_a", "exp_b"])
        assert len(pre) == 1
        assert pre[0].member_experiment_ids == ("exp_a", "exp_b")
        assert pre[0].representative_experiment_id == "exp_a"
        assert pre[0].would_dedup is True
        assert pre[0].deduplicated is True

    def test_without_dedup_groups_record_would_dedup(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.7}})
        result = dedup_sweep([cfg_a, cfg_b], deduplicate=False)
        pre = build_pre_run_groups(result, experiment_ids=["exp_a", "exp_b"])
        assert len(pre) == 1
        assert pre[0].would_dedup is True
        assert pre[0].deduplicated is False

    def test_id_length_mismatch_raises(self):
        cfg = _mk_config()
        result = dedup_sweep([cfg])
        try:
            build_pre_run_groups(result, experiment_ids=[])
        except ValueError as exc:
            assert "does not match" in str(exc)
            return
        raise AssertionError("expected ValueError")


class TestFindH3Groups:
    def test_flags_gap_when_same_h3_distinct_h1(self):
        sidecars = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_a",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_b",  # Distinct H1 — gap!
                "h3_hash": "h3_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_h3_groups(sidecars)
        assert len(groups) == 1
        assert groups[0].gap_detected is True

    def test_no_flag_when_h1_same(self):
        # Same H1 collapsing on the library side is not a gap — the
        # canonicaliser already saw them as equivalent.
        sidecars = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_a",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_a",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_h3_groups(sidecars)
        assert len(groups) == 1
        assert groups[0].gap_detected is False

    def test_distinct_h3_no_group(self):
        sidecars = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_a",
                "h3_hash": "h3_a",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_b",
                "h3_hash": "h3_b",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_h3_groups(sidecars)
        assert groups == []

    def test_different_versions_do_not_cross_groups(self):
        sidecars = [
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_a",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_a",
            },
            {
                "engine": "transformers",
                "library_version": "4.57.0",  # Different version
                "h1_hash": "h1_b",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_b",
            },
        ]
        groups = find_h3_groups(sidecars)
        # Version mismatch — grouped separately, each with len=1, so no gap.
        assert groups == []

    def test_sidecar_missing_h3_skipped(self):
        sidecars = [
            {"engine": "transformers", "library_version": "4.56.0", "h1_hash": "h1_a"},
            {
                "engine": "transformers",
                "library_version": "4.56.0",
                "h1_hash": "h1_b",
                "h3_hash": "h3_shared",
                "experiment_id": "exp_b",
            },
        ]
        # Only one valid sidecar; no group formed.
        groups = find_h3_groups(sidecars)
        assert groups == []
