"""Tests for the sweep library-resolution mechanism — fixpoint iteration + dedup.

Idempotence + shuffle-stability are enforced by
``scripts/miners/_fixpoint_test.py`` (CI-time contract — any corpus PR that
violates them is rejected before this module runs). These tests focus on
the *runtime* behaviour: does the library-resolution mechanism reach fixpoint, does it
collapse measurement-equivalent configs, does it detect cycles, does it
populate equivalence-group metadata.
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.vendored_rules.loader import Rule
from llenergymeasure.study.hashing import build_resolved_view, hash_config
from llenergymeasure.study.library_resolution import (
    LibraryResolutionCycleError,
    _apply_rules_fixpoint,
    resolve_library_effective,
)


def _mk_rule(
    *,
    rule_id: str,
    match_fields: dict,
    normalised_fields: list[str] | None = None,
    severity: str = "dormant",
) -> Rule:
    """Construct a minimal ``Rule`` for library-resolution mechanism tests."""
    return Rule(
        id=rule_id,
        engine="transformers",
        library="transformers",
        rule_under_test="",
        severity=severity,
        native_type="transformers.GenerationConfig",
        match_engine="transformers",
        match_fields=match_fields,
        kwargs_positive={},
        kwargs_negative={},
        expected_outcome={"normalised_fields": normalised_fields or []},
        message_template=None,
        miner_source={},
        references=(),
        added_by="test",
        added_at="2026-04-23",
    )


def _mk_config(**overrides):
    base = {"task": {"model": "gpt2"}, "engine": "transformers"}
    base.update(overrides)
    return ExperimentConfig(**base)


# ---------------------------------------------------------------------------
# _apply_rules_fixpoint()
# ---------------------------------------------------------------------------


class TestCanonicalise:
    def test_empty_rules_returns_deep_copy(self):
        cfg = _mk_config()
        result = _apply_rules_fixpoint(cfg, [])
        assert result is not cfg
        assert result.model_dump() == cfg.model_dump()

    def test_no_match_no_change(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": True, "temperature": 0.5}})
        rule = _mk_rule(
            rule_id="never_fires",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_rules_fixpoint(cfg, [rule])
        # do_sample=True disqualifies the rule; temperature should be untouched.
        assert result.transformers.sampling.temperature == 0.5

    def test_greedy_normalises_temperature_via_not_equal(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        rule = _mk_rule(
            rule_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_rules_fixpoint(cfg, [rule])
        # Canonical form is the not_equal sentinel.
        assert result.transformers.sampling.temperature == 1.0

    def test_idempotence_on_already_canonical(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 1.0}})
        rule = _mk_rule(
            rule_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_rules_fixpoint(cfg, [rule])
        assert result.transformers.sampling.temperature == 1.0

    def test_chained_rules_converge(self):
        # Rule A: if temperature > 0.8, strip top_p (simulating greedy-when-hot
        # semantics — imaginary for this test).
        # Rule B: if top_p is None, strip top_k.
        # Input with temperature 0.9, top_p=0.95, top_k=50 must converge in 2 passes.
        cfg = _mk_config(
            transformers={
                "sampling": {
                    "do_sample": True,
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 50,
                }
            }
        )
        rule_a = _mk_rule(
            rule_id="hot_strips_top_p",
            match_fields={
                "transformers.sampling.temperature": {">": 0.8},
                "transformers.sampling.top_p": {"present": True},
            },
        )
        rule_b = _mk_rule(
            rule_id="no_top_p_strips_top_k",
            match_fields={
                "transformers.sampling.top_p": {"absent": True},
                "transformers.sampling.top_k": {"present": True, "not_equal": 50},
            },
        )
        result = _apply_rules_fixpoint(cfg, [rule_a, rule_b])
        assert result.transformers.sampling.top_p is None
        # top_k unchanged: rule_b only fires when top_k != 50, but input is 50.
        assert result.transformers.sampling.top_k == 50

    def test_cycle_detection(self):
        # Synthetic cycle: two rules that flip a field back and forth.
        # rule_a: if top_k=50, strip it
        # rule_b: if top_k absent, set to 50 (via not_equal on present)
        # The real rule shape is constrained; simulate a cycle by assigning a
        # path that gets reset each iteration.
        cfg = _mk_config(transformers={"sampling": {"do_sample": True, "top_k": 50}})
        rule_a = _mk_rule(
            rule_id="clear_top_k",
            match_fields={
                "transformers.sampling.top_k": {"present": True, "not_equal": 999},
            },
        )
        rule_b = _mk_rule(
            rule_id="restore_top_k",
            match_fields={
                "transformers.sampling.top_k": {"present": True, "not_equal": 50},
            },
        )
        with pytest.raises(LibraryResolutionCycleError):
            _apply_rules_fixpoint(cfg, [rule_a, rule_b])

    def test_non_dormant_rules_ignored(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": True, "temperature": 0.5}})
        rule = _mk_rule(
            rule_id="not_dormant",
            severity="warn",
            match_fields={
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = _apply_rules_fixpoint(cfg, [rule])
        # warn severity must not trigger normalisation.
        assert result.transformers.sampling.temperature == 0.5


# ---------------------------------------------------------------------------
# resolve_library_effective()
# ---------------------------------------------------------------------------


class TestDedupSweep:
    def test_empty_sweep_returns_empty(self):
        result = resolve_library_effective([])
        assert result.canonical_configs == []
        assert result.groups == []

    def test_collapse_equivalent_configs(self):
        # Two configs differ only by a dormant field under greedy decoding; they
        # must collapse into one after canonicalisation.
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.7}})
        rule = _mk_rule(
            rule_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective([cfg_a, cfg_b], rules=[rule])
        assert len(result.canonical_configs) == 1
        assert len(result.groups) == 1
        assert result.groups[0].member_count == 2
        assert result.deduplicated is True
        assert result.would_dedup is True

    def test_distinct_configs_stay_distinct(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": True, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": True, "temperature": 0.7}})
        rule = _mk_rule(
            rule_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective([cfg_a, cfg_b], rules=[rule])
        assert len(result.canonical_configs) == 2
        assert len(result.groups) == 2
        assert result.would_dedup is False
        assert result.deduplicated is False

    def test_dedup_disabled_preserves_all_but_groups_still_populated(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.7}})
        rule = _mk_rule(
            rule_id="greedy_normalises_temperature",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        result = resolve_library_effective([cfg_a, cfg_b], rules=[rule], deduplicate=False)
        # Every declared config still executes.
        assert len(result.canonical_configs) == 2
        # But the groups record the collapse that *would* have happened.
        assert len(result.groups) == 1
        assert result.groups[0].member_count == 2
        assert result.would_dedup is True
        assert result.deduplicated is False

    def test_declared_resolved_hashes_length_matches_input(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": True}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False}})
        result = resolve_library_effective([cfg_a, cfg_b], rules=[])
        assert len(result.declared_resolved_hashes) == 2

    def test_integration_with_real_corpus(self):
        # End-to-end with the actual vendored rules — the original motivating
        # example: do_sample x temperature = [T,F] x [0.5, 1.0, 1.5] -> 6 configs,
        # library-resolution mechanism collapses to 4 (1 greedy canonical + 3 sampling variants).
        from llenergymeasure.config.vendored_rules.loader import VendoredRulesLoader

        loader = VendoredRulesLoader()
        rules = loader.load_rules("transformers").rules

        configs = []
        for do_sample in (True, False):
            for temp in (0.5, 1.0, 1.5):
                configs.append(
                    _mk_config(
                        transformers={"sampling": {"do_sample": do_sample, "temperature": temp}}
                    )
                )
        result = resolve_library_effective(configs, rules=rules)
        # do_sample=True x temps=[0.5, 1.0, 1.5] -> 3 distinct sampling configs
        # do_sample=False x any temp -> 1 canonical greedy
        assert len(result.canonical_configs) == 4

    def test_hashes_stable_across_repeated_dedup(self):
        cfg = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        rule = _mk_rule(
            rule_id="greedy",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        r1 = resolve_library_effective([cfg], rules=[rule])
        r2 = resolve_library_effective([cfg], rules=[rule])
        assert r1.declared_resolved_hashes == r2.declared_resolved_hashes
        assert r1.groups[0].resolved_config_hash == r2.groups[0].resolved_config_hash


# ---------------------------------------------------------------------------
# resolved_config_hashing symmetry
# ---------------------------------------------------------------------------


class TestH1HashSymmetry:
    def test_equivalent_canonical_forms_share_h1(self):
        cfg_a = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.5}})
        cfg_b = _mk_config(transformers={"sampling": {"do_sample": False, "temperature": 0.7}})
        rule = _mk_rule(
            rule_id="greedy",
            match_fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        )
        canon_a = _apply_rules_fixpoint(cfg_a, [rule])
        canon_b = _apply_rules_fixpoint(cfg_b, [rule])
        assert hash_config(build_resolved_view(canon_a)) == hash_config(
            build_resolved_view(canon_b)
        )
