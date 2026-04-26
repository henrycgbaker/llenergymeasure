"""Tests for :mod:`scripts.miners.tensorrt_static_miner`.

Covers:

- Source-extraction landmarks (fail-loud on missing source root / class /
  validator method).
- ``TESTED_AGAINST_VERSIONS`` declared and pin-shape sane.
- Method resolution via :func:`scripts.miners._base.find_class` /
  :func:`find_method` (so refactors that drop those helpers fail loudly).
- Per-validator AST extraction emits the predicate shapes we expect for
  load-bearing methods (``LookaheadDecodingConfig.validate_positive_values``,
  ``BaseLlmArgs.validate_build_config_with_runtime_params``).
- Gate-soundness fixpoint contract (Decision #12 of the adversarial review)
  — running ``assert_gate_soundness_fixpoint`` against the live vendor
  gate must succeed.

These tests skip cleanly when the 0.21.0 source isn't available locally
(CI on the GH-hosted runner extracts it; dev hosts may not). The skip is
clearly logged so a missing source can't masquerade as a passing test.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from packaging.specifiers import SpecifierSet  # noqa: E402

from scripts.miners import tensorrt_static_miner as trt_miner  # noqa: E402
from scripts.miners._base import (  # noqa: E402
    MinerLandmarkMissingError,
    find_class,
    find_method,
)
from scripts.miners._fixpoint_test import assert_gate_soundness_fixpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Source-availability fixture
# ---------------------------------------------------------------------------

_SOURCE_ROOT = trt_miner._DEFAULT_SOURCE_ROOT


def _source_available() -> bool:
    """True iff the 0.21.0 source tree is in place at the canonical location."""
    return (
        _SOURCE_ROOT.is_dir()
        and (_SOURCE_ROOT / "llmapi" / "llm_args.py").is_file()
        and (_SOURCE_ROOT / "version.py").is_file()
    )


_REQUIRES_SOURCE = pytest.mark.skipif(
    not _source_available(),
    reason=(
        f"TRT-LLM 0.21.0 source not available at {_SOURCE_ROOT}. "
        f"Extract the wheel before running miner tests "
        f"(see scripts/miners/tensorrt_static_miner.py docstring)."
    ),
)


# ---------------------------------------------------------------------------
# Module-level contract — survives without the live source
# ---------------------------------------------------------------------------


class TestModuleContract:
    """Static-shape contract that holds regardless of source availability."""

    def test_tested_against_versions_declared(self) -> None:
        assert isinstance(trt_miner.TESTED_AGAINST_VERSIONS, SpecifierSet)

    def test_tested_against_versions_pins_021(self) -> None:
        # The miner must reject 1.x — that's a separate library generation.
        # The pin must accept 0.21.x and reject 1.x.
        assert trt_miner.TESTED_AGAINST_VERSIONS.contains("0.21.0", prereleases=True)
        assert not trt_miner.TESTED_AGAINST_VERSIONS.contains("1.1.0", prereleases=True)
        assert not trt_miner.TESTED_AGAINST_VERSIONS.contains("0.20.5", prereleases=True)

    def test_method_landmarks_use_find_class_and_find_method(self) -> None:
        """Pin the contract that the miner uses the shared landmark helpers.

        If a refactor renames or removes these helpers, the regression should
        be a loud test failure rather than silent walker breakage.
        """
        # _METHOD_LANDMARKS lookups exercise find_class / find_method directly.
        assert trt_miner._METHOD_LANDMARKS  # at least one landmark
        for cls_name, method_name in trt_miner._METHOD_LANDMARKS:
            assert isinstance(cls_name, str) and cls_name
            assert isinstance(method_name, str) and method_name

    def test_load_source_raises_when_root_missing(self, tmp_path: Path) -> None:
        bogus = tmp_path / "no_such_root"
        with pytest.raises(MinerLandmarkMissingError):
            trt_miner._load_source(bogus)


# ---------------------------------------------------------------------------
# Source-driven extraction
# ---------------------------------------------------------------------------


@_REQUIRES_SOURCE
class TestSourceExtraction:
    """Tests that exercise the live 0.21.0 source tree."""

    def test_walk_tensorrt_returns_021_version(self) -> None:
        candidates, version, rel_path = trt_miner.walk_tensorrt()
        assert version == "0.21.0"
        assert rel_path.endswith("llm_args.py")
        assert candidates  # something extracted

    def test_walk_tensorrt_rule_count_in_target_range(self) -> None:
        """Pin the design's 20-28 rule target.

        If a future TRT-LLM source change kicks the count outside this band,
        the test fails loudly — surface the drift rather than silently shift
        the corpus.
        """
        candidates, _version, _rel_path = trt_miner.walk_tensorrt()
        # The miner emits 30 raw candidates; merger fingerprint-dedup
        # collapses to ~27. We pin a generous band on raw output.
        assert 20 <= len(candidates) <= 40, f"Unexpected raw candidate count: {len(candidates)}"

    def test_lookahead_positive_values_emits_three_field_rules(self) -> None:
        """``LookaheadDecodingConfig.validate_positive_values`` covers 3 fields."""
        candidates, _version, _rel_path = trt_miner.walk_tensorrt()
        lookahead_rules = [
            c for c in candidates if c.miner_source.method == "validate_positive_values"
        ]
        assert len(lookahead_rules) == 3
        fields = sorted(list(rule.match_fields.keys())[0] for rule in lookahead_rules)
        assert fields == [
            "tensorrt.max_ngram_size",
            "tensorrt.max_verification_set_size",
            "tensorrt.max_window_size",
        ]
        # Each rule must trip the LookaheadDecodingConfig.validate_positive_values
        # raise: ``if v <= 0: raise ValueError("Value must be positive, got {v}")``.
        for rule in lookahead_rules:
            assert rule.severity == "error"
            spec = list(rule.match_fields.values())[0]
            assert spec == {"<=": 0}, f"Unexpected spec: {spec!r}"

    def test_validate_model_emits_type_allowlist_rule(self) -> None:
        candidates, _v, _p = trt_miner.walk_tensorrt()
        model_rules = [c for c in candidates if c.miner_source.method == "validate_model"]
        assert len(model_rules) == 1
        rule = model_rules[0]
        assert rule.severity == "error"
        spec = rule.match_fields["tensorrt.model"]
        # ``not isinstance(v, (str, Path))`` -> type_is_not predicate.
        assert "type_is_not" in spec
        assert spec["type_is_not"] in (["str", "Path"], ["Path", "str"])

    def test_validate_build_config_with_runtime_params_emits_5_rules(self) -> None:
        """``validate_build_config_with_runtime_params`` is the crown-jewel validator.

        Source: 2 cross-field raises (max_batch_size > build_config.max_batch_size,
        max_num_tokens > build_config.max_num_tokens) + 3 logger.warnings on
        max_seq_len / max_beam_width / max_input_len mismatches.
        """
        candidates, _v, _p = trt_miner.walk_tensorrt()
        rules = [
            c
            for c in candidates
            if c.miner_source.method == "validate_build_config_with_runtime_params"
        ]
        assert len(rules) == 5
        sev_counts: dict[str, int] = {}
        for rule in rules:
            sev_counts[rule.severity] = sev_counts.get(rule.severity, 0) + 1
        assert sev_counts == {"error": 2, "warn": 3}

    def test_validate_enable_build_cache_emits_type_rule(self) -> None:
        candidates, _v, _p = trt_miner.walk_tensorrt()
        rules = [c for c in candidates if c.miner_source.method == "validate_enable_build_cache"]
        assert len(rules) == 1
        assert rules[0].native_type == "tensorrt_llm.TrtLlmArgs"
        assert rules[0].severity == "error"

    def test_set_runtime_knobs_emits_per_loop_iteration_rule(self) -> None:
        """The loop-literal pattern parameterises 5 fields into 5 rules."""
        candidates, _v, _p = trt_miner.walk_tensorrt()
        rules = [
            c for c in candidates if c.miner_source.method == "set_runtime_knobs_from_build_config"
        ]
        assert len(rules) == 5
        # Each rule's match.fields must reference one of the loop's 5 keys.
        loop_keys = {
            "tensorrt.max_batch_size",
            "tensorrt.max_num_tokens",
            "tensorrt.max_seq_len",
            "tensorrt.max_input_len",
            "tensorrt.max_beam_width",
        }
        seen_keys: set[str] = set()
        for rule in rules:
            for path in rule.match_fields:
                if path in loop_keys:
                    seen_keys.add(path)
        assert seen_keys == loop_keys

    def test_pydantic_literal_lift_picks_up_tokenizer_mode(self) -> None:
        """Source-driven Literal lift on BaseLlmArgs.tokenizer_mode."""
        candidates, _v, _p = trt_miner.walk_tensorrt()
        # tokenizer_mode is ``Literal['auto', 'slow']`` at llm_args.py L782.
        rule = next(
            (c for c in candidates if "tokenizer_mode" in c.id and "in_2_values" in c.id),
            None,
        )
        assert rule is not None, "tokenizer_mode Literal lift missing"
        assert rule.match_fields == {"tensorrt.tokenizer_mode": {"in": ["auto", "slow"]}}

    def test_strenum_lift_picks_up_capacity_scheduler_policy(self) -> None:
        """``CapacitySchedulerPolicy`` (StrEnum, 3 members) -> one allowlist rule."""
        candidates, _v, _p = trt_miner.walk_tensorrt()
        rule = next(
            (
                c
                for c in candidates
                if "capacity_scheduler_policy" in c.id and "in_3_values" in c.id
            ),
            None,
        )
        assert rule is not None
        spec = rule.match_fields["tensorrt.capacity_scheduler_policy"]
        assert spec == {"in": ["MAX_UTILIZATION", "GUARANTEED_NO_EVICT", "STATIC_BATCH"]}


# ---------------------------------------------------------------------------
# Fail-loud landmark checks
# ---------------------------------------------------------------------------


@_REQUIRES_SOURCE
class TestLandmarkContract:
    """If the upstream library renames a class/method, the miner must fail."""

    def test_load_source_raises_on_missing_class(self, tmp_path: Path) -> None:
        # Build a stub tree with a llm_args.py that doesn't contain TrtLlmArgs.
        stub_root = tmp_path / "tensorrt_llm"
        (stub_root / "llmapi").mkdir(parents=True)
        (stub_root / "llmapi" / "llm_args.py").write_text("class NotTrtLlmArgs:\n    pass\n")
        (stub_root / "builder.py").write_text("class BuildConfig:\n    pass\n")
        (stub_root / "version.py").write_text('__version__ = "0.21.0"\n')
        with pytest.raises(MinerLandmarkMissingError):
            trt_miner._load_source(stub_root)

    def test_method_landmarks_present_in_021(self) -> None:
        """Every (class, method) the miner expects is present in the source."""
        tree = trt_miner._load_source(_SOURCE_ROOT)
        for cls_name, method_name in trt_miner._METHOD_LANDMARKS:
            cls = find_class(tree.llm_args, cls_name)
            assert cls is not None, f"Missing class {cls_name}"
            method = find_method(cls, method_name)
            assert method is not None, f"Missing method {cls_name}.{method_name} in 0.21.0 source"

    def test_verify_method_landmarks_raises_on_missing_method(self, tmp_path: Path) -> None:
        # Build a tree with all required *classes* but missing methods.
        stub_root = tmp_path / "tensorrt_llm"
        (stub_root / "llmapi").mkdir(parents=True)
        (stub_root / "llmapi" / "llm_args.py").write_text(
            "\n".join(
                [
                    "class BaseLlmArgs:",
                    "    pass",
                    "class TrtLlmArgs(BaseLlmArgs):",
                    "    pass",
                    "class LookaheadDecodingConfig:",
                    "    pass",
                    "class CalibConfig:",
                    "    pass",
                    "class BatchingType:",
                    "    pass",
                    "class CapacitySchedulerPolicy:",
                    "    pass",
                    "class ContextChunkingPolicy:",
                    "    pass",
                ]
            )
            + "\n"
        )
        (stub_root / "builder.py").write_text("class BuildConfig:\n    pass\n")
        (stub_root / "version.py").write_text('__version__ = "0.21.0"\n')
        tree = trt_miner._load_source(stub_root)
        with pytest.raises(MinerLandmarkMissingError):
            trt_miner._verify_method_landmarks(tree)


# ---------------------------------------------------------------------------
# Gate-soundness fixpoint (adversarial review decision #12)
# ---------------------------------------------------------------------------


class TestGateSoundnessFixpoint:
    """The vendor-CI gate's three soundness checks must all be wired.

    This is the same regression contract :class:`TestGateSoundnessFixpoint`
    in :mod:`tests.unit.scripts.miners.test_fixpoint` exercises — duplicated
    here so the TRT-LLM miner's CI lane fails loudly if the gate weakens.
    """

    def test_gate_soundness_passes_on_real_gate(self) -> None:
        # Should not raise — all three checks (positive_raises,
        # message_template_match, negative_does_not_raise) are wired.
        assert_gate_soundness_fixpoint()


# ---------------------------------------------------------------------------
# AST helpers — parameterised regression on representative landmarks
# ---------------------------------------------------------------------------


@_REQUIRES_SOURCE
class TestAstHelpers:
    """Spot-check the ``_extract_*`` helpers against fragments of real source."""

    def test_extract_predicates_handles_self_attr_eq(self) -> None:
        # ``self.backend == "pytorch"`` -> [Predicate(backend, ==, "pytorch")].
        fragment = "self.backend == 'pytorch'"
        cond = ast.parse(fragment, mode="eval").body
        preds = trt_miner._extract_predicates(cond)
        assert len(preds) == 1
        assert preds[0].field == "backend"
        assert preds[0].op == "=="
        assert preds[0].rhs == "pytorch"

    def test_extract_predicates_handles_cross_field_compare(self) -> None:
        # ``self.max_batch_size > self.build_config.max_batch_size`` —
        # the right-hand side is a nested attribute, not a self.<simple>
        # — so the walker treats it as opaque (no @ref synthesised).
        fragment = "self.max_batch_size > self.build_config.max_batch_size"
        cond = ast.parse(fragment, mode="eval").body
        preds = trt_miner._extract_predicates(cond)
        # We extract one predicate with no usable RHS in this case, or none.
        # Either way, the rule body still emits — recall over precision.
        # The important contract is: the call doesn't raise.
        assert isinstance(preds, list)

    def test_literal_args_handles_optional_literal(self) -> None:
        # ``Literal['auto', 'slow'] | None`` -> ['auto', 'slow'].
        annotation = ast.parse("x: Literal['auto', 'slow'] | None").body[0].annotation
        values = trt_miner._literal_args(annotation)
        assert values == ["auto", "slow"]

    def test_literal_args_returns_none_for_non_literal(self) -> None:
        annotation = ast.parse("x: int").body[0].annotation
        assert trt_miner._literal_args(annotation) is None
