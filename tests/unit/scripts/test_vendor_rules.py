"""Tests for :mod:`scripts.vendor_rules` and :mod:`scripts._vendor_common`.

The test strategy: exercise the vendor script via synthetic native types —
we construct small Pydantic / dataclass / ``__slots__`` fixtures and point
the vendor step at them. Tests that touch the real transformers library
live in the workflow-smoke integration test; unit tests stay deterministic
and fast.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _vendor_common, vendor_rules  # noqa: E402
from scripts._vendor_common import (  # noqa: E402
    CaptureBuffers,
    classify_emission_channel,
    classify_outcome,
    compare_expected_vs_observed,
    diff_input_vs_state,
    extract_state,
    message_matches_template,
    message_template_to_substring,
    run_case,
)

# ---------------------------------------------------------------------------
# Synthetic native types for fixture-based testing
# ---------------------------------------------------------------------------


@dataclass
class _DataclassConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    _private: int = 0


class _SlotsConfig:
    __slots__ = ("_internal", "alpha", "beta")

    def __init__(self, alpha: int = 1, beta: str = "x", _internal: bool = False) -> None:
        self.alpha = alpha
        self.beta = beta
        self._internal = _internal


class _DictConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@dataclass
class _NormalisingConfig:
    """Dataclass that silently strips ``temperature`` when ``do_sample=False``."""

    do_sample: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.do_sample and self.temperature != 1.0:
            self.temperature = 1.0


# ---------------------------------------------------------------------------
# extract_state
# ---------------------------------------------------------------------------


class TestExtractState:
    def test_dataclass(self) -> None:
        obj = _DataclassConfig(temperature=0.5)
        state = extract_state(obj)
        assert state == {"temperature": 0.5, "top_p": 1.0}
        assert "_private" not in state

    def test_slots(self) -> None:
        obj = _SlotsConfig(alpha=2, beta="y", _internal=True)
        state = extract_state(obj)
        assert state["alpha"] == 2
        assert state["beta"] == "y"
        assert "_internal" not in state

    def test_dict_class(self) -> None:
        obj = _DictConfig(foo=1, bar="baz", _hidden=99)
        state = extract_state(obj)
        assert state["foo"] == 1
        assert state["bar"] == "baz"
        assert "_hidden" not in state

    def test_private_allowlist(self) -> None:
        obj = _DictConfig(foo=1, _commit_hash="abc123")
        state = extract_state(obj, private_allowlist={"_commit_hash"})
        assert state["_commit_hash"] == "abc123"


# ---------------------------------------------------------------------------
# diff_input_vs_state
# ---------------------------------------------------------------------------


class TestDiffInputVsState:
    def test_no_diff(self) -> None:
        kwargs = {"a": 1, "b": 2}
        state = {"a": 1, "b": 2, "c": 3}
        assert diff_input_vs_state(kwargs, state) == {}

    def test_silent_normalisation(self) -> None:
        kwargs = {"temperature": 0.9}
        state = {"temperature": 1.0}
        diffs = diff_input_vs_state(kwargs, state)
        assert diffs == {"temperature": {"declared": 0.9, "observed": 1.0}}

    def test_missing_from_state_ignored(self) -> None:
        kwargs = {"absent": "value"}
        state = {"other": "thing"}
        assert diff_input_vs_state(kwargs, state) == {}


# ---------------------------------------------------------------------------
# run_case
# ---------------------------------------------------------------------------


class TestRunCase:
    def test_captures_exception(self) -> None:
        def boom() -> None:
            raise ValueError("nope")

        buf = run_case(boom)
        assert buf.exception_type == "ValueError"
        assert buf.exception_message == "nope"
        assert buf.observed_state is None

    def test_captures_warnings(self) -> None:
        import warnings

        def warner() -> _DataclassConfig:
            warnings.warn("heads up", UserWarning, stacklevel=2)
            return _DataclassConfig()

        buf = run_case(warner)
        assert buf.exception_type is None
        assert any("heads up" in str(w) for w in buf.warnings_captured)

    def test_captures_state(self) -> None:
        def ok() -> _DataclassConfig:
            return _DataclassConfig(temperature=0.7)

        buf = run_case(ok)
        assert buf.exception_type is None
        assert buf.observed_state is not None
        assert buf.observed_state["temperature"] == 0.7

    def test_captures_logger_output(self) -> None:
        import logging

        logger_name = "llenergymeasure_test_vendor_rules_capture"

        def emitter() -> _DataclassConfig:
            logging.getLogger(logger_name).warning("observed emission")
            return _DataclassConfig()

        buf = run_case(emitter, logger_names=(logger_name,))
        assert any("observed emission" in m for m in buf.logger_messages)

    def test_preserves_warnings_when_call_raises(self) -> None:
        # Dormant-then-raise paths (e.g. deprecation warning followed by a
        # strict-mode ValueError) must preserve the warning alongside the
        # exception — both are the rule's fingerprint.
        import warnings

        def warn_then_raise() -> None:
            warnings.warn("about to fail", UserWarning, stacklevel=2)
            raise ValueError("strict mode")

        buf = run_case(warn_then_raise)
        assert buf.exception_type == "ValueError"
        assert buf.exception_message == "strict mode"
        assert any("about to fail" in str(w) for w in buf.warnings_captured)


# ---------------------------------------------------------------------------
# classify_outcome / classify_emission_channel
# ---------------------------------------------------------------------------


class TestClassify:
    def test_error_on_exception(self) -> None:
        buf = _vendor_common.CaptureBuffers(
            exception_type="ValueError",
            exception_message="x",
            warnings_captured=(),
            logger_messages=(),
            observed_state=None,
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "error"
        assert classify_emission_channel(buf) == "none"

    def test_warn_on_captured_warning(self) -> None:
        buf = _vendor_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=("heads up",),
            logger_messages=(),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "warn"
        assert classify_emission_channel(buf) == "warnings_warn"

    def test_dormant_announced_on_logger_only(self) -> None:
        buf = _vendor_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=("silent normalisation",),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "dormant_announced"
        assert classify_emission_channel(buf) == "logger_warning"

    def test_logger_warning_once_classified_when_sentinel_present(self) -> None:
        # Any sentinel-tagged line upgrades the classification from
        # logger_warning to logger_warning_once — the dedup-wrapped form is
        # the stricter claim on user visibility.
        sentinel = _vendor_common._WARNING_ONCE_SENTINEL
        buf = _vendor_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(f"{sentinel}one-shot warning from HF", "regular warning"),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_emission_channel(buf) == "logger_warning_once"

    def test_strip_warning_once_sentinel_cleans_messages(self) -> None:
        sentinel = _vendor_common._WARNING_ONCE_SENTINEL
        messages = (f"{sentinel}deprecated kwarg", "plain log")
        cleaned = _vendor_common.strip_warning_once_sentinel(messages)
        assert cleaned == ("deprecated kwarg", "plain log")
        assert all(sentinel not in m for m in cleaned)

    def test_dormant_silent_on_state_change_only(self) -> None:
        buf = _vendor_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(),
            observed_state={"a": 1},
            duration_ms=1,
        )
        assert classify_outcome(buf, {"a": {"declared": 2, "observed": 1}}) == "dormant_silent"

    def test_no_op_when_nothing_observed(self) -> None:
        buf = _vendor_common.CaptureBuffers(
            exception_type=None,
            exception_message=None,
            warnings_captured=(),
            logger_messages=(),
            observed_state={},
            duration_ms=1,
        )
        assert classify_outcome(buf, {}) == "no_op"


# ---------------------------------------------------------------------------
# compare_expected_vs_observed
# ---------------------------------------------------------------------------


class TestCompareExpectedVsObserved:
    def test_exact_match_no_divergence(self) -> None:
        divergences = compare_expected_vs_observed(
            rule_id="r",
            expected={"outcome": "error", "emission_channel": "none"},
            observed_outcome="error",
            observed_emission="none",
            silent_normalisations={},
        )
        assert divergences == []

    def test_outcome_mismatch(self) -> None:
        divergences = compare_expected_vs_observed(
            rule_id="r",
            expected={"outcome": "error"},
            observed_outcome="warn",
            observed_emission="warnings_warn",
            silent_normalisations={},
        )
        assert len(divergences) == 1
        assert divergences[0].field == "outcome"

    def test_normalised_fields_mismatch(self) -> None:
        divergences = compare_expected_vs_observed(
            rule_id="r",
            expected={"outcome": "dormant_silent", "normalised_fields": ["x", "y"]},
            observed_outcome="dormant_silent",
            observed_emission="none",
            silent_normalisations={"x": {"declared": 1, "observed": 0}},
        )
        assert any(d.field == "normalised_fields" for d in divergences)


# ---------------------------------------------------------------------------
# vendor_rule — end-to-end on a synthetic corpus
# ---------------------------------------------------------------------------


class TestVendorRuleSynthetic:
    """Exercise ``vendor_rule`` via a synthetic engine runner.

    We monkeypatch the transformers runner to point at our synthetic configs.
    This covers the full vendor_rule loop without needing transformers installed.
    """

    @pytest.fixture
    def patched_runner(self, monkeypatch: pytest.MonkeyPatch):
        def synthetic_runner(
            native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
        ) -> _vendor_common.CaptureBuffers:
            if native_type == "test.raises":
                return run_case(lambda: (_ for _ in ()).throw(ValueError("expected")))
            if native_type == "test.normalises":
                return run_case(lambda: _NormalisingConfig(**kwargs))
            return run_case(lambda: _DataclassConfig(**kwargs))

        monkeypatch.setitem(vendor_rules._ENGINE_RUNNERS, "transformers", synthetic_runner)
        return synthetic_runner

    def test_error_rule_positive_confirmed(self, patched_runner: Any) -> None:
        rule = {
            "id": "test_raises",
            "severity": "error",
            "native_type": "test.raises",
            "kwargs_positive": {},
            "kwargs_negative": {},
            "expected_outcome": {"outcome": "error", "emission_channel": "none"},
        }
        result = vendor_rules.vendor_rule("transformers", rule, gpu_mode="all")
        assert result.outcome == "error"
        assert result.positive_confirmed is True
        assert result.observed_exception is not None
        assert result.observed_exception["type"] == "ValueError"

    def test_dormant_silent_detected(self, patched_runner: Any) -> None:
        rule = {
            "id": "test_normalises",
            "severity": "dormant",
            "native_type": "test.normalises",
            "kwargs_positive": {"do_sample": False, "temperature": 0.9},
            "kwargs_negative": {"do_sample": True, "temperature": 0.9},
            "expected_outcome": {
                "outcome": "dormant_silent",
                "emission_channel": "none",
                "normalised_fields": ["temperature"],
            },
        }
        result = vendor_rules.vendor_rule("transformers", rule, gpu_mode="all")
        assert result.outcome == "dormant_silent"
        assert "temperature" in result.observed_silent_normalisations

    def test_gpu_mode_skip_skips_gpu_rule(self, patched_runner: Any) -> None:
        rule = {
            "id": "test_gpu",
            "severity": "error",
            "native_type": "test.raises",
            "requires_gpu": True,
            "kwargs_positive": {},
            "kwargs_negative": {},
            "expected_outcome": {"outcome": "error"},
        }
        result = vendor_rules.vendor_rule("transformers", rule, gpu_mode="skip")
        assert result.outcome == "skipped_hardware_dependent"
        assert result.skipped_reason == "requires_gpu_and_gpu_mode_skip"


# ---------------------------------------------------------------------------
# envelope writing
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_assemble_writes_expected_keys(self) -> None:
        envelope = vendor_rules.assemble_envelope(
            engine="transformers",
            engine_version="4.56.0",
            image_ref="test:latest",
            base_image_ref="test:latest",
            vendor_commit="abc",
            cases=[],
            divergences=[],
        )
        assert envelope["schema_version"] == "1.0.0"
        assert envelope["engine"] == "transformers"
        assert "vendored_at" in envelope
        assert envelope["cases"] == []
        assert envelope["divergences"] == []

    def test_vendor_engine_writes_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        corpus_path = tmp_path / "t.yaml"
        corpus_path.write_text(
            "schema_version: 1.0.0\nengine: transformers\nengine_version: 4.56.0\n"
            "walker_pinned_range: <5.0\nmined_at: 2026-01-01T00:00:00Z\n"
            "rules: []\n"
        )
        out_path = tmp_path / "t.json"

        monkeypatch.setattr(vendor_rules, "_resolve_engine_version", lambda _e: "test-ver")

        envelope, divergences = vendor_rules.vendor_engine(
            engine="transformers",
            corpus_path=corpus_path,
            out_path=out_path,
        )
        assert out_path.exists()
        assert envelope["engine_version"] == "test-ver"
        assert divergences == []
        written = json.loads(out_path.read_text())
        assert written["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_main_exits_2_on_missing_corpus(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    out = tmp_path / "out.json"
    exit_code = vendor_rules.main(
        [
            "--engine",
            "transformers",
            "--corpus",
            str(missing),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 2


def test_main_exits_0_on_no_divergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_path = tmp_path / "t.yaml"
    corpus_path.write_text(
        "schema_version: 1.0.0\nengine: transformers\nengine_version: 4.56.0\n"
        "walker_pinned_range: <5.0\nmined_at: 2026-01-01T00:00:00Z\nrules: []\n"
    )
    out_path = tmp_path / "t.json"

    monkeypatch.setattr(vendor_rules, "_resolve_engine_version", lambda _e: "test-ver")

    exit_code = vendor_rules.main(
        [
            "--engine",
            "transformers",
            "--corpus",
            str(corpus_path),
            "--out",
            str(out_path),
            "--fail-on-divergence",
        ]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# message_template_to_substring + message_matches_template
# ---------------------------------------------------------------------------


class TestMessageTemplateSubstring:
    def test_simple_placeholder_drop(self) -> None:
        assert (
            message_template_to_substring("`{flag}` is set to `{value}` but ...") == "` is set to `"
        )

    def test_picks_longest_static_run(self) -> None:
        assert (
            message_template_to_substring("Invalid `cache_implementation` ({val}). Choose one of:")
            == "Invalid `cache_implementation` ("
        )

    def test_only_placeholders_returns_empty(self) -> None:
        assert message_template_to_substring("{a}{b}") == ""

    def test_below_min_length_returns_empty(self) -> None:
        # "is" has only 2 non-whitespace chars - below the floor.
        assert message_template_to_substring("{a} is {b}") == ""

    def test_empty_template_returns_empty(self) -> None:
        assert message_template_to_substring("") == ""

    def test_strips_fstring_quoting_single_quote(self) -> None:
        assert (
            message_template_to_substring("f'Greedy methods do not support {x}.'")
            == "Greedy methods do not support "
        )

    def test_strips_fstring_quoting_double_quote(self) -> None:
        assert (
            message_template_to_substring('f"Greedy methods do not support {x}."')
            == "Greedy methods do not support "
        )

    def test_no_placeholders_returns_full_template(self) -> None:
        assert (
            message_template_to_substring("bnb_4bit_compute_dtype must be torch.dtype")
            == "bnb_4bit_compute_dtype must be torch.dtype"
        )


class TestMessageMatchesTemplate:
    def test_substring_match_case_insensitive(self) -> None:
        matched, fragment = message_matches_template(
            "INVALID `cache_implementation` (got 'foo'). Choose one of: ...",
            "Invalid `cache_implementation` ({val}). Choose one of: ...",
        )
        assert matched is True
        assert "cache_implementation" in fragment

    def test_no_match(self) -> None:
        matched, fragment = message_matches_template(
            "Some unrelated runtime message.",
            "Invalid `cache_implementation` ({val}). Choose one of: ...",
        )
        assert matched is False
        assert fragment != ""

    def test_too_dynamic_template(self) -> None:
        matched, fragment = message_matches_template("anything", "{a}{b}")
        assert matched is False
        assert fragment == ""

    def test_empty_observed_message(self) -> None:
        matched, _ = message_matches_template("", "expected fragment here")
        assert matched is False


# ---------------------------------------------------------------------------
# compute_gate_soundness_divergences
# ---------------------------------------------------------------------------


def _capture(
    *,
    exception_type: str | None = None,
    exception_message: str | None = None,
    warnings_captured: tuple[str, ...] = (),
    logger_messages: tuple[str, ...] = (),
    observed_state: dict[str, Any] | None = None,
) -> CaptureBuffers:
    """Convenience constructor for synthetic capture buffers."""
    return CaptureBuffers(
        exception_type=exception_type,
        exception_message=exception_message,
        warnings_captured=warnings_captured,
        logger_messages=logger_messages,
        observed_state=observed_state,
        duration_ms=0,
    )


class TestComputeGateSoundnessDivergences:
    """Decision #12 of the invariant-miner adversarial review."""

    def test_clean_error_rule_no_divergence(self) -> None:
        rule = {
            "id": "r1",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="field `a` must be non-zero")
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert divergences == []

    def test_positive_did_not_raise_for_error_severity(self) -> None:
        rule = {
            "id": "r2",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "irrelevant",
        }
        pos = _capture(observed_state={"a": 1})  # construction succeeded
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert any(d.check_failed == vendor_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_dormant_severity_accepts_warning(self) -> None:
        rule = {
            "id": "r3",
            "severity": "dormant",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "use `a=0` for stability",
        }
        pos = _capture(logger_messages=("use `a=0` for stability",))
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        # Dormant rule fired (logger.warning) - no positive_raises divergence.
        assert all(d.check_failed != vendor_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_dormant_severity_no_op_is_divergence(self) -> None:
        rule = {
            "id": "r4",
            "severity": "dormant",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "use `a=0` for stability",
        }
        pos = _capture(observed_state={"a": 1})  # nothing fired
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert any(d.check_failed == vendor_rules.CHECK_POSITIVE_RAISES for d in divergences)

    def test_message_template_match_failure(self) -> None:
        rule = {
            "id": "r5",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="totally different message")
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert any(d.check_failed == vendor_rules.CHECK_MESSAGE_TEMPLATE_MATCH for d in divergences)

    def test_message_template_too_dynamic(self) -> None:
        rule = {
            "id": "r6",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "{a}{b}",
        }
        pos = _capture(exception_type="ValueError", exception_message="anything")
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert any(
            d.check_failed == vendor_rules.CHECK_MESSAGE_TEMPLATE_TOO_DYNAMIC for d in divergences
        )

    def test_negative_raised_unexpectedly(self) -> None:
        rule = {
            "id": "r7",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(exception_type="ValueError", exception_message="field `a` must be non-zero")
        neg = _capture(exception_type="TypeError", exception_message="oops")
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        assert any(
            d.check_failed == vendor_rules.CHECK_NEGATIVE_DOES_NOT_RAISE for d in divergences
        )

    def test_divergence_dict_includes_check_failed_field(self) -> None:
        rule = {
            "id": "r8",
            "severity": "error",
            "kwargs_positive": {"a": 1},
            "kwargs_negative": {"a": 0},
            "message_template": "field `a` must be non-zero",
        }
        pos = _capture(observed_state={"a": 1})  # no raise - should trip positive_raises
        neg = _capture(observed_state={"a": 0})
        divergences = vendor_rules.compute_gate_soundness_divergences(rule, pos, neg)
        d = divergences[0].as_dict()
        assert "check_failed" in d
        assert d["check_failed"] == vendor_rules.CHECK_POSITIVE_RAISES
        assert d["rule_id"] == "r8"
        assert d["field"] == "kwargs_positive"
