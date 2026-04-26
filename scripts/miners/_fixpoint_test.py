"""Shuffle-application fixpoint contract test for the dormant-rule corpus.

Enforces three invariants the dormant library-resolution mechanism will depend on:

1. **Idempotent** — applying the same rule twice leaves the state unchanged.
2. **Order-independent at fixpoint** — multiple random application orderings
   converge to the same canonical form.
3. **Cycle-free** — no rule pair alternates values indefinitely.

The test operates on a declarative projection: a dormant rule declares
``normalised_fields`` in ``expected_outcome``, and the "fix" is setting each
normalised field to its declared default (from the predicate's ``not_equal``
operand, falling back to ``None``). This is sufficient to catch the structural
failure modes the library-resolution mechanism would trip on.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_MAX_ITER = 50
"""Maximum passes per ordering before declaring non-convergence.

Each pass is one full sweep over all rules. A corpus of N rules converges in
at most N passes when all rules commute; more iterations tolerate mild
order-dependence without false-positive cycle detection.
"""

_DEFAULT_SHUFFLE_COUNT = 5
"""Number of random orderings to try per input state — empirically enough to
catch cycles without meaningfully slowing CI. Raise via ``shuffle_count`` if
a future corpus exhibits rare order-dependent modes."""


class FixpointError(Exception):
    """Base class for fixpoint-test failures."""


class LibraryResolutionCycleError(FixpointError):
    """A rule ordering failed to converge within ``_MAX_ITER`` passes."""

    def __init__(self, ordering: list[str], final_state: dict[str, Any]) -> None:
        super().__init__(
            f"Canonicaliser cycle: ordering {ordering[:5]}... did not converge "
            f"within {_MAX_ITER} passes. Last state: {final_state!r}"
        )
        self.ordering = ordering
        self.final_state = final_state


class NonIdempotentRuleError(FixpointError):
    """A single rule changed state on second application."""

    def __init__(self, rule_id: str, state_pass1: Any, state_pass2: Any) -> None:
        super().__init__(
            f"Rule {rule_id!r} is non-idempotent: "
            f"pass1 -> {state_pass1!r}, pass2 -> {state_pass2!r}"
        )
        self.rule_id = rule_id


class OrderDependentRuleError(FixpointError):
    """Two orderings produced different fixed points."""

    def __init__(
        self, state_a: dict[str, Any], state_b: dict[str, Any], offending_rules: list[str]
    ) -> None:
        super().__init__(
            f"Order-dependent corpus: different orderings produced different "
            f"fixed points. Diff: {_dict_diff(state_a, state_b)!r}. "
            f"Involved rules: {offending_rules!r}"
        )
        self.state_a = state_a
        self.state_b = state_b
        self.offending_rules = offending_rules


# ---------------------------------------------------------------------------
# Rule representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProjectedRule:
    """Minimal shape the fixpoint test needs from a corpus rule."""

    id: str
    severity: str
    match_fields: dict[str, Any]
    normalised_fields: tuple[str, ...]

    def applies(self, state: dict[str, Any]) -> bool:
        """True iff every ``match_fields`` predicate holds on ``state``."""
        for path, spec in self.match_fields.items():
            actual = state.get(path)
            if not _evaluate(actual, spec):
                return False
        return True

    def apply(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a new state with the normalised fields stripped to defaults."""
        next_state = dict(state)
        for field_path in self.normalised_fields:
            next_state[field_path] = None
        return next_state


def _evaluate(actual: Any, spec: Any) -> bool:
    """Minimal predicate evaluator — supports the operator shapes in the corpus.

    Inequality operators (``<`` / ``<=`` / ``>`` / ``>=``) treat type
    mismatches between ``actual`` and the rule's threshold as "predicate
    does not hold" rather than raising. Cross-rule seed pollution is
    real — one rule seeds ``pad_token_id`` with a string sentinel from
    a ``not_in`` predicate, and a second rule with ``{"<": 0}`` then
    runs against that state. Comparing a string with an int raises in
    Python 3, but the rule simply does not apply.
    """

    def _safe_lt(a: Any, b: Any) -> bool:
        try:
            return a < b
        except TypeError:
            return False

    def _safe_le(a: Any, b: Any) -> bool:
        try:
            return a <= b
        except TypeError:
            return False

    def _safe_gt(a: Any, b: Any) -> bool:
        try:
            return a > b
        except TypeError:
            return False

    def _safe_ge(a: Any, b: Any) -> bool:
        try:
            return a >= b
        except TypeError:
            return False

    if isinstance(spec, dict):
        if not spec:
            return True
        for op, value in spec.items():
            if op == "==" and actual != value:
                return False
            if op == "!=" and actual == value:
                return False
            if op == "<" and not (actual is not None and _safe_lt(actual, value)):
                return False
            if op == "<=" and not (actual is not None and _safe_le(actual, value)):
                return False
            if op == ">" and not (actual is not None and _safe_gt(actual, value)):
                return False
            if op == ">=" and not (actual is not None and _safe_ge(actual, value)):
                return False
            if op == "in" and actual not in value:
                return False
            if op == "not_in" and actual in value:
                return False
            if op == "present" and (actual is None) == bool(value):
                return False
            if op == "absent" and (actual is not None) == bool(value):
                return False
            if op == "equals" and actual != value:
                return False
            if op == "not_equal" and (actual is None or actual == value):
                return False
        return True
    return bool(actual == spec)


# ---------------------------------------------------------------------------
# Corpus ingestion
# ---------------------------------------------------------------------------


def load_dormant_rules(corpus: dict[str, Any]) -> list[_ProjectedRule]:
    """Return the ``_ProjectedRule`` view of every dormant rule in the corpus."""
    rules: list[_ProjectedRule] = []
    for raw in corpus.get("rules", []):
        if str(raw.get("severity", "")).lower() != "dormant":
            continue
        match = raw.get("match") or {}
        fields = match.get("fields") if isinstance(match, dict) else None
        if not isinstance(fields, dict):
            continue
        normalised = tuple(
            str(f) for f in (raw.get("expected_outcome") or {}).get("normalised_fields", [])
        )
        rules.append(
            _ProjectedRule(
                id=str(raw["id"]),
                severity="dormant",
                match_fields=dict(fields),
                normalised_fields=normalised,
            )
        )
    return rules


def construct_seed_states(rules: Iterable[_ProjectedRule]) -> list[dict[str, Any]]:
    """One representative input state per rule, sufficient to activate its match.

    Builds the minimal state from each rule's match predicates — for each
    field, picks a concrete value that satisfies the spec. This keeps the
    fixpoint sweep from needing real ``ExperimentConfig`` objects.
    """
    seeds: list[dict[str, Any]] = []
    for rule in rules:
        state: dict[str, Any] = {}
        for path, spec in rule.match_fields.items():
            state[path] = _value_satisfying(spec)
        # For dormant rules, fields scheduled to normalise must be set to a
        # non-default so the first application has something to strip.
        for path in rule.normalised_fields:
            state.setdefault(path, _value_satisfying({"present": True}))
        seeds.append(state)
    return seeds


def _value_satisfying(spec: Any) -> Any:
    """Pick a concrete value that satisfies ``spec`` (used only for seeding)."""
    if isinstance(spec, dict):
        if "==" in spec or "equals" in spec:
            return spec.get("==", spec.get("equals"))
        if spec.get("in"):
            return spec["in"][0]
        if "not_in" in spec:
            return "__sentinel_not_in_list__"
        if "<" in spec:
            return spec["<"] - 1 if isinstance(spec["<"], (int, float)) else 0
        if "<=" in spec:
            return spec["<="]
        if ">" in spec:
            return spec[">"] + 1 if isinstance(spec[">"], (int, float)) else 1
        if ">=" in spec:
            return spec[">="]
        if "not_equal" in spec:
            sentinel = spec["not_equal"]
            if isinstance(sentinel, bool):
                return not sentinel
            if isinstance(sentinel, (int, float)):
                return sentinel + 1
            return f"__sentinel_not_{sentinel!r}__"
        if spec.get("present"):
            return "__sentinel_present__"
        if spec.get("absent"):
            return None
    return spec


# ---------------------------------------------------------------------------
# Fixpoint sweep
# ---------------------------------------------------------------------------


def apply_to_fixpoint(
    state: dict[str, Any],
    rules: list[_ProjectedRule],
) -> tuple[dict[str, Any], list[str]]:
    """Apply rules in the given order repeatedly until the state stops changing.

    Returns ``(final_state, applied_rule_ids)``. Raises :class:`LibraryResolutionCycleError`
    if ``_MAX_ITER`` passes fail to converge.
    """
    current = dict(state)
    applied: list[str] = []
    for _pass in range(_MAX_ITER):
        changed = False
        for rule in rules:
            if rule.applies(current):
                next_state = rule.apply(current)
                if next_state != current:
                    current = next_state
                    applied.append(rule.id)
                    changed = True
        if not changed:
            return current, applied
    raise LibraryResolutionCycleError([r.id for r in rules], current)


def assert_idempotent(rule: _ProjectedRule, seed: dict[str, Any]) -> None:
    """Confirm that applying ``rule`` twice is the same as applying it once."""
    if not rule.applies(seed):
        return
    pass1 = rule.apply(seed)
    pass2 = rule.apply(pass1)
    if pass1 != pass2:
        raise NonIdempotentRuleError(rule.id, pass1, pass2)


def assert_shuffle_stable(
    rules: list[_ProjectedRule],
    seed: dict[str, Any],
    *,
    shuffle_count: int = _DEFAULT_SHUFFLE_COUNT,
    seed_rng: int = 0,
) -> dict[str, Any]:
    """Confirm that ``shuffle_count`` orderings all produce the same fixed point.

    Returns the canonical final state (shared across orderings). Raises
    :class:`OrderDependentRuleError` on divergence or :class:`LibraryResolutionCycleError`
    on non-convergence.
    """
    rng = random.Random(seed_rng)
    reference, reference_applied = apply_to_fixpoint(seed, rules)
    for _ in range(shuffle_count - 1):
        ordering = list(rules)
        rng.shuffle(ordering)
        candidate, candidate_applied = apply_to_fixpoint(seed, ordering)
        if candidate != reference:
            offending = sorted(set(reference_applied) ^ set(candidate_applied))
            raise OrderDependentRuleError(reference, candidate, offending)
    return reference


def fixpoint_test_corpus(
    corpus: dict[str, Any], *, shuffle_count: int = _DEFAULT_SHUFFLE_COUNT
) -> None:
    """Run the full shuffle-application test suite on a corpus dict.

    Raises on any failure. Returns silently on success.
    """
    rules = load_dormant_rules(corpus)
    if not rules:
        return
    seeds = construct_seed_states(rules)
    for rule, seed in zip(rules, seeds, strict=False):
        assert_idempotent(rule, seed)
    # Shuffle stability runs against each seed — a single failure anywhere
    # surfaces the corpus problem regardless of which rule triggered it.
    for seed in seeds:
        assert_shuffle_stable(rules, seed, shuffle_count=shuffle_count)


# ---------------------------------------------------------------------------
# Gate-soundness structural fixpoint
# ---------------------------------------------------------------------------
#
# Decision #12 of the invariant-miner adversarial review
# (`.product/designs/adversarial-review-invariant-miner-2026-04-26.md`)
# requires the vendor-CI gate to perform three checks on every rule:
#
#   positive_raises          - kwargs_positive must raise (or emit, if dormant)
#   message_template_match   - raised message contains the template's static fragment
#   negative_does_not_raise  - kwargs_negative must construct without raising
#
# Without these checks, a typo in a corpus rule's ``expected_outcome``
# silently passes - the existing per-field comparison treats missing keys
# as "no constraint". The structural fixpoint below pins those checks in
# place: it synthesises one malformed rule per check and asserts the gate
# records exactly the matching divergence. If any of the three checks is
# removed from ``vendor_rules.compute_gate_soundness_divergences``, the
# corresponding fixpoint case fails loudly.


class GateSoundnessRegressionError(FixpointError):
    """One of the three vendor-gate-soundness checks failed to fire.

    Carries the check name + the divergences the gate actually produced
    so the failure message points directly at the missing check.
    """

    def __init__(self, check_name: str, observed_divergences: list[Any]) -> None:
        super().__init__(
            f"Vendor gate-soundness regression: check {check_name!r} did not "
            f"surface a divergence on a malformed rule designed to trip it. "
            f"Observed divergences: {observed_divergences!r}. "
            f"This indicates the gate has been weakened - restore the check "
            f"in scripts/vendor_rules.compute_gate_soundness_divergences."
        )
        self.check_name = check_name
        self.observed_divergences = observed_divergences


def synthesise_malformed_rule_cases() -> list[dict[str, Any]]:
    """Return one malformed-rule scenario per gate-soundness check.

    Each scenario is ``{check_name, rule, pos_capture, neg_capture}`` -
    feed ``rule, pos_capture, neg_capture`` to the gate and assert a
    divergence with ``check_failed == check_name`` comes out.
    """
    # Imported here to avoid a circular path dependency:
    # ``_fixpoint_test`` is loaded eagerly by ``tests/integration/...``, and
    # ``scripts._vendor_common`` is heavy. The deferred import keeps the
    # module load cheap for the corpus-shuffle path.
    from scripts._vendor_common import CaptureBuffers
    from scripts.vendor_rules import (
        CHECK_MESSAGE_TEMPLATE_MATCH,
        CHECK_NEGATIVE_DOES_NOT_RAISE,
        CHECK_POSITIVE_RAISES,
    )

    no_exception = CaptureBuffers(
        exception_type=None,
        exception_message=None,
        warnings_captured=(),
        logger_messages=(),
        observed_state={},
        duration_ms=0,
    )
    raised_matching = CaptureBuffers(
        exception_type="ValueError",
        exception_message="Invalid `cache_implementation` (got 'nonsense'). Choose one of: ...",
        warnings_captured=(),
        logger_messages=(),
        observed_state=None,
        duration_ms=0,
    )
    raised_mismatching = CaptureBuffers(
        exception_type="ValueError",
        exception_message="Some completely unrelated runtime message.",
        warnings_captured=(),
        logger_messages=(),
        observed_state=None,
        duration_ms=0,
    )

    base_rule = {
        "id": "synth_rule",
        "severity": "error",
        "native_type": "synthetic.NativeType",
        "kwargs_positive": {"cache_implementation": "nonsense"},
        "kwargs_negative": {"cache_implementation": "static"},
        "message_template": "Invalid `cache_implementation` ({val}). Choose one of: ...",
    }

    # 1. positive_raises - positive did not raise (no exception captured).
    pos_no_raise = {
        "check_name": CHECK_POSITIVE_RAISES,
        "rule": dict(base_rule, id="synth_positive_did_not_raise"),
        "pos": no_exception,
        "neg": no_exception,
    }

    # 2. message_template_match - positive raised but message doesn't match.
    msg_mismatch = {
        "check_name": CHECK_MESSAGE_TEMPLATE_MATCH,
        "rule": dict(base_rule, id="synth_message_did_not_match"),
        "pos": raised_mismatching,
        "neg": no_exception,
    }

    # 3. negative_does_not_raise - negative raised when it shouldn't have.
    neg_raised = {
        "check_name": CHECK_NEGATIVE_DOES_NOT_RAISE,
        "rule": dict(base_rule, id="synth_negative_raised_unexpectedly"),
        "pos": raised_matching,
        "neg": raised_matching,
    }

    return [pos_no_raise, msg_mismatch, neg_raised]


def assert_gate_soundness_fixpoint() -> None:
    """Assert the vendor-CI gate's three soundness checks are all wired.

    Synthesises one malformed rule per check, runs each through
    :func:`scripts.vendor_rules.compute_gate_soundness_divergences`, and
    raises :class:`GateSoundnessRegressionError` if any check's divergence
    is missing.
    """
    from scripts.vendor_rules import compute_gate_soundness_divergences

    for case in synthesise_malformed_rule_cases():
        divergences = compute_gate_soundness_divergences(case["rule"], case["pos"], case["neg"])
        check_names_observed = {d.check_failed for d in divergences}
        if case["check_name"] not in check_names_observed:
            raise GateSoundnessRegressionError(
                case["check_name"], [d.as_dict() for d in divergences]
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dict_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Shallow diff of two dicts for error messages."""
    keys = set(a) | set(b)
    return {k: {"a": a.get(k), "b": b.get(k)} for k in sorted(keys) if a.get(k) != b.get(k)}
