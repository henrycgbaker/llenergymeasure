"""Shuffle-application fixpoint test for the dormant-rule canonicaliser.

The canonicaliser itself lands in Wave 2 (phase 50.3a). This module ships
the CI-time *contract test* that corpora it consumes must satisfy:

1. **Idempotent** — applying the same rule twice leaves the state unchanged.
2. **Order-independent at fixpoint** — multiple random application orderings
   converge to the same canonical form.
3. **Cycle-free** — no rule pair alternates values indefinitely.

The test operates on a declarative projection of each rule: a dormant rule
declares ``normalised_fields`` in ``expected_outcome``, and the "fix" is
setting each normalised field to its declared default (from the predicate's
``not_equal`` operand, falling back to ``None``). This is sufficient to catch
the structural failure modes the canonicaliser would trip on — we don't need
the canonicaliser itself to prove shuffle-stability of the *rules*.

Imported by ``scripts/vendor_rules.py`` and by ``tests/unit/scripts/walkers/test_fixpoint.py``.
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
"""Number of random orderings to try per input state.

Proposal in PLAN §"Open questions at P2 implementation time" — 5 is fast and
catches cycles empirically. Bumped via ``shuffle_count`` if needed.
"""


class FixpointError(Exception):
    """Base class for fixpoint-test failures."""


class CanonicaliserCycleError(FixpointError):
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
    """Minimal predicate evaluator — supports the operator shapes in the corpus."""
    if isinstance(spec, dict):
        if not spec:
            return True
        for op, value in spec.items():
            if op == "==" and actual != value:
                return False
            if op == "!=" and actual == value:
                return False
            if op == "<" and not (actual is not None and actual < value):
                return False
            if op == "<=" and not (actual is not None and actual <= value):
                return False
            if op == ">" and not (actual is not None and actual > value):
                return False
            if op == ">=" and not (actual is not None and actual >= value):
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

    Returns ``(final_state, applied_rule_ids)``. Raises :class:`CanonicaliserCycleError`
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
    raise CanonicaliserCycleError([r.id for r in rules], current)


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
    :class:`OrderDependentRuleError` on divergence or :class:`CanonicaliserCycleError`
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
# Helpers
# ---------------------------------------------------------------------------


def _dict_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Shallow diff of two dicts for error messages."""
    keys = set(a) | set(b)
    return {k: {"a": a.get(k), "b": b.get(k)} for k in sorted(keys) if a.get(k) != b.get(k)}
