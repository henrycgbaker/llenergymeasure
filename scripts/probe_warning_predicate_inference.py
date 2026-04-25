#!/usr/bin/env python3
"""PoC-RT-1 — warning-feedback predicate inference feasibility.

LOAD-BEARING for M3 scope decision: Option α/α' (capture-only) vs Option β
(capture + report-gaps --source runtime-warnings).

Question: given a sweep of N configs with per-experiment captured warnings +
per-experiment kwargs, can a simple algorithm reliably reverse-engineer the
trigger predicate for each unmatched warning template? If yes → Option β
viable. If no → Option α/α' is the answer.

Method:
    - Synthesise 72 configs varying 4 fields.
    - Plant 6 ground-truth rules covering the predicate-shape spectrum:
        R1: simple equality (single field, single value)
        R2: conjunction (two fields, two values)
        R3: range that doesn't fire — hallucination test
        R4: sentinel value (single field, special value -1)
        R5: range that fires for one discrete value (narrow-predicate test)
        R6: range that fires for multiple discrete values (algorithm should
            fail to propose, not propose wrongly)
    - Generate warnings deterministically per planted rules.
    - Apply normalisation (preview's regex pipeline, inlined).
    - Run inference algorithm: find smallest field-value tuple that ALL of A
      shares AND NO config in B shares. Try arity 1, then 2, then 3.
    - Score recovery vs ground truth.
    - Stretch: rerun with reduced cardinality (10 random configs).

Decision criteria:
    ≥80% recovery on full sweep → Option β VIABLE
    50-80%                       → marginal, lean toward α'
    <50%                         → defaults to α/α'
    Hallucinations on R3 (no firings)  → red flag — drop or harden

Reference: .claude/plans/m3-design-discussion-2026-04-24.md

Written by autonomous overnight PoC run, 2026-04-24.
"""

from __future__ import annotations

import random
import re
from itertools import combinations
from typing import Any

# -----------------------------------------------------------------------------
# Inline message normalisation (preview's regex pipeline, simplified)
# -----------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![A-Za-z_0-9])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_0-9])")
_PATH_RE = re.compile(r"(?:(?<=\s)|^)(?:/[^\s:()\[\]]+|[A-Za-z]:\\[^\s:]+)")
_WS_RE = re.compile(r"\s+")


def normalise(msg: str) -> str:
    n = _PATH_RE.sub(" <PATH>", msg)
    n = _NUMBER_RE.sub("<NUM>", n)
    return _WS_RE.sub(" ", n).strip()


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------


def make_sweep() -> list[dict[str, Any]]:
    """72 configs varying 4 fields."""
    return [
        {"temperature": t, "top_p": tp, "top_k": tk, "do_sample": ds}
        for t in (0.0, 0.3, 0.7, 1.0)
        for tp in (0.5, 0.95, 1.0)
        for tk in (-1, 50, 100)
        for ds in (True, False)
    ]


# -----------------------------------------------------------------------------
# Planted ground-truth rules
# -----------------------------------------------------------------------------


def planted_rules() -> list[tuple]:
    """Each rule: (id, expected_predicate_or_marker, predicate_fn, message)."""
    return [
        (
            "R1_do_sample_false",
            {"do_sample": False},
            lambda c: c["do_sample"] is False,
            "temperature is ignored when do_sample is False",
        ),
        (
            "R2_greedy_with_temp_0",
            {"do_sample": True, "temperature": 0.0},
            lambda c: c["do_sample"] is True and c["temperature"] == 0.0,
            "do_sample=True with temperature=0 is equivalent to greedy decoding",
        ),
        (
            "R3_unfired_range",
            ("range", "temperature", 0.0, 0.01),
            lambda c: 0.0 < c["temperature"] < 0.01,
            "temperature is below epsilon clamp threshold",
        ),
        (
            "R4_top_k_sentinel",
            {"top_k": -1},
            lambda c: c["top_k"] == -1,
            "top_k=-1 means unlimited candidates",
        ),
        (
            "R5_range_fired_one_value",
            ("range", "temperature", 0.0, 0.5),
            lambda c: 0.0 < c["temperature"] < 0.5,
            "temperature 0.3 was clamped to safe range",
        ),
        (
            "R6_range_fired_many_values",
            ("range", "temperature", 0.0, 0.99),
            lambda c: 0.0 < c["temperature"] < 0.99,
            "temperature 0.X is in low-confidence regime",
        ),
    ]


def emit_messages(config: dict, rules: list[tuple]) -> list[str]:
    out = []
    for rule in rules:
        if rule[2](config):
            out.append(rule[3])
    return out


# -----------------------------------------------------------------------------
# Inference algorithm
# -----------------------------------------------------------------------------


def find_distinguishing_predicate(
    A: list[dict[str, Any]],
    B: list[dict[str, Any]],
    fields: list[str],
    max_arity: int = 3,
) -> tuple[str, dict | None]:
    """Find smallest field-value combination such that:
      - every config in A shares it
      - no config in B has it

    Try arity 1 (single field) first, then 2 (pair), then 3 (triple).
    Returns (confidence, predicate). Confidence is "high" / "medium" / "low" /
    "none".
    """
    if not A:
        return ("none", None)

    for arity in range(1, max_arity + 1):
        for fset in combinations(fields, arity):
            a_tuples = {tuple(a.get(f) for f in fset) for a in A}
            if len(a_tuples) != 1:
                continue
            value_tuple = next(iter(a_tuples))
            if any(all(b.get(f) == v for f, v in zip(fset, value_tuple)) for b in B):
                continue
            predicate = dict(zip(fset, value_tuple))
            confidence = {1: "high", 2: "medium", 3: "low"}[arity]
            return (confidence, predicate)
    return ("none", None)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def run_inference(configs: list[dict], rules: list[tuple]) -> dict:
    """For each unique normalised template emitted across configs, infer predicate."""
    # Step 1: per-config emissions, normalised
    fired_per_config: list[set[str]] = [
        {normalise(m) for m in emit_messages(c, rules)} for c in configs
    ]

    # Step 2: enumerate every template that ANY rule could emit (so unfired
    # templates also appear with all-False fired flags — tests R3 hallucination)
    templates_to_check: dict[str, str] = {}  # template -> rule_id
    for r in rules:
        templates_to_check[normalise(r[3])] = r[0]

    # Step 3: per template, compute A/B, infer predicate
    fields = ["temperature", "top_p", "top_k", "do_sample"]
    results = {}
    for template, rule_id in templates_to_check.items():
        A = [c for i, c in enumerate(configs) if template in fired_per_config[i]]
        B = [c for i, c in enumerate(configs) if template not in fired_per_config[i]]
        confidence, predicate = find_distinguishing_predicate(A, B, fields)
        results[rule_id] = {
            "template": template,
            "n_fired": len(A),
            "n_not_fired": len(B),
            "confidence": confidence,
            "proposed_predicate": predicate,
        }
    return results


# -----------------------------------------------------------------------------
# Grading
# -----------------------------------------------------------------------------


def grade(inference_results: dict, rules: list[tuple]) -> dict:
    """Score each rule's recovery.

    Grades:
        fully_recovered      — proposed predicate matches ground truth exactly
        correctly_skipped    — A is empty (rule didn't fire) and no proposal made
        narrow_predicate     — ground truth is a range, algorithm proposes a
                                single equality value within the range. Right
                                for the data, wrong for generalisation.
                                (Per design §4.9.2, this gets needs-generalisation-review label.)
        missed               — A is non-empty but no predicate proposed
        wrong_fields         — proposed fields don't overlap ground truth
        wrong_values         — same fields, wrong values
        false_positive       — A is empty but predicate proposed (HALLUCINATION)
    """
    rules_by_id = {r[0]: r for r in rules}
    out = {}
    for rule_id, res in inference_results.items():
        rule = rules_by_id[rule_id]
        ground = rule[1]
        proposed = res["proposed_predicate"]
        n_fired = res["n_fired"]

        if n_fired == 0:
            grade_str = "false_positive" if proposed is not None else "correctly_skipped"
        elif proposed is None:
            grade_str = "missed"
        elif isinstance(ground, tuple) and ground[0] == "range":
            # Range ground truth, equality proposal
            _, field, lo, hi = ground
            if proposed and field in proposed:
                v = proposed[field]
                if isinstance(v, (int, float)) and lo < v < hi:
                    # Right configs (within range), but proposed equality
                    if len(proposed) == 1:
                        grade_str = "narrow_predicate"
                    else:
                        # Proposed extra fields beyond just the range field
                        grade_str = "narrow_with_extras"
                else:
                    grade_str = "wrong_values"
            else:
                grade_str = "wrong_fields"
        elif isinstance(ground, dict):
            if proposed == ground:
                grade_str = "fully_recovered"
            elif set(proposed.keys()) == set(ground.keys()):
                grade_str = "wrong_values"
            elif set(proposed.keys()) >= set(ground.keys()):
                # Proposed includes ground truth fields plus extras
                grade_str = "over_specified"
            else:
                grade_str = "wrong_fields"
        else:
            grade_str = "wrong_fields"

        out[rule_id] = {
            "grade": grade_str,
            "ground_truth": ground,
            "proposed": proposed,
            "confidence": res["confidence"],
            "n_fired": n_fired,
            "n_not_fired": res["n_not_fired"],
            "template": res["template"],
        }
    return out


def print_scores(scores: dict, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"  {'rule_id':<28s} {'grade':<22s} {'fired':>6s} {'conf':>7s}  proposed")
    for rule_id, s in scores.items():
        proposed_str = str(s["proposed"]) if s["proposed"] else "(none)"
        print(
            f"  {rule_id:<28s} {s['grade']:<22s} {s['n_fired']:>6d} {s['confidence']:>7s}  {proposed_str}"
        )


def summarise(scores: dict, label: str) -> tuple[int, int, int, int, int]:
    full = sum(1 for s in scores.values() if s["grade"] == "fully_recovered")
    skip = sum(1 for s in scores.values() if s["grade"] == "correctly_skipped")
    narrow = sum(
        1 for s in scores.values() if s["grade"] in ("narrow_predicate", "narrow_with_extras")
    )
    missed = sum(1 for s in scores.values() if s["grade"] == "missed")
    wrong = sum(
        1
        for s in scores.values()
        if s["grade"] in ("wrong_fields", "wrong_values", "false_positive", "over_specified")
    )
    total = len(scores)
    print(f"\n  Summary {label}:")
    print(f"    fully_recovered:    {full}/{total}")
    print(f"    correctly_skipped:  {skip}/{total}")
    print(f"    narrow_predicate:   {narrow}/{total}  (right configs, equality-instead-of-range)")
    print(f"    missed:             {missed}/{total}  (A non-empty, no predicate found)")
    print(f"    wrong/false-pos:    {wrong}/{total}")
    strict_pct = (full + skip) / total * 100
    lenient_pct = (full + skip + narrow) / total * 100
    print(f"    strict-recovery:    {strict_pct:.0f}% (fully + skipped)")
    print(
        f"    lenient-recovery:   {lenient_pct:.0f}% (+ narrow, would land with needs-generalisation-review)"
    )
    return (full, skip, narrow, missed, wrong)


def main() -> int:
    rules = planted_rules()
    print("PoC-RT-1 — warning-feedback predicate inference feasibility")
    print(f"\nGround-truth rules ({len(rules)}):")
    for r in rules:
        print(f"  {r[0]:<28s} expected={r[1]}")
        print(f"  {'':<28s} message={r[3]!r}")

    full_sweep = make_sweep()
    print(f"\nFull sweep size: {len(full_sweep)} configs")

    # Per-rule firing counts
    print("\nFiring counts (full sweep):")
    for r in rules:
        n = sum(1 for c in full_sweep if r[2](c))
        print(f"  {r[0]:<28s} fires for {n}/{len(full_sweep)} configs")

    # Full sweep
    inf_full = run_inference(full_sweep, rules)
    scores_full = grade(inf_full, rules)
    print_scores(scores_full, "Full sweep (72 configs)")
    summarise(scores_full, "(full sweep)")

    # Reduced sweep (10 random)
    random.seed(42)
    small = random.sample(full_sweep, 10)
    inf_small = run_inference(small, rules)
    scores_small = grade(inf_small, rules)
    print_scores(scores_small, "Reduced sweep (10 configs)")
    summarise(scores_small, "(small sweep)")

    # Tiny sweep (3 configs — extreme degradation test)
    tiny = random.sample(full_sweep, 3)
    print(f"\nTiny sweep (3 configs): {tiny}")
    inf_tiny = run_inference(tiny, rules)
    scores_tiny = grade(inf_tiny, rules)
    print_scores(scores_tiny, "Tiny sweep (3 configs)")
    summarise(scores_tiny, "(tiny sweep)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
