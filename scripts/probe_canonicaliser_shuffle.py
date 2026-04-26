"""PoC-F: Canonicaliser shuffle-application test.

Hypothesis
----------
sweep-dedup §2.2 asserts the vendored-rule corpus must satisfy three
invariants: rules must be (a) idempotent, (b) order-independent at fixpoint,
and (c) terminating (no cycles). The design proposes a CI shuffle-application
test to validate these. This PoC implements a minimal version against a seeded
rule corpus derived from the AST PoC output, to check whether the invariants
actually hold on real library rules.

Rule set (seeded from AST PoC findings)
---------------------------------------
vLLM:
  r_vllm_greedy_normalises_sampling:
    if temperature == 0 -> top_p=1.0, top_k=0, min_p=0.0
  r_vllm_temp_epsilon_clamp:
    if 0 < temperature < 0.01 -> temperature = 0.01
  r_vllm_seed_sentinel:
    if seed == -1 -> seed = None
  r_vllm_stop_none_to_list:
    if stop is None -> stop = []

transformers (dormancy rules, no actual field mutations — HF's validate()
doesn't mutate, just flags. So for the canonicaliser's purposes, we model
them as "add flagged field to a stable representation"):
  r_tf_greedy_strips_temp:
    if do_sample == False and temperature != 1.0 -> temperature = 1.0
  r_tf_greedy_strips_top_p:
    if do_sample == False and top_p != 1.0 -> top_p = 1.0
  r_tf_greedy_strips_top_k:
    if do_sample == False and top_k != 50 -> top_k = 50

Cross-rule interaction to test:
  {temperature: 0.001, top_p: 0.95}  applied in different orders:
    order [epsilon, greedy]: epsilon fires -> temp=0.01, greedy doesn't fire -> {temp:0.01, top_p:0.95}
    order [greedy, epsilon]: greedy doesn't fire (temp=0.001 != 0), epsilon fires -> {temp:0.01, top_p:0.95}
    Same result. OK.

  {temperature: 0.0, top_p: 0.95}  (greedy):
    order [greedy, epsilon]: greedy -> top_p=1.0,top_k=0; epsilon doesn't fire on temp=0
    order [epsilon, greedy]: epsilon doesn't fire on 0.0, greedy -> top_p=1.0,top_k=0
    Same result. OK.

Decision criteria (pre-committed)
---------------------------------
- All test cases converge to same fixpoint across 30 random rule orderings
  per case, within 10 iterations -> invariants hold on this corpus; CI
  shuffle-test is sufficient gate for P4.
- Any case diverges -> identify the cycle/non-idempotent rule; corpus needs
  structural fix before P4.

Run
---
  /usr/bin/python3.10 scripts/probe_canonicaliser_shuffle.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

random.seed(42)


# --- Rule representation (minimal, models sweep-dedup §2.1 runtime) ---
@dataclass
class Rule:
    id: str
    engine: str
    predicate: Callable[[dict[str, Any]], bool]
    normalise: Callable[[dict[str, Any]], None]  # mutates in place


# --- Seeded corpus (subset — shuffle semantics are what we care about) ---
def _rule_vllm_greedy_normalises(state: dict[str, Any]) -> bool:
    return (
        state.get("engine") == "vllm"
        and state.get("temperature") == 0.0
        and (state.get("top_p") != 1.0 or state.get("top_k") != 0 or state.get("min_p") != 0.0)
    )


def _apply_vllm_greedy_normalises(state: dict[str, Any]) -> None:
    state["top_p"] = 1.0
    state["top_k"] = 0
    state["min_p"] = 0.0


def _rule_vllm_temp_epsilon(state: dict[str, Any]) -> bool:
    t = state.get("temperature")
    return state.get("engine") == "vllm" and isinstance(t, (int, float)) and 0 < t < 0.01


def _apply_vllm_temp_epsilon(state: dict[str, Any]) -> None:
    state["temperature"] = 0.01


def _rule_vllm_seed_sentinel(state: dict[str, Any]) -> bool:
    return state.get("engine") == "vllm" and state.get("seed") == -1


def _apply_vllm_seed_sentinel(state: dict[str, Any]) -> None:
    state["seed"] = None


def _rule_vllm_stop_none(state: dict[str, Any]) -> bool:
    return state.get("engine") == "vllm" and state.get("stop") is None and "stop" in state


def _apply_vllm_stop_none(state: dict[str, Any]) -> None:
    state["stop"] = []


def _rule_tf_greedy_strips_temp(state: dict[str, Any]) -> bool:
    return (
        state.get("engine") == "transformers"
        and state.get("do_sample") is False
        and state.get("temperature") != 1.0
    )


def _apply_tf_greedy_strips_temp(state: dict[str, Any]) -> None:
    state["temperature"] = 1.0


def _rule_tf_greedy_strips_top_p(state: dict[str, Any]) -> bool:
    return (
        state.get("engine") == "transformers"
        and state.get("do_sample") is False
        and state.get("top_p") != 1.0
    )


def _apply_tf_greedy_strips_top_p(state: dict[str, Any]) -> None:
    state["top_p"] = 1.0


def _rule_tf_greedy_strips_top_k(state: dict[str, Any]) -> bool:
    return (
        state.get("engine") == "transformers"
        and state.get("do_sample") is False
        and state.get("top_k") != 50
    )


def _apply_tf_greedy_strips_top_k(state: dict[str, Any]) -> None:
    state["top_k"] = 50


CORPUS: list[Rule] = [
    Rule(
        "vllm_greedy_normalises",
        "vllm",
        _rule_vllm_greedy_normalises,
        _apply_vllm_greedy_normalises,
    ),
    Rule("vllm_temp_epsilon_clamp", "vllm", _rule_vllm_temp_epsilon, _apply_vllm_temp_epsilon),
    Rule("vllm_seed_sentinel", "vllm", _rule_vllm_seed_sentinel, _apply_vllm_seed_sentinel),
    Rule("vllm_stop_none_to_list", "vllm", _rule_vllm_stop_none, _apply_vllm_stop_none),
    Rule(
        "tf_greedy_strips_temp",
        "transformers",
        _rule_tf_greedy_strips_temp,
        _apply_tf_greedy_strips_temp,
    ),
    Rule(
        "tf_greedy_strips_top_p",
        "transformers",
        _rule_tf_greedy_strips_top_p,
        _apply_tf_greedy_strips_top_p,
    ),
    Rule(
        "tf_greedy_strips_top_k",
        "transformers",
        _rule_tf_greedy_strips_top_k,
        _apply_tf_greedy_strips_top_k,
    ),
]


# --- Canonicaliser implementation under test ---
def canonicalise(
    state: dict[str, Any], rules: list[Rule], max_iter: int = 10
) -> tuple[dict[str, Any], int]:
    """Apply rules to fixpoint. Returns (canonical_state, iterations_used)."""
    c = copy.deepcopy(state)
    for i in range(max_iter):
        fired_any = False
        for rule in rules:
            if rule.predicate(c):
                rule.normalise(c)
                fired_any = True
        if not fired_any:
            return c, i + 1
    raise RuntimeError(f"Canonicaliser did not reach fixpoint in {max_iter} iterations: {c}")


def h_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --- Test cases ---
TEST_STATES = [
    {
        "engine": "vllm",
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 50,
        "min_p": 0.1,
        "seed": -1,
        "stop": None,
    },
    {"engine": "vllm", "temperature": 0.001, "top_p": 0.95, "top_k": 50, "min_p": 0.1},
    {"engine": "vllm", "temperature": 0.005, "top_p": 0.5, "top_k": 20},
    {"engine": "vllm", "temperature": 0.7, "top_p": 0.9, "top_k": 40},  # baseline, no rules fire
    {"engine": "transformers", "do_sample": False, "temperature": 0.9, "top_p": 0.95, "top_k": 40},
    {"engine": "transformers", "do_sample": False, "temperature": 0.0, "top_p": 0.5, "top_k": 100},
    {
        "engine": "transformers",
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    },  # baseline
    {
        "engine": "vllm",
        "temperature": 0.005,
        "top_p": 0.5,
        "top_k": 20,
        "seed": -1,
        "stop": None,
    },  # multi-rule chain
]


def main() -> None:
    print("=" * 78)
    print("PoC-F: Canonicaliser shuffle-application test")
    print("=" * 78)
    print()

    N_SHUFFLES = 30
    all_pass = True
    case_results = []

    for idx, state in enumerate(TEST_STATES):
        # Single reference application (arbitrary order)
        try:
            ref_state, ref_iters = canonicalise(state, CORPUS)
            ref_hash = h_hash(ref_state)
        except RuntimeError as e:
            print(f"[{idx}] Reference application FAILED: {e}")
            all_pass = False
            continue

        # Shuffle application
        divergent_hashes = set()
        max_iters_seen = 0
        errors = []
        for shuffle_n in range(N_SHUFFLES):
            rules_shuffled = list(CORPUS)
            random.shuffle(rules_shuffled)
            try:
                shuffled_state, iters = canonicalise(state, rules_shuffled)
                shuffled_hash = h_hash(shuffled_state)
                max_iters_seen = max(max_iters_seen, iters)
                if shuffled_hash != ref_hash:
                    divergent_hashes.add(shuffled_hash)
            except RuntimeError as e:
                errors.append(str(e)[:100])

        status = "PASS" if not divergent_hashes and not errors else "**FAIL**"
        if status != "PASS":
            all_pass = False
        case_results.append(
            (idx, status, ref_iters, max_iters_seen, len(divergent_hashes), len(errors))
        )

    # Report
    print(
        f"{'CASE':<6}  {'STATUS':<10}  {'ref iters':<10}  {'max iters':<10}  "
        f"{'div hashes':<12}  errors"
    )
    print("-" * 78)
    for idx, status, ref_iters, max_iters, div, errs in case_results:
        print(f"[{idx}]   {status:<10}  {ref_iters:<10}  {max_iters:<10}  {div:<12}  {errs}")
        print(f"        input: {TEST_STATES[idx]}")

    print()
    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    if all_pass:
        print("-> PASS: invariants hold across all cases × shuffled orderings.")
        print("        Shuffle-application CI test is a sufficient gate for P4.")
    else:
        print("-> FAIL: one or more cases diverge. Corpus has a cycle, non-")
        print("         idempotent rule, or order-dependence. Fix before P4.")

    print()
    print("Notes:")
    print("  - This corpus is a minimal seed extracted from the AST PoC. A full")
    print("    corpus (~60-80 rules) may expose interactions not visible here.")
    print("    The CI gate at vendor time re-runs this test against the full")
    print("    corpus on every corpus PR.")
    print(f"  - Max iterations observed: {max(r[3] for r in case_results)}; limit is 10.")


if __name__ == "__main__":
    main()
