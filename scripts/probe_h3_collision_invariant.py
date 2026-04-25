"""PoC-K: H3-collision invariant end-to-end demonstration.

Hypothesis
----------
sweep-dedup.md §4.1 states: "After H1-dedup, the post-dedup run set is by
construction a set of configs with distinct H1 hashes. Any two sidecars in
this run set that share an H3 hash prove a canonicaliser gap."

This is claimed as a clean invariant. This PoC demonstrates it end-to-end
on a small sweep that intentionally triggers a rule the canonicaliser
doesn't know about (a canonicaliser gap). It then shows H3-collision
correctly identifies the gap.

Setup
-----
  1. Construct a synthetic sweep of 6 vLLM SamplingParams configs.
  2. Use a CANONICALISER that knows ONE rule (greedy normalises top_p/top_k/min_p).
  3. Let the library also apply its epsilon-clamp rule (not in canonicaliser).
  4. Show: two configs with temperature=0.001 and temperature=0.005 get
     DIFFERENT H1 (canonicaliser didn't know to clamp) but SAME H3 (library
     clamped both to 0.01).
  5. The H3-collision detector flags these as a gap. Correct.

Decision criteria (pre-committed)
---------------------------------
- Invariant holds: gap detector correctly identifies (different H1, same H3)
  pair AND proposes the missing rule from field-diff
    -> sweep-dedup §4 invariant validated.
- Gap detector misses the collision -> invariant framing is broken; redesign.
- Gap detector false-positives (flags gap where library behaviour is
  legitimately divergent) -> Q3 normalisation rules insufficient; tighten.

Run
---
  /usr/bin/python3.10 scripts/probe_h3_collision_invariant.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

# --- Minimal canonicaliser with ONE rule: vLLM greedy normalisation.
#     Intentionally MISSING: the epsilon clamp rule (this is the gap).


def _rule_greedy_normalises_top_p_top_k(state: dict[str, Any]) -> bool:
    return state.get("temperature") == 0.0 and (
        state.get("top_p") != 1.0 or state.get("top_k") != 0
    )


def _apply_greedy_normalises(state: dict[str, Any]) -> None:
    state["top_p"] = 1.0
    state["top_k"] = 0
    state["min_p"] = 0.0


MINIMAL_CORPUS = [
    {
        "id": "vllm_greedy_normalises",
        "pred": _rule_greedy_normalises_top_p_top_k,
        "apply": _apply_greedy_normalises,
    },
]


def canonicalise(state: dict[str, Any], corpus: list[dict]) -> dict[str, Any]:
    c = dict(state)
    for _ in range(10):
        fired = False
        for rule in corpus:
            if rule["pred"](c):
                rule["apply"](c)
                fired = True
        if not fired:
            return c
    return c


# --- Hash helpers ---
def _canonical_json(obj: Any) -> str:
    def default(o):
        if isinstance(o, (set, frozenset)):
            return sorted(list(o))
        if is_dataclass(o):
            return asdict(o)
        return repr(o)

    return json.dumps(obj, sort_keys=True, default=default)


def _h(state: dict[str, Any], label: str) -> str:
    # Mimic sweep-dedup §2.4 schema: hash measurement-relevant fields only.
    RELEVANT = {"temperature", "top_p", "top_k", "min_p"}
    subset = {k: state.get(k) for k in RELEVANT}
    return hashlib.sha256(_canonical_json(subset).encode()).hexdigest()[:12]


# --- "Sidecar" equivalent of what a real run would persist ---
@dataclass
class Sidecar:
    experiment_id: str
    declared_kwargs: dict[str, Any]
    canonical_state: dict[str, Any]
    effective_state: dict[str, Any]  # = what library actually produced
    h1: str = ""
    h3: str = ""


def observe_library_state(declared: dict[str, Any]) -> dict[str, Any]:
    """Run the real vLLM SamplingParams constructor and extract effective state."""
    from vllm import SamplingParams

    # Extract the fields we care about
    sp = SamplingParams(**declared)
    return {
        "temperature": sp.temperature,
        "top_p": sp.top_p,
        "top_k": sp.top_k,
        "min_p": sp.min_p,
    }


def run_sweep() -> list[Sidecar]:
    sweep_configs = [
        {"temperature": 0.0, "top_p": 0.5, "top_k": 30},  # A: greedy; canonicaliser catches
        {"temperature": 0.0, "top_p": 0.9, "top_k": 60},  # B: greedy; canonicaliser catches
        {
            "temperature": 0.001,
            "top_p": 0.95,
            "top_k": 40,
        },  # C: epsilon-clamp; canonicaliser MISSES
        {
            "temperature": 0.005,
            "top_p": 0.95,
            "top_k": 40,
        },  # D: epsilon-clamp; canonicaliser MISSES
        {"temperature": 0.7, "top_p": 0.9, "top_k": 40},  # E: valid sampling; no rules fire
        {"temperature": 0.8, "top_p": 0.85, "top_k": 30},  # F: valid sampling; no rules fire
    ]

    sidecars = []
    for i, declared in enumerate(sweep_configs):
        exp_id = f"exp_{i:03d}"
        canonical = canonicalise(declared, MINIMAL_CORPUS)
        effective = observe_library_state(declared)
        sc = Sidecar(
            experiment_id=exp_id,
            declared_kwargs=declared,
            canonical_state=canonical,
            effective_state=effective,
            h1=_h(canonical, "h1"),
            h3=_h(effective, "h3"),
        )
        sidecars.append(sc)
    return sidecars


def detect_h3_collisions(sidecars: list[Sidecar]) -> list[tuple[Sidecar, Sidecar]]:
    """§4.1 invariant: post-H1-dedup, any H3 collision = proven gap."""
    # Step 1: H1-dedup (keep first per h1)
    seen_h1 = set()
    h1_dedupped: list[Sidecar] = []
    for s in sidecars:
        if s.h1 in seen_h1:
            continue
        seen_h1.add(s.h1)
        h1_dedupped.append(s)

    # Step 2: within H1-dedupped set, find any H3 collision
    by_h3: dict[str, list[Sidecar]] = {}
    for s in h1_dedupped:
        by_h3.setdefault(s.h3, []).append(s)

    collisions = []
    for h3, members in by_h3.items():
        if len(members) >= 2:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    collisions.append((members[i], members[j]))
    return collisions


def infer_bridging_rule(a: Sidecar, b: Sidecar) -> dict[str, Any]:
    """Simple field-diff to propose a candidate bridging rule."""
    # Common trigger fields: those where declared == declared AND H3 == H3
    common_triggers = {}
    for k in a.declared_kwargs:
        if k in b.declared_kwargs:
            if a.declared_kwargs[k] == b.declared_kwargs[k]:
                common_triggers[k] = a.declared_kwargs[k]
    # Affected fields: those where declared differs but H3 converged
    affected = {}
    for k in a.effective_state:
        if a.effective_state[k] == b.effective_state[k]:
            if a.declared_kwargs.get(k) != b.declared_kwargs.get(k):
                affected[k] = a.effective_state[k]

    return {
        "trigger": common_triggers,
        "affected": affected,
        "note": (
            "Simple equality predicate — if range/inequality is the real "
            "rule, maintainer widens predicate in review"
        ),
    }


def main() -> None:
    print("=" * 78)
    print("PoC-K: H3-collision invariant end-to-end demonstration")
    print("=" * 78)
    print()

    try:
        sidecars = run_sweep()
    except ImportError as e:
        print(f"vllm not importable: {e}")
        return
    except Exception as e:
        print(f"ERROR running sweep: {type(e).__name__}: {e}")
        return

    print(f"{'exp':<8}  {'H1':<14}  {'H3':<14}  declared -> effective")
    print("-" * 78)
    for s in sidecars:
        print(
            f"{s.experiment_id:<8}  {s.h1:<14}  {s.h3:<14}  "
            f"{s.declared_kwargs} -> {s.effective_state}"
        )

    print()
    collisions = detect_h3_collisions(sidecars)
    print(f"H3-collisions after H1-dedup: {len(collisions)}")
    print()

    if not collisions:
        print("-> No gaps detected. Either canonicaliser is complete or sweep")
        print("   didn't exercise any rule the canonicaliser missed.")
        return

    for i, (a, b) in enumerate(collisions):
        print(f"--- Gap #{i}: {a.experiment_id} ~ {b.experiment_id} ---")
        print(f"  H1(a)={a.h1}  H1(b)={b.h1}  (distinct -> H1-dedup kept both)")
        print(f"  H3(a)={a.h3}  H3(b)={b.h3}  (same -> library collapsed)")
        print(f"  declared(a): {a.declared_kwargs}")
        print(f"  declared(b): {b.declared_kwargs}")
        print(f"  effective:   {a.effective_state}  (canonical across pair)")
        proposed = infer_bridging_rule(a, b)
        print("  PROPOSED BRIDGING RULE:")
        print(f"    trigger:  {proposed['trigger']}")
        print(f"    affected: {proposed['affected']}")
        print(f"    note:     {proposed['note']}")
        print()

    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    # Check if the KNOWN gap (epsilon clamp) was detected
    epsilon_gap_detected = any(
        (
            a.declared_kwargs.get("temperature") in (0.001, 0.005)
            and b.declared_kwargs.get("temperature") in (0.001, 0.005)
            and a.declared_kwargs != b.declared_kwargs
        )
        for a, b in collisions
    )
    if epsilon_gap_detected:
        print("-> PASS: the deliberately-missing epsilon-clamp rule was detected")
        print("   via H3-collision. Invariant framing in sweep-dedup.md §4.1 holds.")
    else:
        print("-> FAIL: epsilon-clamp gap NOT detected. Invariant framing is broken.")

    false_positives = [
        (a, b)
        for a, b in collisions
        if a.declared_kwargs.get("temperature") == b.declared_kwargs.get("temperature")
    ]
    if false_positives:
        print(f"   WARNING: {len(false_positives)} false-positive-candidate pairs")
        print("   (identical declared kwargs shouldn't have distinct H1; check Q3).")


if __name__ == "__main__":
    main()
