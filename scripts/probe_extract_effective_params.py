"""PoC-C: Does extract_effective_params work uniformly across engine native types?

Hypothesis
----------
sweep-dedup.md §3.2 claims `extract_effective_params(native_obj)` is ~5 LoC
of glue per engine handling Pydantic / dataclass / msgspec / __slots__ uniformly.
But each engine has multiple native types (vLLM: EngineArgs + SamplingParams;
TRT-LLM: LlmArgs + BuildConfig) and the question of "which object represents
'the config'?" is under-specified. Worse: if extraction includes non-deterministic
state (timestamps, memo-caches, UUIDs), H3 won't be byte-stable across runs,
breaking the gap-detection invariant.

This PoC:
  1. Instantiates each engine's candidate native types with known-normalising
     kwargs.
  2. Runs the proposed `extract_effective_params` across all candidates.
  3. Verifies: (a) normalisations are captured, (b) no private `_field`
     leakage, (c) byte-stable across repeated constructions.

Decision criteria (pre-committed)
---------------------------------
- Byte-stable H3 across repeat runs on all engines -> OK for P4.
- Non-stable H3 (any field) -> identify the problem field; either add
  to an explicit exclude-list per engine OR widen Q3's normalisation rules.
- Extraction requires >15 LoC per engine (signal: multi-step state merging,
  per-field allowlists, or custom fallbacks) -> revise the "5 LoC glue"
  claim in sweep-dedup.md.
- Private-state leakage (fields starting with "_") -> add allowlist per engine.

Run
---
  /usr/bin/python3.10 scripts/probe_extract_effective_params.py

Note: runs on host if libraries are importable. For engines not available on
host (TRT-LLM likely), skip gracefully with a note for container-based
follow-up.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


# --- The proposed ~5 LoC helper from sweep-dedup §3.2 ---
def extract_effective_params(native_obj: Any) -> dict[str, Any]:
    """Dump a constructed native type's post-__post_init__ state."""
    if hasattr(native_obj, "model_dump"):  # Pydantic
        return native_obj.model_dump(mode="python")
    if is_dataclass(native_obj):  # dataclasses
        return asdict(native_obj)
    if hasattr(native_obj, "__slots__"):  # msgspec-adjacent / slot-based
        return {s: getattr(native_obj, s) for s in native_obj.__slots__ if hasattr(native_obj, s)}
    return dict(native_obj.__dict__)


# --- Canonical JSON encoder for H3 stability ---
def canonical_json(obj: Any) -> str:
    """JSON-serialise with sorted keys for hash stability. Handles common types."""

    def _default(o: Any) -> Any:
        if isinstance(o, (set, frozenset)):
            return sorted(list(o))
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return repr(o)  # fallback; makes non-serialisable visible

    return json.dumps(obj, sort_keys=True, default=_default)


def h3_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(params).encode()).hexdigest()[:16]


# --- Per-engine test specs ---
# For each engine: list of (name, native_type_factory, kwargs, expected_normalisations)
# factory returns the constructed native object; kwargs go into the factory.


def _test_pydantic() -> list[tuple[str, Any, dict[str, Any], dict[str, Any]]]:
    # transformers: GenerationConfig is the main native type
    try:
        from transformers import GenerationConfig
    except ImportError:
        return []

    def make(kwargs: dict[str, Any]) -> Any:
        return GenerationConfig(**kwargs)

    return [
        (
            "transformers_greedy_strips_temp",
            make,
            {"do_sample": False, "temperature": 0.9, "top_p": 0.95},
            {"temperature": 0.9, "top_p": 0.95},  # declared; validate() leaves them on
        ),
        (
            "transformers_valid_sampling",
            make,
            {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
            {},
        ),
        (
            "transformers_num_beams",
            make,
            {"num_beams": 4, "length_penalty": 1.5},
            {},
        ),
    ]


def _test_vllm() -> list[tuple[str, Any, dict[str, Any], dict[str, Any]]]:
    try:
        from vllm import SamplingParams
    except ImportError:
        return []

    def make(kwargs: dict[str, Any]) -> Any:
        return SamplingParams(**kwargs)

    return [
        (
            "vllm_temp_zero_greedy_clamp",
            make,
            {"temperature": 0.0, "top_p": 0.95, "top_k": 50},
            # Expected post-init: vLLM clamps top_p/top_k when temperature=0
            {"top_p": 1.0, "top_k": 0},
        ),
        (
            "vllm_temp_epsilon_clamp",
            make,
            {"temperature": 0.001},
            # Expected: clamped to 0.01
            {"temperature": 0.01},
        ),
        (
            "vllm_seed_sentinel",
            make,
            {"seed": -1},
            # Expected: -1 -> None
            {"seed": None},
        ),
        (
            "vllm_stop_none_to_list",
            make,
            {"stop": None},
            # Expected: None -> []
            {"stop": []},
        ),
    ]


def _test_tensorrt() -> list[tuple[str, Any, dict[str, Any], dict[str, Any]]]:
    try:
        from tensorrt_llm.llmapi import BuildConfig
    except Exception:
        return []

    def make(kwargs: dict[str, Any]) -> Any:
        return BuildConfig(**kwargs)

    return [
        (
            "tensorrt_buildconfig_minimal",
            make,
            {},
            {},
        ),
    ]


def run_engine_tests(engine: str, tests: list[tuple[str, Any, dict, dict]]) -> dict[str, Any]:
    """Run extract_effective_params + H3 stability tests for one engine."""
    results = {
        "engine": engine,
        "n_tests": len(tests),
        "n_passing": 0,
        "issues": [],
        "private_fields_seen": set(),
        "non_serialisable_fields": set(),
        "loc_estimate": 5,  # baseline claim
    }

    if not tests:
        results["note"] = "library not importable on host — container required"
        return results

    for name, factory, kwargs, expected_normalisations in tests:
        try:
            obj_1 = factory(kwargs)
            obj_2 = factory(kwargs)
        except Exception as e:
            results["issues"].append(
                {"test": name, "kind": "construction_failed", "detail": str(e)[:120]}
            )
            continue

        try:
            params_1 = extract_effective_params(obj_1)
            params_2 = extract_effective_params(obj_2)
        except Exception as e:
            results["issues"].append(
                {"test": name, "kind": "extraction_failed", "detail": str(e)[:120]}
            )
            continue

        # --- (1) Byte-stability check ---
        h3_1 = h3_hash(params_1)
        h3_2 = h3_hash(params_2)
        if h3_1 != h3_2:
            unstable_fields = {
                k
                for k in params_1
                if canonical_json(params_1[k]) != canonical_json(params_2.get(k))
            }
            results["issues"].append(
                {
                    "test": name,
                    "kind": "h3_unstable",
                    "detail": f"fields differing across runs: {sorted(unstable_fields)[:10]}",
                }
            )
            continue

        # --- (2) Private-field leakage check ---
        private = {k for k in params_1 if k.startswith("_")}
        if private:
            results["private_fields_seen"].update(private)

        # --- (3) Non-serialisable check: if canonical_json used repr() as
        #       fallback for any value, we have a non-deterministic field. ---
        # Coarse heuristic: check for "<" in values (repr output typically contains it).
        for k, v in params_1.items():
            if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                results["non_serialisable_fields"].add(k)

        # --- (4) Expected-normalisation check ---
        missing: list[str] = []
        for field, expected_val in expected_normalisations.items():
            actual = params_1.get(field)
            if actual != expected_val:
                missing.append(f"{field}: expected {expected_val!r}, got {actual!r}")
        if missing:
            results["issues"].append(
                {
                    "test": name,
                    "kind": "expected_normalisation_missed",
                    "detail": "; ".join(missing),
                }
            )
            # Don't mark as fail — library-version-specific; informational.

        results["n_passing"] += 1

    # Normalise sets to sorted lists for JSON output
    results["private_fields_seen"] = sorted(results["private_fields_seen"])
    results["non_serialisable_fields"] = sorted(results["non_serialisable_fields"])
    return results


def main() -> None:
    print("=" * 78)
    print("PoC-C: extract_effective_params per-engine integration")
    print("=" * 78)
    print()

    all_results = []
    for engine, tests_fn in [
        ("transformers", _test_pydantic),
        ("vllm", _test_vllm),
        ("tensorrt_llm", _test_tensorrt),
    ]:
        tests = tests_fn()
        print(f"--- {engine} ---")
        if not tests:
            print("  library not importable on host; skip (rerun in engine container)")
            print()
            all_results.append(
                {"engine": engine, "skipped": True, "note": "library not importable"}
            )
            continue
        r = run_engine_tests(engine, tests)
        print(f"  tests:            {r['n_tests']}")
        print(f"  passing:          {r['n_passing']}")
        print(f"  issues found:     {len(r['issues'])}")
        for issue in r["issues"][:5]:
            print(f"    [{issue['kind']}] {issue['test']}: {issue['detail']}")
        if r["private_fields_seen"]:
            print(f"  private-field LEAKAGE: {r['private_fields_seen']}")
        if r["non_serialisable_fields"]:
            print(f"  non-serialisable:  {r['non_serialisable_fields']}")
        print()
        all_results.append(r)

    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    print(f"{'ENGINE':<16}  {'H3-STABLE':<12}  {'LEAKAGE':<14}  {'SERIALISABLE':<14}  STATUS")
    print("-" * 78)
    for r in all_results:
        if r.get("skipped"):
            print(f"{r['engine']:<16}  {'SKIP':<12}  {'-':<14}  {'-':<14}  rerun-in-container")
            continue
        h3_stable = not any(i["kind"] == "h3_unstable" for i in r["issues"])
        leakage = bool(r["private_fields_seen"])
        serialisable = not r["non_serialisable_fields"]
        status = "OK" if (h3_stable and not leakage and serialisable) else "**ATTENTION**"
        print(
            f"{r['engine']:<16}  {'yes' if h3_stable else 'NO':<12}  "
            f"{'NONE' if not leakage else 'YES':<14}  "
            f"{'clean' if serialisable else 'fallback':<14}  "
            f"{status}"
        )

    print()
    print("Interpretation:")
    print("  OK             -> extract_effective_params is sound for this engine")
    print("                    using the 5-LoC generic helper. No changes needed.")
    print("  **ATTENTION**  -> one or more of: H3 not byte-stable (needs field")
    print("                    exclusion or canonicalisation), private fields")
    print("                    leaked (add allowlist), non-serialisable values")
    print("                    (need custom encoder). Revise sweep-dedup.md §3.2")
    print("                    to reflect per-engine LoC estimate.")
    print("  rerun-in-cont  -> library not available on host; run this PoC inside")
    print("                    the engine's Docker container for validation.")


if __name__ == "__main__":
    main()
