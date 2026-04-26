#!/usr/bin/env python3
"""Phase B: run T0-T2-T5 against a synthetic user-error corpus.

The full-suite YAML is a *curated* showcase — its sweep: groups structurally
prevent cross-axis invalid combos. Real user configs don't have those guards.
This PoC measures probe value on deliberately-malformed configs (see
``synthetic_bad_configs.py``) to answer:

- On realistic user errors, what fraction does T0/T1/T2/T5 catch?
- Which category of error does each tier catch?
- Are there errors NO tier catches (would slip through to runtime)?

Output: per-case row with (expected-tier, actual-tiers-firing, ok).
Summary: coverage matrix + recap of misses.
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")


# --- Mock tensorrt_llm.llmapi so TRT probe can reach T5 on hosts with
# partial/broken tensorrt_llm install (libpython mismatch). This mirrors
# the approach in the test suite. Without this, T0 _build_llm_kwargs()
# for any TRT config with quant set raises OSError and the probe returns
# early, skipping T5 entirely. The mock is a PoC artefact; production
# probes run in the proper TRT container where the real library is
# importable.
if "tensorrt_llm.llmapi" not in sys.modules:
    _fake_llmapi = types.ModuleType("tensorrt_llm.llmapi")

    class _Algo:
        FP8 = "FP8"
        INT8 = "INT8"
        W8A16 = "W8A16"
        W4A16_AWQ = "W4A16_AWQ"
        W4A16_GPTQ = "W4A16_GPTQ"
        NVFP4 = "NVFP4"

        def __class_getitem__(cls, item):
            return getattr(cls, item, item)

    class _Cls:
        def __init__(self, **kw):
            self.kw = kw

    _fake_llmapi.QuantAlgo = _Algo
    _fake_llmapi.QuantConfig = _Cls
    _fake_llmapi.KvCacheConfig = _Cls
    _fake_llmapi.SchedulerConfig = _Cls
    _fake_llmapi.CapacitySchedulerPolicy = _Algo
    _fake_llmapi.BuildCacheConfig = _Cls
    sys.modules["tensorrt_llm.llmapi"] = _fake_llmapi

from llenergymeasure.engines import get_engine
from llenergymeasure.engines.protocol import ConfigProbe
from scripts.synthetic_bad_configs import SyntheticCase, build_corpus


def _tier_of_error(err: str) -> str:
    if err.startswith("T0 "):
        return "T0"
    if err.startswith("T1 "):
        return "T1"
    if err.startswith("T1+T2 "):
        return "T1+T2"
    if err.startswith("T2 "):
        return "T2"
    if "SM" in err and "Turing" in err:
        return "T5"
    if "FP8 quantisation requires" in err or "FP8 KV cache" in err:
        return "T5"
    return "other"


def _run_case(case: SyntheticCase) -> dict:
    """Run the probe against one synthetic case; return outcome dict."""
    try:
        cfg = case.factory()
    except Exception as e:
        return {
            "ok_to_construct": False,
            "schema_error": f"{type(e).__name__}: {e}"[:200],
            "probe": None,
        }

    try:
        engine = get_engine(cfg.engine)
        probe: ConfigProbe = engine.probe_config(cfg)
    except Exception as e:
        return {
            "ok_to_construct": True,
            "probe": None,
            "probe_raise": f"{type(e).__name__}: {e}"[:200],
        }

    # Filter environmental noise (partial TRT install libpython error)
    env_errs = [e for e in probe.errors if "libpython" in e or "cannot open shared object" in e]
    real_errs = [e for e in probe.errors if e not in env_errs]
    tiers = sorted({_tier_of_error(e) for e in real_errs})

    return {
        "ok_to_construct": True,
        "probe_is_valid": probe.is_valid and not real_errs,
        "dormant_count": len(probe.dormant_fields),
        "dormant_keys": sorted(probe.dormant_fields.keys()),
        "error_count": len(real_errs),
        "error_tiers": tiers,
        "errors": real_errs[:5],  # truncate for display
        "env_error_count": len(env_errs),
    }


def main() -> int:
    corpus = build_corpus()
    print(f"Synthetic corpus: {len(corpus)} cases")
    print()

    results = []
    for c in corpus:
        outcome = _run_case(c)
        results.append((c, outcome))

    # Per-case table
    print("=" * 110)
    print(f"{'CASE':40s}  {'EXP TIER':8s}  {'PROBE SAYS':40s}  RESULT")
    print("=" * 110)

    hits = 0
    misses = 0
    partials = 0

    for c, r in results:
        if not r.get("ok_to_construct"):
            actual = f"schema: {r.get('schema_error', '')[:30]}"
            ok_expected = c.expected_tier == "schema"
            status = "HIT" if ok_expected else "MISS"
            if ok_expected:
                hits += 1
            else:
                misses += 1
        elif r.get("probe") is None and r.get("probe_raise"):
            actual = f"probe raised: {r['probe_raise'][:30]}"
            status = "PROBE-BUG"
            misses += 1
        else:
            tiers = r.get("error_tiers", [])
            dormant = r.get("dormant_count", 0)
            valid = r.get("probe_is_valid", True)

            actual_parts = []
            if dormant:
                actual_parts.append(f"T0:{dormant}d")
            actual_parts.extend(tiers)
            actual = ",".join(actual_parts) or "clean"

            # Compare
            if c.expected_tier == "none":
                # baseline should be clean (no errors; dormancy OK depending on expectation)
                expected_dormant = set(c.expected_dormant_fields)
                if not tiers and len(expected_dormant) == 0 and dormant == 0:
                    status = "HIT"
                    hits += 1
                else:
                    status = f"UNEXPECTED-SIGNAL ({dormant}d/{len(tiers)}e)"
                    partials += 1
            elif c.expected_tier in tiers:
                status = "HIT"
                hits += 1
            elif c.expected_tier == "T0":
                if set(c.expected_dormant_fields).issubset(set(r.get("dormant_keys", []))):
                    status = "HIT"
                    hits += 1
                else:
                    status = "MISS"
                    misses += 1
            elif c.expected_tier == "T1.5":
                status = "N/A (T1.5 not in base probe)"
                partials += 1
            else:
                status = "MISS"
                misses += 1

        print(f"{c.name:40s}  {c.expected_tier:8s}  {actual[:40]:40s}  {status}")

    # Summary
    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"  Total cases:     {len(corpus)}")
    print(f"  HIT:             {hits}")
    print(f"  MISS:            {misses}")
    print(f"  Other/partial:   {partials}")
    print()

    # Coverage by category
    print("Coverage by category:")
    by_cat = {}
    for c, r in results:
        by_cat.setdefault(c.category, {"total": 0, "caught": 0})
        by_cat[c.category]["total"] += 1
        tiers_or_schema = r.get("error_tiers") or []
        if (
            (not r.get("ok_to_construct") and c.expected_tier == "schema")
            or c.expected_tier in tiers_or_schema
            or (
                c.expected_tier == "T0"
                and set(c.expected_dormant_fields).issubset(set(r.get("dormant_keys", []) or []))
            )
            or (
                c.expected_tier == "none" and not tiers_or_schema and r.get("dormant_count", 0) == 0
            )
        ):
            by_cat[c.category]["caught"] += 1
    for cat, stats in sorted(by_cat.items()):
        pct = 100 * stats["caught"] / stats["total"]
        print(f"  {cat:30s}  {stats['caught']}/{stats['total']}  ({pct:5.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
