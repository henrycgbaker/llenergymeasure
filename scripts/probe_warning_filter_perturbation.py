#!/usr/bin/env python3
"""PoC-RT-0 — does warnings.simplefilter("always") perturb the measurement window?

Adversarial finding B1: the M3 preview wraps ``harness.run()`` (which is inside
the NVML measurement window) with ``warnings.catch_warnings(record=True)`` +
``warnings.simplefilter("always")``. Forcing every ``warnings.warn`` site to
bypass the standard "show once per location" filter could perturb the J/token
number we exist to measure. PoC-D verified capture, NOT non-perturbation.

Three regimes compared:
    (a) baseline: no override, Python's default filter list (dedups by location).
    (b) record-only: ``warnings.catch_warnings(record=True)`` only.
    (c) preview pattern: ``catch_warnings(record=True)`` + ``simplefilter("always")``.

Tests:
    Test 1 — Identical warnings from same location.
        Default filter list dedups, so this exaggerates (c)'s overhead. Real
        engines that emit the same construction warning N times during a sweep
        match this shape.
    Test 2 — Distinct warnings (varying message).
        Dedup doesn't help (a). Isolates pure infrastructure overhead.
    Test 3 — Torch.matmul loop (no warnings in path).
        Tests whether the regime *setup* itself adds visible overhead in an
        inference-style hot loop.
    Test 4 — Capture count comparison.
        Does (b) record-only actually capture as much as (c)? If yes, the
        cleanest fix is to delete the simplefilter line entirely.

Decision criteria (from RT-0 spec):
    (c) <1% slower than (a)        → SAFE; preview can stay.
    (c) 1-5% slower than (a)       → BORDERLINE; scope simplefilter outside inference.
    (c) >5% slower than (a)        → BLOCKING; redesign capture (don't touch globals).
    (b) captures all emissions     → cleanest fix is to delete simplefilter line.

Usage:
    /usr/bin/python3.10 scripts/probe_warning_filter_perturbation.py

Written by autonomous overnight PoC run, 2026-04-24.
Reference: .claude/plans/m3-design-discussion-2026-04-24.md
"""

from __future__ import annotations

import statistics
import sys
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# Three filter regimes
# -----------------------------------------------------------------------------


@contextmanager
def regime_a_baseline() -> Iterator[None]:
    """No override — Python's default filter list (dedups by location)."""
    yield None


@contextmanager
def regime_b_record_only() -> Iterator[list]:
    """``catch_warnings(record=True)`` only — no simplefilter call.

    Per Python 3.10 docs, ``record=True`` resets the showwarning function but
    does NOT explicitly call simplefilter. Whether the default filter list is
    re-initialised to empty (= effectively 'always') or carries over from the
    enclosing context is exactly what this regime tests.
    """
    with warnings.catch_warnings(record=True) as buf:
        yield buf


@contextmanager
def regime_c_preview() -> Iterator[list]:
    """``catch_warnings(record=True)`` + ``simplefilter("always")`` — preview's pattern."""
    with warnings.catch_warnings(record=True) as buf:
        warnings.simplefilter("always")
        yield buf


REGIMES = [
    ("a_baseline", regime_a_baseline),
    ("b_record_only", regime_b_record_only),
    ("c_preview", regime_c_preview),
]


# -----------------------------------------------------------------------------
# Test 1: identical warnings (same location, same message — dedup-favourable)
# -----------------------------------------------------------------------------


def warn_loop_identical(n: int) -> None:
    for _ in range(n):
        warnings.warn("synthetic perturbation probe", UserWarning, stacklevel=2)


def time_identical(regime, n: int) -> tuple[float, int]:
    # Reset the per-module __warningregistry__ so each run starts fresh.
    # Otherwise (a) gets faster on repeat runs (registry already has the entry).
    if "__warningregistry__" in globals():
        globals()["__warningregistry__"].clear()
    with regime() as buf:
        t0 = time.perf_counter()
        warn_loop_identical(n)
        t1 = time.perf_counter()
    captured = len(buf) if buf is not None else -1
    return t1 - t0, captured


# -----------------------------------------------------------------------------
# Test 2: distinct warnings (varying message — dedup-hostile)
# -----------------------------------------------------------------------------


def warn_loop_distinct(n: int) -> None:
    for i in range(n):
        warnings.warn(f"synthetic probe variant {i}", UserWarning, stacklevel=2)


def time_distinct(regime, n: int) -> tuple[float, int]:
    if "__warningregistry__" in globals():
        globals()["__warningregistry__"].clear()
    with regime() as buf:
        t0 = time.perf_counter()
        warn_loop_distinct(n)
        t1 = time.perf_counter()
    captured = len(buf) if buf is not None else -1
    return t1 - t0, captured


# -----------------------------------------------------------------------------
# Test 3: torch matmul loop (no warnings in path — setup overhead only)
# -----------------------------------------------------------------------------


def time_matmul(regime, n: int, dim: int = 256) -> tuple[float, int]:
    try:
        import torch
    except ImportError:
        return float("nan"), -1
    a = torch.randn(dim, dim)
    b = torch.randn(dim, dim)
    with regime() as buf:
        t0 = time.perf_counter()
        for _ in range(n):
            torch.matmul(a, b)
        t1 = time.perf_counter()
    captured = len(buf) if buf is not None else -1
    return t1 - t0, captured


# -----------------------------------------------------------------------------
# Test 4: capture count comparison (load-bearing — does (b) capture all?)
# -----------------------------------------------------------------------------


def capture_count_comparison(n: int = 1000) -> dict:
    """Run a known-N emission set under (b) and (c), compare captured counts."""
    out = {}
    for label, regime in REGIMES:
        # Fresh registry
        if "__warningregistry__" in globals():
            globals()["__warningregistry__"].clear()
        # Mix of identical and distinct warnings
        with regime() as buf:
            for i in range(n):
                # Half identical, half distinct
                if i < n // 2:
                    warnings.warn("identical", UserWarning, stacklevel=2)
                else:
                    warnings.warn(f"distinct-{i}", UserWarning, stacklevel=2)
        captured = len(buf) if buf is not None else None
        out[label] = {
            "emitted": n,
            "captured": captured,
        }
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def run_timed_test(name: str, fn, regimes: list, n: int, repeats: int) -> dict:
    print(f"\n--- {name} (n={n}, repeats={repeats}) ---")
    out = {}
    for label, regime in regimes:
        times = []
        last_captured = -1
        for _ in range(repeats):
            t, c = fn(regime, n)
            times.append(t)
            last_captured = c
        median = statistics.median(times)
        try:
            q = statistics.quantiles(times, n=4)
            iqr_str = f"{q[0] * 1000:.2f}-{q[2] * 1000:.2f}ms"
        except statistics.StatisticsError:
            iqr_str = "n/a"
        per_call_us = (median / n) * 1e6
        out[label] = {
            "median_s": median,
            "median_ms": median * 1000,
            "iqr_str": iqr_str,
            "per_call_us": per_call_us,
            "captured": last_captured,
            "all_runs_s": times,
        }
        print(
            f"  {label:14s}: median {median * 1000:7.2f}ms  IQR {iqr_str:>22s}  "
            f"{per_call_us:6.2f} µs/call  captured={last_captured}"
        )
    return out


def overhead_table(out: dict, baseline: str = "a_baseline") -> None:
    base = out[baseline]["median_s"]
    print(f"  Relative overhead vs {baseline}:")
    for label, data in out.items():
        delta_pct = (data["median_s"] - base) / base * 100
        marker = ""
        if label != baseline:
            if abs(delta_pct) < 1:
                marker = " (SAFE)"
            elif abs(delta_pct) < 5:
                marker = " (BORDERLINE)"
            else:
                marker = " (BLOCKING)"
        print(f"    {label:14s}: {delta_pct:+7.2f}%{marker}")


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print("PoC-RT-0 — warning-filter perturbation, three regimes")

    # Test 1
    out1 = run_timed_test(
        "Test 1: identical warnings (dedup-favourable)",
        time_identical,
        REGIMES,
        n=10000,
        repeats=5,
    )
    overhead_table(out1)

    # Test 2
    out2 = run_timed_test(
        "Test 2: distinct warnings (dedup-hostile)",
        time_distinct,
        REGIMES,
        n=10000,
        repeats=5,
    )
    overhead_table(out2)

    # Test 3
    out3 = run_timed_test(
        "Test 3: torch.matmul (no warnings in path)",
        time_matmul,
        REGIMES,
        n=1000,
        repeats=5,
    )
    overhead_table(out3)

    # Test 4
    print("\n--- Test 4: capture-count comparison ---")
    cap = capture_count_comparison(n=1000)
    print("  500 identical + 500 distinct emitted under each regime:")
    for label, data in cap.items():
        captured = data["captured"]
        if captured is None:
            captured_str = "N/A (no record)"
        else:
            captured_str = str(captured)
        print(f"    {label:14s}: emitted={data['emitted']}  captured={captured_str}")

    # Summary
    print("\n--- Summary ---")
    print(
        f"  Test 1 (identical) c vs a:   {(out1['c_preview']['median_s'] - out1['a_baseline']['median_s']) / out1['a_baseline']['median_s'] * 100:+.2f}%"
    )
    print(
        f"  Test 1 (identical) b vs a:   {(out1['b_record_only']['median_s'] - out1['a_baseline']['median_s']) / out1['a_baseline']['median_s'] * 100:+.2f}%"
    )
    print(
        f"  Test 2 (distinct)  c vs a:   {(out2['c_preview']['median_s'] - out2['a_baseline']['median_s']) / out2['a_baseline']['median_s'] * 100:+.2f}%"
    )
    print(
        f"  Test 2 (distinct)  b vs a:   {(out2['b_record_only']['median_s'] - out2['a_baseline']['median_s']) / out2['a_baseline']['median_s'] * 100:+.2f}%"
    )
    print(
        f"  Test 3 (matmul)    c vs a:   {(out3['c_preview']['median_s'] - out3['a_baseline']['median_s']) / out3['a_baseline']['median_s'] * 100:+.2f}%"
    )
    print(
        f"  Test 4: b captured {cap['b_record_only']['captured']} / 1000  vs c captured {cap['c_preview']['captured']} / 1000"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
