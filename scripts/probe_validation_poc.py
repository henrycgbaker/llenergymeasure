#!/usr/bin/env python3
"""Proof-of-concept: measure ConfigProbe value on a full-suite study YAML.

Purpose
-------
Ground the `.product/designs/runtime-config-probe-revisions.md` discussion in
real numbers. Answers: "how often does the probe actually find something on
a realistic config, and how compressible are those findings into a cache?"

Usage
-----
    python scripts/probe_validation_poc.py [path/to/study.yaml] [--sample N] [--all]

Defaults to configs/example-study-full.yaml, sample size 2000 (seeded).
--all probes every expanded config (slow for large sweeps).

Full-suite YAML expands to ~57k configs. Per-probe AutoConfig calls dominate
runtime even with HF cache hits. A 2000-sample run is statistically adequate
for the decision-criteria thresholds and finishes in a few minutes.
Meta-device construction is disabled by default here (env var set inside
the script) for the same reason — set LLEM_PROBE_META_DEVICE_ENABLED=1 if
you want to include it.

What it measures
----------------
- Expansion: how many ExperimentConfigs the YAML expands into.
- Dormancy rate: % of configs with at least one dormant field at T0.
- Error rate: % of configs the probe rejects (T1/T2 framework errors + T5 hw).
- Distinct signatures: unique (dormant-field-set) and (error-bucket) tuples.
- Cache compression: configs ÷ distinct signatures. High ratio = lazy cache
  pays off; low ratio = each config is its own snowflake, cache barely helps.
- Wall-clock: probe time per config + total.

What it cannot measure (yet)
----------------------------
- T3 runtime pathologies (hangs, cold-start): needs live engines.
- Runtime `generate()` warnings: needs actual inference.
- Cache hit-rate evolution over a sequential run: needs a replay sim.

Scope caveats
-------------
- Runs on the host. T1/T2 library imports for vLLM/TRT likely fail on a host
  without those libraries — the probe silently skips affected tiers (by
  design). Transformers T1/T2 can fully run if transformers is installed.
- T5 (TRT hardware check) depends on nvidia-smi visibility. On this dev
  host NVML sees the GPUs even though CUDA isn't available to torch —
  the probe will correctly report SM-capability checks.
- The PoC is diagnostic, not a blocker. Numbers are illustrative, not
  statistically rigorous.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# Disable expensive T2 meta-device construction by default for the PoC;
# users can re-enable via env var if they want to include it.
os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.engines import get_engine
from llenergymeasure.engines.protocol import ConfigProbe

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "example-study-full.yaml"
DEFAULT_SAMPLE = 2000
DEFAULT_SEED = 42


def _short(txt: str, n: int = 80) -> str:
    """Trim a string for tabular display."""
    return txt if len(txt) <= n else txt[: n - 1] + "…"


def _signature(probe: ConfigProbe) -> tuple[str, ...]:
    """Stable tuple of dormant-field keys for dedup. Empty tuple = no dormancy."""
    return tuple(sorted(probe.dormant_fields.keys()))


_ENVIRONMENTAL_MARKERS = (
    "libpython",
    "cannot open shared object",
    "No module named",
)


def _is_environmental(err: str) -> bool:
    """Heuristic: error stems from broken/partial library install, not config."""
    return any(m in err for m in _ENVIRONMENTAL_MARKERS)


def _error_bucket(err: str) -> str:
    """Coarse-grain error string to its tier prefix."""
    if _is_environmental(err):
        return "ENV (host-partial-install, not config-related)"
    for prefix in ("T0 ", "T1 ", "T2 ", "T5", "FP8 quant", "TensorRT-LLM requires"):
        if err.startswith(prefix) or prefix in err[:30]:
            return prefix.strip()
    return "other"


def _error_signature(probe: ConfigProbe) -> tuple[str, ...]:
    """Stable tuple of error buckets for dedup."""
    return tuple(sorted({_error_bucket(e) for e in probe.errors}))


def run_probes(
    yaml_path: Path,
    sample_size: int | None,
    seed: int,
) -> tuple[list[tuple[str, ConfigProbe, float]], float, int]:
    """Load the YAML, expand, optionally sample, and probe each config.

    Returns
    -------
    probes : list of (engine_name, ConfigProbe, wall_clock_seconds)
    total_wall : total probe time in seconds
    full_count : total configs after expansion (pre-sampling)
    """
    print(f"Loading {yaml_path}...", flush=True)
    study = load_study_config(yaml_path)
    configs = study.experiments
    full_count = len(configs)
    print(f"Expanded into {full_count} ExperimentConfig objects.", flush=True)

    if sample_size is not None and sample_size < full_count:
        rng = random.Random(seed)
        configs = rng.sample(configs, sample_size)
        print(f"Sampling {sample_size} configs (seed={seed}).\n", flush=True)
    else:
        print("Probing all configs (no sampling).\n", flush=True)

    results: list[tuple[str, ConfigProbe, float]] = []
    total_start = time.perf_counter()
    for i, cfg in enumerate(configs, 1):
        if i % 100 == 0 or i == len(configs):
            elapsed = time.perf_counter() - total_start
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  probe {i}/{len(configs)}  ({rate:.1f} probes/sec, {elapsed:.1f}s elapsed)",
                flush=True,
            )
        try:
            engine = get_engine(cfg.engine)
        except Exception as exc:
            print(f"    [skip] config {i}: get_engine failed: {exc}")
            continue
        t0 = time.perf_counter()
        probe = engine.probe_config(cfg)
        elapsed_one = time.perf_counter() - t0
        results.append((cfg.engine, probe, elapsed_one))
    total_wall = time.perf_counter() - total_start
    return results, total_wall, full_count


def summarise(
    results: list[tuple[str, ConfigProbe, float]],
    total_wall: float,
    full_count: int,
) -> None:
    print("=" * 78)
    print("PROBE VALIDATION SUMMARY")
    print("=" * 78)

    if not results:
        print("No probes ran. Nothing to summarise.")
        return

    # Global metrics
    total = len(results)
    if total != full_count:
        print(f"\n(Sample of {total} / {full_count} total configs; results scaled to the sample.)")

    # Partition errors into config-relevant vs environmental
    config_errors = 0
    env_errors = 0
    for _, probe, _ in results:
        for err in probe.errors:
            if _is_environmental(err):
                env_errors += 1
            else:
                config_errors += 1
    configs_with_env_only = sum(
        1 for _, p, _ in results if p.errors and all(_is_environmental(e) for e in p.errors)
    )
    configs_with_config_err = sum(
        1 for _, p, _ in results if any(not _is_environmental(e) for e in p.errors)
    )
    with_dormancy = sum(1 for _, p, _ in results if p.dormant_fields)
    with_errors = sum(1 for _, p, _ in results if not p.is_valid)
    with_warnings = sum(1 for _, p, _ in results if p.warnings)
    per_probe_ms = [w * 1000 for _, _, w in results]

    print(f"\nTotal configs probed:        {total}")
    print(f"Wall-clock total:            {total_wall:.2f}s")
    print(
        f"Wall-clock per probe (ms):   "
        f"p50={statistics.median(per_probe_ms):.1f}  "
        f"p95={sorted(per_probe_ms)[int(0.95 * len(per_probe_ms)) - 1]:.1f}  "
        f"max={max(per_probe_ms):.1f}"
    )
    print(
        f"\nConfigs with T0 dormancy:    {with_dormancy:5d} ({100 * with_dormancy / total:5.1f}%)"
    )
    print(f"Configs with errors (not valid): {with_errors:5d} ({100 * with_errors / total:5.1f}%)")
    print(
        f"  ... of which config-relevant: "
        f"{configs_with_config_err:5d} "
        f"({100 * configs_with_config_err / total:5.1f}%)"
    )
    print(
        f"  ... of which environmental-only: "
        f"{configs_with_env_only:5d} "
        f"({100 * configs_with_env_only / total:5.1f}%) [noise; filter out]"
    )
    print(f"Configs with warnings:       {with_warnings:5d} ({100 * with_warnings / total:5.1f}%)")

    # Per-engine breakdown
    by_engine: dict[str, list[ConfigProbe]] = {}
    for name, probe, _ in results:
        by_engine.setdefault(name, []).append(probe)

    print("\n" + "-" * 78)
    print("PER-ENGINE BREAKDOWN")
    print("-" * 78)
    for engine_name, probes in sorted(by_engine.items()):
        n = len(probes)
        n_d = sum(1 for p in probes if p.dormant_fields)
        n_e = sum(1 for p in probes if not p.is_valid)
        sigs = {_signature(p) for p in probes}
        err_sigs = {_error_signature(p) for p in probes if not p.is_valid}
        print(f"\n  {engine_name}: {n} configs")
        print(f"    dormancy:        {n_d:5d} ({100 * n_d / n:5.1f}%)")
        print(f"    errors:          {n_e:5d} ({100 * n_e / n:5.1f}%)")
        print(f"    distinct dormancy signatures: {len(sigs)}")
        print(f"    distinct error   signatures:  {len(err_sigs)}")
        if n > 0:
            compression = n / max(len(sigs), 1)
            print(
                f"    cache compression ratio:      {compression:6.1f}x "
                f"({n} configs -> {len(sigs)} signatures)"
            )

    # Global signature dedup
    all_sigs = {_signature(p) for _, p, _ in results}
    all_err_sigs = {_error_signature(p) for _, p, _ in results if not p.is_valid}
    print("\n" + "-" * 78)
    print("CACHE COMPRESSION (global, across all engines)")
    print("-" * 78)
    print(f"Distinct dormancy signatures: {len(all_sigs)}")
    print(f"Distinct error signatures:    {len(all_err_sigs)}")
    if all_sigs:
        print(f"Compression ratio (dormancy): {total / len(all_sigs):6.1f}x")
    if all_err_sigs:
        print(f"Compression ratio (errors):   {with_errors / max(len(all_err_sigs), 1):6.1f}x")

    # Top dormant keys
    dormant_keys = Counter()
    for _, probe, _ in results:
        for key in probe.dormant_fields:
            dormant_keys[key] += 1

    if dormant_keys:
        print("\n" + "-" * 78)
        print("TOP DORMANT FIELDS (global)")
        print("-" * 78)
        for key, count in dormant_keys.most_common(15):
            pct = 100 * count / total
            print(f"  {count:5d} ({pct:5.1f}%)  {key}")

    # Top error buckets
    error_buckets = Counter()
    for _, probe, _ in results:
        for err in probe.errors:
            error_buckets[_error_bucket(err)] += 1

    if error_buckets:
        print("\n" + "-" * 78)
        print("ERROR BUCKETS (by tier)")
        print("-" * 78)
        for bucket, count in error_buckets.most_common():
            pct = 100 * count / total
            print(f"  {count:5d} ({pct:5.1f}%)  {bucket}")

    # Sample error messages
    sample_errors: list[str] = []
    seen_buckets: set[str] = set()
    for _, probe, _ in results:
        for err in probe.errors:
            bucket = _error_bucket(err)
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                sample_errors.append(f"[{bucket}] {_short(err, 120)}")
    if sample_errors:
        print("\n" + "-" * 78)
        print("SAMPLE ERROR MESSAGES (one per bucket)")
        print("-" * 78)
        for s in sample_errors[:10]:
            print(f"  {s}")

    # Decision-criteria check against the revisions doc (§6)
    print("\n" + "=" * 78)
    print("DECISION-CRITERIA READOUT (see .product/designs/runtime-config-probe-revisions.md §6)")
    print("=" * 78)
    dormancy_rate = 100 * with_dormancy / total
    error_rate = 100 * configs_with_config_err / total  # exclude environmental noise
    compression = total / max(len(all_sigs), 1)

    def verdict(cond: bool) -> str:
        return "✓ YES" if cond else "✗ NO"

    print(
        f"  >30% dormancy?             {verdict(dormancy_rate > 30):6s}  "
        f"(observed: {dormancy_rate:.1f}%)"
    )
    print(
        f"  >10% errors?               {verdict(error_rate > 10):6s}  (observed: {error_rate:.1f}%)"
    )
    print(
        f"  cache compression >50x?    {verdict(compression > 50):6s}  "
        f"(observed: {compression:.1f}x)"
    )
    print(
        f"  cache compression <5x?     {verdict(compression < 5):6s}  "
        f"(observed: {compression:.1f}x)"
    )
    print(
        f"  both <5% dormancy AND <2% errors?  "
        f"{verdict(dormancy_rate < 5 and error_rate < 2):6s}  "
        f"(overengineering warning)"
    )

    print("\nInterpretation guide:")
    print("  All green 'YES' on first three → probe has strong value; ship the design")
    print("  Compression <5x → lazy exact-tuple cache doesn't pay; drop cache or generalise")
    print("  Overengineering flag → consider descoping probe to runtime-only")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml",
        nargs="?",
        default=str(DEFAULT_YAML),
        help=f"Study YAML path (default: {DEFAULT_YAML.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Random sample size (default: {DEFAULT_SAMPLE}; ignored with --all)",
    )
    parser.add_argument("--all", action="store_true", help="Probe every expanded config (slow)")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed (default {DEFAULT_SEED})"
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"ERROR: YAML not found: {yaml_path}", file=sys.stderr)
        return 1

    sample_size = None if args.all else args.sample

    try:
        results, total_wall, full_count = run_probes(yaml_path, sample_size, args.seed)
    except Exception as exc:
        print(f"ERROR during probe run: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2

    summarise(results, total_wall, full_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
