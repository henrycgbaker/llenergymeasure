#!/usr/bin/env python3
"""Measure per-tier probe latency to scope what's viable as preflight.

The n=2000 PoC showed 545ms/probe median. Of that, vLLM's T2
(create_engine_config) dominates. Question: if we disable T2, can T0 alone
run fast enough to preflight the full 56k sweep?

Timing strategy
---------------
For each of N sampled configs:
  - Time T0 alone (declared/effective diff)
  - Time T0+T1 (add native-type construction)
  - Time T0+T1+T2 (full probe)
Report p50/p95/max for each slice.
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.engines import get_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "example-study-full.yaml"


def time_t0(cfg, engine) -> float:
    """Time just the T0 dormancy diff (wrapper internal, no library imports)."""
    t = time.perf_counter()
    if cfg.engine == "transformers":
        declared = engine._declared_sampling_params(cfg)
        effective = engine._build_generate_kwargs(cfg)
    elif cfg.engine == "vllm" or cfg.engine == "tensorrt":
        declared = engine._declared_sampling_params(cfg)
        effective = engine._build_sampling_kwargs(cfg)
    # Compute diff
    for k, v in declared.items():
        _ = effective.get(k) != v
    return time.perf_counter() - t


def time_t0_t1(cfg, engine) -> float:
    """Time T0 + T1 (construct native sampling-params class)."""
    t = time.perf_counter()
    if cfg.engine == "transformers":
        from transformers import GenerationConfig

        gk = engine._build_generate_kwargs(cfg)
        allowed = set(GenerationConfig().__dict__.keys())
        filt = {k: v for k, v in gk.items() if k in allowed}
        try:
            GenerationConfig(**filt)
        except Exception:
            pass
    elif cfg.engine == "vllm":
        try:
            from vllm import SamplingParams
        except Exception:
            return time.perf_counter() - t
        sk = engine._build_sampling_kwargs(cfg)
        try:
            SamplingParams(**sk)
        except Exception:
            pass
    # TRT T1 fails on this host (partial install), skip timing
    return time.perf_counter() - t


def time_full_probe(cfg, engine) -> float:
    t = time.perf_counter()
    engine.probe_config(cfg)
    return time.perf_counter() - t


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    yaml_path = DEFAULT_YAML
    print(f"Loading {yaml_path}...", flush=True)
    study = load_study_config(yaml_path)
    configs = random.Random(42).sample(study.experiments, n)
    print(f"Sampled {n} configs.\n", flush=True)

    engines = {
        "transformers": get_engine("transformers"),
        "vllm": get_engine("vllm"),
        "tensorrt": get_engine("tensorrt"),
    }

    by_engine: dict[str, dict[str, list[float]]] = {
        name: {"t0": [], "t0_t1": [], "full": []} for name in engines
    }

    for i, cfg in enumerate(configs, 1):
        engine = engines[cfg.engine]
        by_engine[cfg.engine]["t0"].append(time_t0(cfg, engine) * 1000)
        by_engine[cfg.engine]["t0_t1"].append(time_t0_t1(cfg, engine) * 1000)
        by_engine[cfg.engine]["full"].append(time_full_probe(cfg, engine) * 1000)
        if i % 50 == 0:
            print(f"  {i}/{n}", flush=True)

    print("\n" + "=" * 78)
    print("PROBE LATENCY PER TIER (ms)")
    print("=" * 78)
    for engine_name, data in by_engine.items():
        print(f"\n{engine_name}: {len(data['t0'])} configs")
        for slice_name in ("t0", "t0_t1", "full"):
            samples = data[slice_name]
            if not samples:
                print(f"  {slice_name:10s}  (no samples)")
                continue
            p50 = statistics.median(samples)
            p95 = (
                sorted(samples)[int(0.95 * len(samples)) - 1]
                if len(samples) >= 20
                else max(samples)
            )
            mx = max(samples)
            print(f"  {slice_name:10s}  p50={p50:7.3f}  p95={p95:7.3f}  max={mx:7.3f}")

    # Extrapolate to full 56k
    print("\n" + "=" * 78)
    print("EXTRAPOLATION TO FULL 56,832-CONFIG SWEEP")
    print("=" * 78)
    full_counts = {"transformers": 9600, "vllm": 41472, "tensorrt": 5760}
    for engine_name in by_engine:
        t0_p50 = (
            statistics.median(by_engine[engine_name]["t0"]) if by_engine[engine_name]["t0"] else 0
        )
        full_p50 = (
            statistics.median(by_engine[engine_name]["full"])
            if by_engine[engine_name]["full"]
            else 0
        )
        n_full = full_counts[engine_name]
        print(f"\n{engine_name} ({n_full} configs):")
        print(f"  T0-only preflight:  {t0_p50 * n_full / 1000:.1f}s")
        print(
            f"  Full probe (T0-T5): {full_p50 * n_full / 1000:.1f}s = {full_p50 * n_full / 60_000:.1f} min"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
