#!/usr/bin/env python3
"""Phase D.1: build-tuple cardinality analysis across the full 56k YAML expansion.

Answers:
- How many unique engine-build-config tuples exist per engine?
- What is the dedup compression ratio?
- If T3 runs once per unique build, what's the compute budget?
- If cache is keyed on (build + sampling), what's the cache size?

Runs in seconds (no probe execution).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.loader import load_study_config
from scripts.probe_t3_canary_poc import BUILD_TUPLE_KEYS, _get_dotted

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "example-study-full.yaml"


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, default=str, sort_keys=True).encode()).hexdigest()[:12]


def sampling_tuple_keys(engine_name: str) -> list[str]:
    """Sampling-axis keys per engine (parallel to BUILD_TUPLE_KEYS in probe_t3_canary_poc)."""
    if engine_name == "transformers":
        return [
            "transformers.sampling.do_sample",
            "transformers.sampling.temperature",
            "transformers.sampling.top_p",
            "transformers.sampling.top_k",
            "transformers.sampling.min_p",
            "transformers.sampling.repetition_penalty",
        ]
    if engine_name == "vllm":
        return [
            "vllm.sampling.temperature",
            "vllm.sampling.top_p",
            "vllm.sampling.top_k",
            "vllm.sampling.repetition_penalty",
            "vllm.sampling.min_p",
            "vllm.sampling.presence_penalty",
            "vllm.sampling.frequency_penalty",
            "vllm.sampling.min_tokens",
        ]
    if engine_name == "tensorrt":
        return [
            "tensorrt.sampling.temperature",
            "tensorrt.sampling.top_p",
            "tensorrt.sampling.top_k",
            "tensorrt.sampling.repetition_penalty",
            "tensorrt.sampling.min_p",
        ]
    return []


def main() -> int:
    yaml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_YAML
    print(f"Loading {yaml_path} ...", flush=True)
    study = load_study_config(yaml_path)
    configs = study.experiments
    total = len(configs)
    print(f"Total expanded configs: {total}\n", flush=True)

    # Per-engine counts
    by_engine = Counter(c.engine for c in configs)
    print("Configs per engine:")
    for name, count in by_engine.most_common():
        print(f"  {name}: {count}")
    print()

    print("=" * 78)
    print("CARDINALITY BY DEDUP SCHEME")
    print("=" * 78)

    for engine_name in ("transformers", "vllm", "tensorrt"):
        engine_configs = [c for c in configs if c.engine == engine_name]
        if not engine_configs:
            continue

        build_keys_list = BUILD_TUPLE_KEYS.get(engine_name, [])
        sampling_keys_list = sampling_tuple_keys(engine_name)

        build_tuples = Counter()
        sampling_tuples = Counter()
        build_plus_sampling = Counter()

        for c in engine_configs:
            b = {k: _get_dotted(c, k) for k in build_keys_list}
            s = {k: _get_dotted(c, k) for k in sampling_keys_list}
            b_hash = _hash(b)
            s_hash = _hash(s)
            build_tuples[b_hash] += 1
            sampling_tuples[s_hash] += 1
            build_plus_sampling[(b_hash, s_hash)] += 1

        n = len(engine_configs)
        n_builds = len(build_tuples)
        n_sampling = len(sampling_tuples)
        n_both = len(build_plus_sampling)

        print(f"\n{engine_name}: {n} configs")
        print(
            f"  unique build-tuples:              {n_builds:6d}  (compression {n / n_builds:.1f}x)"
        )
        print(
            f"  unique sampling-tuples:           {n_sampling:6d}  (compression {n / n_sampling:.1f}x)"
        )
        print(f"  unique (build × sampling):        {n_both:6d}  (compression {n / n_both:.1f}x)")

        # T3 cost estimates
        secs_per_canary = {"transformers": 20, "vllm": 45, "tensorrt": 300}.get(engine_name, 60)
        t3_per_build = n_builds * secs_per_canary
        print(f"  est. T3 cost (at {secs_per_canary}s/canary, dedup on build):")
        print(f"    sequential: {t3_per_build / 60:.1f} min = {t3_per_build / 3600:.2f} h")

        # What fraction of canary-failures does T3 learn per experiment?
        # If T3 runs per build and we amortize over runtime axes:
        avg_runs_per_build = n / n_builds
        print(f"  avg configs sharing each build-tuple: {avg_runs_per_build:.1f}")

    print("\n" + "=" * 78)
    print("CACHE POPULATION (replay simulation)")
    print("=" * 78)
    # If we replay configs in YAML order, how fast does a (build × sampling) cache fill?
    for engine_name in ("transformers", "vllm", "tensorrt"):
        engine_configs = [c for c in configs if c.engine == engine_name]
        if not engine_configs:
            continue
        build_keys_list = BUILD_TUPLE_KEYS.get(engine_name, [])
        sampling_keys_list = sampling_tuple_keys(engine_name)

        seen_build_sampling: set[tuple[str, str]] = set()
        hits = 0
        # Sample checkpoints
        checkpoints = [
            1,
            10,
            100,
            500,
            len(engine_configs) // 4,
            len(engine_configs) // 2,
            len(engine_configs),
        ]
        checkpoints = sorted(set(c for c in checkpoints if c > 0 and c <= len(engine_configs)))
        milestones = []
        for i, c in enumerate(engine_configs, 1):
            b = _hash({k: _get_dotted(c, k) for k in build_keys_list})
            s = _hash({k: _get_dotted(c, k) for k in sampling_keys_list})
            k = (b, s)
            if k in seen_build_sampling:
                hits += 1
            else:
                seen_build_sampling.add(k)
            if i in checkpoints:
                hit_rate = 100 * hits / i
                milestones.append((i, len(seen_build_sampling), hit_rate))

        print(f"\n{engine_name}:")
        print(f"  {'after N configs':>18s}  {'unique keys':>12s}  {'hit rate':>10s}")
        for i, uniq, hr in milestones:
            print(f"  {i:18d}  {uniq:12d}  {hr:9.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
