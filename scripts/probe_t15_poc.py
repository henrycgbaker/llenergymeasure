#!/usr/bin/env python3
"""T1.5 PoC — input-vs-output diff on native config objects.

Motivation
----------
T1 in the current probe only catches construction *exceptions*
(`SamplingParams(**kwargs)` raising). It doesn't catch silent normalisations:
vLLM's ``SamplingParams.__post_init__`` forces ``top_p=1.0`` when
``temperature < epsilon``; HF's ``GenerationConfig`` has similar logic. The
user wrote kwargs, the library silently rewrote them, no exception fires.

T1.5 proposal: after native-type construction succeeds, diff the declared
kwargs against the constructed object's field values. Any discrepancy is
dormancy caused by library-internal normalisation — a class of dormancy T0
(our wrapper-side diff) cannot see.

Scope
-----
- Transformers: diff kwargs against ``GenerationConfig(**kwargs).__dict__``.
- vLLM: diff kwargs against ``vllm.SamplingParams(**kwargs)`` attributes.
- TRT-LLM: diff kwargs against ``tensorrt_llm.llmapi.LlmArgs(**kwargs)`` fields.

Runs host-side. vLLM T1 works locally. Transformers T1 works locally.
TRT-LLM fails on this host (partial install) — skip and note.

Usage
-----
    python scripts/probe_t15_poc.py [--sample N]

Writes results to stdout + optional archive file.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "example-study-full.yaml"


def _extract_object_fields(obj: Any) -> dict[str, Any]:
    """Extract introspectable field values from a native config object.

    Handles:
      - Pydantic v2 models (model_dump)
      - Dataclasses (__dataclass_fields__)
      - msgspec.Struct / __slots__-based objects (iterate __slots__ + getattr)
      - Plain objects with __dict__

    Returns only fields with primitive-ish values (ignores nested objects).
    """
    result: dict[str, Any] = {}

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump(exclude_none=False)
            if dumped:
                return dumped
        except Exception:
            pass

    # __slots__-based (msgspec.Struct, and vLLM's SamplingParams)
    # Walk the MRO so inherited slots are picked up too.
    slot_names: set[str] = set()
    for cls in type(obj).__mro__:
        for name in getattr(cls, "__slots__", ()):
            if not name.startswith("_"):
                slot_names.add(name)
    for name in slot_names:
        try:
            v = getattr(obj, name)
            if isinstance(v, (int, float, bool, str, type(None), list, tuple)):
                result[name] = v
        except AttributeError:
            continue

    # Merge __dict__ on top (for mixed slot/dict classes)
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, bool, str, type(None), list, tuple)):
                result[k] = v

    return result


def _diff_kwargs_vs_object(
    kwargs: dict[str, Any], obj_fields: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return fields whose declared value differs from the constructed object's value.

    Only considers keys present in BOTH kwargs and obj_fields. Keys declared by
    the user but missing from the object (dropped entirely) count as dormant;
    keys present only in the object (added by defaults) do not.
    """
    diffs = {}
    for k, declared in kwargs.items():
        if k not in obj_fields:
            diffs[k] = (declared, None)  # key dropped
            continue
        effective = obj_fields[k]
        if declared != effective:
            diffs[k] = (declared, effective)
    return diffs


def t15_transformers(engine: TransformersEngine, cfg: Any) -> dict[str, tuple[Any, Any]]:
    """T1.5 for transformers: diff against GenerationConfig."""
    try:
        from transformers import GenerationConfig
    except Exception:
        return {}

    try:
        generate_kwargs = engine._build_generate_kwargs(cfg)
    except Exception:
        return {}

    # Filter to fields GenerationConfig knows about
    try:
        allowed = set(GenerationConfig().__dict__.keys())
    except Exception:
        return {}

    filtered = {k: v for k, v in generate_kwargs.items() if k in allowed}
    if not filtered:
        return {}

    try:
        gc_obj = GenerationConfig(**filtered)
    except Exception:
        return {}

    obj_fields = _extract_object_fields(gc_obj)
    return _diff_kwargs_vs_object(filtered, obj_fields)


def t15_vllm(engine: VLLMEngine, cfg: Any) -> dict[str, tuple[Any, Any]]:
    """T1.5 for vLLM: diff against SamplingParams attributes."""
    try:
        from vllm import SamplingParams
    except Exception:
        return {}

    # Beam search uses a different params class; skip T1.5 in that branch
    if cfg.vllm is not None and cfg.vllm.beam_search is not None:
        return {}

    try:
        sampling_kwargs = engine._build_sampling_kwargs(cfg)
    except Exception:
        return {}

    if not sampling_kwargs:
        return {}

    try:
        sp_obj = SamplingParams(**sampling_kwargs)
    except Exception:
        return {}

    obj_fields = _extract_object_fields(sp_obj)
    return _diff_kwargs_vs_object(sampling_kwargs, obj_fields)


def t15_tensorrt(engine: TensorRTEngine, cfg: Any) -> dict[str, tuple[Any, Any]]:
    """T1.5 for TRT-LLM: diff against SamplingParams attributes.

    Falls back silently if tensorrt_llm is not importable on this host.
    """
    try:
        from tensorrt_llm import SamplingParams  # noqa: F401
    except Exception:
        return {}

    try:
        sampling_kwargs = engine._build_sampling_kwargs(cfg)
    except Exception:
        return {}

    if not sampling_kwargs:
        return {}

    try:
        from tensorrt_llm import SamplingParams as TRT_SP

        sp_obj = TRT_SP(**sampling_kwargs)
    except Exception:
        return {}

    obj_fields = _extract_object_fields(sp_obj)
    return _diff_kwargs_vs_object(sampling_kwargs, obj_fields)


T15_DISPATCH = {
    "transformers": t15_transformers,
    "vllm": t15_vllm,
    "tensorrt": t15_tensorrt,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", nargs="?", default=str(DEFAULT_YAML))
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    print(f"Loading {yaml_path}...", flush=True)
    study = load_study_config(yaml_path)
    full_count = len(study.experiments)
    print(f"Expanded into {full_count} configs.", flush=True)

    rng = random.Random(args.seed)
    configs = (
        rng.sample(study.experiments, args.sample)
        if args.sample < full_count
        else study.experiments
    )
    print(f"Probing {len(configs)} configs (T1.5 diff).\n", flush=True)

    engines = {
        "transformers": TransformersEngine(),
        "vllm": VLLMEngine(),
        "tensorrt": TensorRTEngine(),
    }

    results: dict[str, list[dict[str, tuple[Any, Any]]]] = {
        "transformers": [],
        "vllm": [],
        "tensorrt": [],
    }

    t0 = time.perf_counter()
    for i, cfg in enumerate(configs, 1):
        if i % 200 == 0:
            print(f"  t1.5 probe {i}/{len(configs)}  ({time.perf_counter() - t0:.1f}s)", flush=True)
        name = cfg.engine
        engine = engines[name]
        diffs = T15_DISPATCH[name](engine, cfg)
        results[name].append(diffs)

    total = time.perf_counter() - t0
    print(f"\nT1.5 total time: {total:.1f}s\n", flush=True)

    # Summary
    print("=" * 78)
    print("T1.5 (input-vs-output diff) RESULTS")
    print("=" * 78)
    for engine_name, per_cfg_diffs in results.items():
        n = len(per_cfg_diffs)
        if n == 0:
            continue
        with_diff = sum(1 for d in per_cfg_diffs if d)
        total_diffs = sum(len(d) for d in per_cfg_diffs)
        key_counter: Counter[str] = Counter()
        for d in per_cfg_diffs:
            for k in d:
                key_counter[k] += 1
        print(f"\n{engine_name}: {n} configs")
        print(f"  configs with T1.5 diff: {with_diff:5d} ({100 * with_diff / n:5.1f}%)")
        print(f"  total diff entries:     {total_diffs}")
        if key_counter:
            print("  top fields with diffs:")
            for k, c in key_counter.most_common(10):
                print(f"    {c:5d}  {k}")
        else:
            print("  (no diffs found)")

    # Sample diffs for eyeballing
    print("\n" + "-" * 78)
    print("SAMPLE DIFFS (one per engine)")
    print("-" * 78)
    for engine_name, per_cfg_diffs in results.items():
        for d in per_cfg_diffs:
            if d:
                print(f"\n  {engine_name}:")
                for k, (declared, effective) in list(d.items())[:5]:
                    print(f"    {k}: declared={declared!r} -> effective={effective!r}")
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
