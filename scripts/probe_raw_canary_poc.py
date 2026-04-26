#!/usr/bin/env python3
"""Phase C: raw canary — does HF surface the dormancy our wrapper catches?

Test: pick a handful of greedy+sampling configs (the class T0 catches via
wrapper-strip). For each, pass the user's *declared* kwargs directly to
HF's generate() — no wrapper preprocessing. Capture HF's warnings. Compare
to what T0 reports as dormant.

If HF surfaces the same dormant fields T0 catches:
  → Wrapper strip is redundant with HF's runtime warnings.
  → T0 could be removed if T3/runtime log capture shipped.
  → Symmetry with vLLM/TRT wrappers becomes achievable.

If HF surfaces NOTHING or different fields:
  → T0 is catching something HF only surfaces silently.
  → Wrapper strip earns its keep as the only detection path for these cases.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.engine_configs import (
    TransformersConfig,
    TransformersSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.transformers import TransformersEngine

REPO_ROOT = Path(__file__).resolve().parent.parent


# Hand-crafted representative configs covering the dormancy-triggering classes.
CASES = [
    {
        "name": "greedy_with_temp_top_p",
        "cfg": ExperimentConfig(
            task={"model": "Qwen/Qwen2.5-0.5B"},
            engine="transformers",
            transformers=TransformersConfig(
                sampling=TransformersSamplingConfig(
                    do_sample=False,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=40,
                )
            ),
        ),
    },
    {
        "name": "greedy_with_min_p",
        "cfg": ExperimentConfig(
            task={"model": "Qwen/Qwen2.5-0.5B"},
            engine="transformers",
            transformers=TransformersConfig(
                sampling=TransformersSamplingConfig(
                    do_sample=False,
                    temperature=0.5,
                    top_p=0.9,
                    top_k=50,
                    min_p=0.1,
                )
            ),
        ),
    },
    {
        "name": "temp_zero_sampling_on",
        "cfg": ExperimentConfig(
            task={"model": "Qwen/Qwen2.5-0.5B"},
            engine="transformers",
            transformers=TransformersConfig(
                sampling=TransformersSamplingConfig(
                    do_sample=True,
                    temperature=0.0,
                    top_p=0.9,
                    top_k=40,
                )
            ),
        ),
    },
    {
        "name": "clean_sampling",
        "cfg": ExperimentConfig(
            task={"model": "Qwen/Qwen2.5-0.5B"},
            engine="transformers",
            transformers=TransformersConfig(
                sampling=TransformersSamplingConfig(
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                )
            ),
        ),
    },
]


def declared_sampling_kwargs(cfg: ExperimentConfig) -> dict:
    """The user's raw sampling kwargs, NOT wrapper-stripped."""
    pt = cfg.transformers
    sampling = pt.sampling if pt and pt.sampling else None
    if sampling is None:
        return {}
    return sampling.model_dump(exclude_none=True)


def run_raw(cfg: ExperimentConfig, raw_kwargs: dict, timeout: int = 120) -> dict:
    payload = json.dumps(
        {"cfg": json.loads(cfg.model_dump_json()), "raw_generate_kwargs": raw_kwargs}
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-i",
        "-e",
        "PYTHONPATH=/workspace/src",
        "-e",
        "HF_HUB_CACHE=/root/.cache/huggingface/hub",
        "-e",
        "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-v",
        f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-w",
        "/workspace",
        "--entrypoint",
        "python",
        "llenergymeasure:pytorch",
        "/workspace/scripts/canaries/canary_transformers_raw.py",
    ]
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    if proc.returncode != 0:
        return {"ok": False, "error": f"rc={proc.returncode}", "stderr": proc.stderr[-500:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {"ok": False, "error": f"parse: {e}", "stdout": proc.stdout[-500:]}


def main() -> int:
    engine = TransformersEngine()
    print("=" * 90)
    print("PHASE C — raw canary vs T0 comparison")
    print("=" * 90)

    for case in CASES:
        cfg = case["cfg"]
        name = case["name"]
        print(f"\n--- {name} ---")

        # What T0 says
        probe = engine.probe_config(cfg)
        t0_dormant = sorted(probe.dormant_fields.keys())
        print(f"  T0 reports dormant: {t0_dormant}")

        # Raw canary
        raw_kwargs = declared_sampling_kwargs(cfg)
        print(f"  Raw kwargs passed to HF: {raw_kwargs}")
        t_start = time.perf_counter()
        result = run_raw(cfg, raw_kwargs, timeout=300)
        elapsed = time.perf_counter() - t_start
        if not result.get("ok"):
            print(f"  RAW CANARY FAILED ({elapsed:.1f}s): {result.get('error', '')[:100]}")
            continue

        hf_hints = result.get("dormancy_hints", [])
        hf_warnings = result.get("captured_warnings", [])
        hf_stderr = result.get("relevant_stderr", [])

        print(f"  HF warnings captured: {len(hf_warnings)}")
        for w in hf_warnings[:5]:
            print(f"    W: {w[:140]}")
        print(f"  HF stderr relevant:   {len(hf_stderr)}")
        for s in hf_stderr[:8]:
            print(f"    S: {s[:140]}")
        print(f"  HF dormancy hints:    {hf_hints}")
        print(f"  canary wall time:     {elapsed:.1f}s")

        # Direct comparison
        t0_short = set(k.rsplit(".", 1)[-1] for k in t0_dormant)
        hf_set = set(hf_hints)
        both = t0_short & hf_set
        t0_only = t0_short - hf_set
        hf_only = hf_set - t0_short

        print("  COMPARISON:")
        print(f"    Both T0 and HF catch: {sorted(both)}")
        print(f"    T0 catches, HF silent: {sorted(t0_only)}")
        print(f"    HF catches, T0 silent: {sorted(hf_only)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
