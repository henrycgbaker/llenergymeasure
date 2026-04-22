#!/usr/bin/env python3
"""Does T3 (vLLM canary) catch the normalizations T1.5 catches?

vLLM's SamplingParams.__post_init__ normalizes:
- temperature < 0.01 → clamped to 0.01 (WITH logger warning)
- temperature==0 → top_p=1.0, top_k=0 (silent)

T1.5 verifies this by introspecting the constructed object. T3 runs the
engine for real. If T3 captures the same via stderr/warnings, T1.5 is
redundant with T3 (but cheaper). If T3 misses any, T1.5 has unique value.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.engine_configs import VLLMConfig, VLLMSamplingConfig
from llenergymeasure.config.models import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


CASES = [
    ("temp_epsilon", VLLMSamplingConfig(temperature=0.001, top_p=0.9, top_k=50)),
    ("temp_zero", VLLMSamplingConfig(temperature=0.0, top_p=0.95, top_k=50)),
    ("clean", VLLMSamplingConfig(temperature=0.7, top_p=0.9, top_k=50)),
]


def run_canary(cfg: ExperimentConfig, timeout: int = 180) -> dict:
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
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-v",
        f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-w",
        "/workspace",
        "--entrypoint",
        "python3",
        "llenergymeasure:vllm",
        "/workspace/scripts/canaries/canary_vllm.py",
    ]
    try:
        proc = subprocess.run(
            cmd, input=cfg.model_dump_json(), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    if proc.returncode != 0:
        return {"ok": False, "error": f"rc={proc.returncode}", "stderr": proc.stderr[-400:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {"ok": False, "error": f"parse: {e}", "stdout_tail": proc.stdout[-400:]}


def main() -> int:
    print("=" * 90)
    print("vLLM T3 canary vs T1.5: does T3 catch the same normalizations?")
    print("=" * 90)
    for name, sampling in CASES:
        cfg = ExperimentConfig(
            task={"model": "Qwen/Qwen2.5-0.5B"},
            engine="vllm",
            vllm=VLLMConfig(sampling=sampling),
        )
        print(f"\n--- {name}: temperature={sampling.temperature} ---")
        result = run_canary(cfg)
        if not result.get("ok"):
            print(f"  FAILED: {result.get('error')}")
            continue

        log_lines = result.get("relevant_log_lines", [])
        hints = result.get("dormancy_hints", [])
        normalization_lines = [
            ln
            for ln in log_lines
            if any(m in ln for m in ("temperature", "maxed", "top_p", "top_k"))
        ]
        print(
            f"  load+fwd: {result.get('t_load_sec', 0):.1f}+{result.get('first_forward_sec', 0):.2f}s"
        )
        print(f"  captured log lines: {len(log_lines)}")
        print(f"  normalization-related: {len(normalization_lines)}")
        for line in normalization_lines[:5]:
            print(f"    {line[:150]}")
        print(f"  parsed dormancy hints: {hints}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
