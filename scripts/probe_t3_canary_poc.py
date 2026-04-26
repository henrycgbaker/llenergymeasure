#!/usr/bin/env python3
"""T3 canary PoC — actually launches inference per unique engine-build-tuple.

Complements T0-T2 (`probe_validation_poc.py`) and T1.5 (`probe_t15_poc.py`).
T3 runs the engine for real — loading model, calling generate(), capturing
warnings and first-forward timing that preflight tiers cannot see.

Architecture
------------
- Loads the study YAML, deduplicates configs on engine-build-tuple (unique
  combination of kwargs that would produce a distinct engine build).
- For each unique tuple, runs a canary:
  - Transformers: via `docker run --gpus all` on llenergymeasure:transformers.
  - vLLM: via `docker run --gpus all` on llenergymeasure:vllm.
  - TRT-LLM: via `docker run --gpus all` on llenergymeasure:tensorrt.
- Each container executes `scripts/canaries/canary_{engine}.py` with the
  config piped on stdin; result comes back as JSON on stdout.
- Orchestrator aggregates: how many dormancy hints T3 finds that T0-T2 missed.

Caps (for sanity):
  --transformers-limit  (default 30)
  --vllm-limit          (default 20)
  --tensorrt-limit      (default 10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("LLEM_PROBE_META_DEVICE_ENABLED", "0")

from llenergymeasure.config.loader import load_study_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "configs" / "example-study-full.yaml"
CANARY_DIR = Path(__file__).parent / "canaries"

IMAGES = {
    "transformers": "llenergymeasure:pytorch",
    "vllm": "llenergymeasure:vllm",
    # NOTE: the newer llenergymeasure:tensorrt image (8 days, 4 weeks) has a
    # cuKernelGetName symbol mismatch / missing-libnvinfer issue. The older
    # llm-energy-measure:tensorrt image (pre-rename, 2 months old) has a
    # working tensorrt_llm 0.21.0 and we mount current source via PYTHONPATH.
    # TODO: rebuild the current image; for PoC purposes we use the older one.
    "tensorrt": "llm-energy-measure:tensorrt",
}
# Each image uses a different python binary on $PATH.
PYTHON_BIN = {
    "transformers": "python",
    "vllm": "python3",
    "tensorrt": "python3",
}

# Fields that DEFINE the engine build (changes here force rebuild).
# Keys are dotted paths into ExperimentConfig's model_dump(exclude_none=True).
BUILD_TUPLE_KEYS = {
    "transformers": [
        "task.model",
        "transformers.dtype",
        "transformers.attn_implementation",
        "transformers.tp_plan",
        "transformers.tp_size",
        "transformers.device_map",
        "transformers.load_in_4bit",
        "transformers.load_in_8bit",
        "transformers.bnb_4bit_compute_dtype",
        "transformers.bnb_4bit_quant_type",
        "transformers.bnb_4bit_use_double_quant",
        "transformers.torch_compile",
        "transformers.torch_compile_mode",
        "transformers.torch_compile_backend",
    ],
    "vllm": [
        "task.model",
        "vllm.dtype",
        "vllm.engine.tensor_parallel_size",
        "vllm.engine.pipeline_parallel_size",
        "vllm.engine.quantization",
        "vllm.engine.kv_cache_dtype",
        "vllm.engine.enforce_eager",
        "vllm.engine.enable_chunked_prefill",
        "vllm.engine.enable_prefix_caching",
        "vllm.engine.block_size",
        "vllm.engine.max_num_seqs",
        "vllm.engine.max_num_batched_tokens",
        "vllm.engine.max_model_len",
        "vllm.attention.backend",
    ],
    "tensorrt": [
        "task.model",
        "tensorrt.dtype",
        "tensorrt.tensor_parallel_size",
        "tensorrt.pipeline_parallel_size",
        "tensorrt.max_batch_size",
        "tensorrt.max_input_len",
        "tensorrt.max_seq_len",
        "tensorrt.quant.quant_algo",
        "tensorrt.quant.kv_cache_quant_algo",
    ],
}


def _get_dotted(obj: Any, path: str) -> Any:
    """Navigate a dotted path through Pydantic model/dict structures."""
    parts = path.split(".")
    cur = obj
    for p in parts:
        if cur is None:
            return None
        if hasattr(cur, p):
            cur = getattr(cur, p)
        elif isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def build_tuple_key(cfg: Any) -> str:
    """SHA-hash the build-axis values for this config."""
    keys = BUILD_TUPLE_KEYS.get(cfg.engine, [])
    values = {k: _get_dotted(cfg, k) for k in keys}
    return hashlib.sha256(json.dumps(values, default=str, sort_keys=True).encode()).hexdigest()[:16]


def _canary_runnable(cfg: Any) -> tuple[bool, str]:
    """Skip configs whose build path needs non-trivial canary support.

    Returns (runnable, reason_if_skipped). Known hazards for a
    shoestring canary:
      - torch_compile=True: first-forward triggers dynamo, which can OOM or
        raise on tiny prompts; needs warmup before a real probe.
      - bnb quantisation: bitsandbytes requires CUDA context + proper
        compute_dtype alignment; our canary's simplified input prep trips it.

    These are exactly the paths T3 *should* cover in production, but fixing
    the canary to support them is out of PoC scope — we'd need to reuse
    the engine's own _run_batch input prep. Filtering here lets the PoC
    answer "does T3 find dormancy T0-T2 missed?" on the rest.
    """
    pt = getattr(cfg, "transformers", None)
    if pt is not None:
        if getattr(pt, "torch_compile", False):
            return False, "torch_compile=True (canary doesn't prep dynamo inputs)"
        if getattr(pt, "load_in_4bit", False) or getattr(pt, "load_in_8bit", False):
            return False, "bnb quant (canary doesn't prep compute_dtype)"
        # flash_attention_3 isn't installed in pytorch image → would error
        if getattr(pt, "attn_implementation", None) == "flash_attention_3":
            return False, "flash_attention_3 (not installed in pytorch image)"
    return True, ""


def run_canary_in_docker(engine_name: str, cfg: Any, timeout: int) -> dict[str, Any]:
    """Dispatch a canary into the appropriate container."""
    image = IMAGES[engine_name]
    canary_script = CANARY_DIR / f"canary_{engine_name}.py"
    cfg_json = cfg.model_dump_json()

    # Mount the repo into the container at /workspace so both the canary
    # script and the llenergymeasure package are accessible.
    # The container images carry a pre-installed llenergymeasure from before
    # the Phase 49 schema restructure. Prepend the mounted source to
    # PYTHONPATH so the canary uses the CURRENT codebase.
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-i",
        "-e",
        "LLEM_TRT_BUILD_CACHE_ENABLED=1",
        "-e",
        "HF_HUB_CACHE=/root/.cache/huggingface/hub",
        "-e",
        "PYTHONPATH=/workspace/src",
        "-e",
        "LLEM_TRANSFORMERS_DEFAULT_DEVICE_MAP=auto",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-v",
        f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-w",
        "/workspace",
        "--entrypoint",
        PYTHON_BIN[engine_name],
        image,
        f"/workspace/scripts/canaries/canary_{engine_name}.py",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            input=cfg_json,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"docker canary timed out after {timeout}s",
            "wall_sec": time.perf_counter() - t0,
        }
    wall_sec = time.perf_counter() - t0

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"docker exited {proc.returncode}",
            "stderr_tail": proc.stderr[-2000:],
            "wall_sec": wall_sec,
        }

    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {
            "ok": False,
            "error": f"stdout parse: {e}",
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "wall_sec": wall_sec,
        }

    result["wall_sec"] = wall_sec
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", nargs="?", default=str(DEFAULT_YAML))
    parser.add_argument("--transformers-limit", type=int, default=30)
    parser.add_argument("--vllm-limit", type=int, default=20)
    parser.add_argument("--tensorrt-limit", type=int, default=10)
    parser.add_argument(
        "--transformers-timeout", type=int, default=180, help="Per-canary timeout (s); default 180"
    )
    parser.add_argument("--vllm-timeout", type=int, default=300)
    parser.add_argument(
        "--tensorrt-timeout",
        type=int,
        default=900,
        help="TRT first-time compile can be slow; default 15 min",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engines", nargs="+", default=["transformers", "vllm", "tensorrt"])
    args = parser.parse_args()

    print(f"Loading {args.yaml}...", flush=True)
    study = load_study_config(args.yaml)
    print(f"Expanded into {len(study.experiments)} configs.", flush=True)

    # Group configs by engine, then dedup on build-tuple. Filter out builds
    # that the shoestring canary can't exercise cleanly.
    by_engine: dict[str, dict[str, Any]] = {name: {} for name in IMAGES}
    skipped: dict[str, int] = {name: 0 for name in IMAGES}
    skip_reasons: dict[str, list[str]] = {name: [] for name in IMAGES}
    for cfg in study.experiments:
        if cfg.engine not in by_engine:
            continue
        runnable, reason = _canary_runnable(cfg)
        key = build_tuple_key(cfg)
        if not runnable:
            if reason not in skip_reasons[cfg.engine]:
                skip_reasons[cfg.engine].append(reason)
            skipped[cfg.engine] += 1
            continue
        if key not in by_engine[cfg.engine]:
            by_engine[cfg.engine][key] = cfg  # keep first representative

    for name in IMAGES:
        print(
            f"  {name}: {len(by_engine[name])} unique canary-runnable build-tuples "
            f"(skipped {skipped[name]} configs: {skip_reasons[name]})",
            flush=True,
        )

    rng = random.Random(args.seed)
    limits = {
        "transformers": args.transformers_limit,
        "vllm": args.vllm_limit,
        "tensorrt": args.tensorrt_limit,
    }
    timeouts = {
        "transformers": args.transformers_timeout,
        "vllm": args.vllm_timeout,
        "tensorrt": args.tensorrt_timeout,
    }

    all_results: dict[str, list[dict[str, Any]]] = {name: [] for name in IMAGES}

    for name in args.engines:
        if name not in IMAGES:
            print(f"  skip unknown engine: {name}")
            continue

        tuples = list(by_engine[name].values())
        if limits[name] < len(tuples):
            sampled = rng.sample(tuples, limits[name])
        else:
            sampled = tuples

        print(f"\n=== {name} canaries ({len(sampled)} unique builds) ===", flush=True)
        for i, cfg in enumerate(sampled, 1):
            key = build_tuple_key(cfg)
            print(f"  [{i}/{len(sampled)}] build={key} ...", flush=True, end=" ")
            t0 = time.perf_counter()
            result = run_canary_in_docker(name, cfg, timeout=timeouts[name])
            elapsed = time.perf_counter() - t0
            ok = result.get("ok", False)
            hints = result.get("dormancy_hints", [])
            err = result.get("error", "")[:100] if not ok else ""
            print(f"{elapsed:6.1f}s ok={ok} hints={hints} err={err}", flush=True)
            result["build_key"] = key
            all_results[name].append(result)

    # Summary
    print("\n" + "=" * 78)
    print("T3 CANARY SUMMARY")
    print("=" * 78)
    for name in IMAGES:
        rs = all_results[name]
        if not rs:
            continue
        ok_n = sum(1 for r in rs if r.get("ok"))
        fail_n = len(rs) - ok_n
        with_hints = sum(1 for r in rs if r.get("dormancy_hints"))
        total_hint_list: list[str] = []
        for r in rs:
            total_hint_list.extend(r.get("dormancy_hints", []))

        print(f"\n{name}: {len(rs)} canaries")
        print(f"  ok:               {ok_n}")
        print(f"  failed:           {fail_n}")
        print(f"  with hints:       {with_hints}")
        print(f"  unique hint keys: {sorted(set(total_hint_list))}")

        if fail_n:
            print(
                f"  first failure: {next((r for r in rs if not r.get('ok')), {}).get('error', 'n/a')[:120]}"
            )

    # Dump raw JSON for later analysis
    raw_path = REPO_ROOT / ".product/designs/runtime-config-probe-t3-raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results: {raw_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
