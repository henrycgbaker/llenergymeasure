#!/usr/bin/env python3
"""PoC-N: TensorRT-LLM stdout-silence profile (issue #366).

Sibling of ``probe_vllm_stdout_silence.py``. Runs a TRT-LLM study with
the engine cache cleared upfront so the build phase actually compiles —
this is the worst-case scenario for stdout-silence (engine compilation
can go several minutes between progress lines).

Usage:

    /usr/bin/python3.10 scripts/probe_trt_stdout_silence.py [--config PATH] [--out PATH] [--keep-cache]

By default the script clears ``~/.cache/llm-energy-measure/tensorrt-engines/``
before running so the compile path is exercised. Pass ``--keep-cache``
to skip the clear (useful for inference-only profiling on the second
run).

Writes results to ``scripts/probe_trt_stdout_silence_results.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_KNOWN_STEPS = (
    "container_start",
    "container_preflight",
    "model_load",
    "engine_build",
    "warmup",
    "measurement",
    "inference",
    "saving",
)

_TRT_CACHE_DIR = Path.home() / ".cache" / "llm-energy-measure" / "tensorrt-engines"


def _build_minimal_config(out_path: Path) -> Path:
    """Write a minimal TRT-LLM-only study config for the PoC.

    Single experiment, single cycle, 4 prompts, max_output_tokens=8.
    """
    out_path.write_text(
        """\
study_name: poc-n-trt-stdout-silence

runners:
  tensorrt: docker

study_execution:
  n_cycles: 1
  experiment_order: sequential
  experiment_gap_seconds: 0.0
  experiment_timeout_seconds: 1800.0

task:
  model: Qwen/Qwen2.5-0.5B
  random_seed: 42
  dataset:
    source: aienergyscore
    n_prompts: 4
    order: interleaved
  max_input_tokens: 64
  max_output_tokens: 8

measurement:
  energy_sampler: nvml
  baseline:
    enabled: false
    duration_seconds: 5.0
  warmup:
    enabled: true
    n_warmup: 1
    thermal_floor_seconds: 30.0

output:
  results_dir: ./results/poc-n-trt-stdout-silence
  save_timeseries: false

experiments:
  - engine: tensorrt
    tensorrt:
      dtype: bfloat16
      tp_size: 1
      max_seq_len: 256
      max_batch_size: 1
""",
        encoding="utf-8",
    )
    return out_path


def _clear_trt_cache() -> None:
    """Remove the local TRT-LLM engine cache so the build phase compiles fresh."""
    if _TRT_CACHE_DIR.exists():
        print(f"# clearing TRT engine cache at {_TRT_CACHE_DIR}", flush=True)
        shutil.rmtree(_TRT_CACHE_DIR)
        _TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"# TRT cache absent at {_TRT_CACHE_DIR} — nothing to clear", flush=True)


def _run_and_capture(config_path: Path) -> list[tuple[float, str]]:
    cmd = [str(Path.home() / ".local" / "bin" / "llem"), "run", str(config_path)]
    print(f"# launching: {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/src"
    env["LLEM_SKIP_IMAGE_CHECK"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        env=env,
    )

    captured: list[tuple[float, str]] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        offset = time.monotonic() - start
        captured.append((offset, line.rstrip("\n")))
        print(f"[{offset:7.2f}s] {line}", end="", flush=True)
    proc.wait()
    print(f"\n# subprocess exited rc={proc.returncode} after {time.monotonic() - start:.1f}s")
    return captured


def _classify_phase(line: str, current: str) -> str:
    stripped = line.strip()
    if not stripped.startswith('{"event":'):
        return current
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        return current
    if event.get("event") != "step_start":
        return current
    step = str(event.get("step", "")).strip()
    return step or current


def _gap_stats(gaps: list[float]) -> dict[str, float]:
    if not gaps:
        return {"count": 0}
    return {
        "count": len(gaps),
        "mean_s": statistics.fmean(gaps),
        "median_s": statistics.median(gaps),
        "p95_s": statistics.quantiles(gaps, n=20)[18] if len(gaps) >= 20 else max(gaps),
        "p99_s": statistics.quantiles(gaps, n=100)[98] if len(gaps) >= 100 else max(gaps),
        "max_s": max(gaps),
    }


def _analyse(captured: list[tuple[float, str]]) -> dict:
    if len(captured) < 2:
        return {"error": "fewer than 2 lines captured", "raw_count": len(captured)}
    overall: list[float] = []
    by_phase: dict[str, list[float]] = {}
    phase = "container_start"
    last_t = captured[0][0]
    for t, line in captured[1:]:
        gap = t - last_t
        overall.append(gap)
        by_phase.setdefault(phase, []).append(gap)
        phase = _classify_phase(line, phase)
        last_t = t
    return {
        "wall_clock_s": captured[-1][0] - captured[0][0],
        "line_count": len(captured),
        "overall": _gap_stats(overall),
        "by_phase": {k: _gap_stats(v) for k, v in sorted(by_phase.items())},
    }


def _print_report(results: dict, out_path: Path | None) -> None:
    print()
    print("=" * 72)
    print("PoC-N  TRT-LLM stdout-silence gap distribution")
    print("=" * 72)
    overall = results.get("overall", {})
    print(
        f"wall_clock={results.get('wall_clock_s', 0):.1f}s  "
        f"lines={results.get('line_count', 0)}  "
        f"max_gap={overall.get('max_s', 0):.2f}s  "
        f"p99={overall.get('p99_s', 0):.2f}s  "
        f"p95={overall.get('p95_s', 0):.2f}s  "
        f"median={overall.get('median_s', 0):.3f}s"
    )
    print()
    print(f"{'phase':<22} {'count':>7} {'median':>9} {'p95':>9} {'max':>9}")
    print("-" * 72)
    for phase, stats in results.get("by_phase", {}).items():
        if not stats.get("count"):
            continue
        print(
            f"{phase:<22} {stats['count']:>7} "
            f"{stats['median_s']:>8.3f}s "
            f"{stats['p95_s']:>8.2f}s "
            f"{stats['max_s']:>8.2f}s"
        )
    print()
    if out_path is not None:
        out_path.write_text(json.dumps(results, indent=2))
        print(f"# results written to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "probe_trt_stdout_silence_results.json",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Skip clearing the TRT engine cache (useful for inference-only re-runs).",
    )
    args = parser.parse_args()

    if not args.keep_cache:
        _clear_trt_cache()

    if args.config is None:
        cfg_path = REPO_ROOT / "scripts" / "_poc_n_minimal.yaml"
        _build_minimal_config(cfg_path)
    else:
        cfg_path = args.config

    captured = _run_and_capture(cfg_path)
    results = _analyse(captured)
    _print_report(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
