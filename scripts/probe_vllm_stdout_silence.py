#!/usr/bin/env python3
"""PoC-M: vLLM stdout-silence profile (issue #366).

Subprocess-launches a real ``llem run`` against a minimal vLLM study,
captures every line of combined stdout+stderr with monotonic-time
timestamps, then post-processes the inter-line gap distribution to
inform the default ``stdout_silence_timeout_seconds`` value for the
unified watchdog refactor.

Why this shape (not in-container instrumentation): the watchdog the
default will protect lives in ``docker_runner.py:_run_container_streaming``
on the *host* side, reading the container's stdout pipe. Measuring on
that exact pipe (via host-side subprocess.Popen mirroring the runner)
reflects what the watchdog actually observes — including container
startup latency, NVML init, vLLM's torch.compile warmup, and steady-state
inference output rate.

Usage:

    /usr/bin/python3.10 scripts/probe_vllm_stdout_silence.py [--config PATH] [--out PATH]

Defaults to a derived minimal vLLM-only config from ``configs/test.yaml``.
Writes results JSON to ``scripts/probe_vllm_stdout_silence_results.json``
and a percentile table to stdout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Phases delineated by container progress events emitted by the existing
# `_run_container_streaming` JSON event protocol. We classify each line by
# the most recent step_start observed, so per-phase percentiles are clean.
_KNOWN_STEPS = (
    "container_start",
    "container_preflight",
    "model_load",
    "warmup",
    "measurement",
    "inference",
    "saving",
)


def _build_minimal_config(out_path: Path) -> Path:
    """Write a minimal vLLM-only study config for the PoC.

    Single experiment, single cycle, 4 prompts, max_output_tokens=8.
    Uses Qwen2.5-0.5B (already configured as the test default).
    """
    out_path.write_text(
        """\
study_name: poc-m-vllm-stdout-silence

runners:
  vllm: docker

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
  results_dir: ./results/poc-m-vllm-stdout-silence
  save_timeseries: false

experiments:
  - engine: vllm
    vllm:
      dtype: bfloat16
      engine:
        tensor_parallel_size: 1
        max_model_len: 256
        gpu_memory_utilization: 0.5
""",
        encoding="utf-8",
    )
    return out_path


def _run_and_capture(config_path: Path) -> list[tuple[float, str]]:
    """Run ``llem run <config>`` and capture (monotonic_offset, line) for every line."""
    cmd = [
        str(Path.home() / ".local" / "bin" / "llem"),
        "run",
        str(config_path),
    ]
    print(f"# launching: {' '.join(cmd)}", flush=True)
    start = time.monotonic()
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT}/src"
    # Skew between host package version and pre-built image schema is fine for
    # this PoC — we're measuring stdout pacing, not config-schema correctness.
    env["LLEM_SKIP_IMAGE_CHECK"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge — we want the *same* stream the watchdog will see
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
        # Also echo so the operator can watch progress.
        print(f"[{offset:7.2f}s] {line}", end="", flush=True)
    proc.wait()
    print(f"\n# subprocess exited with rc={proc.returncode} after {time.monotonic() - start:.1f}s")
    return captured


def _classify_phase(line: str, current: str) -> str:
    """Return new phase if line emits a step_start; else current phase."""
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
    if not step:
        return current
    return step if step in _KNOWN_STEPS else step


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
    """Return overall + per-phase gap statistics."""
    if len(captured) < 2:
        return {"error": "fewer than 2 lines captured", "raw_count": len(captured)}
    overall_gaps: list[float] = []
    by_phase: dict[str, list[float]] = {}
    phase = "container_start"
    last_t = captured[0][0]
    for t, line in captured[1:]:
        gap = t - last_t
        overall_gaps.append(gap)
        by_phase.setdefault(phase, []).append(gap)
        phase = _classify_phase(line, phase)
        last_t = t
    return {
        "wall_clock_s": captured[-1][0] - captured[0][0],
        "line_count": len(captured),
        "overall": _gap_stats(overall_gaps),
        "by_phase": {k: _gap_stats(v) for k, v in sorted(by_phase.items())},
    }


def _print_report(results: dict, out_path: Path | None) -> None:
    print()
    print("=" * 72)
    print("PoC-M  vLLM stdout-silence gap distribution")
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
        default=REPO_ROOT / "scripts" / "probe_vllm_stdout_silence_results.json",
    )
    args = parser.parse_args()

    if args.config is None:
        cfg_path = REPO_ROOT / "scripts" / "_poc_m_minimal.yaml"
        _build_minimal_config(cfg_path)
    else:
        cfg_path = args.config

    captured = _run_and_capture(cfg_path)
    results = _analyse(captured)
    _print_report(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
