"""Container-side entry point for baseline-only idle power measurement.

Invoked by the host study runner via a short-lived container whose CUDA state
must match the experiment container's — a host-measured baseline understates
container idle power by ~8.7 W/GPU on A100 because the host has no CUDA context
and no torch memory pool seeded. See
``.product/research/baseline-measurement-location.md`` for the controlled
experiment and statistics.

Volume layout (managed by the host helper ``study/baseline_container.py``):
    /run/llem/baseline_spec.json    — written by host before container start
    /run/llem/baseline_result.json  — written here on success
    /run/llem/baseline_error.json   — written here on failure

Environment variable:
    LLEM_BASELINE_SPEC_PATH  — absolute path to the spec JSON inside the container.

Spec JSON shape::

    {
        "mode": "measure" | "spot_check",
        "gpu_indices": [0, 1],
        "duration_sec": 30.0
    }

Result JSON shape (both modes — callers read power_w for spot_check)::

    {
        "power_w": 42.590,
        "timestamp": 1712737123.45,
        "gpu_indices": [0, 1],
        "sample_count": 576,
        "duration_sec": 30.0,
        "mode": "measure"
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

from llenergymeasure.config.ssot import ENV_BASELINE_SPEC_PATH

logger = logging.getLogger(__name__)

__all__ = ["_emit_stage", "_prime_cuda", "main", "run_baseline_measurement"]


# Line prefix for stage markers emitted to stdout. The host dispatcher
# (``study/baseline_container.py``) matches this prefix when streaming the
# subprocess output line-by-line, so it can surface live sub-step progress
# in the CLI instead of the user staring at a single spinner for ~30s.
STAGE_LINE_PREFIX = "[llem.baseline]"


def _emit_stage(name: str, **kv: object) -> None:
    """Print a stage marker to stdout for the host to parse.

    Wire format: ``[llem.baseline] stage=<name> [k=v ...]``. Keys and values
    must be whitespace-free — this is a purpose-built internal protocol, not
    a general key-value serialiser. ``flush=True`` is non-negotiable: without
    it, container stdio buffering delays each marker until the pipe's 4-8KB
    block fills, which hides the whole point of streaming.
    """
    parts = [f"stage={name}"]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(f"{STAGE_LINE_PREFIX} " + " ".join(parts), flush=True)


def _prime_cuda(gpu_indices: list[int]) -> None:
    """Initialise the CUDA runtime and seed the torch caching allocator.

    Matches the experiment container's pre-inference GPU state so NVML reads
    capture the same idle draw. A bare ``torch.cuda.init()`` is not enough —
    the allocator only pins its baseline block on the first real tensor
    allocation, so we do a throw-away ``torch.zeros(1024)``.

    Tolerant of environments without torch or without CUDA: logs a warning
    and returns cleanly so host-side fallback paths and CI can still invoke
    this module.
    """
    try:
        import torch
    except ImportError:
        logger.warning(
            "baseline_measure: torch not available; skipping CUDA prime "
            "(measurement will reflect un-primed state)"
        )
        return

    try:
        if not torch.cuda.is_available():
            logger.warning("baseline_measure: torch reports CUDA unavailable; skipping prime")
            return
        torch.cuda.init()
        device_index = gpu_indices[0] if gpu_indices else 0
        # Allocation is what seeds the caching allocator — bare init does not.
        _ = torch.zeros(1024, device=f"cuda:{device_index}")
    except Exception as exc:
        logger.warning("baseline_measure: CUDA prime failed: %s", exc)


def run_baseline_measurement(spec_path: Path) -> Path:
    """Read the spec, prime CUDA, measure, and write the result JSON.

    Args:
        spec_path: Path to the spec JSON inside the container.

    Returns:
        Path to the written ``baseline_result.json`` file.

    Raises:
        Any exception propagates up so ``main`` can serialise it into
        ``baseline_error.json``.
    """
    # First line of output after Python has finished importing the package —
    # the host interprets this as "container launched, runtime ready" and
    # prints a dim sub-bullet with the elapsed time from subprocess start.
    _emit_stage("container_ready")

    from llenergymeasure.harness.baseline import (
        measure_baseline_power,
        measure_spot_check,
    )

    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    mode = raw.get("mode")
    gpu_indices = list(raw.get("gpu_indices") or [])
    duration_sec = float(raw.get("duration_sec", 30.0))

    if mode not in ("measure", "spot_check"):
        raise ValueError(
            f"baseline_measure: invalid mode {mode!r}, expected 'measure' or 'spot_check'"
        )

    _prime_cuda(gpu_indices)
    _emit_stage("cuda_primed")

    result_dir = spec_path.parent
    result_path = result_dir / "baseline_result.json"

    if mode == "measure":
        _emit_stage("sampling_started", mode="measure", duration=f"{duration_sec:.1f}")
        measured = measure_baseline_power(
            gpu_indices=gpu_indices or None,
            duration_sec=duration_sec,
            cache_ttl_sec=0.0,
        )
        if measured is None:
            raise RuntimeError(
                "baseline_measure: measure_baseline_power returned None "
                "(NVML unavailable or no samples collected)"
            )
        _emit_stage(
            "sampling_done",
            power_w=f"{measured.power_w:.2f}",
            samples=measured.sample_count,
            duration=f"{measured.duration_sec:.2f}",
        )
        payload = {
            "power_w": measured.power_w,
            "timestamp": measured.timestamp,
            "gpu_indices": measured.gpu_indices,
            "sample_count": measured.sample_count,
            "duration_sec": measured.duration_sec,
            "mode": "measure",
        }
    else:
        _emit_stage("sampling_started", mode="spot_check", duration=f"{duration_sec:.1f}")
        power_w = measure_spot_check(
            gpu_indices=gpu_indices or None,
            duration_sec=duration_sec,
        )
        if power_w is None:
            raise RuntimeError(
                "baseline_measure: measure_spot_check returned None "
                "(NVML unavailable or no samples collected)"
            )
        import time

        _emit_stage(
            "sampling_done",
            power_w=f"{power_w:.2f}",
            samples=0,
            duration=f"{duration_sec:.2f}",
        )
        payload = {
            "power_w": power_w,
            "timestamp": time.time(),
            "gpu_indices": gpu_indices,
            "sample_count": 0,
            "duration_sec": duration_sec,
            "mode": "spot_check",
        }

    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _emit_stage("result_written")
    return result_path


def main() -> None:
    """Entry point invoked when the baseline container starts."""
    spec_path_env = os.environ.get(ENV_BASELINE_SPEC_PATH)
    if not spec_path_env:
        raise RuntimeError(
            f"{ENV_BASELINE_SPEC_PATH} environment variable is not set. "
            "The baseline container dispatcher must set this before starting the container."
        )

    spec_path = Path(spec_path_env)
    result_dir = spec_path.parent

    try:
        result_path = run_baseline_measurement(spec_path)
        print(f"[llem] Baseline measurement complete. Result written to {result_path}")
    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_path = result_dir / "baseline_error.json"
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
            print(f"[llem] Baseline measurement failed. Error written to {error_path}")
        except Exception as write_exc:  # pragma: no cover
            print(f"[llem] CRITICAL: could not write baseline error payload: {write_exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
