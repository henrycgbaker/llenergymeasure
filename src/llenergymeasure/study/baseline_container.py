"""Host-side helper that dispatches a short-lived baseline-only Docker container.

Why this exists: a host-measured baseline underestimates the container's idle
GPU power by ~8.7 W per A100 (~19% on typical 4-GPU, 2-minute runs) because
the host has no CUDA context and no torch memory pool seeded. Measuring the
baseline inside a container whose CUDA state matches the experiment container
eliminates the bias. See ``.product/research/baseline-measurement-location.md``
for the controlled experiment.

This helper deliberately does not use ``infra.docker_runner.DockerRunner``:
baseline dispatch needs none of DockerRunner's experiment-specific machinery
(config-hash indirection, streamed stdout progress, timeseries rescue, result
deserialisation) — just a spec file in, a result file out, single subprocess
invocation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import subprocess
import tempfile
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    ENV_BASELINE_SPEC_PATH,
    TEMP_PREFIX_EXCHANGE,
)
from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)

__all__ = [
    "StageCallback",
    "build_baseline_docker_cmd",
    "parse_stage_line",
    "run_baseline_container",
]

# Host callback signature for stage markers parsed from the container's stdout.
# Positional args: (stage_name, elapsed_since_subprocess_start, kv_tags).
StageCallback = Callable[[str, float, "dict[str, str]"], None]

_STAGE_LINE_PREFIX = "[llem.baseline] stage="


BASELINE_SPEC_FILENAME = "baseline_spec.json"


def build_baseline_docker_cmd(
    image: str,
    exchange_dir: str,
    gpu_indices: list[int],
) -> list[str]:
    """Build the ``docker run`` command list for a baseline-only container.

    Kept separate from ``run_baseline_container`` so tests can assert on the
    command shape without mocking subprocess internals.
    """
    cuda_visible = ",".join(str(i) for i in gpu_indices) if gpu_indices else ""
    spec_container_path = f"{CONTAINER_EXCHANGE_DIR}/{BASELINE_SPEC_FILENAME}"
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{exchange_dir}:{CONTAINER_EXCHANGE_DIR}",
        "-e",
        f"{ENV_BASELINE_SPEC_PATH}={spec_container_path}",
        "-e",
        f"CUDA_VISIBLE_DEVICES={cuda_visible}",
        image,
        "python3",
        "-m",
        "llenergymeasure.entrypoints.baseline_measure",
    ]


def parse_stage_line(line: str) -> tuple[str, dict[str, str]] | None:
    """Parse a ``[llem.baseline] stage=NAME k=v ...`` line.

    Returns ``(stage_name, kv)`` or ``None`` if the line is not a stage
    marker. Malformed key-value tokens (no ``=``) are skipped silently;
    we treat this wire format as a best-effort progress channel, not a
    critical data path.
    """
    if not line.startswith(_STAGE_LINE_PREFIX):
        return None
    payload = line[len(_STAGE_LINE_PREFIX) :].strip()
    if not payload:
        return None
    parts = payload.split()
    stage_name = parts[0]
    kv: dict[str, str] = {}
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            kv[k] = v
    return stage_name, kv


def run_baseline_container(
    image: str,
    mode: str,
    duration_sec: float,
    gpu_indices: list[int],
    timeout_sec: float | None = None,
    on_stage: StageCallback | None = None,
) -> BaselineCache | None:
    """Spawn a short-lived baseline container and return the measurement.

    Writes a minimal spec JSON to a temp exchange dir, runs the container, and
    reads back ``baseline_result.json``. Returns ``None`` if the container
    failed to produce a result (matches ``harness.baseline.measure_baseline_power``
    None semantics so callers can degrade gracefully).

    Args:
        image: Docker image tag of the backend the baseline is being measured
            for. Must match the experiment container's image so the CUDA init
            footprint lines up.
        mode: ``"measure"`` for a full baseline measurement, ``"spot_check"``
            for a short drift check.
        duration_sec: Sampling duration inside the container.
        gpu_indices: GPUs to measure. Translated to ``CUDA_VISIBLE_DEVICES`` so
            the baseline container sees exactly the same GPUs the experiment
            container will.
        timeout_sec: Subprocess timeout. Defaults to
            ``max(duration_sec * 2 + 60, 120)`` — enough for cold-start,
            sampling, and teardown with headroom.
        on_stage: Optional callback invoked for each stage marker streamed on
            the container's stdout (``container_ready``, ``cuda_primed``,
            ``sampling_started``, ``sampling_done``, ``result_written``). The
            callback is passed ``(stage_name, elapsed_since_popen, kv_tags)``
            and is what the CLI hooks to emit live sub-bullets while the
            container is still running.

    Returns:
        A ``BaselineCache`` with ``method=None`` on success (the caller sets
        the method based on the strategy), or ``None`` on any failure.
    """
    if timeout_sec is None:
        timeout_sec = max(duration_sec * 2.0 + 60.0, 120.0)

    exchange_dir = Path(tempfile.mkdtemp(prefix=f"{TEMP_PREFIX_EXCHANGE}baseline-"))
    spec_path = exchange_dir / BASELINE_SPEC_FILENAME
    result_path = exchange_dir / "baseline_result.json"
    error_path = exchange_dir / "baseline_error.json"

    spec_payload = {
        "mode": mode,
        "gpu_indices": list(gpu_indices),
        "duration_sec": float(duration_sec),
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2), encoding="utf-8")

    cmd = build_baseline_docker_cmd(
        image=image,
        exchange_dir=str(exchange_dir),
        gpu_indices=list(gpu_indices),
    )

    logger.debug(
        "Dispatching baseline container: image=%s mode=%s duration=%.1fs gpus=%s",
        image,
        mode,
        duration_sec,
        gpu_indices,
    )

    # stderr merged into stdout so one iterator covers both streams and stderr's
    # pipe buffer can't fill up and deadlock the child.
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        logger.warning("Baseline container dispatch failed: `docker` binary not found on PATH")
        _cleanup_exchange_dir(exchange_dir)
        return None

    start = time.monotonic()
    # Bounded tail — only the last 10 lines are logged on failure; torch/CUDA
    # init can emit thousands of lines we never read.
    output_tail: deque[str] = deque(maxlen=64)

    try:
        if process.stdout is None:
            raise RuntimeError("Popen returned no stdout pipe")
        for line in process.stdout:
            output_tail.append(line)
            stripped = line.rstrip("\n")
            parsed = parse_stage_line(stripped)
            if parsed is not None and on_stage is not None:
                stage_name, kv = parsed
                elapsed = time.monotonic() - start
                try:
                    on_stage(stage_name, elapsed, kv)
                except Exception:
                    # A broken CLI callback must never take down the measurement.
                    logger.debug(
                        "baseline on_stage callback raised; continuing",
                        exc_info=True,
                    )
            # Kill a wedged container mid-stream rather than only when wait() fires.
            if (time.monotonic() - start) > timeout_sec:
                process.kill()
                with contextlib.suppress(Exception):
                    process.wait(timeout=5)
                logger.warning(
                    "Baseline container timed out after %.0fs (image=%s, mode=%s). "
                    "Exchange dir preserved at %s for post-mortem.",
                    timeout_sec,
                    image,
                    mode,
                    exchange_dir,
                )
                return None
        # Stdout closed → child is essentially done; a small grace is enough.
        returncode = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)
        logger.warning(
            "Baseline container timed out after %.0fs (image=%s, mode=%s). "
            "Exchange dir preserved at %s for post-mortem.",
            timeout_sec,
            image,
            mode,
            exchange_dir,
        )
        return None

    if returncode != 0:
        tail = [line.rstrip("\n") for line in list(output_tail)[-10:]]
        logger.warning(
            "Baseline container exited non-zero (code=%d, image=%s, mode=%s). "
            "Output tail: %s. Exchange dir preserved at %s.",
            returncode,
            image,
            mode,
            tail,
            exchange_dir,
        )
        if error_path.exists():
            try:
                err = json.loads(error_path.read_text(encoding="utf-8"))
                logger.warning(
                    "Baseline container error payload: type=%s message=%s",
                    err.get("type"),
                    err.get("message"),
                )
            except Exception:
                pass
        return None

    if not result_path.exists():
        logger.warning(
            "Baseline container exited 0 but no result file at %s. "
            "Exchange dir preserved for post-mortem.",
            result_path,
        )
        return None

    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Baseline container wrote a malformed result: %s. Exchange dir preserved at %s.",
            exc,
            exchange_dir,
        )
        return None

    try:
        cache = BaselineCache(
            power_w=float(raw["power_w"]),
            timestamp=float(raw.get("timestamp", time.time())),
            gpu_indices=list(raw.get("gpu_indices") or gpu_indices),
            sample_count=int(raw.get("sample_count", 0)),
            duration_sec=float(raw.get("duration_sec", duration_sec)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Baseline container result missing/invalid fields: %s (payload=%s). "
            "Exchange dir preserved at %s.",
            exc,
            raw,
            exchange_dir,
        )
        return None

    _cleanup_exchange_dir(exchange_dir)
    return cache


def _cleanup_exchange_dir(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except Exception as exc:
        logger.debug("Could not remove baseline exchange dir %s: %s", path, exc)
