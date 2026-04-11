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

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    ENV_BASELINE_SPEC_PATH,
    TEMP_PREFIX_EXCHANGE,
)
from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)

__all__ = ["build_baseline_docker_cmd", "run_baseline_container"]


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


def run_baseline_container(
    image: str,
    mode: str,
    duration_sec: float,
    gpu_indices: list[int],
    timeout_sec: float | None = None,
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

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Baseline container timed out after %.0fs (image=%s, mode=%s). "
            "Exchange dir preserved at %s for post-mortem.",
            timeout_sec,
            image,
            mode,
            exchange_dir,
        )
        return None
    except FileNotFoundError:
        logger.warning("Baseline container dispatch failed: `docker` binary not found on PATH")
        _cleanup_exchange_dir(exchange_dir)
        return None

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "").strip().splitlines()[-10:]
        logger.warning(
            "Baseline container exited non-zero (code=%d, image=%s, mode=%s). "
            "Stderr tail: %s. Exchange dir preserved at %s.",
            completed.returncode,
            image,
            mode,
            stderr_tail,
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
