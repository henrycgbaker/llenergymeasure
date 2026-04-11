"""StudyRunner — subprocess dispatch core for experiment isolation.

Each experiment in a study runs in a freshly spawned subprocess with a clean CUDA
context. Results travel parent←child via multiprocessing.Pipe. The parent survives
experiment failures, timeouts, and SIGINT without data corruption.

Key design decisions (locked in .product/decisions/experiment-isolation.md):
- spawn context: CUDA-safe; fork causes silent CUDA corruption (CP-1)
- daemon=False: clean CUDA teardown if parent exits unexpectedly (CP-4)
- Pipe-only IPC: ExperimentResult fits in Pipe buffer for typical experiment sizes
- SIGKILL on timeout: SIGTERM may be ignored by hung CUDA operations
- Process group kill: worker calls os.setpgrp() to become group leader so all
  descendant processes (vLLM workers, MPI ranks, etc.) are killed together
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    DOCKER_PULL_TIMEOUT,
    RUNNER_DOCKER,
    TEMP_PREFIX_TIMESERIES,
    TIMEOUT_DOCKER_INSPECT,
    TIMEOUT_ENV_SNAPSHOT,
    TIMEOUT_INTERRUPT_POLL,
    TIMEOUT_SIGTERM_GRACE,
    TIMEOUT_THREAD_JOIN,
)
from llenergymeasure.domain.progress import STEP_BASELINE, STEPS_LOCAL, docker_steps
from llenergymeasure.study.gaps import run_gap

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter
    from llenergymeasure.utils.exceptions import DockerError

__all__ = [
    "StudyRunner",
    "_kill_process_group",
    "_run_experiment_worker",
    "_save_and_record",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level helpers
# =============================================================================


def _sanitize_image_for_filename(image: str) -> str:
    """Return a filename-safe slug for a Docker image tag.

    Replaces path separators, tag markers, digest markers, and whitespace with
    underscores. Clipped to 128 chars and stripped of trailing underscores so
    the resulting basename stays well under typical filesystem limits.
    """
    sanitized = image
    for ch in ("/", ":", "@", " ", "\t", "\n"):
        sanitized = sanitized.replace(ch, "_")
    sanitized = sanitized[:128].rstrip("_")
    return sanitized or "unknown"


def _save_and_record(
    result: Any,
    study_dir: Path,
    manifest: ManifestWriter,
    config_hash: str,
    cycle: int,
    result_files: list[str],
    experiment_index: int | None = None,
    ts_source_dir: Path | None = None,
    environment_snapshot: Any | None = None,
    resolution_log: dict[str, Any] | None = None,
) -> None:
    """Save result to disk and update manifest. Appends result path to result_files.

    Resolves the timeseries parquet sidecar from the result object and passes it
    to save_result() so it is copied into the experiment subdirectory. The stale
    flat file written by MeasurementHarness is removed after the copy.

    Args:
        ts_source_dir: Directory where the harness wrote timeseries.parquet.
        environment_snapshot: EnvironmentSnapshot for per-experiment environment.json sidecar.
        resolution_log: Pre-built resolution log for this experiment (written as _resolution.json).

    On save failure, marks the experiment as completed with empty path.
    """
    try:
        from llenergymeasure.results.persistence import save_environment, save_result

        # Resolve timeseries sidecar from result fields.
        # MeasurementHarness writes timeseries.parquet to the output_dir and
        # sets result.timeseries = "timeseries.parquet". Both must be present for
        # the copy to proceed.
        ts_source: Path | None = None
        ts_filename = getattr(result, "timeseries", None)
        if ts_filename and ts_source_dir is not None:
            candidate = ts_source_dir / ts_filename
            if candidate.exists():
                ts_source = candidate

        result_path = save_result(
            result,
            study_dir,
            timeseries_source=ts_source,
            experiment_index=experiment_index,
            cycle=cycle,
            resolution_log=resolution_log,
        )

        # Write per-experiment environment.json sidecar
        if environment_snapshot is not None:
            save_environment(
                environment_snapshot,
                result.experiment_id,
                config_hash,
                result_path.parent,
            )

        # Clean up the stale flat parquet file after it has been copied into the
        # experiment subdirectory (mirrors cli/run.py line 288).
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)

        result_files.append(str(result_path))
        rel_path = str(result_path.relative_to(study_dir))

        # Extract summary metrics for manifest (used by --resume display).
        elapsed_sec = getattr(result, "total_inference_time_sec", None)
        manifest.mark_completed(
            config_hash,
            cycle,
            rel_path,
            elapsed_seconds=elapsed_sec,
            inference_seconds=elapsed_sec,
            energy_joules=getattr(result, "total_energy_j", None),
            adj_energy_joules=getattr(result, "energy_adjusted_j", None),
            throughput_tok_s=getattr(result, "avg_tokens_per_second", None),
            mj_per_tok=getattr(result, "mj_per_tok_adjusted", None)
            or getattr(result, "mj_per_tok_total", None),
        )
    except Exception as exc:
        manifest.mark_failed(config_hash, cycle, type(exc).__name__, str(exc))


def _kill_process_group(pid: int, sig: int) -> None:
    """Send signal to the entire process group rooted at pid.

    Uses os.killpg so that all descendant processes (vLLM workers, MPI ranks, etc.)
    receive the signal, not just the parent. Errors are suppressed because the
    process group may already be dead by the time this is called.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


# =============================================================================
# Queue-based progress callback (runs inside child process)
# =============================================================================


class _QueueProgressCallback:
    """ProgressCallback that serialises step events onto a multiprocessing.Queue.

    Created inside the worker subprocess. Events are dicts consumed by the
    parent's _consume_progress_events thread and forwarded to StudyStepDisplay.
    """

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def _put(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._queue.put(event)

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        self._put(
            {"event": "step_start", "step": step, "description": description, "detail": detail}
        )

    def on_step_update(self, step: str, detail: str) -> None:
        self._put({"event": "step_update", "step": step, "detail": detail})

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        self._put({"event": "step_done", "step": step, "elapsed_sec": elapsed_sec})

    def on_step_skip(self, step: str, reason: str = "") -> None:
        self._put({"event": "step_skip", "step": step, "reason": reason})

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        self._put({"event": "substep", "step": step, "text": text, "elapsed_sec": elapsed_sec})


# =============================================================================
# Worker function (runs inside child process)
# =============================================================================


def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Any,  # multiprocessing.Connection (child end)
    progress_queue: Any,  # multiprocessing.Queue
    snapshot: EnvironmentSnapshot | None = None,
    output_dir: str | None = None,
    save_timeseries: bool = True,
    baseline: Any = None,  # BaselineCache | None (avoids import at module level)
) -> None:
    """Entry point for the child process. Runs one experiment and returns result via Pipe.

    Signal handling:
        Installs SIGINT → SIG_IGN so the child ignores Ctrl+C.
        # parent owns SIGINT; child ignores it
        The parent handles SIGINT and decides whether to kill the child.

    IPC protocol:
        On success: sends ExperimentResult (or result dict) via conn.
        On failure: sends {"type": ..., "message": ..., "traceback": ...} via conn.
        Progress events are put to progress_queue for the consumer thread.

    Args:
        output_dir: Directory for timeseries parquet output. Passed through to harness.
        save_timeseries: Whether to persist GPU timeseries. Passed through to harness.
        baseline: Pre-measured baseline power from parent process (study-level cache).
    """
    # Become process group leader so all descendants (vLLM workers, MPI ranks, etc.)
    # share this PGID. The parent can then kill the whole group via os.killpg().
    os.setpgrp()

    # parent owns SIGINT; child ignores it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        progress_queue.put({"event": "started", "config_hash": config_hash})

        # Create progress callback that serialises step events to queue
        progress_cb = _QueueProgressCallback(progress_queue)

        # Run the actual experiment in-process (within the spawned subprocess)
        from llenergymeasure.backends import get_backend
        from llenergymeasure.harness import MeasurementHarness
        from llenergymeasure.harness.preflight import run_preflight

        # Pre-flight inside subprocess: CUDA availability must be checked in the
        # process that will use the GPU.
        progress_cb.on_step_start("container_preflight", "Checking", "CUDA, model access")
        t0_pf = time.perf_counter()
        run_preflight(config)
        progress_cb.on_step_done("container_preflight", time.perf_counter() - t0_pf)

        backend = get_backend(config.backend)
        harness = MeasurementHarness()
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        gpu_indices = _resolve_gpu_indices(config)
        result = harness.run(
            backend,
            config,
            snapshot=snapshot,
            gpu_indices=gpu_indices,
            progress=progress_cb,
            output_dir=output_dir,
            save_timeseries=save_timeseries,
            baseline=baseline,
        )

        # Send result back to parent via Pipe
        conn.send(result)
        progress_queue.put({"event": "completed", "config_hash": config_hash})

    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with contextlib.suppress(Exception):
            # Pipe may be broken (e.g. parent killed). Best-effort only.
            conn.send(error_payload)

        with contextlib.suppress(Exception):
            progress_queue.put({"event": "failed", "error": str(exc)})

        raise

    finally:
        conn.close()


# =============================================================================
# Progress consumer (daemon thread in parent)
# =============================================================================


def _consume_progress_events(
    q: Any,
    study_progress: StudyProgressCallback | None = None,
) -> None:
    """Consume progress events from the queue and forward to study display.

    Runs as a daemon thread in the parent process. Receives step events from
    the child subprocess via multiprocessing.Queue and forwards them to the
    StudyProgressCallback (typically StudyStepDisplay).

    Coarse events (started/completed/failed) are ignored here - study-level
    begin/end experiment tracking is handled directly by _run_one().
    """
    while True:
        event = q.get()
        if event is None:
            break

        if not isinstance(event, dict) or study_progress is None:
            continue

        event_type = event.get("event")

        # Forward step-level events to study display
        if event_type == "step_start":
            study_progress.on_step_start(
                event["step"], event.get("description", ""), event.get("detail", "")
            )
        elif event_type == "step_update":
            study_progress.on_step_update(event["step"], event.get("detail", ""))
        elif event_type == "step_done":
            study_progress.on_step_done(event["step"], event.get("elapsed_sec", 0))
        elif event_type == "step_skip":
            study_progress.on_step_skip(event["step"], event.get("reason", ""))
        elif event_type == "substep":
            study_progress.on_substep(
                event["step"], event.get("text", ""), event.get("elapsed_sec", 0)
            )


# =============================================================================
# Result collection (parent, after p.join)
# =============================================================================

# Sentinel object used to distinguish "no payload provided" from a None payload.
_UNSET = object()


def _collect_result(
    p: Any,  # multiprocessing.Process
    parent_conn: Any,  # multiprocessing.Connection (parent end)
    config: ExperimentConfig,
    timeout: float,
    pipe_payload: Any = _UNSET,
) -> Any:
    """Inspect process outcome and return either a result or a failure dict.

    Called after the pipe has been drained and p.join() has returned.

    Args:
        p: The child process.
        parent_conn: Parent end of the Pipe (read-only).
        config: Experiment configuration.
        timeout: Timeout used for the experiment (for error messages).
        pipe_payload: Pre-drained pipe value from the recv-before-join pattern
            (H5 deadlock fix). When provided, skips calling recv() again.
            Pass _UNSET (default) to fall back to reading from the pipe directly.

    Returns:
        ExperimentResult on success, dict with keys (type, message) on failure.
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(config)

    if p.is_alive():
        # Timed out — kill with SIGKILL
        # SIGKILL: SIGTERM may be ignored by hung CUDA operations
        _kill_process_group(p.pid, signal.SIGKILL)
        p.join()
        return {
            "type": "TimeoutError",
            "message": f"Experiment exceeded timeout of {timeout}s and was killed.",
            "config_hash": config_hash,
        }

    if p.exitcode != 0:
        # Non-zero exit — try to read error payload from pipe
        # Use pre-drained payload if available; otherwise poll/recv
        if pipe_payload is not _UNSET:
            payload = pipe_payload
            if isinstance(payload, dict) and "type" in payload and "message" in payload:
                payload["config_hash"] = config_hash
                return payload
        elif parent_conn.poll():
            try:
                payload = parent_conn.recv()
                if isinstance(payload, dict) and "type" in payload and "message" in payload:
                    payload["config_hash"] = config_hash
                    return payload
            except Exception:
                pass

        return {
            "type": "ProcessCrash",
            "message": f"Subprocess exited with code {p.exitcode} and no error data in Pipe.",
            "config_hash": config_hash,
        }

    # Success path — use pre-drained payload if available
    if pipe_payload is not _UNSET:
        try:
            payload = pipe_payload
            # If payload is an error dict (exception in worker), treat as failure
            if isinstance(payload, dict) and "type" in payload and "message" in payload:
                payload["config_hash"] = config_hash
                return payload
            return payload
        except Exception as exc:
            return {
                "type": "PipeError",
                "message": f"Failed to process pre-drained pipe payload: {exc}",
                "config_hash": config_hash,
            }

    # Fallback: read from pipe directly (no pre-drained payload)
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
            # If payload is an error dict (exception in worker), treat as failure
            if isinstance(payload, dict) and "type" in payload and "message" in payload:
                payload["config_hash"] = config_hash
                return payload
            return payload
        except Exception as exc:
            return {
                "type": "PipeError",
                "message": f"Failed to receive result from subprocess: {exc}",
                "config_hash": config_hash,
            }

    return {
        "type": "ProcessCrash",
        "message": "Subprocess exited 0 but sent no data through Pipe.",
        "config_hash": config_hash,
    }


# =============================================================================
# StudyRunner
# =============================================================================


class StudyRunner:
    """Dispatcher: runs each experiment in a freshly spawned subprocess.

    Uses multiprocessing.get_context('spawn') — never fork.
    Results travel via Pipe. Failures are structured and non-fatal.
    Handles SIGINT (Ctrl+C) with two-stage escalation: SIGTERM → 2s grace → SIGKILL.
    """

    def __init__(
        self,
        study: StudyConfig,
        manifest_writer: ManifestWriter,
        study_dir: Path,
        runner_specs: dict[str, RunnerSpec] | None = None,
        progress: StudyProgressCallback | None = None,
        no_lock: bool = False,
        skip_set: set[tuple[str, int]] | None = None,
        resolution_logs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.study = study
        self.manifest = manifest_writer
        self.study_dir = study_dir
        self.result_files: list[str] = []
        # Pre-resolved runner specs per backend (None = all experiments use subprocess path)
        self._runner_specs = runner_specs
        # Live study progress display (None = no live output)
        self._progress = progress
        # When True, skip GPU advisory lock acquisition
        self._no_lock = no_lock
        # Set of (config_hash, cycle) pairs to skip (resume mode)
        self._skip_set: set[tuple[str, int]] = skip_set or set()
        # Pre-built resolution logs keyed by config_hash (written as _resolution.json sidecar)
        self._resolution_logs: dict[str, dict[str, Any]] = resolution_logs or {}
        # SIGINT state — initialised here, set live in run()
        self._interrupt_event: threading.Event = threading.Event()
        self._active_process: Any = None  # multiprocessing.Process | None
        self._interrupt_count: int = 0
        # Per-config_hash cycle counters — reset at the start of each run()
        self._cycle_counters: dict[str, int] = {}
        # Study-level environment snapshot cache — collected once, reused across experiments
        self._env_snapshot_future: Future[EnvironmentSnapshot] | None = None
        # Study-level baseline cache, keyed per runner target ("local" or
        # "image_<sanitized>") so multi-backend studies don't cross-contaminate.
        self._baselines: dict[str, Any] = {}  # dict[str, BaselineCache]
        self._experiments_since_validation: dict[str, int] = {}
        # Study-level image preparation: True after _prepare_images() succeeds
        self._images_prepared: bool = False

    def run(self) -> list[Any]:
        """Run all experiments in order; return list of results or failure dicts.

        Installs a SIGINT handler for the duration of the run. First Ctrl+C sends
        SIGTERM to the active subprocess and sets interrupt_event. Second Ctrl+C (or
        grace period expiry) sends SIGKILL. After the loop exits, if interrupted,
        calls manifest.mark_interrupted() and sys.exit(130).

        Acquires per-GPU advisory file locks before image preparation (unless
        no_lock=True). Releases locks in the finally block regardless of outcome.

        Integrates circuit breaker (closed -> open -> half-open -> closed/abort) and
        wall-clock timeout: both mark remaining experiments as skipped and update
        the manifest status before returning.

        Note: study.experiments is already the fully-ordered execution sequence produced
        by apply_cycles() in load_study_config(). The runner must not call apply_cycles()
        again — doing so would multiply the count by n_cycles a second time.
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash
        from llenergymeasure.study.circuit_breaker import CircuitBreaker

        # study.experiments is already cycled by load_study_config(); use as-is.
        ordered = self.study.experiments

        # n_unique: count of distinct configs (for cycle-gap detection).
        # Do not use len(ordered) — that includes repetitions.
        seen_hashes = {compute_measurement_config_hash(c) for c in self.study.experiments}
        n_unique = len(seen_hashes)

        # spawn: CUDA-safe; fork causes silent CUDA corruption (CP-1)
        mp_ctx = multiprocessing.get_context("spawn")

        # Reset interrupt state for this run
        self._interrupt_event.clear()
        self._interrupt_count = 0
        self._active_process = None
        self._cycle_counters = {}

        def _sigint_handler(signum: int, frame: Any) -> None:
            self._interrupt_count += 1
            self._interrupt_event.set()
            if self._interrupt_count == 1:
                print(
                    "\nInterrupt received. Waiting for experiment to finish cleanly "
                    "(Ctrl+C again to force)..."
                )
                if self._active_process is not None and self._active_process.is_alive():
                    _kill_process_group(
                        self._active_process.pid, signal.SIGTERM
                    )  # SIGTERM — gentle first attempt
            else:
                print("\nForce-killing experiment subprocess...")
                if self._active_process is not None and self._active_process.is_alive():
                    _kill_process_group(self._active_process.pid, signal.SIGKILL)  # SIGKILL

        original_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        # Acquire per-GPU advisory locks before image preparation.
        # Sorted acquisition prevents deadlocks when multiple studies share GPUs.
        gpu_locks: list[Any] = []
        if not self._no_lock and ordered:
            from llenergymeasure.device.gpu_info import _resolve_gpu_indices
            from llenergymeasure.study.gpu_locks import acquire_gpu_locks

            gpu_indices = _resolve_gpu_indices(ordered[0])
            gpu_locks = acquire_gpu_locks(gpu_indices)

        # Container lifecycle: reap orphaned containers, register cleanup, install SIGTERM bridge.
        # Only activated for studies that use Docker runners.
        original_sigterm: signal.Handlers | None = None
        if self._runner_specs and any(s.mode == RUNNER_DOCKER for s in self._runner_specs.values()):
            from llenergymeasure.study.container_lifecycle import (
                install_sigterm_bridge,
                reap_orphaned_containers,
                register_container_cleanup,
            )

            study_id = self.study.study_design_hash or "unknown"
            reap_orphaned_containers()
            register_container_cleanup(study_id)
            original_sigterm = install_sigterm_bridge()

        self._prepare_images()

        # Circuit breaker: tracks consecutive failures, decides abort/probe.
        breaker = CircuitBreaker(
            max_failures=self.study.study_execution.max_consecutive_failures,
            cooldown_seconds=self.study.study_execution.circuit_breaker_cooldown_seconds,
        )

        # Wall-clock deadline: computed once before the loop.
        deadline: float | None = None
        if self.study.study_execution.wall_clock_timeout_hours:
            deadline = time.monotonic() + (
                self.study.study_execution.wall_clock_timeout_hours * 3600
            )

        try:
            results: list[Any] = []
            # Track whether the loop was aborted by timeout or circuit breaker.
            # Used to skip mark_study_completed() on non-clean exits.
            _aborted = False

            for i, config in enumerate(ordered):
                if self._interrupt_event.is_set():
                    break

                # Resume skip-set: skip experiments that completed in a previous run.
                if self._skip_set:
                    config_hash_pre = compute_measurement_config_hash(config)
                    next_cycle = self._cycle_counters.get(config_hash_pre, 0) + 1
                    if (config_hash_pre, next_cycle) in self._skip_set:
                        self._cycle_counters[config_hash_pre] = next_cycle
                        logger.info(
                            "Skipping completed experiment %d/%d (resumed)",
                            i + 1,
                            len(ordered),
                        )
                        continue

                # Wall-clock timeout check: mark remaining experiments skipped.
                if deadline is not None and time.monotonic() > deadline:
                    self._mark_remaining_skipped(ordered, i, compute_measurement_config_hash)
                    self.manifest.mark_study_timed_out()
                    logger.warning(
                        "Study timed out after %.1f hours",
                        self.study.study_execution.wall_clock_timeout_hours,
                    )
                    _aborted = True
                    break

                # Config gap: between every consecutive experiment pair
                if i > 0:
                    gap_secs = float(self.study.study_execution.experiment_gap_seconds or 0)
                    if gap_secs > 0:
                        self._run_gap(gap_secs, "Experiment gap")
                        if self._interrupt_event.is_set():
                            break

                # Cycle gap: after every complete round of N unique configs
                if n_unique > 0 and i > 0 and i % n_unique == 0:
                    cycle_gap_secs = float(self.study.study_execution.cycle_gap_seconds or 0)
                    if cycle_gap_secs > 0:
                        self._run_gap(cycle_gap_secs, "Cycle gap")
                        if self._interrupt_event.is_set():
                            break

                result = self._run_one(config, mp_ctx, index=i + 1, total=len(ordered))
                results.append(result)

                # Circuit breaker integration: update state based on result.
                if isinstance(result, dict) and "type" in result:
                    error_type = result.get("type", "UnknownError")
                    error_msg = result.get("message", "")
                    action = breaker.record_failure(error_type, error_msg)

                    if action == "tripped":
                        for line in breaker.get_failure_summary():
                            logger.warning("Circuit breaker: %s", line)
                        if breaker.cooldown_seconds > 0:
                            logger.info("Circuit breaker cooldown: %.0fs", breaker.cooldown_seconds)
                            time.sleep(breaker.cooldown_seconds)
                        breaker.start_probe()
                        # Next loop iteration is the probe experiment.

                    elif action == "abort":
                        # Probe failed — abort the study immediately.
                        self._mark_remaining_skipped(
                            ordered, i + 1, compute_measurement_config_hash
                        )
                        self.manifest.mark_study_circuit_breaker()
                        logger.error("Circuit breaker: probe experiment failed, aborting study")
                        _aborted = True
                        break

                else:
                    # Success path: reset circuit breaker (if not disabled).
                    if not breaker.is_disabled:
                        breaker.record_success()

            # Mark study completed on clean exit (no interrupt, timeout, or circuit break).
            if not self._interrupt_event.is_set() and not _aborted:
                self.manifest.mark_study_completed()

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            if original_sigterm is not None:
                signal.signal(signal.SIGTERM, original_sigterm)
            if gpu_locks:
                from llenergymeasure.study.gpu_locks import release_gpu_locks

                release_gpu_locks(gpu_locks)

        if self._interrupt_event.is_set():
            completed = sum(1 for r in results if not isinstance(r, dict))
            total = len(ordered)
            print(
                f"\n{completed}/{total} experiments completed. "
                "Results in study directory. Manifest: interrupted."
            )
            self.manifest.mark_interrupted()
            sys.exit(130)

        return results

    def _mark_remaining_skipped(
        self,
        ordered: list[Any],
        start_index: int,
        hash_fn: Any,
    ) -> None:
        """Mark all experiments from start_index onwards as skipped in the manifest.

        Increments cycle counters to assign the correct cycle number for each
        remaining experiment before marking it skipped.

        Args:
            ordered: Full ordered experiment list (study.experiments).
            start_index: Index of the first experiment to mark as skipped.
            hash_fn: compute_measurement_config_hash callable.
        """
        for j in range(start_index, len(ordered)):
            cfg = ordered[j]
            h = hash_fn(cfg)
            c = self._cycle_counters.get(h, 0) + 1
            self._cycle_counters[h] = c
            self.manifest.mark_skipped(h, c)

    def _get_env_snapshot(self) -> EnvironmentSnapshot:
        """Return cached environment snapshot, collecting on first call.

        Uses background-threaded collection on first call. Subsequent calls
        return the resolved snapshot immediately (study-level cache).
        """
        if self._env_snapshot_future is None:
            from llenergymeasure.harness.environment import collect_environment_snapshot_async

            self._env_snapshot_future = collect_environment_snapshot_async()
        return self._env_snapshot_future.result(timeout=TIMEOUT_ENV_SNAPSHOT)

    def _baseline_cache_key(self, config: ExperimentConfig) -> str:
        """Return the cache key for this experiment's runner target.

        Baselines are keyed per runner target because the container's CUDA init
        footprint (~8.7 W/GPU on A100) is process-local and may differ between
        backend images. See ``.product/research/baseline-measurement-location.md``.

        The ``image_`` prefix uses an underscore (not ``:``) so the key is
        safe to embed directly in both filesystem paths and Docker bind-mount
        sources. A ``:`` in the mount source string would be parsed by Docker
        as the mount-mode separator and fail with ``invalid mode``.
        """
        spec = self._runner_specs.get(config.backend) if self._runner_specs else None
        if spec is None or spec.mode != RUNNER_DOCKER or not spec.image:
            return "local"
        return f"image_{_sanitize_image_for_filename(spec.image)}"

    def _get_baseline(self, config: ExperimentConfig) -> Any:
        """Return baseline power according to the configured strategy.

        Strategies: ``cached`` (measure once per runner target, persist to disk,
        reuse within TTL), ``validated`` (same, with periodic drift spot-check),
        and ``fresh`` (returns None; harness measures in-container per experiment).

        For Docker targets the measurement runs inside a short-lived container
        of the same backend image, so the CUDA init state matches the experiment
        container (see ``.product/research/baseline-measurement-location.md``).
        """
        strategy = config.baseline.strategy

        if strategy == "fresh":
            return None

        cache_key = self._baseline_cache_key(config)
        location = "host" if cache_key == "local" else "baseline container"
        cached = self._baselines.get(cache_key)

        # TTL expiry check
        if cached is not None:
            age = time.time() - cached.timestamp
            if age >= config.baseline.cache_ttl_seconds:
                logger.info(
                    "Baseline expired (age=%.0fs > ttl=%.0fs). Re-measuring.",
                    age,
                    config.baseline.cache_ttl_seconds,
                )
                cached = None
                self._baselines.pop(cache_key, None)

        # validated: periodic spot-check for drift (only if baseline still valid)
        if strategy == "validated" and cached is not None:
            self._experiments_since_validation[cache_key] = (
                self._experiments_since_validation.get(cache_key, 0) + 1
            )
            if self._experiments_since_validation[cache_key] >= config.baseline.validation_interval:
                self._validate_baseline(config, cache_key)
                cached = self._baselines.get(cache_key)  # may have been re-measured

        # In-memory hit → emit "Reusing" and return
        if cached is not None:
            if self._progress is not None:
                self._progress.on_step_start(
                    STEP_BASELINE,
                    "Reusing",
                    f"cached {cached.power_w:.1f}W · {location}",
                )
                self._emit_baseline_result_substeps(cached, elapsed=0.0, mode="cached")
                self._progress.on_step_done(STEP_BASELINE, 0.0)
            return cached

        # Try loading from disk first (handles mid-study restarts)
        disk_path = self._get_baseline_cache_path(cache_key)
        if disk_path.exists():
            from llenergymeasure.harness.baseline import load_baseline_cache

            if self._progress is not None:
                self._progress.on_step_start(
                    STEP_BASELINE, "Loading", f"baseline cache · {location}"
                )
                t0_load = time.perf_counter()

            loaded = load_baseline_cache(disk_path, ttl=config.baseline.cache_ttl_seconds)

            if self._progress is not None:
                load_elapsed = time.perf_counter() - t0_load
                if loaded is not None:
                    self._emit_baseline_result_substeps(loaded, elapsed=load_elapsed, mode="disk")
                self._progress.on_step_done(STEP_BASELINE, load_elapsed)

            if loaded is not None:
                if loaded.method is None:
                    loaded.method = strategy
                self._baselines[cache_key] = loaded
                self._experiments_since_validation.setdefault(cache_key, 0)
                logger.debug("Loaded baseline from disk cache: %.1fW", loaded.power_w)
                return loaded

        # Measure fresh baseline (in-container for Docker targets)
        dur = config.baseline.duration_seconds
        if self._progress is not None:
            self._progress.on_step_start(
                STEP_BASELINE, "Measuring", f"{location} · idle power ({dur:.0f}s)"
            )
            t0_meas = time.perf_counter()

        on_stage = self._make_baseline_stage_callback() if self._progress is not None else None
        measured = self._measure_baseline(config, cache_key, on_stage=on_stage)

        if measured is not None:
            measured.method = strategy
            self._baselines[cache_key] = measured
            self._experiments_since_validation[cache_key] = 0

            from llenergymeasure.harness.baseline import save_baseline_cache

            save_baseline_cache(disk_path, measured)

        if self._progress is not None:
            elapsed = time.perf_counter() - t0_meas
            if measured is None:
                # Surface the failure so users don't see a silent tick hiding a
                # dispatch crash. The experiment container falls back to measuring
                # its own baseline; the substep tells the user *why*.
                self._progress.on_substep(
                    STEP_BASELINE,
                    f"measurement failed after {elapsed:.1f}s — see log warnings "
                    f"(experiment container will re-measure fresh)",
                    elapsed,
                )
            elif cache_key == "local":
                # No container subprocess → emit a retroactive sampling substep
                # so the local path still gets a breakdown. The Docker path
                # already emitted substeps live via the on_stage callback.
                self._emit_baseline_result_substeps(
                    measured,
                    elapsed=elapsed,
                    mode="fresh",
                    is_containerised=False,
                )
            self._progress.on_step_done(STEP_BASELINE, elapsed)

        return measured

    def _make_baseline_stage_callback(self) -> Any:
        """Build a stage-marker callback that emits live baseline sub-bullets.

        Each sub-bullet reports the duration of *that stage* (delta since the
        previous marker), so users see "launch took Xs, CUDA init took Ys,
        sampling took Zs" rather than ever-increasing cumulative totals.
        ``container_ready`` is the exception — its elapsed IS the container
        launch cost (no prior marker to diff against).
        """
        last_t = 0.0

        def on_stage(name: str, elapsed: float, kv: dict[str, str]) -> None:
            nonlocal last_t
            if self._progress is None:
                return
            delta = max(0.0, elapsed - last_t)
            last_t = elapsed
            if name == "container_ready":
                self._progress.on_substep(
                    STEP_BASELINE,
                    "dispatched baseline container · Python runtime ready",
                    elapsed,
                )
            elif name == "cuda_primed":
                self._progress.on_substep(
                    STEP_BASELINE,
                    "initialised CUDA runtime · seeded torch allocator",
                    delta,
                )
            elif name == "sampling_done":
                power_w = kv.get("power_w", "?")
                samples = kv.get("samples", "?")
                sampled = kv.get("duration", "?")
                self._progress.on_substep(
                    STEP_BASELINE,
                    f"sampled idle GPU power · {sampled}s ({power_w}W · {samples} samples)",
                    delta,
                )
            elif name == "result_written" and delta > 0.1:
                self._progress.on_substep(
                    STEP_BASELINE,
                    "wrote result & container teardown",
                    delta,
                )

        return on_stage

    def _emit_baseline_result_substeps(
        self,
        baseline: Any,  # BaselineCache
        *,
        elapsed: float,
        mode: str,  # "fresh" | "cached" | "disk"
        is_containerised: bool = False,
    ) -> None:
        """Emit dim-bullet substeps explaining where the baseline time went.

        For a fresh Docker measurement we split the wall-clock into container
        launch/teardown (the residual) vs the NVML sampling window (recorded
        inside ``measure_baseline_power``). This answers "why did a 30s
        measurement take 37.7s?" without the user having to dig through logs.

        For in-memory and disk-loaded reuse we only emit the result summary —
        there is no sampling window to describe.
        """
        if self._progress is None:
            return
        if mode == "fresh" and is_containerised:
            # Residual captures docker run startup + CUDA prime + result write
            # + container teardown.
            overhead = max(0.0, elapsed - baseline.duration_sec)
            self._progress.on_substep(
                STEP_BASELINE,
                f"container launch + teardown: {overhead:.1f}s",
                overhead,
            )
            self._progress.on_substep(
                STEP_BASELINE,
                f"sampling: {baseline.duration_sec:.1f}s "
                f"({baseline.power_w:.1f}W · {baseline.sample_count} samples)",
                baseline.duration_sec,
            )
        elif mode == "fresh":
            self._progress.on_substep(
                STEP_BASELINE,
                f"sampling: {baseline.duration_sec:.1f}s "
                f"({baseline.power_w:.1f}W · {baseline.sample_count} samples)",
                baseline.duration_sec,
            )
        else:
            source = "in-memory" if mode == "cached" else "disk"
            self._progress.on_substep(
                STEP_BASELINE,
                f"reused from {source} cache: "
                f"{baseline.power_w:.1f}W ({baseline.sample_count} samples)",
            )

    def _measure_baseline(
        self,
        config: ExperimentConfig,
        cache_key: str,
        on_stage: Any = None,  # StageCallback | None
    ) -> Any:
        """Measure a fresh baseline on host or inside a baseline container.

        Local runner targets measure on host (no process boundary, no bias).
        Docker targets dispatch a short-lived baseline container of the backend
        image so the CUDA init state matches the experiment container. See
        ``.product/research/baseline-measurement-location.md`` for the
        empirical rationale (~8.7 W/GPU bias on A100).

        Args:
            config: Experiment config with baseline settings.
            cache_key: "local" or "image_<slug>" — chooses dispatch path.
            on_stage: Optional callback forwarded to ``run_baseline_container``
                for streaming stage markers. Ignored on the local path (there
                is no subprocess to stream from).
        """
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        gpu_indices = _resolve_gpu_indices(config)

        if cache_key == "local":
            from llenergymeasure.harness.baseline import measure_baseline_power

            return measure_baseline_power(
                duration_sec=config.baseline.duration_seconds,
                gpu_indices=gpu_indices,
            )

        # Docker: _baseline_cache_key already guaranteed a resolved image when
        # it returned a non-"local" key.
        assert self._runner_specs is not None
        spec = self._runner_specs[config.backend]
        assert spec.image is not None
        from llenergymeasure.study.baseline_container import run_baseline_container

        return run_baseline_container(
            image=spec.image,
            mode="measure",
            duration_sec=config.baseline.duration_seconds,
            gpu_indices=gpu_indices,
            on_stage=on_stage,
        )

    def _spot_check_baseline(
        self,
        config: ExperimentConfig,
        cache_key: str,
        gpu_indices: list[int],
    ) -> float | None:
        """Quick drift-check measurement for the validated strategy.

        Dispatches to host or baseline container matching the cache key. Returns
        the measured power in watts, or None on failure.
        """
        if cache_key == "local":
            from llenergymeasure.harness.baseline import measure_spot_check

            return measure_spot_check(gpu_indices=gpu_indices, duration_sec=5.0)

        spec = self._runner_specs.get(config.backend) if self._runner_specs else None
        if spec is None or spec.image is None:
            return None

        from llenergymeasure.study.baseline_container import run_baseline_container

        result = run_baseline_container(
            image=spec.image,
            mode="spot_check",
            duration_sec=5.0,
            gpu_indices=gpu_indices,
        )
        return result.power_w if result is not None else None

    def _validate_baseline(self, config: ExperimentConfig, cache_key: str) -> None:
        """Spot-check baseline for drift (strategy='validated' only).

        Performs a short measurement and compares with the cached baseline. On
        drift beyond the configured threshold, re-measures the full baseline and
        updates the disk cache. Emits a single ``Validating`` step (with an
        ``on_step_update`` mid-step when a re-measure is triggered) so the
        display step counter stays clean.
        """
        cached = self._baselines.get(cache_key)
        if cached is None:
            return

        from llenergymeasure.device.gpu_info import _resolve_gpu_indices
        from llenergymeasure.harness.baseline import save_baseline_cache

        gpu_indices = _resolve_gpu_indices(config)
        location = "host" if cache_key == "local" else "baseline container"

        if self._progress is not None:
            self._progress.on_step_start(
                STEP_BASELINE, "Validating", f"{location} · drift check (5s)"
            )
            t0 = time.perf_counter()

        spot = self._spot_check_baseline(config, cache_key, gpu_indices)
        self._experiments_since_validation[cache_key] = 0

        if spot is None:
            logger.warning("Baseline validation: spot-check measurement failed")
            if self._progress is not None:
                self._progress.on_step_done(STEP_BASELINE, time.perf_counter() - t0)
            return

        drift = abs(spot - cached.power_w) / cached.power_w
        if drift > config.baseline.drift_threshold:
            dur = config.baseline.duration_seconds
            logger.info(
                "Baseline drift detected: %.1fW -> %.1fW (%.1f%% > %.1f%% threshold). "
                "Re-measuring full baseline.",
                cached.power_w,
                spot,
                drift * 100,
                config.baseline.drift_threshold * 100,
            )
            if self._progress is not None:
                self._progress.on_step_update(
                    STEP_BASELINE,
                    f"{location} · drift {drift * 100:.1f}% > "
                    f"{config.baseline.drift_threshold * 100:.0f}%, "
                    f"re-measuring ({dur:.0f}s)",
                )

            remeasured = self._measure_baseline(config, cache_key)
            if remeasured is not None:
                remeasured.method = "validated"
                self._baselines[cache_key] = remeasured
                save_baseline_cache(self._get_baseline_cache_path(cache_key), remeasured)
        else:
            cached.method = "validated"
            logger.debug(
                "Baseline validation passed: drift=%.1f%% (threshold=%.1f%%)",
                drift * 100,
                config.baseline.drift_threshold * 100,
            )

        if self._progress is not None:
            self._progress.on_step_done(STEP_BASELINE, time.perf_counter() - t0)

    def _get_baseline_cache_path(self, cache_key: str) -> Path:
        """Return the disk path for the baseline cache file keyed by runner target.

        File lives at ``{study_dir}/_study-artefacts/baseline_cache_{cache_key}.json``.
        Creates the artefacts directory if needed.
        """
        artefacts_dir = self.study_dir / "_study-artefacts"
        artefacts_dir.mkdir(parents=True, exist_ok=True)
        return artefacts_dir / f"baseline_cache_{cache_key}.json"

    def _prepare_images(self) -> None:
        """Check/pull Docker images for all Docker backends before experiments.

        Runs once at the start of the study. Each backend's image is verified
        (or pulled) sequentially. On failure, raises so the study aborts early.
        Sets ``_images_prepared`` so per-experiment image_check is skipped.
        """

        if not self._runner_specs:
            return

        docker_backends = [
            (backend, spec)
            for backend, spec in self._runner_specs.items()
            if spec.mode == RUNNER_DOCKER and spec.image
        ]
        if not docker_backends:
            return

        if self._progress:
            self._progress.begin_image_prep([b for b, _ in docker_backends])

        for backend, spec in docker_backends:
            image = spec.image
            assert image is not None  # narrowing for type checker
            t0 = time.monotonic()

            # Check if image exists locally
            try:
                check = subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True,
                    timeout=TIMEOUT_DOCKER_INSPECT,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                check = None

            if check is not None and check.returncode == 0:
                elapsed = time.monotonic() - t0
                metadata = self._parse_image_metadata(check.stdout)
                if self._progress:
                    self._progress.image_ready(
                        backend, image, cached=True, elapsed=elapsed, metadata=metadata
                    )
                continue

            # Image not found locally — try to pull
            logger.info("Image %s not found locally, pulling...", image)
            try:
                pull = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    timeout=DOCKER_PULL_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                if self._progress:
                    self._progress.image_failed(backend, image, "pull timed out (30min)")
                    self._progress.end_image_prep()
                from llenergymeasure.infra.docker_errors import DockerImagePullError

                raise DockerImagePullError(
                    message=f"Image pull timed out: {image}",
                    fix_suggestion=f"COMPOSE_BAKE=true docker compose build {backend}",
                ) from exc

            if pull.returncode != 0:
                tip = f"COMPOSE_BAKE=true docker compose build {backend}"
                if self._progress:
                    self._progress.image_failed(backend, image, f"not found \u2014 run: {tip}")
                    self._progress.end_image_prep()
                from llenergymeasure.infra.docker_errors import DockerImagePullError

                raise DockerImagePullError(
                    message=f"Image not found: {image}",
                    fix_suggestion=tip,
                )

            elapsed = time.monotonic() - t0
            # Re-inspect for metadata after pull
            try:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True,
                    timeout=TIMEOUT_DOCKER_INSPECT,
                )
                metadata = self._parse_image_metadata(inspect.stdout)
            except Exception:
                metadata = None

            if self._progress:
                self._progress.image_ready(
                    backend, image, cached=False, elapsed=elapsed, metadata=metadata
                )

        if self._progress:
            self._progress.end_image_prep()

        self._images_prepared = True

    @staticmethod
    def _parse_image_metadata(inspect_stdout: bytes) -> dict[str, str] | None:
        """Extract human-readable metadata from docker image inspect JSON."""
        try:
            data = json.loads(inspect_stdout)
            if not data:
                return None
            info = data[0]
            meta: dict[str, str] = {}

            image_id = info.get("Id", "")
            if image_id.startswith("sha256:"):
                image_id = image_id[7:19]
            if image_id:
                meta["id"] = image_id

            size_bytes = info.get("Size", 0)
            if size_bytes:
                if size_bytes >= 1_073_741_824:
                    meta["size"] = f"{size_bytes / 1_073_741_824:.1f} GB"
                else:
                    meta["size"] = f"{size_bytes / 1_048_576:.0f} MB"

            created = info.get("Created", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created[:26].rstrip("Z"))
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - created_dt
                    if age.days > 0:
                        meta["built"] = f"{age.days}d ago"
                    elif age.seconds >= 3600:
                        meta["built"] = f"{age.seconds // 3600}h ago"
                    else:
                        meta["built"] = f"{age.seconds // 60}m ago"
                except (ValueError, TypeError):
                    pass

            layers = info.get("RootFS", {}).get("Layers", [])
            if layers:
                meta["layers"] = str(len(layers))

            return meta if meta else None
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def _run_gap(self, seconds: float, label: str) -> None:
        """Run a thermal gap, rendering countdown in the live display or terminal."""
        if self._progress:
            from llenergymeasure.study.gaps import format_gap_duration

            for remaining in range(int(seconds), 0, -1):
                if self._interrupt_event.is_set():
                    break
                self._progress.show_gap(f"{label}: {format_gap_duration(remaining)}")
                self._interrupt_event.wait(timeout=TIMEOUT_INTERRUPT_POLL)
            self._progress.clear_gap()
        else:
            # Fall back to terminal countdown
            run_gap(seconds, label, self._interrupt_event)

    def _run_one(self, config: ExperimentConfig, mp_ctx: Any, index: int, total: int) -> Any:
        """Dispatch one experiment via Docker or subprocess, collect result or failure dict.

        Checks runner_specs for this experiment's backend first. If a Docker spec is
        found, delegates to _run_one_docker(). Otherwise falls through to the existing
        subprocess dispatch path.

        If interrupt_event is set after join, attempts graceful SIGTERM → 2s grace →
        SIGKILL before collecting whatever result is available.
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        # Increment per-config_hash counter: 1st run → cycle=1, 2nd → cycle=2, etc.
        current = self._cycle_counters.get(config_hash, 0) + 1
        self._cycle_counters[config_hash] = current
        cycle = current

        # Docker dispatch path — check runner spec for this backend
        spec = self._runner_specs.get(config.backend) if self._runner_specs else None
        if spec is not None and spec.mode == RUNNER_DOCKER:
            return self._run_one_docker(
                config, spec, config_hash=config_hash, cycle=cycle, index=index
            )

        timeout = self.study.study_execution.experiment_timeout_seconds

        # Signal study display: new experiment starting (subprocess = local steps)
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            local_spec = self._runner_specs.get(config.backend) if self._runner_specs else None
            self._progress.begin_experiment(
                index,
                format_experiment_header(config),
                list(STEPS_LOCAL),
                runner_info=local_spec.to_runner_info() if local_spec else None,
            )

        exp_start = time.monotonic()

        # Create a temp dir for timeseries parquet output. The harness receives
        # output_dir as a runtime param (not from config). The temp dir is
        # cleaned up after _handle_result copies the parquet into the study directory.
        save_ts = self.study.output.save_timeseries
        ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES)) if save_ts else None

        # Resolve cached snapshot in parent — serialised to subprocess via Pipe
        snapshot = self._get_env_snapshot()

        # Resolve cached baseline in parent — avoids 30s re-measurement per subprocess
        baseline = self._get_baseline(config) if config.baseline.enabled else None

        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue, snapshot),
            kwargs={
                "output_dir": str(ts_tmpdir) if ts_tmpdir else None,
                "save_timeseries": save_ts,
                "baseline": baseline,
            },
            daemon=False,  # daemon=False: clean CUDA teardown if parent exits unexpectedly
        )

        consumer = threading.Thread(
            target=_consume_progress_events,
            args=(progress_queue, self._progress),
            daemon=True,
        )
        consumer.start()

        self.manifest.mark_running(config_hash, cycle)
        self._active_process = p

        # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

        check_gpu_memory_residual()

        p.start()
        child_conn.close()

        # Drain pipe BEFORE join to prevent buffer deadlock (H5).
        # If pickled ExperimentResult > 64 KB, child blocks on conn.send()
        # while parent blocks in p.join() — classic deadlock.
        pipe_payload = _UNSET
        if parent_conn.poll(timeout=timeout):
            try:
                pipe_payload = parent_conn.recv()
            except Exception:
                pipe_payload = _UNSET

        # Non-blocking join after pipe is drained (grace for teardown)
        p.join(timeout=TIMEOUT_THREAD_JOIN)

        # SIGINT was received during join: SIGTERM was already sent by handler.
        # Grace period for clean CUDA teardown, then SIGKILL.
        if self._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=TIMEOUT_SIGTERM_GRACE)
            if p.is_alive():
                _kill_process_group(p.pid, signal.SIGKILL)
                p.join()

        self._active_process = None

        # Sentinel stops consumer thread — covers SIGKILL path too
        progress_queue.put(None)
        consumer.join()

        result = _collect_result(p, parent_conn, config, timeout, pipe_payload=pipe_payload)
        parent_conn.close()

        exp_elapsed = time.monotonic() - exp_start
        self._handle_result(
            result,
            config_hash,
            cycle,
            index,
            exp_elapsed,
            ts_source_dir=ts_tmpdir,
            environment_snapshot=self._get_env_snapshot() if not isinstance(result, dict) else None,
        )

        # Clean up the temp dir created for timeseries parquet output.
        # _save_and_record already copied the parquet into the study dir.
        if ts_tmpdir is not None:
            shutil.rmtree(ts_tmpdir, ignore_errors=True)

        return result

    def _handle_result(
        self,
        result: Any,
        config_hash: str,
        cycle: int,
        index: int,
        elapsed: float,
        ts_source_dir: Path | None = None,
        environment_snapshot: Any | None = None,
    ) -> None:
        """Update manifest and signal study display based on experiment outcome."""
        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_message = result.get("message", "")
            log_file = result.get("log_file")
            self.manifest.mark_failed(
                config_hash, cycle, error_type, error_message, log_file=log_file
            )
            if self._progress:
                self._progress.end_experiment_fail(index, elapsed, error=error_message)
        else:
            _save_and_record(
                result,
                self.study_dir,
                self.manifest,
                config_hash,
                cycle,
                self.result_files,
                experiment_index=index,
                ts_source_dir=ts_source_dir,
                environment_snapshot=environment_snapshot,
                resolution_log=self._resolution_logs.get(config_hash),
            )
            if self._progress:
                # Emit save paths as substeps BEFORE end_experiment_ok (which clears
                # inner step display). This makes paths visible inline in TTY mode.
                if self.result_files:
                    host_path = self.result_files[-1]
                    # Docker experiments: show container path first, then host path.
                    # The original container path is /run/llem; by this point output_dir
                    # has been rewritten to the host temp dir, so use the known constant.
                    spec = (
                        self._runner_specs.get(getattr(result, "backend", None) or "")
                        if self._runner_specs
                        else None
                    )
                    is_docker = spec is not None and spec.mode == RUNNER_DOCKER
                    if is_docker:
                        self._progress.on_substep("save", f"container: {CONTAINER_EXCHANGE_DIR}")
                    self._progress.on_substep("save", f"host: {host_path}")

                energy_j = getattr(result, "total_energy_j", None)
                throughput = getattr(result, "avg_tokens_per_second", None)
                infer_sec = getattr(result, "total_inference_time_sec", None)
                adj_energy_j = getattr(result, "energy_adjusted_j", None)
                mj_per_tok_adjusted = getattr(result, "mj_per_tok_adjusted", None)
                mj_per_tok_total = getattr(result, "mj_per_tok_total", None)
                self._progress.end_experiment_ok(
                    index,
                    elapsed,
                    energy_j=energy_j if energy_j and energy_j > 0 else None,
                    throughput_tok_s=throughput if throughput and throughput > 0 else None,
                    inference_time_sec=infer_sec if infer_sec and infer_sec > 0 else None,
                    adj_energy_j=adj_energy_j if adj_energy_j and adj_energy_j > 0 else None,
                    mj_per_tok_adjusted=mj_per_tok_adjusted,
                    mj_per_tok_total=mj_per_tok_total,
                )
                # Also store for finish() footer
                if self.result_files:
                    host_path = self.result_files[-1]
                    self._progress.on_experiment_saved(index, host_path)

    def _run_one_docker(
        self,
        config: ExperimentConfig,
        spec: RunnerSpec,
        *,
        config_hash: str,
        cycle: int,
        index: int,
    ) -> Any:
        """Dispatch one experiment to a Docker container via DockerRunner.

        Blocking dispatch — no subprocess or thread overhead.
        DockerErrors are caught and converted to non-fatal failure dicts so the
        study continues even when a container fails.

        Args:
            config:      ExperimentConfig to run.
            spec:        Resolved RunnerSpec (mode="docker") for this backend.
            config_hash: Pre-computed config hash (avoids recomputing).
            cycle:       Current cycle number for manifest tracking.
            index:       1-based position in study for progress display.

        Returns:
            ExperimentResult on success, or a failure dict on error.
        """
        from llenergymeasure.infra.docker_errors import DockerTimeoutError
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.container_lifecycle import (
            generate_container_labels,
            generate_container_name,
        )
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual
        from llenergymeasure.utils.exceptions import DockerError

        # Image is pre-resolved during preflight (resolve_image precedence chain).
        # Fall back to get_default_image() only for direct DockerRunner usage
        # outside the study path.
        image = spec.image if spec.image is not None else get_default_image(config.backend)

        study_id = self.study.study_design_hash or "unknown"
        container_name = generate_container_name(study_id, index)
        labels = generate_container_labels(study_id)

        # begin_experiment MUST run before _get_baseline so baseline step events
        # fire against a registered experiment index.
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            host_baseline = config.baseline.enabled and config.baseline.strategy != "fresh"
            steps = docker_steps(
                images_prepared=self._images_prepared,
                host_baseline=host_baseline,
            )
            self._progress.begin_experiment(
                index,
                format_experiment_header(config),
                steps,
                runner_info=spec.to_runner_info(),
            )
            # Host-side preflight doesn't run in Docker path — checked inside container
            self._progress.on_step_skip("preflight", "checked inside container")

        extra_mounts = list(spec.extra_mounts) if spec.extra_mounts else []
        cache_key = self._baseline_cache_key(config)
        baseline = self._get_baseline(config) if config.baseline.enabled else None
        if baseline is not None:
            # Experiment container reads /run/llem/baseline_cache.json; the host
            # picks the right per-cache-key file at dispatch time. Docker parses
            # relative bind-mount sources as named volumes, so resolve first.
            baseline_cache_path = self._get_baseline_cache_path(cache_key)
            extra_mounts.append(
                (
                    str(baseline_cache_path.resolve()),
                    f"{CONTAINER_EXCHANGE_DIR}/baseline_cache.json",
                )
            )

        docker_runner = DockerRunner(
            image=image,
            timeout=self.study.study_execution.experiment_timeout_seconds,
            source=spec.source,
            extra_mounts=extra_mounts,
            container_name=container_name,
            labels=labels,
        )

        # Pre-dispatch GPU memory residual check (same as local path)
        check_gpu_memory_residual()

        self.manifest.mark_running(config_hash, cycle)

        exp_start = time.monotonic()

        result: Any
        docker_ts_dir: Path | None = None
        try:
            # Pass study progress as step callback — DockerRunner calls on_step_*
            # skip_image_check=True when images were verified at study level.
            result, docker_ts_dir = docker_runner.run(
                config,
                progress=self._progress,
                save_timeseries=self.study.output.save_timeseries,
                skip_image_check=self._images_prepared,
            )
        except DockerTimeoutError as exc:
            # Normalise to "TimeoutError" so the circuit breaker sees the same
            # failure class as the subprocess path (see _collect_result).
            result = {
                "type": "TimeoutError",
                "message": str(exc),
                "config_hash": config_hash,
            }
            self._persist_failure_artefacts(exc, config_hash, cycle, result)
        except DockerError as exc:
            # Use structured error payload from container entrypoint when available,
            # falling back to the exception type/message for stderr-based errors.
            payload = getattr(exc, "error_payload", None)
            if payload:
                result = {
                    "type": payload.get("type", type(exc).__name__),
                    "message": payload.get("message", str(exc)),
                    "config_hash": config_hash,
                }
            else:
                result = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "config_hash": config_hash,
                }
            self._persist_failure_artefacts(exc, config_hash, cycle, result)

        exp_elapsed = time.monotonic() - exp_start
        self._handle_result(
            result,
            config_hash,
            cycle,
            index,
            exp_elapsed,
            ts_source_dir=docker_ts_dir,
            environment_snapshot=self._get_env_snapshot() if not isinstance(result, dict) else None,
        )

        # Clean up the temp dir after _save_and_record has copied the parquet.
        if docker_ts_dir is not None:
            shutil.rmtree(docker_ts_dir, ignore_errors=True)

        return result

    def _persist_failure_artefacts(
        self,
        exc: DockerError,
        config_hash: str,
        cycle: int,
        result: dict[str, Any],
    ) -> None:
        """Copy failure artefacts from the Docker exchange dir into ``failed-runs/``.

        Copies ``container.log`` and any ``*_error.json`` from the exchange
        directory. Adds a ``log_file`` key to *result* so the manifest records
        where the log can be found.
        """
        exchange_dir_str = getattr(exc, "exchange_dir", None)
        if not exchange_dir_str:
            return

        exchange_dir = Path(exchange_dir_str)
        failed_runs_dir = self.study_dir / "failed-runs"
        prefix = f"{config_hash}_cycle{cycle}"

        try:
            failed_runs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as mkdir_exc:
            logger.warning("Failed to create failed-runs/: %s", mkdir_exc)
            return

        # Copy container.log (Docker stderr capture)
        log_file = self._copy_artefact(
            exchange_dir / "container.log",
            failed_runs_dir / f"{prefix}_container.log",
        )
        if log_file:
            result["log_file"] = f"failed-runs/{log_file}"

        # Copy error JSON (structured traceback from container entrypoint).
        # The error JSON uses the Docker config hash (output_dir=/run/llem),
        # which differs from the study-level config_hash, so glob for it.
        for src in exchange_dir.glob("*_error.json"):
            self._copy_artefact(src, failed_runs_dir / f"{prefix}_error.json")
            break  # only one expected

    @staticmethod
    def _copy_artefact(src: Path, dest: Path) -> str | None:
        """Copy a single file, returning the dest filename on success or None."""
        if not src.exists():
            return None
        try:
            shutil.copy2(src, dest)
            logger.debug("Artefact persisted to %s", dest)
            return dest.name
        except Exception as copy_exc:
            logger.warning("Failed to persist %s to %s: %s", src.name, dest, copy_exc)
            return None
