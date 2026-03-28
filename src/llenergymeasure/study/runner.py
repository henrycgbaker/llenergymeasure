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
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import RUNNER_DOCKER
from llenergymeasure.domain.progress import STEPS_DOCKER, STEPS_LOCAL
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
    "_calculate_timeout",
    "_kill_process_group",
    "_resolve_ts_source_dir",
    "_run_experiment_worker",
    "_save_and_record",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level helpers
# =============================================================================


def _resolve_ts_source_dir(
    result: Any,
    spec: RunnerSpec | None,
    local_ts_tmpdir: Path | None,
) -> Path | None:
    """Resolve the directory containing timeseries.parquet for _save_and_record.

    Docker path: DockerRunner rescues parquet to a temp dir and writes the
    host path into effective_config["output_dir"].
    Local path: the caller passes the temp dir it created for the harness.
    """
    if isinstance(result, dict):
        return None
    if spec is not None and spec.mode == RUNNER_DOCKER and hasattr(result, "effective_config"):
        ts_dir_str = result.effective_config.get("output_dir")
        if ts_dir_str and Path(ts_dir_str).exists():
            return Path(ts_dir_str)
        return None
    return local_ts_tmpdir


def _calculate_timeout(config: ExperimentConfig) -> int:
    """Generous timeout heuristic: 2 seconds per prompt, minimum 10 minutes.

    No model-size scaling — keep it simple.
    """
    return max(config.dataset.n_prompts * 2, 600)


def _save_and_record(
    result: Any,
    study_dir: Path,
    manifest: ManifestWriter,
    config_hash: str,
    cycle: int,
    result_files: list[str],
    experiment_index: int | None = None,
    ts_source_dir: Path | None = None,
) -> None:
    """Save result to disk and update manifest. Appends result path to result_files.

    Resolves the timeseries parquet sidecar from the result object and passes it
    to save_result() so it is copied into the experiment subdirectory. The stale
    flat file written by MeasurementHarness is removed after the copy.

    Args:
        ts_source_dir: Directory where the harness wrote timeseries.parquet.
            When provided, overrides the legacy effective_config["output_dir"] lookup.

    On save failure, marks the experiment as completed with empty path.
    """
    try:
        from llenergymeasure.results.persistence import save_result

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
            result, study_dir, timeseries_source=ts_source, experiment_index=experiment_index
        )

        # Clean up the stale flat parquet file after it has been copied into the
        # experiment subdirectory (mirrors cli/run.py line 288).
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)

        result_files.append(str(result_path))
        rel_path = str(result_path.relative_to(study_dir))
        manifest.mark_completed(config_hash, cycle, rel_path)
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
    timeout: int,
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
    ) -> None:
        self.study = study
        self.manifest = manifest_writer
        self.study_dir = study_dir
        self.result_files: list[str] = []
        # Pre-resolved runner specs per backend (None = all experiments use subprocess path)
        self._runner_specs = runner_specs
        # Live study progress display (None = no live output)
        self._progress = progress
        # SIGINT state — initialised here, set live in run()
        self._interrupt_event: threading.Event = threading.Event()
        self._active_process: Any = None  # multiprocessing.Process | None
        self._interrupt_count: int = 0
        # Per-config_hash cycle counters — reset at the start of each run()
        self._cycle_counters: dict[str, int] = {}
        # Study-level environment snapshot cache — collected once, reused across experiments
        self._env_snapshot_future: Future[EnvironmentSnapshot] | None = None

    def run(self) -> list[Any]:
        """Run all experiments in order; return list of results or failure dicts.

        Installs a SIGINT handler for the duration of the run. First Ctrl+C sends
        SIGTERM to the active subprocess and sets interrupt_event. Second Ctrl+C (or
        grace period expiry) sends SIGKILL. After the loop exits, if interrupted,
        calls manifest.mark_interrupted() and sys.exit(130).

        Note: study.experiments is already the fully-ordered execution sequence produced
        by apply_cycles() in load_study_config(). The runner must not call apply_cycles()
        again — doing so would multiply the count by n_cycles a second time.
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

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

        try:
            results: list[Any] = []

            for i, config in enumerate(ordered):
                if self._interrupt_event.is_set():
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

        finally:
            signal.signal(signal.SIGINT, original_sigint)

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

    def _get_env_snapshot(self) -> EnvironmentSnapshot:
        """Return cached environment snapshot, collecting on first call.

        Uses background-threaded collection on first call. Subsequent calls
        return the resolved snapshot immediately (study-level cache).
        """
        if self._env_snapshot_future is None:
            from llenergymeasure.harness.environment import collect_environment_snapshot_async

            self._env_snapshot_future = collect_environment_snapshot_async()
        return self._env_snapshot_future.result(timeout=10)

    def _run_gap(self, seconds: float, label: str) -> None:
        """Run a thermal gap, rendering countdown in the live display or terminal."""
        if self._progress:
            from llenergymeasure.study.gaps import format_gap_duration

            for remaining in range(int(seconds), 0, -1):
                if self._interrupt_event.is_set():
                    break
                self._progress.show_gap(f"{label}: {format_gap_duration(remaining)}")
                self._interrupt_event.wait(timeout=1)
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

        timeout = _calculate_timeout(config)

        # Signal study display: new experiment starting (subprocess = local steps)
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            # Build runner info for display — local path uses spec if available
            local_spec = self._runner_specs.get(config.backend) if self._runner_specs else None
            runner_info: dict[str, str | None] = {
                "mode": "local",
                "source": local_spec.source if local_spec else "default",
                "image": None,
                "image_source": None,
            }
            self._progress.begin_experiment(
                index, format_experiment_header(config), list(STEPS_LOCAL), runner_info=runner_info
            )

        exp_start = time.monotonic()

        # Create a temp dir for timeseries parquet output. The harness receives
        # output_dir as a runtime param (not from config). The temp dir is
        # cleaned up after _handle_result copies the parquet into the study directory.
        save_ts = self.study.output.save_timeseries
        ts_tmpdir = Path(tempfile.mkdtemp(prefix="llem-ts-")) if save_ts else None

        # Resolve cached snapshot in parent — serialised to subprocess via Pipe
        snapshot = self._get_env_snapshot()

        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue, snapshot),
            kwargs={
                "output_dir": str(ts_tmpdir) if ts_tmpdir else None,
                "save_timeseries": save_ts,
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

        # Non-blocking join after pipe is drained (5s grace for teardown)
        p.join(timeout=5)

        # SIGINT was received during join: SIGTERM was already sent by handler.
        # Give child 2s grace for clean CUDA teardown, then SIGKILL.
        if self._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=2)  # 2s grace after SIGTERM
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
        self._handle_result(result, config_hash, cycle, index, exp_elapsed, ts_source_dir=ts_tmpdir)

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
                        from llenergymeasure.config.ssot import CONTAINER_EXCHANGE_DIR

                        self._progress.on_substep("save", f"container: {CONTAINER_EXCHANGE_DIR}")
                    self._progress.on_substep("save", f"host: {host_path}")

                energy_j = getattr(result, "total_energy_j", None)
                throughput = getattr(result, "avg_tokens_per_second", None)
                infer_sec = getattr(result, "total_inference_time_sec", None)
                self._progress.end_experiment_ok(
                    index,
                    elapsed,
                    energy_j=energy_j if energy_j and energy_j > 0 else None,
                    throughput_tok_s=throughput if throughput and throughput > 0 else None,
                    inference_time_sec=infer_sec if infer_sec and infer_sec > 0 else None,
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
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual
        from llenergymeasure.utils.exceptions import DockerError

        # Image is pre-resolved during preflight (resolve_image precedence chain).
        # Fall back to get_default_image() only for direct DockerRunner usage
        # outside the study path.
        image = spec.image if spec.image is not None else get_default_image(config.backend)

        docker_runner = DockerRunner(
            image=image,
            timeout=_calculate_timeout(config),
            source=spec.source,
            extra_mounts=spec.extra_mounts,
        )

        # Pre-dispatch GPU memory residual check (same as local path)
        check_gpu_memory_residual()

        self.manifest.mark_running(config_hash, cycle)

        # Signal study display: new experiment starting (Docker steps)
        if self._progress:
            from llenergymeasure.utils.formatting import format_experiment_header

            runner_info: dict[str, str | None] = {
                "mode": "docker",
                "source": spec.source,
                "image": image,
                "image_source": spec.image_source,
            }
            self._progress.begin_experiment(
                index,
                format_experiment_header(config),
                list(STEPS_DOCKER),
                runner_info=runner_info,
            )
            # Host-side preflight doesn't run in Docker path — mark as skipped
            self._progress.on_step_skip("preflight", "Docker path")

        exp_start = time.monotonic()

        result: Any
        try:
            # Pass study progress as step callback — DockerRunner calls on_step_*
            result = docker_runner.run(
                config,
                progress=self._progress,
                save_timeseries=self.study.output.save_timeseries,
            )
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

        # Resolve the timeseries temp dir that DockerRunner created for the
        # rescued parquet (now copied into the study dir by _save_and_record).
        docker_ts_dir = _resolve_ts_source_dir(result, spec, None)

        exp_elapsed = time.monotonic() - exp_start
        self._handle_result(
            result, config_hash, cycle, index, exp_elapsed, ts_source_dir=docker_ts_dir
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
        prefix = f"{config_hash[:8]}_cycle{cycle}"

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
