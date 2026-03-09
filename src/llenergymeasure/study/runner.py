"""StudyRunner — subprocess dispatch core for experiment isolation.

Each experiment in a study runs in a freshly spawned subprocess with a clean CUDA
context. Results travel parent←child via multiprocessing.Pipe. The parent survives
experiment failures, timeouts, and SIGINT without data corruption.

Key design decisions (locked in .product/decisions/experiment-isolation.md):
- spawn context: CUDA-safe; fork causes silent CUDA corruption (CP-1)
- daemon=False: clean CUDA teardown if parent exits unexpectedly (CP-4)
- Pipe-only IPC: ExperimentResult fits in Pipe buffer for M2 experiment sizes
- SIGKILL on timeout: SIGTERM may be ignored by hung CUDA operations
"""

from __future__ import annotations

import contextlib
import multiprocessing
import signal
import sys
import threading
import traceback
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.study.gaps import run_gap

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter

__all__ = ["StudyRunner", "_calculate_timeout", "_run_experiment_worker", "_save_and_record"]


# =============================================================================
# Module-level helpers
# =============================================================================


def _calculate_timeout(config: ExperimentConfig) -> int:
    """Generous timeout heuristic: 2 seconds per prompt, minimum 10 minutes.

    No model-size scaling — keep it simple for M2.
    """
    return max(config.n * 2, 600)


def _save_and_record(
    result: Any,
    study_dir: Path,
    manifest: ManifestWriter,
    config_hash: str,
    cycle: int,
    result_files: list[str],
) -> None:
    """Save result to disk and update manifest. Appends result path to result_files.

    On save failure, marks the experiment as completed with empty path.
    """
    try:
        from llenergymeasure.results.persistence import save_result

        result_path = save_result(result, study_dir)
        result_files.append(str(result_path))
        rel_path = str(result_path.relative_to(study_dir))
        manifest.mark_completed(config_hash, cycle, rel_path)
    except Exception:
        manifest.mark_completed(config_hash, cycle, result_file="")


# =============================================================================
# Worker function (runs inside child process)
# =============================================================================


def _run_experiment_worker(
    config: ExperimentConfig,
    conn: Any,  # multiprocessing.Connection (child end)
    progress_queue: Any,  # multiprocessing.Queue
    snapshot: EnvironmentSnapshot | None = None,
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
    """
    # parent owns SIGINT; child ignores it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        progress_queue.put({"event": "started", "config_hash": config_hash})

        # Run the actual experiment in-process (within the spawned subprocess)
        from llenergymeasure.core.backends import get_backend
        from llenergymeasure.core.harness import MeasurementHarness
        from llenergymeasure.orchestration.preflight import run_preflight

        # Pre-flight inside subprocess: CUDA availability must be checked in the
        # process that will use the GPU.
        run_preflight(config)

        backend = get_backend(config.backend)
        harness = MeasurementHarness()
        result = harness.run(backend, config, snapshot=snapshot)

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
    index: int,
    total: int,
    config: Any,  # ExperimentConfig
) -> None:
    """Consume progress events from the queue and forward to display.

    Runs as a daemon thread in the parent process. Receives events from the
    child subprocess via multiprocessing.Queue and calls print_study_progress()
    for each meaningful event.
    """
    while True:
        event = q.get()
        if event is None:
            break

        if not isinstance(event, dict):
            continue

        event_type = event.get("event")
        if event_type == "started":
            from llenergymeasure.cli._display import print_study_progress

            print_study_progress(index, total, config, status="running")
        elif event_type == "completed":
            from llenergymeasure.cli._display import print_study_progress

            print_study_progress(index, total, config, status="completed")
        elif event_type == "failed":
            from llenergymeasure.cli._display import print_study_progress

            print_study_progress(index, total, config, status="failed")


# =============================================================================
# Result collection (parent, after p.join)
# =============================================================================


def _collect_result(
    p: Any,  # multiprocessing.Process
    parent_conn: Any,  # multiprocessing.Connection (parent end)
    config: ExperimentConfig,
    timeout: int,
) -> Any:
    """Inspect process outcome and return either a result or a failure dict.

    Called after p.join(timeout=...) has returned.

    Returns:
        ExperimentResult on success, dict with keys (type, message) on failure.
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(config)

    if p.is_alive():
        # Timed out — kill with SIGKILL
        # SIGKILL: SIGTERM may be ignored by hung CUDA operations
        p.kill()
        p.join()
        return {
            "type": "TimeoutError",
            "message": f"Experiment exceeded timeout of {timeout}s and was killed.",
            "config_hash": config_hash,
        }

    if p.exitcode != 0:
        # Non-zero exit — try to read error payload from pipe
        if parent_conn.poll():
            try:
                payload = parent_conn.recv()
                if isinstance(payload, dict) and "type" in payload:
                    payload["config_hash"] = config_hash
                    return payload
            except Exception:
                pass

        return {
            "type": "ProcessCrash",
            "message": f"Subprocess exited with code {p.exitcode} and no error data in Pipe.",
            "config_hash": config_hash,
        }

    # Success path — read result from pipe
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
            # If payload is an error dict (exception in worker), treat as failure
            if isinstance(payload, dict) and "type" in payload and "traceback" in payload:
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
    ) -> None:
        self.study = study
        self.manifest = manifest_writer
        self.study_dir = study_dir
        self.result_files: list[str] = []
        # Pre-resolved runner specs per backend (None = all experiments use subprocess path)
        self._runner_specs = runner_specs
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
                    self._active_process.terminate()  # SIGTERM — gentle first attempt
            else:
                print("\nForce-killing experiment subprocess...")
                if self._active_process is not None and self._active_process.is_alive():
                    self._active_process.kill()  # SIGKILL

        original_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        try:
            results: list[Any] = []

            for i, config in enumerate(ordered):
                if self._interrupt_event.is_set():
                    break

                # Config gap: between every consecutive experiment pair
                if i > 0:
                    gap_secs = float(self.study.execution.experiment_gap_seconds or 0)
                    if gap_secs > 0:
                        run_gap(gap_secs, "Experiment gap", self._interrupt_event)
                        if self._interrupt_event.is_set():
                            break

                # Cycle gap: after every complete round of N unique configs
                if n_unique > 0 and i > 0 and i % n_unique == 0:
                    cycle_gap_secs = float(self.study.execution.cycle_gap_seconds or 0)
                    if cycle_gap_secs > 0:
                        run_gap(cycle_gap_secs, "Cycle gap", self._interrupt_event)
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
            from llenergymeasure.domain.environment import collect_environment_snapshot_async

            self._env_snapshot_future = collect_environment_snapshot_async()
        return self._env_snapshot_future.result(timeout=10)

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
        if spec is not None and spec.mode == "docker":
            return self._run_one_docker(
                config, spec, config_hash=config_hash, cycle=cycle, index=index, total=total
            )

        timeout = _calculate_timeout(config)

        # Resolve cached snapshot in parent — serialised to subprocess via Pipe
        snapshot = self._get_env_snapshot()

        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
        progress_queue = mp_ctx.Queue()

        p = mp_ctx.Process(
            target=_run_experiment_worker,
            args=(config, child_conn, progress_queue, snapshot),
            daemon=False,  # daemon=False: clean CUDA teardown if parent exits unexpectedly
        )

        consumer = threading.Thread(
            target=_consume_progress_events,
            args=(progress_queue, index, total, config),
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
        p.join(timeout=timeout)

        # SIGINT was received during join: SIGTERM was already sent by handler.
        # Give child 2s grace for clean CUDA teardown, then SIGKILL.
        if self._interrupt_event.is_set() and p.is_alive():
            p.join(timeout=2)  # 2s grace after SIGTERM
            if p.is_alive():
                p.kill()
                p.join()

        self._active_process = None

        # Sentinel stops consumer thread — covers SIGKILL path too
        progress_queue.put(None)
        consumer.join()

        result = _collect_result(p, parent_conn, config, timeout)

        # Update manifest based on outcome
        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_message = result.get("message", "")
            self.manifest.mark_failed(config_hash, cycle, error_type, error_message)
        else:
            # Save result to study directory and track path (RES-15)
            _save_and_record(
                result, self.study_dir, self.manifest, config_hash, cycle, self.result_files
            )

        return result

    def _run_one_docker(
        self,
        config: ExperimentConfig,
        spec: RunnerSpec,
        *,
        config_hash: str,
        cycle: int,
        index: int,
        total: int,
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
            total:       Total experiment count for progress display.

        Returns:
            ExperimentResult on success, or a failure dict on error.
        """
        from llenergymeasure.cli._display import print_study_progress
        from llenergymeasure.exceptions import DockerError
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

        # Resolve image — use explicit image from spec or fall back to built-in default
        image = spec.image if spec.image is not None else get_default_image(config.backend)

        docker_runner = DockerRunner(
            image=image,
            timeout=_calculate_timeout(config),
            source=spec.source,
        )

        # Pre-dispatch GPU memory residual check (same as local path)
        check_gpu_memory_residual()

        self.manifest.mark_running(config_hash, cycle)
        print_study_progress(index, total, config, status="running")

        result: Any
        try:
            result = docker_runner.run(config)
        except DockerError as exc:
            result = {
                "type": type(exc).__name__,
                "message": str(exc),
                "config_hash": config_hash,
            }

        # Update manifest based on outcome
        if isinstance(result, dict) and "type" in result:
            error_type = result.get("type", "UnknownError")
            error_message = result.get("message", "")
            self.manifest.mark_failed(config_hash, cycle, error_type, error_message)
            print_study_progress(index, total, config, status="failed")
        else:
            # Save result to study directory and track path (RES-15)
            _save_and_record(
                result, self.study_dir, self.manifest, config_hash, cycle, self.result_files
            )
            print_study_progress(index, total, config, status="completed")

        return result
