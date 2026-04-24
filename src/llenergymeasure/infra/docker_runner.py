"""DockerRunner — dispatches a single experiment to an ephemeral Docker container.

The DockerRunner manages the full container lifecycle:

1. Create a temporary exchange directory (``tempfile.mkdtemp(prefix='llem-')``)
2. Serialise ExperimentConfig to JSON in the exchange dir
3. Start ``docker run --rm --gpus all`` with the exchange dir mounted as /run/llem
4. Block until the container exits (subprocess.run)
5. Read the result JSON written by the container entrypoint
6. Clean up the exchange dir on success; preserve it on failure for debugging

This module is consumed by StudyRunner as the dispatch mechanism when
``runner=docker`` is resolved by runner_resolution.resolve_runner().
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import ProgressCallback

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import (
    CONTAINER_EXCHANGE_DIR,
    DOCKER_PULL_TIMEOUT,
    ENV_CONFIG_PATH,
    ENV_HF_TOKEN,
    ENV_OUTPUT_DIR,
    ENV_SAVE_TIMESERIES,
    TEMP_PREFIX_ENV_FILE,
    TEMP_PREFIX_EXCHANGE,
    TEMP_PREFIX_TIMESERIES,
    TIMEOUT_DOCKER_INSPECT,
    TIMEOUT_THREAD_JOIN,
    Engine,
)
from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerTimeoutError,
    capture_stderr_snippet,
    translate_docker_error,
)
from llenergymeasure.utils.exceptions import DockerError

__all__ = ["DockerRunner"]

logger = logging.getLogger(__name__)


@contextmanager
def _env_file(secrets: dict[str, str]) -> Iterator[Path | None]:
    """Write secrets to a temp env-file, yield path, delete on exit.

    Creates a temp file with mode 0600 (owner read-write only) via mkstemp.
    Yields None if secrets dict is empty.
    Cleanup is guaranteed via finally block (crash/SIGINT/normal exit).

    Args:
        secrets: Dict of env var name -> value pairs.

    Yields:
        Path to temp file, or None if no secrets.
    """
    if not secrets:
        yield None
        return

    fd, path_str = tempfile.mkstemp(prefix=TEMP_PREFIX_ENV_FILE, suffix=".env")
    path = Path(path_str)
    try:
        with os.fdopen(fd, "w") as f:
            for key, value in secrets.items():
                f.write(f"{key}={value}\n")
        yield path
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


def _mask_secrets(text: str, secrets: dict[str, str]) -> str:
    """Replace secret values with *** in a string."""
    for v in secrets.values():
        if v and len(v) > 4:
            text = text.replace(v, "***")
    return text


class DockerRunner:
    """Dispatches a single experiment to an ephemeral Docker container.

    Lifecycle:
        1. Create temp exchange dir (``tempfile.mkdtemp(prefix='llem-')``)
        2. Write ExperimentConfig as JSON to ``{config_hash}_config.json``
        3. ``docker run --rm --gpus all -v {exchange_dir}:/run/llem``
               ``-e LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json``
               ``--shm-size 8g {image}``
               ``python3 -m llenergymeasure.entrypoints.container``
        4. Read ``{config_hash}_result.json`` from exchange dir
        5. Clean up temp dir on success; preserve on failure with debug path logged

    Args:
        image:   Docker image to run (e.g. ``"ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12"``).
        timeout: Optional wall-clock timeout in seconds. None = no timeout.
        source:  Runner resolution source string (e.g. ``"yaml"``, ``"auto_detected"``).
                 Recorded in result effective_config for traceability.
    """

    def __init__(
        self,
        image: str,
        timeout: float | None = None,
        source: str = "unknown",
        extra_mounts: list[tuple[str, str]] | None = None,
        container_name: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.source = source
        self.extra_mounts = extra_mounts or []
        self._container_name = container_name
        self._labels = labels or {}

    @property
    def short_image(self) -> str:
        """Short image tag for display (e.g. 'transformers:v0.9.0')."""
        from llenergymeasure.utils.formatting import short_name

        return short_name(self.image)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: Any,
        progress: ProgressCallback | None = None,
        save_timeseries: bool = True,
        skip_image_check: bool = False,
    ) -> tuple[Any, Path | None]:
        """Run an experiment inside an ephemeral Docker container.

        When a progress callback is provided, streams container stdout line by
        line and forwards JSON progress events to the callback. Lines starting
        with ``{"event":`` are parsed as progress events; other lines are
        forwarded as container log output (visible at -v).

        Args:
            config: ExperimentConfig to dispatch.
            progress: Optional ProgressCallback for step-by-step progress reporting.

        Returns:
            Tuple of (result, ts_tmpdir):
            - result: ExperimentResult on success, or a dict error payload if the
              container wrote an error JSON.
            - ts_tmpdir: Path to temp dir containing rescued timeseries.parquet,
              or None. Caller is responsible for cleanup.

        Raises:
            DockerTimeoutError:    Container exceeded ``self.timeout`` seconds.
            DockerImagePullError:  Image not found or could not be pulled.
            DockerGPUAccessError:  NVIDIA Container Toolkit misconfigured.
            DockerOOMError:        Container ran out of memory.
            DockerPermissionError: Permission denied on Docker socket.
            DockerContainerError:  Generic container failure (non-zero exit).
        """
        # Lazy import to avoid heavy domain imports at module load time
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        exchange_dir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_EXCHANGE))

        # Collect secrets for env-file (never pass as CLI args)
        secrets: dict[str, str] = {}
        hf_token = os.environ.get(ENV_HF_TOKEN)
        if hf_token:
            secrets[ENV_HF_TOKEN] = hf_token

        _p = progress  # short alias

        try:
            # --- Ensure image is available (pull with visible output if needed) ---
            if not skip_image_check:
                self._ensure_image(progress=_p)

            # --- Write config JSON ---
            # Compute config_hash from the clean config (no output path mutation).
            # Output dir and save_timeseries are passed via env vars, not config.
            config_hash = compute_declared_config_hash(config)
            config_path = exchange_dir / f"{config_hash}_config.json"
            config_path.write_text(
                config.model_dump_json(),
                encoding="utf-8",
            )

            # Pass output params via env vars so the container entrypoint can
            # forward them to the harness as runtime params.
            secrets[ENV_OUTPUT_DIR] = CONTAINER_EXCHANGE_DIR
            secrets[ENV_SAVE_TIMESERIES] = "1" if save_timeseries else "0"

            # --- Build and execute docker command ---
            t0_container: float | None = None
            if _p:
                # Show short image tag (e.g. "transformers:v0.9.0") not the full registry path
                short_image = self.short_image
                _p.on_step_start("container_start", "Starting", short_image)
                t0_container = time.perf_counter()

            # Secrets are passed via a temp env-file (mode 0600) that is deleted after
            # the container exits — they never appear in the command argument list.
            with _env_file(secrets) as env_path:
                cmd = self._build_docker_cmd(
                    config, config_hash, str(exchange_dir), env_path=env_path
                )
                logger.debug("Running docker command: %s", _mask_secrets(str(cmd), secrets))

                # Use Popen streaming when progress callback is provided.
                # Container inner events (baseline, model, warmup, measure, save)
                # are forwarded as top-level steps for granular progress display.
                if _p:
                    returncode, stderr_text = self._run_container_streaming(
                        cmd,
                        _p,
                        _mask_secrets_fn=lambda t: _mask_secrets(t, secrets),
                        container_start_time=t0_container,
                    )
                else:
                    # Classic mode: blocking subprocess.run (backward compatible)
                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=self.timeout,
                        )
                    except subprocess.TimeoutExpired as exc:
                        logger.debug(
                            "Docker container timed out after %ss. Debug artifacts at %s",
                            self.timeout,
                            exchange_dir,
                        )
                        raise DockerTimeoutError(
                            message=f"Container timed out after {self.timeout}s.",
                            fix_suggestion="Increase timeout or reduce experiment size.",
                        ) from exc
                    returncode = proc.returncode
                    stderr_text = proc.stderr

            # --- Handle non-zero exit ---
            if returncode != 0:
                logger.debug(
                    "Container failed (exit %d). Debug artifacts at %s",
                    returncode,
                    exchange_dir,
                )
                # Persist container stderr to a log file in the exchange dir
                # so it survives for post-mortem debugging.
                container_log_path = exchange_dir / "container.log"
                try:
                    container_log_path.write_text(
                        stderr_text or "(no stderr captured)", encoding="utf-8"
                    )
                    logger.debug("Container log written to %s", container_log_path)
                except Exception as write_exc:
                    logger.warning("Failed to write container.log: %s", write_exc)

                # Prefer structured error JSON written by the container entrypoint
                # over Docker's stderr, which can contain misleading daemon messages.
                error_json_path = exchange_dir / f"{config_hash}_error.json"
                error: DockerError
                if error_json_path.exists():
                    payload = json.loads(error_json_path.read_text(encoding="utf-8"))
                    error = DockerContainerError(
                        message=f"{payload.get('type', 'UnknownError')}: {payload.get('message', '')}",
                        fix_suggestion="Check the error traceback in the error JSON for details.",
                        stderr_snippet=capture_stderr_snippet(stderr_text) if stderr_text else None,
                    )
                    error.error_payload = payload
                else:
                    error = translate_docker_error(returncode, stderr_text, self.image)

                error.exchange_dir = str(exchange_dir)
                # Do NOT clean up — preserve for debugging
                exchange_dir = None  # type: ignore[assignment]
                raise error

            # --- Read result ---
            result = self._read_result(exchange_dir, config_hash)

            # --- Rescue timeseries parquet before cleanup ---
            # The harness inside the container wrote timeseries.parquet to
            # /run/llem (= exchange_dir on host). Move it to a temp dir so
            # the caller can copy it into the study directory.
            ts_parquet = exchange_dir / "timeseries.parquet"
            ts_tmpdir: Path | None = None
            if ts_parquet.exists() and not isinstance(result, dict):
                ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES))
                shutil.move(str(ts_parquet), str(ts_tmpdir / "timeseries.parquet"))

            # --- Success: clean up ---
            self._cleanup_exchange_dir(exchange_dir)
            exchange_dir = None  # type: ignore[assignment]

            # Error payload dicts ({type, message, traceback}) are returned as-is.
            if isinstance(result, dict):
                return result, None

            return result, ts_tmpdir

        finally:
            # Exchange dir is set to None when we've handed off or already cleaned up.
            # If it's still set here, an unexpected exception occurred — preserve for debugging.
            if exchange_dir is not None:
                logger.debug("Preserving exchange dir for debugging: %s", exchange_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_image(self, progress: ProgressCallback | None = None) -> None:
        """Check if the Docker image exists locally; pull with visible output if not.

        Always emits an ``image_check`` step so the user sees the cache lookup.
        If the image is not cached, emits a separate ``pull`` step.
        Substeps report image metadata (ID, size, age) for provenance visibility.
        """
        short_image = self.short_image

        if progress:
            progress.on_step_start("image_check", "Inspecting", short_image)
        t0 = time.perf_counter()

        check = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True,
            timeout=TIMEOUT_DOCKER_INSPECT,
        )
        if check.returncode == 0:
            if progress:
                progress.on_step_update("image_check", f"{short_image} (cached)")
                progress.on_step_done("image_check", time.perf_counter() - t0)
                progress.on_step_skip("pull", "cached")
            return

        if progress:
            progress.on_step_done("image_check", time.perf_counter() - t0)

        # Image not cached — pull it
        if progress:
            progress.on_step_start("pull", "Pulling", self.image)
        t0_pull = time.perf_counter()

        print(f"Pulling image: {self.image}", file=sys.stderr)
        try:
            pull = subprocess.run(
                ["docker", "pull", self.image],
                stdout=sys.stderr,
                stderr=sys.stderr,
                timeout=DOCKER_PULL_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            if progress:
                progress.on_step_done("pull", time.perf_counter() - t0_pull)
            from llenergymeasure.infra.docker_errors import DockerImagePullError

            raise DockerImagePullError(
                message=f"Image pull timed out after {DOCKER_PULL_TIMEOUT}s: {self.image}",
                fix_suggestion=f"Pull manually: docker pull {self.image}",
            ) from exc
        if pull.returncode != 0:
            if progress:
                progress.on_step_done("pull", time.perf_counter() - t0_pull)
            from llenergymeasure.infra.docker_errors import DockerImagePullError

            raise DockerImagePullError(
                message=f"Image not found or could not be pulled: {self.image}",
                fix_suggestion=f"docker pull {self.image}",
            )

        if progress:
            progress.on_step_done("pull", time.perf_counter() - t0_pull)

    def _run_container_streaming(
        self,
        cmd: list[str],
        progress: ProgressCallback | None = None,
        _mask_secrets_fn: Callable[[str], str] | None = None,
        container_start_time: float | None = None,
    ) -> tuple[int, str]:
        """Run container with Popen, streaming stdout for progress events.

        Container inner events (step_start, step_update, step_done) are
        forwarded as top-level progress steps so the CLI can display each
        measurement phase individually (Docker BuildKit-style granularity).

        The ``container_start`` step (started by the caller) is ended when
        the first inner event arrives, capturing the container boot time.

        Args:
            cmd: Docker command list.
            progress: Optional ProgressCallback.
            _mask_secrets_fn: Optional callable to mask secrets in log output.
            container_start_time: perf_counter timestamp of container_start step.

        Returns:
            Tuple of (returncode, stderr_text).
        """
        stderr_lines: list[str] = []
        # Keywords in container stderr that indicate meaningful activity.
        # When no JSON progress events arrive (old images), surface these
        # as on_step_update to show the container is alive and working.
        _ACTIVITY_KEYWORDS = (
            "loading",
            "downloading",
            "measuring",
            "warmup",
            "warming",
            "inference",
            "saving",
            "running",
            "model",
            "tokenizer",
        )

        def _read_stderr(pipe: Any) -> None:
            """Read stderr in a background thread to prevent blocking.

            For old images that don't emit JSON progress events, surfaces
            interesting log lines as step updates on container_start.
            """
            for line in pipe:
                stderr_lines.append(line)
                stripped = line.strip()
                logger.debug("container stderr: %s", stripped)
                # Surface activity from old images as step updates
                if (
                    progress is not None
                    and not container_start_done_event.is_set()
                    and stripped
                    and any(kw in stripped.lower() for kw in _ACTIVITY_KEYWORDS)
                ):
                    # Truncate long log lines and strip log prefix (e.g. "INFO:root:")
                    display_text = stripped
                    if ":" in display_text and display_text.split(":")[0].isupper():
                        display_text = display_text.split(":", 2)[-1].strip()
                    progress.on_step_update("container_start", display_text[:50])
            pipe.close()

        # Thread-safe flag shared with stderr thread to track if JSON events arrived
        container_start_done_event = threading.Event()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise DockerContainerError(
                message=f"Failed to start docker process: {exc}",
                fix_suggestion="Is Docker installed and running?",
            ) from exc

        # Read stderr in background thread to avoid deadlock
        stderr_thread = threading.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True)
        stderr_thread.start()

        # Stream stdout line by line — forward inner events as top-level steps.
        # The container's "preflight" step is translated to "container_preflight"
        # to avoid collision with the host-level preflight.
        container_start_done = False
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.strip()
            if stripped.startswith('{"event":') and progress is not None:
                try:
                    event = json.loads(stripped)
                    event_type = event.get("event")
                    step = event.get("step", "")

                    # End "container_start" step on first inner event (boot time)
                    if not container_start_done and container_start_time is not None:
                        container_start_done = True
                        container_start_done_event.set()
                        progress.on_step_done(
                            "container_start", time.perf_counter() - container_start_time
                        )

                    # Translate container's "preflight" to avoid host collision
                    if step == "preflight":
                        step = "container_preflight"

                    # Forward inner events as top-level steps
                    if event_type == "step_start":
                        progress.on_step_start(
                            step,
                            event.get("description", ""),
                            event.get("detail", ""),
                        )
                    elif event_type == "step_update":
                        progress.on_step_update(step, event.get("detail", ""))
                    elif event_type == "step_done":
                        progress.on_step_done(step, event.get("elapsed_sec", 0.0))
                    elif event_type == "step_skip":
                        progress.on_step_skip(step, event.get("reason", ""))
                    elif event_type == "substep":
                        progress.on_substep(
                            step,
                            event.get("text", ""),
                            event.get("elapsed_sec", 0.0),
                        )
                    elif event_type == "substep_start":
                        progress.on_substep_start(step, event.get("text", ""))
                    elif event_type == "substep_done":
                        progress.on_substep_done(
                            step,
                            event.get("text"),
                            event.get("elapsed_sec"),
                        )
                except (json.JSONDecodeError, KeyError):
                    logger.debug("Unparseable progress line: %s", stripped)
            else:
                # Non-progress line — log as container output
                if stripped:
                    masked = _mask_secrets_fn(stripped) if _mask_secrets_fn else stripped
                    logger.debug("container stdout: %s", masked)

        proc.stdout.close()

        # If no inner events arrived, end container_start now (old images)
        if not container_start_done and container_start_time is not None and progress is not None:
            progress.on_step_done("container_start", time.perf_counter() - container_start_time)

        # Wait for process to finish
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise DockerTimeoutError(
                message=f"Container timed out after {self.timeout}s.",
                fix_suggestion="Increase timeout or reduce experiment size.",
            ) from exc

        stderr_thread.join(timeout=TIMEOUT_THREAD_JOIN)
        stderr_text = "".join(stderr_lines)

        return proc.returncode, stderr_text

    def _build_docker_cmd(
        self,
        config: Any,
        config_hash: str,
        exchange_dir: str,
        env_path: Path | None = None,
    ) -> list[str]:
        """Build the ``docker run`` command list.

        For TRT-LLM tensor parallelism (tensor_parallel_size > 1), ``mpirun -n {n}
        --allow-run-as-root`` is injected between the image name and ``python3``.
        MPI worker ranks re-import the module but do not call ``main()`` because
        ``container_entrypoint.py`` is guarded by ``if __name__ == "__main__"``.

        Args:
            config:       ExperimentConfig for the current experiment. Used to
                          detect TRT-LLM engine and read ``tensorrt.tensor_parallel_size``.
            config_hash:  Hash prefix for config/result file names.
            exchange_dir: Host path of the temporary exchange directory.
            env_path:     Path to a temp env-file (written by ``_env_file``), or None.
                          When set, ``--env-file <path>`` is added to the command.
                          Secrets (e.g. HF_TOKEN) are never passed as ``-e KEY=VALUE``
                          arguments to avoid exposure in ``/proc/<pid>/cmdline``.

        Returns:
            List of strings suitable for ``subprocess.run``.
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{exchange_dir}:{CONTAINER_EXCHANGE_DIR}",
            "-e",
            f"{ENV_CONFIG_PATH}={CONTAINER_EXCHANGE_DIR}/{config_hash}_config.json",
            "--shm-size",
            "8g",
        ]

        # Propagate secrets via --env-file (never as -e KEY=VALUE CLI args)
        if env_path is not None:
            cmd.extend(["--env-file", str(env_path)])

        # TRT-LLM engine cache: persist compiled engines across ephemeral containers
        if config.engine == Engine.TENSORRT:
            cache_host = str(Path.home() / ".cache" / "trt-llm")
            cache_container = "/root/.cache/trt-llm"
            # Only add if not already in extra_mounts (user may override path)
            if not any(cp == cache_container for _, cp in self.extra_mounts):
                cmd.extend(["-v", f"{cache_host}:{cache_container}"])

        # Extra volume mounts (engine cache, model cache, etc.)
        for host_path, container_path in self.extra_mounts:
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Container name and labels for lifecycle management (cleanup, reaper).
        # These must appear before the image name in the docker run command.
        if self._container_name:
            cmd.extend(["--name", self._container_name])
        for key, value in self._labels.items():
            cmd.extend(["--label", f"{key}={value}"])

        # Determine TRT-LLM tensor parallel size for MPI injection
        tp_size = None
        if config.engine == "tensorrt" and config.tensorrt is not None:
            tp_size = config.tensorrt.tensor_parallel_size

        cmd.append(self.image)

        # Inject mpirun for TRT-LLM tensor parallelism > 1.
        # MPI workers re-import the module but don't hit the __main__ guard,
        # so only rank 0 runs the experiment. See entrypoints/container.py.
        if tp_size is not None and tp_size > 1:
            cmd.extend(["mpirun", "-n", str(tp_size), "--allow-run-as-root"])

        cmd.extend(["python3", "-m", "llenergymeasure.entrypoints.container"])

        return cmd

    def _read_result(self, exchange_dir: Path, config_hash: str) -> Any:
        """Read and parse the result JSON written by the container.

        Args:
            exchange_dir: Host path of the temporary exchange directory.
            config_hash:  Hash prefix for the result file name.

        Returns:
            ExperimentResult if the file contains a valid result, or a dict
            error payload if the container wrote an error JSON.

        Raises:
            DockerContainerError: If the result file does not exist.
        """
        from llenergymeasure.domain.experiment import ExperimentResult

        result_path = exchange_dir / f"{config_hash}_result.json"
        if not result_path.exists():
            raise DockerContainerError(
                message=f"Container exited 0 but no result file found at {result_path}",
                fix_suggestion="Check container logs for errors during experiment execution.",
            )

        raw = json.loads(result_path.read_text(encoding="utf-8"))

        # Container may write an error payload even on exit 0 (defensive check).
        # Error payloads have "type" and "traceback" keys (mirror StudyRunner worker format).
        if isinstance(raw, dict) and "type" in raw and "traceback" in raw:
            return raw

        # Strip fields unknown to the host schema (container may run an older/newer
        # version that produces fields the host model doesn't expect).
        known_fields = set(ExperimentResult.model_fields)
        extra_keys = set(raw.keys()) - known_fields
        if extra_keys:
            for key in extra_keys:
                raw.pop(key)
            logger.debug("Stripped unknown fields from container result: %s", extra_keys)

        result = ExperimentResult.model_validate(raw)

        container_version = result.llenergymeasure_version
        if container_version is None or container_version != __version__:
            logger.warning(
                "Container result version %s differs from host %s — rebuild Docker images",
                container_version,
                __version__,
            )

        return result

    def _cleanup_exchange_dir(self, exchange_dir: Path) -> None:
        """Remove the temporary exchange directory.

        Logs a warning on failure but never raises — cleanup must not mask
        real errors from the caller.

        Args:
            exchange_dir: Path to remove.
        """
        try:
            shutil.rmtree(exchange_dir)
        except Exception as exc:
            logger.warning("Could not remove exchange dir %s: %s", exchange_dir, exc)

    def get_runner_metadata(self) -> dict[str, str]:
        """Return runner metadata dict for inclusion in effective_config sidecar."""
        return {
            "runner_type": "docker",
            "runner_image": self.image,
            "runner_source": self.source,
        }
