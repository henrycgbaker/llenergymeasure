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
import tempfile
from pathlib import Path
from typing import Any

from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerTimeoutError,
    translate_docker_error,
)

__all__ = ["DockerRunner"]

logger = logging.getLogger(__name__)


class DockerRunner:
    """Dispatches a single experiment to an ephemeral Docker container.

    Lifecycle:
        1. Create temp exchange dir (``tempfile.mkdtemp(prefix='llem-')``)
        2. Write ExperimentConfig as JSON to ``{config_hash}_config.json``
        3. ``docker run --rm --gpus all -v {exchange_dir}:/run/llem``
               ``-e LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json``
               ``--shm-size 8g {image}``
               ``python -m llenergymeasure.infra.container_entrypoint``
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
        timeout: int | None = None,
        source: str = "unknown",
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.source = source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: Any) -> Any:
        """Run an experiment inside an ephemeral Docker container.

        Args:
            config: ExperimentConfig to dispatch.

        Returns:
            ExperimentResult on success, or a dict error payload if the container
            wrote an error JSON (same format as StudyRunner worker errors).

        Raises:
            DockerTimeoutError:    Container exceeded ``self.timeout`` seconds.
            DockerImagePullError:  Image not found or could not be pulled.
            DockerGPUAccessError:  NVIDIA Container Toolkit misconfigured.
            DockerOOMError:        Container ran out of memory.
            DockerPermissionError: Permission denied on Docker socket.
            DockerContainerError:  Generic container failure (non-zero exit).
        """
        # Lazy import to avoid heavy domain imports at module load time
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        config_hash = compute_measurement_config_hash(config)
        exchange_dir = Path(tempfile.mkdtemp(prefix="llem-"))

        try:
            # --- Write config JSON ---
            config_path = exchange_dir / f"{config_hash}_config.json"
            config_path.write_text(config.model_dump_json(), encoding="utf-8")

            # --- Build and execute docker command ---
            cmd = self._build_docker_cmd(config_hash, str(exchange_dir))
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

            # --- Handle non-zero exit ---
            if proc.returncode != 0:
                logger.debug(
                    "Container failed (exit %d). Debug artifacts at %s",
                    proc.returncode,
                    exchange_dir,
                )
                error = translate_docker_error(proc.returncode, proc.stderr, self.image)
                # Do NOT clean up — preserve for debugging
                exchange_dir = None  # type: ignore[assignment]
                raise error

            # --- Read result ---
            result = self._read_result(exchange_dir, config_hash)

            # --- Success: clean up ---
            self._cleanup_exchange_dir(exchange_dir)
            exchange_dir = None  # type: ignore[assignment]

            # Error payload dicts ({type, message, traceback}) are returned as-is —
            # they have no effective_config to annotate with runner metadata.
            if isinstance(result, dict):
                return result

            return self._inject_runner_metadata(result)

        finally:
            # Exchange dir is set to None when we've handed off or already cleaned up.
            # If it's still set here, an unexpected exception occurred — preserve for debugging.
            if exchange_dir is not None:
                logger.debug("Preserving exchange dir for debugging: %s", exchange_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_docker_cmd(self, config_hash: str, exchange_dir: str) -> list[str]:
        """Build the ``docker run`` command list.

        Args:
            config_hash:  Hash prefix for config/result file names.
            exchange_dir: Host path of the temporary exchange directory.

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
            f"{exchange_dir}:/run/llem",
            "-e",
            f"LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json",
            "--shm-size",
            "8g",
        ]

        # Propagate HF_TOKEN for gated models inside the container
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

        cmd.extend(
            [
                self.image,
                "python",
                "-m",
                "llenergymeasure.infra.container_entrypoint",
            ]
        )

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

        return ExperimentResult.model_validate(raw)

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

    def _inject_runner_metadata(self, result: Any) -> Any:
        """Inject runner metadata into the result's effective_config.

        Since ExperimentResult is frozen, this creates a copy via model_copy().

        Args:
            result: ExperimentResult to annotate.

        Returns:
            A new ExperimentResult with runner_type, runner_image, and
            runner_source added to effective_config.
        """
        runner_metadata = {
            "runner_type": "docker",
            "runner_image": self.image,
            "runner_source": self.source,
        }
        updated_effective_config = {**result.effective_config, **runner_metadata}
        return result.model_copy(update={"effective_config": updated_effective_config})
