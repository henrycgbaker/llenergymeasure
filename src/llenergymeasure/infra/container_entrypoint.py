"""Container-side entry point for running an experiment inside Docker.

This module is invoked by the DockerRunner (Plan 02) inside the container.
It reads an ExperimentConfig from a JSON file on the shared volume, runs the
experiment via the library API (not CLI re-entry — DOCK-11), and writes the
ExperimentResult back to the same volume for the host to collect.

Volume layout (managed by DockerRunner on the host):
    /run/llem/{config_hash}_config.json   — written by host before container start
    /run/llem/{config_hash}_result.json   — written here on success
    /run/llem/{config_hash}_error.json    — written here on failure

Environment variable:
    LLEM_CONFIG_PATH  — absolute path to config JSON inside the container
                        (e.g. /run/llem/abc123_config.json)

Invocation::

    docker run ... -e LLEM_CONFIG_PATH=/run/llem/abc123_config.json {image}

or directly for testing::

    python -m llenergymeasure.infra.container_entrypoint
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

__all__ = ["main", "run_container_experiment"]


def run_container_experiment(config_path: Path, result_dir: Path) -> Path:
    """Read config JSON, run experiment via library API, write result JSON.

    Uses the same execution path as the StudyRunner worker
    (``core.backends.get_backend`` + ``orchestration.preflight.run_preflight``)
    so measurement behaviour is identical whether the experiment runs locally
    or inside a container.

    Args:
        config_path: Path to the config JSON file (inside the container).
        result_dir: Directory in which to write the result JSON.

    Returns:
        Path to the written result JSON file.

    Raises:
        Any exception from pre-flight or backend execution propagates up so
        ``main()`` can catch it and write an error payload.
    """
    # Lazy imports — only needed at runtime, not import time
    from llenergymeasure.api._gpu import _resolve_gpu_indices
    from llenergymeasure.api.preflight import run_preflight
    from llenergymeasure.backends import get_backend
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.experiment import compute_measurement_config_hash
    from llenergymeasure.harness import MeasurementHarness

    # --- Deserialise config ---
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(raw)

    # --- Stable file name derived from config content ---
    config_hash = compute_measurement_config_hash(config)

    # --- Run experiment via library API (not CLI) ---
    run_preflight(config)
    backend = get_backend(config.backend)
    harness = MeasurementHarness()
    gpu_indices = _resolve_gpu_indices(config)
    result = harness.run(backend, config, gpu_indices=gpu_indices)

    # --- Write result ---
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{config_hash}_result.json"
    result_path.write_text(result.model_dump_json(), encoding="utf-8")

    return result_path


def main() -> None:
    """Entry point invoked when the container starts.

    Reads LLEM_CONFIG_PATH from the environment, runs the experiment, and
    writes either a result JSON or an error JSON to the same directory.

    Error JSON format (mirrors StudyRunner worker error payloads)::

        {
            "type": "ExceptionClassName",
            "message": "human-readable description",
            "traceback": "full traceback string"
        }
    """
    config_path_env = os.environ.get("LLEM_CONFIG_PATH")
    if not config_path_env:
        raise RuntimeError(
            "LLEM_CONFIG_PATH environment variable is not set. "
            "The DockerRunner must set this before starting the container."
        )

    config_path = Path(config_path_env)
    result_dir = config_path.parent  # same volume mount (e.g. /run/llem)

    try:
        result_path = run_container_experiment(config_path, result_dir)
        print(f"[llem] Experiment complete. Result written to {result_path}")

    except Exception as exc:
        # Derive the error file name from the config file stem so the host can
        # correlate it even without knowing the config hash.
        stem = config_path.stem  # e.g. "abc123_config"
        # Replace trailing "_config" → "_error" for clean naming
        if stem.endswith("_config"):
            error_stem = stem[: -len("_config")] + "_error"
        else:
            error_stem = stem + "_error"

        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        error_path = result_dir / f"{error_stem}.json"
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
            print(f"[llem] Experiment failed. Error written to {error_path}")
        except Exception as write_exc:  # pragma: no cover
            # Last resort: stderr only
            print(f"[llem] CRITICAL: could not write error payload: {write_exc}")

        raise


if __name__ == "__main__":
    main()
