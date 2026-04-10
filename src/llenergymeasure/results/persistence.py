"""v3.0 results persistence — save, load, atomic writes.

Handles directory lifecycle, collision avoidance,
JSON serialisation (primary), and Parquet sidecar management.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import ExperimentResult

logger = logging.getLogger(__name__)


def _experiment_dir_name(
    result: ExperimentResult,
    *,
    experiment_index: int | None = None,
    cycle: int = 1,
) -> str:
    """Generate a human-readable directory name for an experiment result.

    Format: ``[{index:03d}_]c{cycle}_{model_short}-{backend}_{hash[:8]}``

    When ``experiment_index`` is provided (study context), the directory is
    prefixed with a zero-padded index for natural sort ordering.

    Examples:
        ``001_c1_Qwen2.5-0.5B-pytorch_abcdef01``
        ``c1_gpt2-vllm_fedcba98``  (single experiment, no index)
    """
    from llenergymeasure.utils.formatting import model_short_name

    model_short = model_short_name(result.model_name)
    backend = result.backend
    config_hash = result.measurement_config_hash[:8]

    # Build slug: model_short-backend
    slug = f"{model_short}-{backend}"
    # Sanitise for filesystem: replace spaces, slashes, special chars
    slug = slug.replace(" ", "_").replace("/", "-").replace(":", "-")
    # Truncate overly long slugs (filesystem limits)
    if len(slug) > 120:
        slug = slug[:120]

    if experiment_index is not None:
        return f"{experiment_index:03d}_c{cycle}_{slug}_{config_hash}"
    return f"c{cycle}_{slug}_{config_hash}"


def _find_collision_free_dir(base: Path) -> Path:
    """Return base or base_1, base_2, etc. — never overwrites.

    Creates the directory atomically to avoid race conditions.
    """
    target = base
    counter = 0
    while target.exists():
        counter += 1
        target = Path(f"{base}_{counter}")
    target.mkdir(parents=True)
    return target


def _atomic_write(content: str, path: Path) -> None:
    """Write content to path atomically via temp file + os.replace().

    Uses POSIX rename semantics — atomic on same filesystem.
    Calls fsync before replace to ensure durability on power loss.
    Cleans up temp file on failure.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def save_environment(
    snapshot: EnvironmentSnapshot,
    experiment_id: str,
    measurement_config_hash: str,
    experiment_dir: Path,
) -> Path:
    """Write per-experiment environment.json sidecar.

    Contains hardware/runtime metadata for the experiment. Software package
    listings live in the study-level environment.json instead.

    Args:
        snapshot: EnvironmentSnapshot with hardware/runtime metadata.
        experiment_id: Unique experiment identifier.
        measurement_config_hash: Config hash for orphan attribution.
        experiment_dir: Experiment result directory (must already exist).

    Returns:
        Path to the written environment.json file.
    """
    env_data: dict[str, object] = {
        "experiment_id": experiment_id,
        "measurement_config_hash": measurement_config_hash,
    }
    snapshot_dict = snapshot.model_dump()
    env_data["hardware"] = snapshot_dict["hardware"]
    env_data["python_version"] = snapshot_dict["python_version"]
    env_data["tool_version"] = snapshot_dict["tool_version"]
    env_data["cuda_version"] = snapshot_dict.get("cuda_version")
    env_data["cuda_version_source"] = snapshot_dict.get("cuda_version_source")

    path = experiment_dir / "environment.json"
    _atomic_write(json.dumps(env_data, indent=2, default=str), path)
    logger.debug("Saved environment to %s", path)
    return path


def save_result(
    result: ExperimentResult,
    output_dir: Path,
    timeseries_source: Path | None = None,
    experiment_index: int | None = None,
    cycle: int = 1,
    resolution_log: dict[str, object] | None = None,
) -> Path:
    """Save ExperimentResult to a collision-safe subdirectory of output_dir.

    Creates: ``{output_dir}/[{index}_]c{cycle}_{model}-{backend}_{hash}/result.json``
    Also writes ``_resolution.json`` sidecar when ``resolution_log`` is provided.
    If timeseries_source provided: copies to ``{dir}/timeseries.parquet``.

    Args:
        result: The experiment result to persist.
        output_dir: Parent directory. Created if missing.
        timeseries_source: Optional path to existing .parquet file to copy in.
        experiment_index: Optional 1-based experiment index for directory prefix
            (used in study context for natural sort ordering).
        cycle: Cycle number (1-based). Embedded in directory name.
        resolution_log: Per-experiment config resolution log showing which fields
            were overridden and why (CLI flag, sweep, YAML).

    Returns:
        Path to the result.json file (usable with load_result() directly).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_name = _experiment_dir_name(result, experiment_index=experiment_index, cycle=cycle)
    base_dir = output_dir / dir_name
    target_dir = _find_collision_free_dir(base_dir)

    result_path = target_dir / "result.json"
    _atomic_write(result.model_dump_json(indent=2), result_path)
    logger.debug("Saved result to %s", result_path)

    # Write _resolution.json sidecar
    if resolution_log:
        res_path = target_dir / "_resolution.json"
        _atomic_write(json.dumps(resolution_log, indent=2, default=str), res_path)
        logger.debug("Saved resolution log to %s", res_path)

    if timeseries_source is not None:
        timeseries_source = Path(timeseries_source)
        if timeseries_source.exists():
            dest = target_dir / "timeseries.parquet"
            shutil.copy2(timeseries_source, dest)
            logger.debug("Copied timeseries sidecar to %s", dest)
        else:
            logger.warning("timeseries_source %s does not exist — skipping copy", timeseries_source)

    return result_path


def load_result(path: Path) -> ExperimentResult:
    """Load ExperimentResult from a result.json path.

    Auto-discovers timeseries.parquet and environment.json sidecars
    in the same directory. If the result references a timeseries sidecar
    but the file is missing, loads successfully and emits a UserWarning
    (graceful degradation).

    Args:
        path: Path to result.json (as returned by save_result()).

    Returns:
        ExperimentResult loaded from disk.
    """
    from llenergymeasure.domain.experiment import ExperimentResult

    path = Path(path)
    content = path.read_text(encoding="utf-8")
    result = ExperimentResult.model_validate_json(content)

    sidecar = path.parent / "timeseries.parquet"
    if result.timeseries is not None and not sidecar.exists():
        warnings.warn(
            f"Timeseries sidecar missing at {sidecar}. "
            "result.timeseries field preserved but file is not present.",
            UserWarning,
            stacklevel=2,
        )

    return result
