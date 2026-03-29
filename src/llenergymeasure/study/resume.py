"""Study resume capability — skip completed experiments and re-run the rest."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from llenergymeasure.utils.exceptions import StudyError

if TYPE_CHECKING:
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.study.manifest import ManifestWriter, StudyManifest

logger = logging.getLogger(__name__)

# Statuses that indicate a study can be resumed.
_RESUMABLE_STATUSES = {"interrupted", "failed", "circuit_breaker", "timed_out"}


def find_resumable_study(output_dir: Path) -> Path | None:
    """Search output_dir for the most recent resumable study directory.

    A study is resumable when its manifest.json status is one of:
    ``interrupted``, ``failed``, ``circuit_breaker``, ``timed_out``.

    Args:
        output_dir: Root directory to search (e.g. ``results/``).

    Returns:
        Path to the most recent resumable study directory, or None if none found.
    """
    candidates: list[tuple[datetime, Path]] = []

    if not output_dir.is_dir():
        return None

    for subdir in output_dir.iterdir():
        if not subdir.is_dir():
            continue
        manifest_path = subdir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text())
            status = data.get("status", "")
            if status not in _RESUMABLE_STATUSES:
                continue
            started_at_str = data.get("started_at")
            if started_at_str is None:
                continue
            # Parse ISO 8601 timestamp (with or without trailing Z)
            started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
            candidates.append((started_at, subdir))
        except Exception:
            # Best-effort: skip directories with unreadable/unparseable manifests
            continue

    if not candidates:
        return None

    # Return the directory with the most recent started_at timestamp
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_resume_state(
    study_dir: Path,
) -> tuple[StudyManifest, set[tuple[str, int]]]:
    """Load manifest from study_dir and build the skip-set of completed experiments.

    Args:
        study_dir: Directory containing manifest.json.

    Returns:
        A tuple of (manifest, skip_set) where skip_set contains
        ``(config_hash, cycle)`` pairs for experiments with status ``completed``.

    Raises:
        StudyError: If manifest.json is missing or cannot be parsed.
    """
    from llenergymeasure.study.manifest import StudyManifest

    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        raise StudyError(
            f"Cannot resume: manifest.json not found in {study_dir}. "
            "Is this a valid study directory?"
        )

    try:
        data = json.loads(manifest_path.read_text())
        manifest = StudyManifest.model_validate(data)
    except Exception as exc:
        raise StudyError(
            f"Cannot resume: failed to parse manifest.json in {study_dir}: {exc}"
        ) from exc

    skip_set: set[tuple[str, int]] = {
        (entry.config_hash, entry.cycle)
        for entry in manifest.experiments
        if entry.status == "completed"
    }

    logger.info(
        "Resume: loaded manifest from %s — %d completed, %d to re-run",
        study_dir,
        len(skip_set),
        len(manifest.experiments) - len(skip_set),
    )
    return manifest, skip_set


def validate_config_drift(manifest: StudyManifest, new_study: StudyConfig) -> None:
    """Raise StudyError if the study config has changed since the original run.

    Compares ``manifest.study_design_hash`` against ``new_study.study_design_hash``.
    A mismatch means the config was modified after the original run started.

    Args:
        manifest: Manifest loaded from the interrupted study directory.
        new_study: Freshly loaded StudyConfig to resume with.

    Raises:
        StudyError: On hash mismatch with a clear diff message.
    """
    old_hash = manifest.study_design_hash
    new_hash = new_study.study_design_hash or ""

    if old_hash != new_hash:
        raise StudyError(
            f"Config drift detected. "
            f"Original hash: {old_hash}, current hash: {new_hash}. "
            "The study config has changed since the original run. "
            "Cannot resume with a different config."
        )


def prepare_resume_manifest(study_dir: Path, manifest: StudyManifest) -> ManifestWriter:
    """Re-initialise a ManifestWriter that reuses the existing study directory.

    Resets all non-completed experiment entries to ``pending`` status and sets
    the overall manifest status back to ``running``. Writes the updated manifest
    to disk before returning the writer.

    Args:
        study_dir: Path to the existing study directory.
        manifest: StudyManifest loaded from the previous run.

    Returns:
        A ManifestWriter whose internal manifest reflects the reset state.
    """
    from datetime import timezone

    from llenergymeasure.study.manifest import ManifestWriter

    # Reset non-completed entries to pending
    reset_experiments = []
    for entry in manifest.experiments:
        if entry.status != "completed":
            reset_experiments.append(entry.model_copy(update={"status": "pending"}))
        else:
            reset_experiments.append(entry)

    # Reset overall study status to running; keep original started_at
    now = datetime.now(timezone.utc)
    updated_manifest = manifest.model_copy(
        update={
            "status": "running",
            "completed_at": None,
            # Recount pending: all non-completed
            "pending": sum(1 for e in reset_experiments if e.status in ("pending", "running")),
            "interrupted": 0,
            "failed": sum(1 for e in reset_experiments if e.status == "failed"),
            "skipped": sum(1 for e in reset_experiments if e.status == "skipped"),
            "experiments": reset_experiments,
        }
    )

    # Build a ManifestWriter that wraps the existing study_dir without creating a new dir
    writer = ManifestWriter.__new__(ManifestWriter)
    writer._study_dir = study_dir
    writer.path = study_dir / "manifest.json"
    writer.manifest = updated_manifest
    writer._write()

    logger.info(
        "Resume: prepared manifest in %s — %d pending, %d completed (skipping)",
        study_dir,
        updated_manifest.pending,
        updated_manifest.completed,
    )
    return writer
