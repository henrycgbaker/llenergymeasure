"""Study manifest — checkpoint model and atomic writer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from llenergymeasure.results.persistence import _atomic_write
from llenergymeasure.utils.exceptions import StudyError

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------


class ExperimentManifestEntry(BaseModel):
    """Checkpoint record for a single experiment + cycle execution."""

    model_config = {"extra": "forbid"}

    config_hash: str
    config_summary: str
    cycle: int
    status: Literal["pending", "running", "completed", "failed"]
    result_file: str | None = None
    log_file: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class StudyManifest(BaseModel):
    """In-progress checkpoint written after every experiment state transition.

    Distinct from StudyResult (the final return value of _run()).
    StudyManifest is always-on and provides the foundation for --resume support.
    """

    model_config = {"extra": "forbid"}

    schema_version: str = "2.0"
    study_name: str
    study_design_hash: str
    llenergymeasure_version: str
    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["running", "completed", "interrupted", "failed"] = Field(
        default="running",
        description="Overall study status. 'interrupted' = user Ctrl+C (not an error).",
    )
    total_experiments: int
    completed: int = 0
    failed: int = 0
    pending: int
    experiments: list[ExperimentManifestEntry]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_config_summary(experiment: ExperimentConfig) -> str:
    """Return a human-readable summary string for an experiment config.

    Delegates to ``format_experiment_header()`` for consistent naming across
    CLI display, manifest summaries, and directory names.

    Format: "model_short / backend / non_default_params..."
    """
    from llenergymeasure.utils.formatting import format_experiment_header

    return format_experiment_header(experiment)


def study_dir_name(name: str | None) -> str:
    """Return the study directory basename: ``{name}_{timestamp}``."""
    prefix = name if name else "study"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{prefix}_{timestamp}"


def create_study_dir(name: str | None, output_dir: Path) -> Path:
    """Create study output directory with {name}_{timestamp}/ layout.

    Args:
        name: Study name prefix. Uses "study" if None.
        output_dir: Parent directory to create the study directory in.

    Returns:
        Path to the newly created study directory.

    Raises:
        StudyError: If directory creation fails.
    """
    study_dir = output_dir / study_dir_name(name)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        study_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StudyError(f"Failed to create study directory: {exc}") from exc
    return study_dir


def experiment_result_filename(
    model: str,
    backend: str,
    precision: str,
    config_hash: str,
    extension: str = ".json",
) -> str:
    """Return flat filename for an experiment result file.

    Format: "{model_short}-{backend}_{hash[:8]}{extension}"
    model_short: last component after '/' (preserves casing).
    """
    from llenergymeasure.utils.formatting import model_short_name

    model_short = model_short_name(model)
    return f"{model_short}-{backend}_{config_hash[:8]}{extension}"


# ---------------------------------------------------------------------------
# ManifestWriter
# ---------------------------------------------------------------------------


class ManifestWriter:
    """Writes and maintains manifest.json in the study directory.

    Writes after every state transition (mark_running / mark_completed /
    mark_failed). Uses atomic os.replace() via _atomic_write.

    Write failures are logged as warnings — they never abort the study.
    Directory creation failure raises StudyError immediately (fast-fail).
    """

    def __init__(self, study: StudyConfig, study_dir: Path) -> None:
        self._study_dir = study_dir
        self.path = study_dir / "manifest.json"
        self.manifest = self._build_manifest(study)
        self._write()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mark_running(self, config_hash: str, cycle: int) -> None:
        """Mark a pending experiment as running."""
        entry = self._find(config_hash, cycle)
        entry.status = "running"
        entry.started_at = datetime.now(timezone.utc)
        self._recount()
        self._write()

    def mark_completed(self, config_hash: str, cycle: int, result_file: str) -> None:
        """Mark a running experiment as completed."""
        entry = self._find(config_hash, cycle)
        entry.status = "completed"
        entry.result_file = result_file
        entry.completed_at = datetime.now(timezone.utc)
        self._recount()
        self._write()

    def mark_failed(
        self,
        config_hash: str,
        cycle: int,
        error_type: str,
        error_message: str,
        log_file: str | None = None,
    ) -> None:
        """Mark a running experiment as failed."""
        entry = self._find(config_hash, cycle)
        entry.status = "failed"
        entry.error_type = error_type
        entry.error_message = error_message
        entry.log_file = log_file
        entry.completed_at = datetime.now(timezone.utc)
        self._recount()
        self._write()

    def mark_interrupted(self) -> None:
        """Set manifest status to 'interrupted'. Called on SIGINT before sys.exit(130)."""
        self.manifest = self.manifest.model_copy(update={"status": "interrupted"})
        self._write()

    def mark_study_completed(self) -> None:
        """Set manifest status to 'completed' and record completion time.

        Called by _run() after all experiments finish successfully (no SIGINT).
        Only reached on the success path — SIGINT triggers mark_interrupted() then
        sys.exit(130) before _run() returns.
        """
        self.manifest = self.manifest.model_copy(
            update={
                "status": "completed",
                "completed_at": datetime.now(timezone.utc),
            }
        )
        self._write()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find(self, config_hash: str, cycle: int) -> ExperimentManifestEntry:
        """Find entry by (config_hash, cycle). Raises KeyError if not found."""
        for entry in self.manifest.experiments:
            if entry.config_hash == config_hash and entry.cycle == cycle:
                return entry
        raise KeyError(f"No manifest entry for config_hash={config_hash!r}, cycle={cycle}")

    def _recount(self) -> None:
        """Recompute aggregate counters from entries."""
        completed = sum(1 for e in self.manifest.experiments if e.status == "completed")
        failed = sum(1 for e in self.manifest.experiments if e.status == "failed")
        pending = sum(1 for e in self.manifest.experiments if e.status in ("pending", "running"))
        self.manifest = self.manifest.model_copy(
            update={"completed": completed, "failed": failed, "pending": pending}
        )

    def _write(self) -> None:
        """Write manifest to disk atomically. Logs warning on failure."""
        try:
            _atomic_write(self.manifest.model_dump_json(indent=2), self.path)
        except Exception as exc:
            logger.warning("Failed to write manifest to %s: %s", self.path, exc)

    def _build_manifest(self, study: StudyConfig) -> StudyManifest:
        """Build initial StudyManifest from a StudyConfig."""
        from llenergymeasure._version import __version__

        entries = self._build_entries(study)
        return StudyManifest(
            study_name=study.study_name or "unnamed-study",
            study_design_hash=study.study_design_hash or "",
            llenergymeasure_version=__version__,
            started_at=datetime.now(timezone.utc),
            total_experiments=len(entries),
            completed=0,
            failed=0,
            pending=len(entries),
            experiments=entries,
        )

    @staticmethod
    def _build_entries(study: StudyConfig) -> list[ExperimentManifestEntry]:
        """Build pending entries for all (experiment, cycle) combinations.

        study.experiments is already the cycled execution list from apply_cycles()
        in load_study_config() (e.g. 6 entries for 2 configs x 3 cycles). Iterating
        it directly and then looping over n_cycles would produce 18 entries instead
        of 6. Deduplicate by config_hash first to recover the unique configs, then
        build one entry per (config_hash, cycle) pair.
        """
        from llenergymeasure.domain.experiment import compute_measurement_config_hash

        # Deduplicate: preserve first-seen order, discard repetitions from cycling.
        seen: dict[str, ExperimentConfig] = {}
        for exp in study.experiments:
            h = compute_measurement_config_hash(exp)
            if h not in seen:
                seen[h] = exp

        entries: list[ExperimentManifestEntry] = []
        n_cycles = study.study_execution.n_cycles
        for config_hash, exp in seen.items():
            summary = build_config_summary(exp)
            for cycle in range(1, n_cycles + 1):
                entries.append(
                    ExperimentManifestEntry(
                        config_hash=config_hash,
                        config_summary=summary,
                        cycle=cycle,
                        status="pending",
                    )
                )
        return entries
