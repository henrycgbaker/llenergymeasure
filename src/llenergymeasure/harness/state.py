"""Experiment state machine for llenergymeasure v2.0.

Tracks experiment lifecycle for resume and deduplication.
Uses a 3-state machine (INITIALISING, MEASURING, DONE) with an orthogonal
failed:bool flag, replacing the v1.x 6-state design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path

import platformdirs
from pydantic import BaseModel, Field

from llenergymeasure.utils.exceptions import ConfigError, InvalidStateTransitionError
from llenergymeasure.utils.security import is_safe_path, sanitize_experiment_id

logger = logging.getLogger(__name__)


class ExperimentPhase(str, Enum):
    """Lifecycle phase of an experiment.

    The state machine has exactly 3 phases:
        INITIALISING -> MEASURING -> DONE

    The ``failed`` flag on ExperimentState is orthogonal to these phases —
    any phase can be marked failed without changing the phase value.
    """

    INITIALISING = "initialising"
    MEASURING = "measuring"
    DONE = "done"


# Valid forward transitions
_VALID_TRANSITIONS: dict[ExperimentPhase, frozenset[ExperimentPhase]] = {
    ExperimentPhase.INITIALISING: frozenset({ExperimentPhase.MEASURING}),
    ExperimentPhase.MEASURING: frozenset({ExperimentPhase.DONE}),
    ExperimentPhase.DONE: frozenset(),  # Terminal
}


class ExperimentState(BaseModel):
    """Persistent experiment state with 3-phase lifecycle.

    The ``failed`` flag is orthogonal to ``phase``: an experiment can fail
    at any point without leaving the current phase. This simplifies the
    state machine compared to the v1.x 6-state design.
    """

    model_config = {"extra": "forbid"}

    experiment_id: str = Field(..., description="Unique experiment identifier")
    phase: ExperimentPhase = Field(
        default=ExperimentPhase.INITIALISING,
        description="Current lifecycle phase",
    )
    failed: bool = Field(default=False, description="Whether the experiment has failed")
    config_hash: str | None = Field(
        default=None,
        description="Hash of experiment config for deduplication",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed=True",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When the experiment started measuring",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last state update",
    )
    subprocess_pid: int | None = Field(
        default=None,
        description="PID of subprocess (for stale state detection)",
    )

    def can_transition_to(self, new_phase: ExperimentPhase) -> bool:
        """Return True if transition to ``new_phase`` is valid.

        Args:
            new_phase: Target phase to check.

        Returns:
            True if the transition is allowed.
        """
        return new_phase in _VALID_TRANSITIONS.get(self.phase, frozenset())

    def transition_to(self, new_phase: ExperimentPhase) -> None:
        """Transition to ``new_phase``, updating ``last_updated``.

        Args:
            new_phase: Target phase.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        if not self.can_transition_to(new_phase):
            raise InvalidStateTransitionError(
                from_state=self.phase.value,
                to_state=new_phase.value,
            )
        self.phase = new_phase
        self.last_updated = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Set the failed flag and record the error message.

        Does not change ``phase`` — failure is orthogonal to lifecycle phase.

        Args:
            error: Human-readable error description.
        """
        self.failed = True
        self.error_message = error
        self.last_updated = datetime.now()

    def is_subprocess_running(self) -> bool:
        """Return True if the tracked subprocess is still alive.

        Uses signal 0 to check process existence without sending a signal.
        Returns False if no PID is tracked or the process is gone.
        """
        if self.subprocess_pid is None:
            return False
        try:
            os.kill(self.subprocess_pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def compute_config_hash(config_dict: dict[str, object]) -> str:
    """Compute a stable 16-character SHA-256 hash of an experiment config.

    Excludes volatile fields (``experiment_id``, ``_metadata``) so that
    the same logical config always produces the same hash.

    Args:
        config_dict: Experiment configuration as a plain dict.

    Returns:
        16-character lowercase hex string.
    """
    stable = {k: v for k, v in config_dict.items() if k not in ("experiment_id", "_metadata")}
    content = json.dumps(stable, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _default_state_dir() -> Path:
    """Return the platform-appropriate default state directory."""
    return Path(platformdirs.user_state_path("llenergymeasure"))


class StateManager:
    """Manages persistent experiment state with atomic file operations.

    State files are stored as JSON in ``state_dir``. Writes use a
    write-to-temp-then-rename pattern for atomicity.

    Args:
        state_dir: Directory for state files. Defaults to the platform
            user state directory (e.g. ~/.local/state/llenergymeasure).
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = Path(state_dir) if state_dir is not None else _default_state_dir()
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _state_path(self, experiment_id: str) -> Path:
        safe_id = sanitize_experiment_id(experiment_id)
        return self._state_dir / f"{safe_id}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        experiment_id: str,
        config_hash: str | None = None,
    ) -> ExperimentState:
        """Create and persist a new experiment state in INITIALISING phase.

        Args:
            experiment_id: Unique experiment identifier.
            config_hash: Optional config hash for deduplication.

        Returns:
            The newly created ExperimentState.
        """
        state = ExperimentState(
            experiment_id=experiment_id,
            config_hash=config_hash,
        )
        self.save(state)
        return state

    def load(self, experiment_id: str) -> ExperimentState | None:
        """Load persisted state for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            ExperimentState if found, None if no state file exists.

        Raises:
            ConfigError: If the state file exists but cannot be parsed, or
                if the resolved path is outside the state directory.
        """
        path = self._state_path(experiment_id)
        if not path.exists():
            return None

        if not is_safe_path(self._state_dir, path):
            raise ConfigError(f"Invalid state path: {path}")

        try:
            content = path.read_text()
            return ExperimentState.model_validate_json(content)
        except Exception as exc:
            raise ConfigError(f"Failed to load state for {experiment_id!r}: {exc}") from exc

    def save(self, state: ExperimentState) -> Path:
        """Atomically persist ``state`` to disk.

        Writes to a temporary file then renames to the final path so that
        readers never see a partial write.

        Args:
            state: State to save.

        Returns:
            Path to the saved state file.

        Raises:
            ConfigError: If the write fails.
        """
        path = self._state_path(state.experiment_id)
        temp_path = path.with_suffix(".tmp")

        try:
            state.last_updated = datetime.now()
            temp_path.write_text(state.model_dump_json(indent=2))
            temp_path.rename(path)
            return path
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise ConfigError(f"Failed to save state for {state.experiment_id!r}: {exc}") from exc

    def delete(self, experiment_id: str) -> bool:
        """Delete persisted state for ``experiment_id``.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            True if a state file was deleted, False if none existed.
        """
        path = self._state_path(experiment_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def list_experiments(self) -> list[str]:
        """Return all experiment IDs that have persisted state.

        Returns:
            List of experiment ID strings (file stems).
        """
        return [p.stem for p in self._state_dir.glob("*.json")]

    def find_by_config_hash(self, config_hash: str) -> ExperimentState | None:
        """Find an incomplete experiment matching ``config_hash``.

        An experiment is considered incomplete if its phase is not DONE.

        Args:
            config_hash: Config hash to match against.

        Returns:
            The first matching ExperimentState, or None.
        """
        for exp_id in self.list_experiments():
            try:
                state = self.load(exp_id)
            except ConfigError:
                logger.warning("Skipping corrupt state file for experiment %r", exp_id)
                continue
            if state is None:
                continue
            if state.phase != ExperimentPhase.DONE and state.config_hash == config_hash:
                return state
        return None

    def cleanup_stale(self) -> list[str]:
        """Mark MEASURING states with dead subprocesses as failed.

        Iterates all persisted states, and for each that is in the
        MEASURING phase with a subprocess PID that is no longer alive,
        calls ``mark_failed`` and persists the updated state.

        Returns:
            List of experiment IDs that were updated.
        """
        cleaned: list[str] = []
        for exp_id in self.list_experiments():
            try:
                state = self.load(exp_id)
            except ConfigError:
                continue
            if state is None:
                continue
            if (
                state.phase == ExperimentPhase.MEASURING
                and not state.failed
                and not state.is_subprocess_running()
            ):
                state.mark_failed("Subprocess no longer running (stale state detected)")
                self.save(state)
                cleaned.append(exp_id)
                logger.debug("Marked stale experiment %r as failed", exp_id)
        return cleaned
