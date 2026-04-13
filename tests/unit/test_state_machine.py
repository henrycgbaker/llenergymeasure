"""Unit tests for the 3-state experiment state machine.

Covers ExperimentPhase enum, ExperimentState transitions, StateManager
persist/load roundtrip, and compute_config_hash determinism.
"""

from __future__ import annotations

import pytest

from llenergymeasure.harness.state import (
    ExperimentPhase,
    ExperimentState,
    StateManager,
    compute_config_hash,
)
from llenergymeasure.utils.exceptions import InvalidStateTransitionError

# ---------------------------------------------------------------------------
# ExperimentPhase enum
# ---------------------------------------------------------------------------


def test_three_phases_exist():
    """All 3 expected phase values exist."""
    assert ExperimentPhase.INITIALISING.value == "initialising"
    assert ExperimentPhase.MEASURING.value == "measuring"
    assert ExperimentPhase.DONE.value == "done"


# ---------------------------------------------------------------------------
# ExperimentState initial state
# ---------------------------------------------------------------------------


def test_initial_phase_is_initialising():
    """Newly created ExperimentState starts in INITIALISING phase."""
    state = ExperimentState(experiment_id="test-001")
    assert state.phase == ExperimentPhase.INITIALISING


def test_initial_failed_is_false():
    """Newly created ExperimentState has failed=False."""
    state = ExperimentState(experiment_id="test-001")
    assert state.failed is False


# ---------------------------------------------------------------------------
# Valid transitions
# ---------------------------------------------------------------------------


def test_transition_initialising_to_measuring():
    """INITIALISING -> MEASURING is a valid transition."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    assert state.phase == ExperimentPhase.MEASURING


def test_transition_measuring_to_done():
    """MEASURING -> DONE is a valid transition."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    state.transition_to(ExperimentPhase.DONE)
    assert state.phase == ExperimentPhase.DONE


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


def test_invalid_transition_initialising_to_done():
    """INITIALISING -> DONE is invalid (cannot skip MEASURING)."""
    state = ExperimentState(experiment_id="test-001")
    with pytest.raises(InvalidStateTransitionError):
        state.transition_to(ExperimentPhase.DONE)


def test_invalid_transition_done_to_measuring():
    """DONE -> MEASURING is invalid (terminal state, cannot go backwards)."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    state.transition_to(ExperimentPhase.DONE)
    with pytest.raises(InvalidStateTransitionError):
        state.transition_to(ExperimentPhase.MEASURING)


def test_invalid_transition_done_to_initialising():
    """DONE -> INITIALISING is invalid (terminal state)."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    state.transition_to(ExperimentPhase.DONE)
    with pytest.raises(InvalidStateTransitionError):
        state.transition_to(ExperimentPhase.INITIALISING)


def test_invalid_transition_measuring_to_initialising():
    """MEASURING -> INITIALISING is invalid (no backwards transition)."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    with pytest.raises(InvalidStateTransitionError):
        state.transition_to(ExperimentPhase.INITIALISING)


def test_invalid_state_transition_error_has_from_to_attributes():
    """InvalidStateTransitionError stores from_state and to_state."""
    state = ExperimentState(experiment_id="test-001")
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        state.transition_to(ExperimentPhase.DONE)
    err = exc_info.value
    assert err.from_state == "initialising"
    assert err.to_state == "done"


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


def test_mark_failed_sets_flag():
    """mark_failed() sets failed=True on the state."""
    state = ExperimentState(experiment_id="test-001")
    state.mark_failed("something went wrong")
    assert state.failed is True


def test_mark_failed_preserves_phase():
    """mark_failed() does not change the current phase."""
    state = ExperimentState(experiment_id="test-001")
    state.transition_to(ExperimentPhase.MEASURING)
    state.mark_failed("something went wrong")
    assert state.phase == ExperimentPhase.MEASURING


def test_mark_failed_records_error_message():
    """mark_failed() stores the error message."""
    state = ExperimentState(experiment_id="test-001")
    state.mark_failed("out of memory")
    assert state.error_message == "out of memory"


# ---------------------------------------------------------------------------
# StateManager: create + load roundtrip
# ---------------------------------------------------------------------------


def test_state_manager_create_and_load_roundtrip(tmp_path):
    """create() then load() returns a state with the same fields."""
    manager = StateManager(state_dir=tmp_path)
    created = manager.create(experiment_id="exp-abc", config_hash="hashval1234567")

    loaded = manager.load("exp-abc")
    assert loaded is not None
    assert loaded.experiment_id == created.experiment_id
    assert loaded.config_hash == created.config_hash
    assert loaded.phase == ExperimentPhase.INITIALISING
    assert loaded.failed is False


def test_state_manager_load_missing_returns_none(tmp_path):
    """load() returns None for a non-existent experiment."""
    manager = StateManager(state_dir=tmp_path)
    result = manager.load("nonexistent-experiment")
    assert result is None


def test_state_manager_save_and_reload_after_transition(tmp_path):
    """Transitions are persisted and loaded correctly."""
    manager = StateManager(state_dir=tmp_path)
    state = manager.create(experiment_id="exp-002")

    state.transition_to(ExperimentPhase.MEASURING)
    manager.save(state)

    loaded = manager.load("exp-002")
    assert loaded is not None
    assert loaded.phase == ExperimentPhase.MEASURING


def test_state_manager_save_returns_path(tmp_path):
    """save() returns a valid Path to the state file."""
    manager = StateManager(state_dir=tmp_path)
    state = manager.create(experiment_id="exp-003")
    path = manager.save(state)
    assert path.exists()
    assert path.suffix == ".json"


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


def test_compute_config_hash_deterministic():
    """Same dict always produces the same hash."""
    config = {"model": "gpt2", "engine": "transformers", "dtype": "bfloat16"}
    h1 = compute_config_hash(config)
    h2 = compute_config_hash(config)
    assert h1 == h2


def test_compute_config_hash_different_for_different_configs():
    """Different dicts produce different hashes."""
    config_a = {"model": "gpt2", "engine": "transformers"}
    config_b = {"model": "llama", "engine": "transformers"}
    assert compute_config_hash(config_a) != compute_config_hash(config_b)


def test_compute_config_hash_excludes_experiment_id():
    """experiment_id is excluded from the hash (volatile field)."""
    config_a = {"model": "gpt2", "experiment_id": "run-1"}
    config_b = {"model": "gpt2", "experiment_id": "run-999"}
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_compute_config_hash_returns_16_chars():
    """Hash is exactly 16 lowercase hex characters."""
    h = compute_config_hash({"model": "gpt2"})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
