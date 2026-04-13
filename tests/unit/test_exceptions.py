"""Unit tests for the exception hierarchy in llenergymeasure.exceptions.

Confirms the flat hierarchy: LLEMError + 5 direct subclasses +
InvalidStateTransitionError under ExperimentError.
"""

from __future__ import annotations

import pytest

from llenergymeasure.utils.exceptions import (
    ConfigError,
    EngineError,
    ExperimentError,
    InvalidStateTransitionError,
    LLEMError,
    PreFlightError,
    StudyError,
)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_llem_error_is_base():
    """LLEMError is a subclass of Exception."""
    assert issubclass(LLEMError, Exception)


# ---------------------------------------------------------------------------
# Direct subclasses of LLEMError
# ---------------------------------------------------------------------------


def test_config_error_inherits_llem_error():
    """ConfigError is a subclass of LLEMError."""
    assert issubclass(ConfigError, LLEMError)


def test_engine_error_inherits_llem_error():
    """EngineError is a subclass of LLEMError."""
    assert issubclass(EngineError, LLEMError)


def test_preflight_error_inherits_llem_error():
    """PreFlightError is a subclass of LLEMError."""
    assert issubclass(PreFlightError, LLEMError)


def test_experiment_error_inherits_llem_error():
    """ExperimentError is a subclass of LLEMError."""
    assert issubclass(ExperimentError, LLEMError)


def test_study_error_inherits_llem_error():
    """StudyError is a subclass of LLEMError."""
    assert issubclass(StudyError, LLEMError)


# ---------------------------------------------------------------------------
# InvalidStateTransitionError sub-hierarchy
# ---------------------------------------------------------------------------


def test_invalid_state_transition_inherits_experiment_error():
    """InvalidStateTransitionError is under ExperimentError, not directly LLEMError."""
    assert issubclass(InvalidStateTransitionError, ExperimentError)


def test_invalid_state_transition_not_direct_llem_error_subclass():
    """InvalidStateTransitionError is NOT a direct subclass of LLEMError."""
    # It inherits through ExperimentError, but its direct parent is ExperimentError
    assert InvalidStateTransitionError.__bases__ == (ExperimentError,)


def test_invalid_state_transition_also_catchable_as_llem_error():
    """InvalidStateTransitionError is also catchable as LLEMError (by inheritance)."""
    assert issubclass(InvalidStateTransitionError, LLEMError)


# ---------------------------------------------------------------------------
# Catchability via base class
# ---------------------------------------------------------------------------


def test_all_errors_catchable_via_llem_error():
    """Catching LLEMError catches all 5 direct subclass instances."""
    subclasses = [
        ConfigError("config"),
        EngineError("engine"),
        PreFlightError("preflight"),
        ExperimentError("experiment"),
        StudyError("study"),
    ]
    for exc in subclasses:
        try:
            raise exc
        except LLEMError:
            pass  # expected
        else:
            pytest.fail(f"{type(exc).__name__} was not caught by LLEMError")


def test_invalid_state_transition_catchable_via_llem_error():
    """InvalidStateTransitionError is also caught by LLEMError."""
    exc = InvalidStateTransitionError(from_state="initialising", to_state="done")
    try:
        raise exc
    except LLEMError:
        pass  # expected
    else:
        pytest.fail("InvalidStateTransitionError was not caught by LLEMError")


def test_invalid_state_transition_catchable_via_experiment_error():
    """InvalidStateTransitionError is caught by ExperimentError."""
    exc = InvalidStateTransitionError(from_state="measuring", to_state="initialising")
    try:
        raise exc
    except ExperimentError:
        pass  # expected
    else:
        pytest.fail("InvalidStateTransitionError was not caught by ExperimentError")


# ---------------------------------------------------------------------------
# Message preservation
# ---------------------------------------------------------------------------


def test_error_messages_preserved():
    """Constructing with a message preserves str(e)."""
    for cls in [ConfigError, EngineError, PreFlightError, ExperimentError, StudyError]:
        msg = f"test message for {cls.__name__}"
        exc = cls(msg)
        assert str(exc) == msg


def test_invalid_state_transition_message_format():
    """InvalidStateTransitionError formats a message from from/to states."""
    exc = InvalidStateTransitionError(from_state="initialising", to_state="done")
    assert "initialising" in str(exc)
    assert "done" in str(exc)


def test_invalid_state_transition_stores_from_to():
    """InvalidStateTransitionError stores from_state and to_state attributes."""
    exc = InvalidStateTransitionError(from_state="measuring", to_state="initialising")
    assert exc.from_state == "measuring"
    assert exc.to_state == "initialising"
