"""Tests for CircuitBreaker 3-state machine."""

import pytest

from llenergymeasure.study.circuit_breaker import CircuitBreaker


class TestInitialState:
    def test_default_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"

    def test_default_max_failures(self):
        cb = CircuitBreaker()
        assert cb.consecutive_failures == 0

    def test_default_cooldown(self):
        cb = CircuitBreaker()
        assert cb.cooldown_seconds == 60.0

    def test_custom_threshold_and_cooldown(self):
        cb = CircuitBreaker(max_failures=5, cooldown_seconds=30.0)
        assert cb.state == "closed"
        assert cb.cooldown_seconds == 30.0

    def test_is_not_disabled_by_default(self):
        cb = CircuitBreaker()
        assert cb.is_disabled is False

    def test_recent_failures_empty_initially(self):
        cb = CircuitBreaker()
        assert cb.recent_failures == []


class TestDisabledMode:
    def test_zero_threshold_is_disabled(self):
        cb = CircuitBreaker(max_failures=0)
        assert cb.is_disabled is True

    def test_disabled_never_trips(self):
        cb = CircuitBreaker(max_failures=0)
        for _ in range(100):
            result = cb.record_failure(error_type="RuntimeError", error_message="boom")
            assert result == "continue"

    def test_disabled_state_stays_closed(self):
        cb = CircuitBreaker(max_failures=0)
        for _ in range(50):
            cb.record_failure()
        assert cb.state == "closed"

    def test_disabled_success_is_noop(self):
        cb = CircuitBreaker(max_failures=0)
        cb.record_failure()
        cb.record_success()  # should not raise
        assert cb.state == "closed"


class TestClosedState:
    def test_single_failure_returns_continue(self):
        cb = CircuitBreaker(max_failures=10)
        result = cb.record_failure()
        assert result == "continue"

    def test_consecutive_failures_increments(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(5):
            cb.record_failure()
        assert cb.consecutive_failures == 5

    def test_threshold_minus_one_returns_continue(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(9):
            result = cb.record_failure()
            assert result == "continue"
        assert cb.state == "closed"

    def test_threshold_reached_returns_tripped(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(9):
            cb.record_failure()
        result = cb.record_failure()
        assert result == "tripped"

    def test_threshold_reached_transitions_to_open(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(10):
            cb.record_failure()
        assert cb.state == "open"

    def test_success_resets_counter(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(5):
            cb.record_failure()
        assert cb.consecutive_failures == 5
        cb.record_success()
        assert cb.consecutive_failures == 0

    def test_success_stays_closed(self):
        cb = CircuitBreaker(max_failures=10)
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"

    def test_failure_after_success_restarts_count(self):
        cb = CircuitBreaker(max_failures=10)
        for _ in range(5):
            cb.record_failure()
        cb.record_success()
        result = cb.record_failure()
        assert result == "continue"
        assert cb.consecutive_failures == 1


class TestFailFastMode:
    def test_threshold_one_trips_on_first_failure(self):
        cb = CircuitBreaker(max_failures=1)
        result = cb.record_failure(error_type="OOM", error_message="out of memory")
        assert result == "tripped"

    def test_threshold_one_transitions_to_open(self):
        cb = CircuitBreaker(max_failures=1)
        cb.record_failure()
        assert cb.state == "open"

    def test_threshold_one_is_not_disabled(self):
        cb = CircuitBreaker(max_failures=1)
        assert cb.is_disabled is False


class TestOpenState:
    def test_record_failure_in_open_state_raises(self):
        cb = CircuitBreaker(max_failures=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        with pytest.raises(RuntimeError, match="open"):
            cb.record_failure()

    def test_record_success_in_open_state_raises(self):
        cb = CircuitBreaker(max_failures=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        with pytest.raises(RuntimeError, match="open"):
            cb.record_success()

    def test_start_probe_transitions_to_half_open(self):
        cb = CircuitBreaker(max_failures=3)
        for _ in range(3):
            cb.record_failure()
        cb.start_probe()
        assert cb.state == "half_open"


class TestHalfOpenState:
    def _trip(self, cb: CircuitBreaker) -> CircuitBreaker:
        for _ in range(cb._max_failures):
            cb.record_failure()
        cb.start_probe()
        return cb

    def test_successful_probe_resets_to_closed(self):
        cb = CircuitBreaker(max_failures=3)
        self._trip(cb)
        cb.record_success()
        assert cb.state == "closed"

    def test_successful_probe_resets_counter(self):
        cb = CircuitBreaker(max_failures=3)
        self._trip(cb)
        cb.record_success()
        assert cb.consecutive_failures == 0

    def test_failed_probe_returns_abort(self):
        cb = CircuitBreaker(max_failures=3)
        self._trip(cb)
        result = cb.record_failure(error_type="RuntimeError", error_message="still broken")
        assert result == "abort"

    def test_failed_probe_state_unchanged(self):
        cb = CircuitBreaker(max_failures=3)
        self._trip(cb)
        cb.record_failure()
        # State stays half_open after abort signal (caller decides to stop)
        assert cb.state == "half_open"

    def test_start_probe_in_non_open_state_raises(self):
        cb = CircuitBreaker(max_failures=3)
        with pytest.raises(RuntimeError, match="open"):
            cb.start_probe()


class TestRecentFailures:
    def test_failures_stored_with_type_and_message(self):
        cb = CircuitBreaker(max_failures=10)
        cb.record_failure(error_type="OOMError", error_message="CUDA out of memory")
        assert len(cb.recent_failures) == 1
        assert cb.recent_failures[0] == ("OOMError", "CUDA out of memory")

    def test_failures_stored_up_to_threshold(self):
        cb = CircuitBreaker(max_failures=5)
        for i in range(5):
            cb.record_failure(error_type="Err", error_message=f"msg-{i}")
        assert len(cb.recent_failures) == 5

    def test_older_failures_not_stored_beyond_threshold(self):
        # Record failures, then a success (resets), then more failures.
        # After the second run of failures, the earlier "OldError" must not appear.
        cb = CircuitBreaker(max_failures=5)
        cb.record_failure(error_type="OldError", error_message="old")
        cb.record_success()  # resets list
        cb.record_failure(error_type="Err1", error_message="msg-1")
        cb.record_failure(error_type="Err2", error_message="msg-2")
        types = [f[0] for f in cb.recent_failures]
        assert "OldError" not in types
        assert "Err1" in types
        assert "Err2" in types

    def test_recent_failures_reset_on_success(self):
        cb = CircuitBreaker(max_failures=10)
        cb.record_failure(error_type="Err", error_message="broken")
        cb.record_success()
        assert cb.recent_failures == []

    def test_empty_defaults_for_failure_metadata(self):
        cb = CircuitBreaker(max_failures=3)
        cb.record_failure()
        assert cb.recent_failures[0] == ("", "")


class TestGetFailureSummary:
    def test_summary_returns_one_line_per_failure(self):
        cb = CircuitBreaker(max_failures=5)
        for i in range(3):
            cb.record_failure(error_type=f"ErrType{i}", error_message=f"message {i}")
        summary = cb.get_failure_summary()
        assert len(summary) == 3
        for line in summary:
            assert isinstance(line, str)

    def test_summary_includes_error_type_and_message(self):
        cb = CircuitBreaker(max_failures=5)
        cb.record_failure(error_type="RuntimeError", error_message="segfault")
        summary = cb.get_failure_summary()
        assert any("RuntimeError" in line for line in summary)
        assert any("segfault" in line for line in summary)

    def test_summary_empty_when_no_failures(self):
        cb = CircuitBreaker()
        assert cb.get_failure_summary() == []


class TestFullLifecycle:
    def test_full_cycle_closed_open_halfopen_closed(self):
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=0.0)
        # closed -> 3 failures -> open
        for _ in range(3):
            cb.record_failure(error_type="Err", error_message="fail")
        assert cb.state == "open"
        # open -> probe -> half_open
        cb.start_probe()
        assert cb.state == "half_open"
        # half_open -> success -> closed
        cb.record_success()
        assert cb.state == "closed"
        assert cb.consecutive_failures == 0

    def test_full_cycle_closed_open_halfopen_abort(self):
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=0.0)
        for _ in range(3):
            cb.record_failure()
        cb.start_probe()
        result = cb.record_failure(error_type="Err", error_message="still failing")
        assert result == "abort"
        assert cb.state == "half_open"

    def test_can_recover_multiple_times(self):
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=0.0)
        for cycle in range(3):
            # Trip
            for _ in range(2):
                cb.record_failure()
            assert cb.state == "open", f"Expected open at cycle {cycle}"
            # Recover
            cb.start_probe()
            cb.record_success()
            assert cb.state == "closed", f"Expected closed at cycle {cycle}"
