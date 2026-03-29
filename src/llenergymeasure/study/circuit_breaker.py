"""Circuit breaker 3-state machine for study runner failure tracking.

Implements Nygard's Release It! pattern (closed/open/half-open), adapted from
Ray Tune's FailureConfig. The caller (StudyRunner) is responsible for the
cooldown sleep; this class handles only state transitions and counters.
"""

from __future__ import annotations

from typing import Literal


class CircuitBreaker:
    """Track consecutive experiment failures and provide abort/probe decisions.

    States:
    - ``closed``: Normal operation. Failures increment counter; successes reset it.
    - ``open``: Threshold reached. Caller should sleep (cooldown_seconds), then
      call :meth:`start_probe` to enter half-open.
    - ``half_open``: One probe experiment runs. Success -> closed. Failure -> abort.

    Args:
        max_failures: Number of consecutive failures that trip the breaker.
            ``0`` disables the breaker entirely. ``1`` is fail-fast mode.
            Defaults to ``10``.
        cooldown_seconds: How long the caller should pause before probing.
            Stored here for the caller's convenience; not enforced by this class.
            Defaults to ``60.0``.
    """

    def __init__(self, max_failures: int = 10, cooldown_seconds: float = 60.0) -> None:
        self._max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._state: Literal["closed", "open", "half_open"] = "closed"
        self._consecutive_failures: int = 0
        self._recent_failures: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def is_disabled(self) -> bool:
        """True when max_failures == 0 (circuit breaker turned off)."""
        return self._max_failures == 0

    @property
    def recent_failures(self) -> list[tuple[str, str]]:
        """The most recent N failure tuples of (error_type, error_message)."""
        return list(self._recent_failures)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful experiment outcome.

        - ``closed``: resets consecutive failure counter.
        - ``half_open``: transitions back to ``closed``, resets counter.
        - ``open``: invalid — raises :class:`RuntimeError`.
        """
        if self._state == "open":
            raise RuntimeError("record_success() called in open state; call start_probe() first.")
        self._state = "closed"
        self._consecutive_failures = 0
        self._recent_failures = []

    def record_failure(
        self,
        error_type: str = "",
        error_message: str = "",
    ) -> str:
        """Record a failed experiment outcome.

        Returns:
            ``"continue"`` — below threshold, keep going.
            ``"tripped"``  — threshold reached, breaker is now open.
            ``"abort"``    — probe failed in half-open state; caller should stop.

        Raises:
            RuntimeError: if called in ``open`` state.
        """
        if self._state == "open":
            raise RuntimeError("record_failure() called in open state; call start_probe() first.")

        if self._state == "half_open":
            return "abort"

        # closed state
        if self.is_disabled:
            return "continue"

        self._consecutive_failures += 1
        self._recent_failures.append((error_type, error_message))
        # Keep only the last max_failures entries for the summary
        if len(self._recent_failures) > self._max_failures:
            self._recent_failures = self._recent_failures[-self._max_failures :]

        if self._consecutive_failures >= self._max_failures:
            self._state = "open"
            return "tripped"

        return "continue"

    def start_probe(self) -> None:
        """Transition from ``open`` to ``half_open`` to run a probe experiment.

        Raises:
            RuntimeError: if not currently in ``open`` state.
        """
        if self._state != "open":
            raise RuntimeError(f"start_probe() requires open state, but state is '{self._state}'.")
        self._state = "half_open"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_failure_summary(self) -> list[str]:
        """Return one-line summary strings for each recent failure.

        Intended for CLI output when the breaker trips.
        """
        return [
            f"{error_type}: {error_message}" if error_type else error_message
            for error_type, error_message in self._recent_failures
        ]
