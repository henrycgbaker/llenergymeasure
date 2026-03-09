"""Exception hierarchy for llenergymeasure."""


class LLEMError(Exception):
    """Base exception for llenergymeasure."""


class ConfigError(LLEMError):
    """Invalid or missing configuration."""


class BackendError(LLEMError):
    """Error from an inference backend (load, run, timeout)."""


class PreFlightError(LLEMError):
    """Pre-flight check failed before GPU allocation."""


class ExperimentError(LLEMError):
    """Error during experiment execution."""


class StudyError(LLEMError):
    """Error during study orchestration."""


class DockerError(LLEMError):
    """Base class for Docker container dispatch errors."""

    def __init__(self, message: str, fix_suggestion: str = "", stderr_snippet: str | None = None):
        super().__init__(message)
        self.fix_suggestion = fix_suggestion
        self.stderr_snippet = stderr_snippet


class DockerPreFlightError(PreFlightError):
    """Docker pre-flight check failed before any container is launched.

    Inherits from PreFlightError so it is caught by the existing CLI error
    handler (PreFlightError catch block in cli/run.py).
    """


class InvalidStateTransitionError(ExperimentError):
    """Invalid state machine transition."""

    def __init__(self, from_state: str, to_state: str):
        super().__init__(f"Invalid transition: {from_state} -> {to_state}")
        self.from_state = from_state
        self.to_state = to_state


# ---------------------------------------------------------------------------
# v1.x compatibility aliases — removed in a later phase when consumers migrate
# ---------------------------------------------------------------------------
ConfigurationError = ConfigError
AggregationError = ExperimentError
BackendInferenceError = BackendError
BackendInitializationError = BackendError
BackendNotAvailableError = BackendError
BackendConfigError = ConfigError
BackendTimeoutError = BackendError
