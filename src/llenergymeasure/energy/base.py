"""Base protocol for energy measurement backends.

Defines the EnergyBackend protocol - the interface contract that all energy
measurement backends (NVML, Zeus, CodeCarbon) must satisfy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol for energy measurement backends.

    Implementations include Zeus, NVML, CodeCarbon, etc.
    """

    @property
    def name(self) -> str:
        """Backend name for identification."""
        ...

    def start_tracking(self) -> Any:
        """Start energy tracking. Returns tracker handle."""
        ...

    def stop_tracking(self, tracker: Any) -> Any:
        """Stop energy tracking and return metrics."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...
