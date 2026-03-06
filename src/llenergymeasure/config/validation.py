"""Centralised validation types and utilities.

This module provides the Single Source of Truth (SSOT) for configuration
warnings and validation-related types used across the codebase.

All other modules should import ConfigWarning from here rather than
defining their own versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ConfigWarning:
    """Warning about configuration validity or compatibility.

    Used for:
    - Config validation during loading
    - Backend compatibility checks
    - Parameter semantic differences

    Severity levels:
    - error: Invalid config (blocks execution without --force)
    - warning: Problematic config that may cause unexpected behaviour
    - info: Suboptimal but valid configuration, or semantic differences

    Attributes:
        field: Parameter/field name that triggered the warning.
        message: Human-readable description of the issue.
        severity: One of 'error', 'warning', 'info'.
        suggestion: Optional suggestion for fixing the issue.
        migration_hint: Hint for deprecated/changed params.
    """

    field: str
    message: str
    severity: Literal["error", "warning", "info"] = "warning"
    suggestion: str | None = None
    migration_hint: str | None = None

    # Alias for backwards compatibility with protocols.py usage
    @property
    def param(self) -> str:
        """Alias for field (backwards compatibility)."""
        return self.field

    def __str__(self) -> str:
        """Format for logging/display."""
        return f"[{self.severity.upper()}] {self.field}: {self.message}"

    def to_result_string(self) -> str:
        """Format for embedding in results."""
        return f"{self.severity}: {self.field} - {self.message}"


# Type alias for validation function return type
ValidationWarnings = list[ConfigWarning]
