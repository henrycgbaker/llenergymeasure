"""Warning categories emitted by the config layer."""

from __future__ import annotations


class ConfigValidationWarning(UserWarning):
    """Emitted when a vendored validation rule matches with ``severity=warn``.

    Distinct subclass so callers can elevate only config-validation warnings
    to errors via ``warnings.simplefilter("error", ConfigValidationWarning)``.
    """
