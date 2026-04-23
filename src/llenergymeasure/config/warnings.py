"""Warning categories emitted by the config layer.

Kept separate from :mod:`llenergymeasure.utils.exceptions` so consumers can
filter ``warnings.catch_warnings`` by category without importing engine-layer
warning types. Pair with :func:`warnings.warn` at the emission site and
``warnings.simplefilter("error", ConfigValidationWarning)`` at the test site.
"""

from __future__ import annotations


class ConfigValidationWarning(UserWarning):
    """Non-fatal observation from the generic vendored-rules validator.

    Emitted when a ``warn``-severity rule in the vendored corpus matches a
    user's ``ExperimentConfig``. The warning carries the rule's rendered
    message prefixed with ``[rule_id]``. Errors raise :class:`ValueError`
    instead; dormancy populates the config's ``_dormant_observations`` list.
    """


__all__ = ["ConfigValidationWarning"]
