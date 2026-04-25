"""Resolved-config view construction from resolved ExperimentConfig (study layer).

The pure hashing primitives live in :mod:`llenergymeasure.domain.hashing`
(Layer 0).  This module contains only :func:`build_resolved_view`, which
needs ``ExperimentConfig`` and therefore belongs at Layer 4 (study).

Re-exports the domain primitives so callers that previously imported from
``study.hashing`` continue to work without changes.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.config.models import ExperimentConfig

# Re-export domain primitives — study.library_resolution and tests import from here.
from llenergymeasure.domain.hashing import (
    _FLOAT_SIG_DIGITS,
    ConfigHashView,
    _normalise,
    build_observed_view,
    canonical_serialise,
    hash_config,
)

__all__ = [
    "_FLOAT_SIG_DIGITS",
    "ConfigHashView",
    "_normalise",
    "build_observed_view",
    "build_resolved_view",
    "canonical_serialise",
    "hash_config",
]


# ---------------------------------------------------------------------------
# Resolved-config view construction — from library-resolution mechanism output
# ---------------------------------------------------------------------------


def build_resolved_view(config: ExperimentConfig) -> ConfigHashView:
    """Project a (post-library-resolution) ``ExperimentConfig`` into a resolved-config view.

    Reads the active engine section's full post-normalisation state; the
    library-resolution mechanism has already applied dormant rules to fixpoint before this
    runs.  Callers pass the resolved config, not the declared one — resolved_config_hash is
    meaningless on a pre-resolved config.

    Engine-specific sub-models carry a ``sampling`` attribute; it is lifted
    into its own dict so the resolved-config / observed-config ordering separates
    "how the engine constructs" from "what it generates with".
    """
    engine_name = config.engine.value if hasattr(config.engine, "value") else str(config.engine)
    section: Any = getattr(config, engine_name, None)
    dump: dict[str, Any] = section.model_dump(mode="python") if section is not None else {}
    sampling = dump.pop("sampling", None) or {}

    return ConfigHashView(
        engine=engine_name,
        task=config.task.model_dump(mode="python"),
        observed_engine_params=dump,
        observed_sampling_params=sampling,
        lora=None,  # No LoRA field yet on ExperimentConfig — reserved for future.
        passthrough_kwargs=dict(config.passthrough_kwargs or {}),
    )
