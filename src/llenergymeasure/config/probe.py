"""Configuration-probe data types: dormancy and probe results.

These are foundation-level validation-outcome data — produced by the
generic ``@model_validator`` (for dormancy observations) and by
:mod:`llenergymeasure.engines.probe_adapter` (for assembled probes).
They live in :mod:`config` rather than :mod:`engines` so the validator
can emit :class:`DormantField` without crossing the engines-layer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DormantField:
    """A user-declared config field that the engine ignored or overrode.

    Distinguishes two shapes of dormancy:
      - Stripped: declared_value set, effective_value is None (field absent from
        effective kwargs, e.g. temperature under greedy decoding).
      - Overridden: effective_value != declared_value (engine remapped it).
    """

    declared_value: Any
    effective_value: Any | None
    reason: str | None = None


@dataclass(frozen=True)
class ConfigProbe:
    """Outcome of probing an ExperimentConfig against an engine.

    The probe observes what the engine would do with this config without
    loading weights, allocating GPU memory, or initialising engine contexts.

    Attributes:
        effective_engine_params: Kwargs that would be passed to the engine
            constructor (vllm.LLM, AutoModelForCausalLM, tensorrt_llm.LLM).
        effective_sampling_params: Kwargs that would be passed to the
            sampling-params constructor after any greedy stripping.
        dormant_fields: Keyed by dotted path (e.g. ``"vllm.sampling.top_p"``)
            — fields the user declared that the engine will silently ignore
            or override.
        errors: Engine-reported framework errors (T1/T2 construction,
            hardware checks). Non-empty means the config will not run as-is.
        warnings: Non-fatal observations.
    """

    effective_engine_params: dict[str, Any]
    effective_sampling_params: dict[str, Any]
    dormant_fields: dict[str, DormantField]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the probe captured no framework errors."""
        return len(self.errors) == 0
