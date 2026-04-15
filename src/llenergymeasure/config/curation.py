"""Structured curation metadata for typed engine config fields.

Every typed field in engine_configs.py carries a CurationMetadata instance attached
via json_schema_extra. This makes the rubric verdict machine-readable and is the
source of truth for the generated inclusion/exclusion table (Plan E).

Usage::

    from llenergymeasure.config.curation import CurationMetadata

    some_field: int | None = Field(
        default=None,
        ge=1,
        description="...",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="AMD MLPerf v5.1 multi-step throughput axis.",
            native_mapping="EngineArgs.num_scheduler_steps",
        ).to_schema_extra(),
    )
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RubricClause = Literal[
    "R1",
    "R2",  # required
    "E1",
    "E2",
    "E3",
    "E4",  # elevating
    "D1",
    "D2",
    "D3",
    "D4",  # demoting (for drop decisions recorded for posterity)
    "T1",
    "T2",
    "T3",
    "T4",
    "T5",  # tie-breakers
]


@dataclass(frozen=True, slots=True)
class CurationMetadata:
    """Structured justification attached to every typed field in engine_configs.py.

    Becomes the source of truth for the generated inclusion/exclusion table (Plan E).

    Attributes:
        clauses:        Tuple of rubric clause codes that fired for this field.
                        At minimum R1 (plausible measurement path) must be present
                        unless R2 (non-duplication) caused a drop.
        rationale:      One-line justification. This text appears verbatim in the
                        generated curation doc as the "reason" column.
        native_mapping: Dotted path to the corresponding native engine argument,
                        e.g. "EngineArgs.num_scheduler_steps". None when the field
                        is engine-local only (no direct native counterpart).
        notes:          Optional caveats — e.g. "churn-prone, watch vLLM v0.8 release
                        notes", or "defer — confirm HF deprecation before adding".
    """

    clauses: tuple[RubricClause, ...]
    rationale: str
    native_mapping: str | None = None
    notes: str | None = None

    def to_schema_extra(self) -> dict[str, object]:
        """Render to the dict shape Pydantic expects in ``json_schema_extra``."""
        return {"curation": asdict(self)}
