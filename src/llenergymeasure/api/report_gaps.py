"""Public API for ``llem report-gaps`` — keeps CLI out of the study layer.

Thin façade around :mod:`llenergymeasure.study.report_gaps`. CLI callers must
not import ``study.`` directly (see cli-boundary contract in
``pyproject.toml``); this module is the architectural seam.
"""

from __future__ import annotations

from pathlib import Path

from llenergymeasure.study._gh_automation import (
    DraftPRRequest,
    GhNotFoundError,
    open_draft_pr,
    plan_request,
)
from llenergymeasure.study.report_gaps import (
    GapReport,
    RuleCandidate,
    generate_report,
    render_candidates_yaml,
)
from llenergymeasure.study.runtime_observations import default_cache_path

__all__ = [
    "DraftPRRequest",
    "GapReport",
    "GhNotFoundError",
    "RuleCandidate",
    "default_cache_path",
    "generate_report",
    "open_draft_pr",
    "plan_request",
    "render_candidates_yaml",
    "run_report_gaps",
]


def run_report_gaps(
    *,
    source: str = "both",
    results_dir: Path | None = None,
    cache_path: Path | None = None,
    engine: str | None = None,
) -> GapReport:
    """API wrapper mirroring the CLI's ``report-gaps`` invocation.

    Callers pass raw ``Path`` values; the study layer handles filesystem
    walks and cache parsing.
    """
    return generate_report(
        source=source,
        results_dir=results_dir,
        cache_path=cache_path or default_cache_path(),
        engine=engine,
    )
