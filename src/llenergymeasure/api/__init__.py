"""Public library API for llenergymeasure."""

from llenergymeasure.api._impl import run_experiment, run_study
from llenergymeasure.api.report_gaps import (
    GapProposal,
    ReportGapsError,
    find_runtime_gaps,
    load_rules_corpus,
    render_yaml_fragment,
)
from llenergymeasure.results.persistence import save_result
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.study.resume import find_resumable_study, load_resume_state

__all__ = [
    "GapProposal",
    "ReportGapsError",
    "find_resumable_study",
    "find_runtime_gaps",
    "load_resume_state",
    "load_rules_corpus",
    "probe_energy_sampler",
    "render_yaml_fragment",
    "run_experiment",
    "run_study",
    "run_study_preflight",
    "save_result",
]


def probe_energy_sampler() -> str | None:
    """Best-effort probe for the auto-selected energy sampler on this host.

    Returns the class name of the selected sampler, or None if unavailable.
    """
    try:
        from llenergymeasure.energy import select_energy_sampler

        sampler = select_energy_sampler("auto")
        return type(sampler).__name__ if sampler else None
    except Exception:
        return None
