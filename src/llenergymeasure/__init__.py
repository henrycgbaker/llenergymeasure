"""LLenergyMeasure -- LLM inference efficiency measurement framework.

Public API:
    run_experiment, run_study, ExperimentConfig, StudyConfig,
    ExperimentResult, StudyResult, __version__

Stability contract: exports in __all__ follow SemVer. Names not in __all__
are internal and may change without notice. One minor version deprecation
window before removing any __all__ export (removed in v2.x+1 at earliest).
"""

from llenergymeasure._api import run_experiment, run_study
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult

__version__: str = "0.8.0"

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "StudyConfig",
    "StudyResult",
    "__version__",
    "run_experiment",
    "run_study",
]
