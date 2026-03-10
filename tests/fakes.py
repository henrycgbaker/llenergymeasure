"""Protocol injection fakes for unit tests.

Each fake implements a Protocol structurally (duck typing). Behaviour is explicit
in the class body -- no implicit MagicMock returns. Tests inject fakes via
constructor args or function parameters, never via unittest.mock.patch on
internal modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult


class FakeInferenceBackend:
    """Minimal InferenceBackend fake -- returns a pre-built ExperimentResult."""

    name = "fake"

    def __init__(self, result: ExperimentResult | None = None):
        self._result = result
        self.run_calls: list[ExperimentConfig] = []

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        self.run_calls.append(config)
        if self._result is None:
            raise ValueError("FakeInferenceBackend: set result before calling run()")
        return self._result


class FakeEnergyBackend:
    """Minimal EnergyBackend fake -- returns fixed EnergyMeasurement."""

    name = "fake-energy"

    def __init__(self, total_j: float = 10.0, duration_sec: float = 5.0):
        self._total_j = total_j
        self._duration_sec = duration_sec
        self._tracking = False

    def start_tracking(self) -> str:
        self._tracking = True
        return "fake-tracker"

    def stop_tracking(self, tracker: Any) -> Any:
        self._tracking = False
        from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement

        return EnergyMeasurement(total_j=self._total_j, duration_sec=self._duration_sec)

    def is_available(self) -> bool:
        return True


class FakeResultsRepository:
    """Minimal ResultsRepository fake -- stores results in memory, keyed by path."""

    def __init__(self):
        self._store: dict[Path, ExperimentResult] = {}
        self.saved: list[tuple[ExperimentResult, Path]] = []

    def save(self, result: ExperimentResult, output_dir: Path) -> Path:
        path = output_dir / "result.json"
        self._store[path] = result
        self.saved.append((result, output_dir))
        return path

    def load(self, path: Path) -> ExperimentResult:
        if path not in self._store:
            raise FileNotFoundError(f"No result at {path}")
        return self._store[path]
