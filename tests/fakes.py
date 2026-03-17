"""Protocol injection fakes for unit tests.

Each fake implements a Protocol structurally (duck typing). Behaviour is explicit
in the class body -- no implicit MagicMock returns. Tests inject fakes via
constructor args or function parameters, never via unittest.mock.patch on
internal modules.
"""

from __future__ import annotations

from typing import Any


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
        from llenergymeasure.energy.nvml import EnergyMeasurement

        return EnergyMeasurement(total_j=self._total_j, duration_sec=self._duration_sec)

    def is_available(self) -> bool:
        return True
