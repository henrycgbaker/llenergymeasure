"""Zeus energy measurement backend.

Wraps ZeusMonitor to measure per-GPU energy consumption during inference.
Zeus provides more accurate GPU energy readings than NVML power polling
by integrating with the GPU's hardware energy counters.

All zeus imports are deferred — this module is safe to import without zeus installed.
"""

from __future__ import annotations

from typing import Any

from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement


class ZeusBackend:
    """Energy backend using the Zeus GPU energy monitor.

    Wraps ``zeus.monitor.ZeusMonitor`` to track per-GPU energy over a named
    measurement window.  Zeus is the preferred backend when available, as it
    reads hardware energy registers directly.

    All zeus imports are deferred — safe to import without zeus installed.

    Args:
        gpu_indices: GPU indices to monitor.  Defaults to ``[0]`` when None.
    """

    WINDOW_NAME = "llem_measurement"

    def __init__(self, gpu_indices: list[int] | None = None) -> None:
        self._gpu_indices = gpu_indices if gpu_indices is not None else [0]

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "zeus"

    def is_available(self) -> bool:
        """Return True if the zeus package is installed and importable."""
        try:
            from zeus.monitor import ZeusMonitor  # noqa: F401

            return True
        except ImportError:
            return False

    def start_tracking(self) -> Any:
        """Begin a Zeus measurement window.

        Returns:
            A ZeusMonitor instance with an active window.
        """
        from zeus.monitor import ZeusMonitor

        monitor = ZeusMonitor(gpu_indices=self._gpu_indices)
        monitor.begin_window(self.WINDOW_NAME)
        return monitor

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        """Close the measurement window and return energy totals.

        Args:
            tracker: ZeusMonitor returned by start_tracking().

        Returns:
            EnergyMeasurement with total_j and per-GPU breakdown.
        """
        measurement = tracker.end_window(self.WINDOW_NAME)

        # measurement.energy is dict[int, float]: gpu_index -> joules
        per_gpu_j: dict[int, float] = dict(measurement.energy)
        total_j: float = sum(per_gpu_j.values())
        duration_sec: float = float(measurement.time)

        return EnergyMeasurement(
            total_j=total_j,
            duration_sec=duration_sec,
            samples=[],  # Zeus does not expose raw samples
            per_gpu_j=per_gpu_j if per_gpu_j else None,
        )
