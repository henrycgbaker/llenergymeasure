"""NVML energy measurement backend.

Uses PowerThermalSampler (pynvml-based) to poll GPU power at 100ms intervals
and integrates samples to joules via the trapezoidal rule.

All pynvml imports are deferred — this module is safe to import on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnergyMeasurement:
    """Energy measurement result from a tracking session.

    Shared dataclass used by both NVMLBackend and ZeusBackend.

    Attributes:
        total_j: Total energy consumed in joules.
        duration_sec: Duration of the measurement window in seconds.
        samples: Raw power samples (backend-specific type).
        per_gpu_j: Per-GPU energy breakdown (joules per GPU index), if available.
    """

    total_j: float
    duration_sec: float
    samples: list[Any] = field(default_factory=list)
    per_gpu_j: dict[int, float] | None = None


class NVMLBackend:
    """Energy backend using NVML via PowerThermalSampler.

    Polls GPU power at 100ms intervals during the measurement window and
    integrates consecutive sample pairs using the trapezoidal rule to produce
    total energy in joules.

    All pynvml access is fully deferred — safe to import on CPU-only hosts.

    Args:
        device_index: CUDA device index to monitor (default: 0).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "nvml"

    def is_available(self) -> bool:
        """Return True if pynvml can initialise on this machine."""
        import importlib.util

        if importlib.util.find_spec("pynvml") is None:
            return False
        try:
            import pynvml

            from llenergymeasure.core.gpu_info import nvml_context

            # Use a sentinel to detect whether nvml_context yielded successfully
            # (nvml_context silently yields even on failure — we probe via nvmlDeviceGetCount)
            with nvml_context():
                pynvml.nvmlDeviceGetCount()
            return True
        except Exception:
            return False

    def start_tracking(self) -> Any:
        """Start NVML power polling.

        Returns:
            A running PowerThermalSampler instance.
        """
        from llenergymeasure.core.power_thermal import PowerThermalSampler

        sampler = PowerThermalSampler(
            device_index=self._device_index,
            sample_interval_ms=100,
        )
        sampler.start()
        return sampler

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        """Stop polling and integrate power samples to joules.

        Uses the trapezoidal rule: for each consecutive sample pair,
        ``energy += (p[i] + p[i+1]) / 2 * dt``.  Pairs where either
        power_w is None are skipped rather than failing.

        Args:
            tracker: PowerThermalSampler returned by start_tracking().

        Returns:
            EnergyMeasurement with total_j, duration_sec, and raw samples.
        """
        tracker.stop()
        samples = tracker.get_samples()

        total_j = 0.0
        duration_sec = 0.0

        if len(samples) >= 2:
            # Overall window duration from first to last sample timestamp
            duration_sec = samples[-1].timestamp - samples[0].timestamp

            for i in range(len(samples) - 1):
                s_a = samples[i]
                s_b = samples[i + 1]
                if s_a.power_w is None or s_b.power_w is None:
                    continue
                dt = s_b.timestamp - s_a.timestamp
                avg_power = (s_a.power_w + s_b.power_w) / 2.0
                total_j += avg_power * dt

        # Per-device breakdown is device_index -> total_j (single GPU)
        per_gpu_j = {self._device_index: total_j} if total_j > 0.0 else None

        return EnergyMeasurement(
            total_j=total_j,
            duration_sec=duration_sec,
            samples=samples,
            per_gpu_j=per_gpu_j,
        )
