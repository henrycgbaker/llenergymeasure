"""NVML energy measurement backend.

Uses PowerThermalSampler (pynvml-based) to poll GPU power at 100ms intervals
and integrates samples to joules via the trapezoidal rule.

All pynvml imports are deferred — this module is safe to import on CPU-only machines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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
        gpu_indices: CUDA device indices to monitor. Defaults to [0] when None.
    """

    def __init__(self, gpu_indices: list[int] | None = None) -> None:
        self._gpu_indices = gpu_indices if gpu_indices is not None else [0]

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

            from llenergymeasure.device.gpu_info import nvml_context

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
        from llenergymeasure.device.power_thermal import PowerThermalSampler

        sampler = PowerThermalSampler(
            gpu_indices=self._gpu_indices,
            sample_interval_ms=100,
        )
        sampler.start()
        return sampler

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        """Stop polling and integrate power samples to joules per GPU.

        Uses the trapezoidal rule: for each consecutive sample pair (per GPU),
        ``energy += (p[i] + p[i+1]) / 2 * dt``.  Pairs where either
        power_w is None are skipped rather than failing.

        Args:
            tracker: PowerThermalSampler returned by start_tracking().

        Returns:
            EnergyMeasurement with total_j (sum across all GPUs), duration_sec,
            per_gpu_j breakdown, and raw samples.
        """
        tracker.stop()
        samples = tracker.get_samples()

        total_j = 0.0
        duration_sec = 0.0
        per_gpu_j: dict[int, float] = {}

        if len(samples) >= 2:
            # Overall window duration from first to last sample timestamp across all GPUs
            duration_sec = samples[-1].timestamp - samples[0].timestamp

            # Check for sampling rate drift: warn if any consecutive gap exceeds 200ms
            # (2x the 100ms target interval), which would degrade energy accuracy.
            timestamps = [s.timestamp for s in samples]
            gaps_ms = [
                (timestamps[i + 1] - timestamps[i]) * 1000.0 for i in range(len(timestamps) - 1)
            ]
            if gaps_ms:
                max_gap_ms = max(gaps_ms)
                if max_gap_ms > 200.0:
                    logger.warning(
                        "Power sampling gap of %.0fms detected (target: 100ms). "
                        "Energy calculation may be less accurate.",
                        max_gap_ms,
                    )

            # Group samples by gpu_index for per-GPU integration
            by_gpu: dict[int, list[Any]] = {}
            for s in samples:
                gpu_idx = getattr(s, "gpu_index", self._gpu_indices[0])
                by_gpu.setdefault(gpu_idx, []).append(s)

            for gpu_idx, gpu_samples in by_gpu.items():
                gpu_j = 0.0
                for i in range(len(gpu_samples) - 1):
                    s_a = gpu_samples[i]
                    s_b = gpu_samples[i + 1]
                    if s_a.power_w is None or s_b.power_w is None:
                        continue
                    dt = s_b.timestamp - s_a.timestamp
                    avg_power = (s_a.power_w + s_b.power_w) / 2.0
                    gpu_j += avg_power * dt
                per_gpu_j[gpu_idx] = gpu_j
                total_j += gpu_j

        return EnergyMeasurement(
            total_j=total_j,
            duration_sec=duration_sec,
            samples=samples,
            per_gpu_j=per_gpu_j if per_gpu_j else None,
        )
