"""Power and thermal sampling during inference.

Provides background sampling of GPU power, memory, temperature, and thermal
throttle state using NVML via the nvidia-ml-py package (imports as pynvml).

Gracefully handles unavailability - returns empty samples and default
ThermalThrottleInfo if NVML is not available (e.g., no GPU, CUDA context
conflicts with vLLM).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from llenergymeasure.core.gpu_info import nvml_context
from llenergymeasure.domain.metrics import ThermalThrottleInfo

logger = logging.getLogger(__name__)


@dataclass
class PowerThermalSample:
    """Single power/thermal sample from GPU."""

    timestamp: float
    power_w: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    sm_utilisation: float | None = None
    thermal_throttle: bool = False
    throttle_reasons: int = 0
    gpu_index: int = 0


class PowerThermalSampler:
    """Background sampler for GPU power, memory, temperature, and throttle state.

    Uses pynvml to sample GPU metrics during inference. Thread-safe context
    manager pattern. Gracefully handles pynvml unavailability.

    Supports monitoring multiple GPUs concurrently. Each sample tick produces
    one PowerThermalSample per monitored GPU, tagged with its gpu_index.

    Usage:
        with PowerThermalSampler(gpu_indices=[0, 1]) as sampler:
            # ... run inference ...
            pass
        samples = sampler.get_samples()
        mean_power = sampler.get_mean_power()  # sum of per-GPU means
        throttle_info = sampler.get_thermal_throttle_info()
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        sample_interval_ms: int = 100,
        device_index: int | None = None,
    ) -> None:
        """Initialise power/thermal sampler.

        Args:
            gpu_indices: CUDA device indices to monitor. Defaults to [0] when None.
            sample_interval_ms: Interval between samples in milliseconds.
            device_index: Deprecated. Use gpu_indices instead. If provided and
                gpu_indices is None, treated as gpu_indices=[device_index].
        """
        if gpu_indices is not None:
            self._gpu_indices = gpu_indices
        elif device_index is not None:
            logger.warning(
                "PowerThermalSampler: device_index is deprecated, use gpu_indices instead"
            )
            self._gpu_indices = [device_index]
        else:
            self._gpu_indices = [0]

        self._sample_interval = sample_interval_ms / 1000.0
        self._sample_interval_ms = sample_interval_ms
        self._samples: list[PowerThermalSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._pynvml_available = False

    def __enter__(self) -> PowerThermalSampler:
        """Start sampling on context entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop sampling on context exit."""
        self.stop()

    def start(self) -> None:
        """Start background sampling thread."""
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for thread to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _sample_loop(self) -> None:
        """Background sampling loop using pynvml — polls all gpu_indices per tick."""
        try:
            import pynvml
        except ImportError:
            logger.debug("Power/thermal: pynvml not available")
            return

        with nvml_context():
            self._pynvml_available = True

            # Get handles for all monitored GPUs
            handles: list[tuple[int, object]] = []
            for gpu_idx in self._gpu_indices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                    handles.append((gpu_idx, handle))
                except Exception as e:
                    logger.debug("Power/thermal: failed to get handle for GPU %d: %s", gpu_idx, e)

            if not handles:
                logger.debug("Power/thermal: no GPU handles obtained")
                return

            # Resolve NVML throttle reason constants (prefer non-deprecated names)
            thermal_bits = 0
            for attr_new, attr_old in [
                (
                    "nvmlClocksEventReasonSwThermalSlowdown",
                    "nvmlClocksThrottleReasonSwThermalSlowdown",
                ),
                (
                    "nvmlClocksEventReasonHwThermalSlowdown",
                    "nvmlClocksThrottleReasonHwThermalSlowdown",
                ),
            ]:
                thermal_bits |= getattr(pynvml, attr_new, getattr(pynvml, attr_old, 0))

            # Prefer non-deprecated clock reasons query (NVML 12+)
            _get_clocks_reasons = getattr(
                pynvml,
                "nvmlDeviceGetCurrentClocksEventReasons",
                getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None),
            )

            try:
                while self._running:
                    for gpu_idx, handle in handles:
                        try:
                            sample = PowerThermalSample(
                                timestamp=time.perf_counter(),
                                gpu_index=gpu_idx,
                            )

                            # Power (milliwatts -> watts)
                            try:
                                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                                sample.power_w = power_mw / 1000.0
                            except pynvml.NVMLError:
                                pass

                            # Memory
                            try:
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                sample.memory_used_mb = mem_info.used / (1024 * 1024)
                                sample.memory_total_mb = mem_info.total / (1024 * 1024)
                            except pynvml.NVMLError:
                                pass

                            # Temperature
                            try:
                                temp = pynvml.nvmlDeviceGetTemperature(
                                    handle, pynvml.NVML_TEMPERATURE_GPU
                                )
                                sample.temperature_c = float(temp)
                            except pynvml.NVMLError:
                                pass

                            # Utilisation
                            try:
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                sample.sm_utilisation = float(util.gpu)
                            except pynvml.NVMLError:
                                pass

                            # Throttle reasons
                            try:
                                if _get_clocks_reasons is not None:
                                    reasons = _get_clocks_reasons(handle)
                                    sample.throttle_reasons = reasons
                                    sample.thermal_throttle = bool(reasons & thermal_bits)
                            except pynvml.NVMLError:
                                pass

                            self._samples.append(sample)

                        except pynvml.NVMLError:
                            # Entire sample for this GPU failed, skip
                            pass

                    time.sleep(self._sample_interval)
            except Exception as e:
                logger.debug("Power/thermal sampling failed: %s", e)

    def get_samples(self) -> list[PowerThermalSample]:
        """Get all collected samples."""
        return list(self._samples)

    def get_power_samples(self) -> list[float]:
        """Get power values only (non-None).

        Returns:
            List of power draw values in Watts (all GPUs interleaved).
        """
        return [s.power_w for s in self._samples if s.power_w is not None]

    def get_mean_power(self) -> float | None:
        """Get mean power draw summed across all monitored GPUs.

        Groups samples by gpu_index, computes the mean per GPU, then sums
        those per-GPU means. This gives total power draw across all GPUs.

        Returns:
            Total mean power in Watts across all GPUs, or None if no samples.
        """
        if not self._samples:
            return None

        # Group by gpu_index
        by_gpu: dict[int, list[float]] = {}
        for s in self._samples:
            if s.power_w is not None:
                by_gpu.setdefault(s.gpu_index, []).append(s.power_w)

        if not by_gpu:
            return None

        # Sum of per-GPU means
        total = 0.0
        for gpu_values in by_gpu.values():
            total += sum(gpu_values) / len(gpu_values)
        return total

    def get_thermal_throttle_info(self) -> ThermalThrottleInfo:
        """Summarise thermal throttle state from collected samples (all GPUs).

        Returns:
            ThermalThrottleInfo with aggregated throttle data across all GPUs.
        """
        if not self._samples:
            return ThermalThrottleInfo()

        any_throttle = any(s.thermal_throttle for s in self._samples)
        throttled_timestamps = [s.timestamp for s in self._samples if s.thermal_throttle]
        throttle_duration = len(throttled_timestamps) * (self._sample_interval_ms / 1000.0)

        temperatures = [s.temperature_c for s in self._samples if s.temperature_c is not None]
        max_temp = max(temperatures) if temperatures else None

        # Check individual throttle reason bits across all samples
        # NVML throttle reason bitmask constants
        try:
            import pynvml

            combined_reasons = 0
            for s in self._samples:
                combined_reasons |= s.throttle_reasons

            hw_thermal_bit = getattr(
                pynvml,
                "nvmlClocksEventReasonHwThermalSlowdown",
                getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0),
            )
            sw_thermal_bit = getattr(
                pynvml,
                "nvmlClocksEventReasonSwThermalSlowdown",
                getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0),
            )
            power_bit = getattr(
                pynvml,
                "nvmlClocksEventReasonSwPowerCap",
                getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", 0),
            )
            hw_power_bit = getattr(
                pynvml,
                "nvmlClocksEventReasonHwPowerBrakeSlowdown",
                getattr(pynvml, "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0),
            )
            # Combined "any thermal" bit: True if either hw or sw thermal throttling occurred
            thermal_bit = hw_thermal_bit | sw_thermal_bit

            return ThermalThrottleInfo(
                detected=any_throttle,
                thermal=bool(combined_reasons & thermal_bit),
                power=bool(combined_reasons & power_bit),
                sw_thermal=bool(combined_reasons & sw_thermal_bit),
                hw_thermal=bool(combined_reasons & hw_thermal_bit),
                hw_power=bool(combined_reasons & hw_power_bit),
                throttle_duration_sec=throttle_duration,
                max_temperature_c=max_temp,
                throttle_timestamps=throttled_timestamps,
            )
        except ImportError:
            # pynvml not available — return basic info from sample flags
            return ThermalThrottleInfo(
                detected=any_throttle,
                throttle_duration_sec=throttle_duration,
                max_temperature_c=max_temp,
                throttle_timestamps=throttled_timestamps,
            )

    @property
    def sample_count(self) -> int:
        """Number of samples collected."""
        return len(self._samples)

    @property
    def is_available(self) -> bool:
        """Whether pynvml sampling is available."""
        return self._pynvml_available


@dataclass
class PowerThermalResult:
    """Result from power/thermal sampling."""

    power_samples: list[float] = field(default_factory=list)
    memory_samples: list[float] = field(default_factory=list)
    temperature_samples: list[float] = field(default_factory=list)
    thermal_throttle_info: ThermalThrottleInfo = field(default_factory=ThermalThrottleInfo)
    sample_count: int = 0
    available: bool = False

    @classmethod
    def from_sampler(cls, sampler: PowerThermalSampler) -> PowerThermalResult:
        """Create result from sampler."""
        samples = sampler.get_samples()
        return cls(
            power_samples=sampler.get_power_samples(),
            memory_samples=[s.memory_used_mb for s in samples if s.memory_used_mb is not None],
            temperature_samples=[s.temperature_c for s in samples if s.temperature_c is not None],
            thermal_throttle_info=sampler.get_thermal_throttle_info(),
            sample_count=sampler.sample_count,
            available=sampler.is_available,
        )
