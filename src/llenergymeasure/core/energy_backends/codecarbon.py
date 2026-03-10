"""CodeCarbon energy tracking backend."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import torch

from llenergymeasure.domain.metrics import EnergyMetrics

logger = logging.getLogger(__name__)

# Suppress FutureWarning from codecarbon's pandas usage
# (DataFrame concatenation with empty columns deprecation)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries",
    category=FutureWarning,
    module="codecarbon",
)


@dataclass
class CodeCarbonData:
    """Structured CodeCarbon emissions data."""

    cpu_power: float | None
    gpu_power: float | None
    ram_power: float | None
    cpu_energy: float | None
    gpu_energy: float | None
    ram_energy: float | None
    total_energy_kwh: float
    emissions_kg: float | None


class CodeCarbonBackend:
    """Energy tracking backend using CodeCarbon.

    CodeCarbon tracks energy consumption at the process level,
    measuring CPU, GPU, and RAM power usage.

    Implements the EnergyBackend protocol.
    """

    def __init__(
        self,
        measure_power_secs: int = 1,
        tracking_mode: str = "process",
    ) -> None:
        """Initialize CodeCarbon backend.

        Args:
            measure_power_secs: Interval between power measurements.
            tracking_mode: Either 'process' or 'machine'.
        """
        self._measure_power_secs = measure_power_secs
        self._tracking_mode = tracking_mode

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return "codecarbon"

    def is_available(self) -> bool:
        """Check if CodeCarbon is available on this system.

        Returns:
            True if codecarbon package is installed.
        """
        try:
            import codecarbon  # noqa: F401

            return True
        except ImportError:
            return False

    def start_tracking(self) -> Any:
        """Start energy tracking.

        Returns:
            EmissionsTracker instance for stop_tracking, or None if unavailable.
        """
        from codecarbon import EmissionsTracker

        # Suppress codecarbon's verbose logging
        logging.getLogger("codecarbon").setLevel(logging.ERROR)

        try:
            tracker = EmissionsTracker(
                measure_power_secs=self._measure_power_secs,
                allow_multiple_runs=True,
                tracking_mode=self._tracking_mode,
                log_level=logging.ERROR,
            )
            tracker.start()
            logger.debug("CodeCarbon tracking started")
            return tracker
        except Exception as e:
            # Handle NVML permission errors in containers, etc.
            logger.warning("Energy tracking unavailable: %s", e)
            logger.warning("Continuing without energy metrics (common in containers)")
            return None

    def stop_tracking(self, tracker: Any) -> EnergyMetrics:
        """Stop tracking and return energy metrics.

        Args:
            tracker: EmissionsTracker from start_tracking.

        Returns:
            EnergyMetrics with collected data.
        """
        if tracker is None:
            logger.debug("Energy tracker was unavailable, returning empty metrics")
            return self._empty_metrics()

        try:
            tracker.stop()
            data = self._extract_data(tracker)
            return self._convert_to_metrics(data)
        except Exception as e:
            logger.warning("Failed to stop energy tracking: %s", e)
            return self._empty_metrics()

    def _empty_metrics(self) -> EnergyMetrics:
        """Return empty metrics for error cases."""
        return EnergyMetrics(
            total_energy_j=0.0,
            gpu_energy_j=0.0,
            cpu_energy_j=0.0,
            ram_energy_j=0.0,
            gpu_power_w=0.0,
            cpu_power_w=0.0,
            duration_sec=0.0,
            emissions_kg_co2=0.0,
            energy_per_token_j=0.0,
        )

    def _extract_data(self, tracker: Any) -> CodeCarbonData:
        """Extract data from the tracker."""
        try:
            emissions_data = tracker._prepare_emissions_data()
        except AttributeError:
            logger.warning("Could not extract emissions data")
            return CodeCarbonData(
                cpu_power=None,
                gpu_power=None,
                ram_power=None,
                cpu_energy=None,
                gpu_energy=None,
                ram_energy=None,
                total_energy_kwh=0.0,
                emissions_kg=None,
            )

        energy_kwh = getattr(emissions_data, "energy_consumed", 0.0) or 0.0

        return CodeCarbonData(
            cpu_power=getattr(emissions_data, "cpu_power", None),
            gpu_power=getattr(emissions_data, "gpu_power", None),
            ram_power=getattr(emissions_data, "ram_power", None),
            cpu_energy=getattr(emissions_data, "cpu_energy", None),
            gpu_energy=getattr(emissions_data, "gpu_energy", None),
            ram_energy=getattr(emissions_data, "ram_energy", None),
            total_energy_kwh=energy_kwh,
            emissions_kg=getattr(emissions_data, "emissions", None),
        )

    def _convert_to_metrics(self, data: CodeCarbonData) -> EnergyMetrics:
        """Convert CodeCarbon data to our EnergyMetrics format."""
        # Convert kWh to Joules (1 kWh = 3.6e6 J)
        total_energy_j = data.total_energy_kwh * 3.6e6

        # Convert individual energy values (if available)
        gpu_energy_j = (data.gpu_energy or 0.0) * 3.6e6 if data.gpu_energy else 0.0
        cpu_energy_j = (data.cpu_energy or 0.0) * 3.6e6 if data.cpu_energy else 0.0
        ram_energy_j = (data.ram_energy or 0.0) * 3.6e6 if data.ram_energy else 0.0

        # Duration is difficult to get from CodeCarbon directly
        # We'll set it to 0 here - the caller should set it from their timing
        duration_sec = 0.0

        return EnergyMetrics(
            total_energy_j=total_energy_j,
            gpu_energy_j=gpu_energy_j,
            cpu_energy_j=cpu_energy_j,
            ram_energy_j=ram_energy_j,
            gpu_power_w=data.gpu_power or 0.0,
            cpu_power_w=data.cpu_power or 0.0,
            duration_sec=duration_sec,
            emissions_kg_co2=data.emissions_kg or 0.0,
            energy_per_token_j=0.0,  # Caller should set this
        )

    def get_raw_data(self, tracker: Any) -> CodeCarbonData | None:
        """Get the raw CodeCarbon data for detailed analysis.

        Args:
            tracker: EmissionsTracker from start_tracking.

        Returns:
            CodeCarbonData if tracking was started, None otherwise.
        """
        if tracker is None:
            return None
        return self._extract_data(tracker)


def warm_up(
    model: Any,
    tokenizer: Any,
    max_input_tokens: int,
    num_warmup_runs: int = 3,
    warmup_prompt: str = "This is a warm-up run.",
) -> None:
    """Run warm-up iterations to initialize GPU and caches.

    Args:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
        max_input_tokens: Maximum input token length.
        num_warmup_runs: Number of warm-up iterations.
        warmup_prompt: Prompt text to use for warm-up.
    """
    logger.debug("Running %d warm-up iterations", num_warmup_runs)

    for i in range(num_warmup_runs):
        dummy_input = tokenizer(
            warmup_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        # Move to model's device
        dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}

        with torch.no_grad():
            _ = model(**dummy_input)

        logger.debug("Warm-up iteration %d/%d completed", i + 1, num_warmup_runs)

    logger.debug("Warm-up complete")
