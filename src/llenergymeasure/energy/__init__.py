"""Energy measurement samplers for llenergymeasure.

Provides the ``select_energy_sampler()`` auto-selection function.

Sampler priority for auto-selection: Zeus > NVML > CodeCarbon > None.

Usage::

    from llenergymeasure.energy import select_energy_sampler

    # Auto-select best available sampler
    sampler = select_energy_sampler("auto")

    # Explicit sampler (raises ConfigError if unavailable)
    sampler = select_energy_sampler("zeus")

    # Intentional disable — returns None, no warnings
    sampler = select_energy_sampler(None)

    if sampler is not None:
        tracker = sampler.start_tracking()
        # ... run inference ...
        measurement = sampler.stop_tracking(tracker)

"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from llenergymeasure.energy.base import EnergySampler  # canonical definition
from llenergymeasure.energy.nvml import EnergyMeasurement, NVMLSampler
from llenergymeasure.energy.zeus import ZeusSampler
from llenergymeasure.utils.exceptions import ConfigError

if TYPE_CHECKING:
    from llenergymeasure.energy.codecarbon import (
        CodeCarbonData as CodeCarbonData,
    )
    from llenergymeasure.energy.codecarbon import (
        CodeCarbonSampler as CodeCarbonSampler,
    )
    from llenergymeasure.energy.codecarbon import (
        warm_up as warm_up,
    )


# ---------------------------------------------------------------------------
# v2.0 auto-selection API
# ---------------------------------------------------------------------------

_INSTALL_GUIDANCE: dict[str, str] = {
    "zeus": "pip install 'llenergymeasure[zeus]'",
    "nvml": "pip install 'llenergymeasure'  # nvidia-ml-py is a base dependency",
    "codecarbon": "pip install 'llenergymeasure[codecarbon]'",
}


def select_energy_sampler(
    explicit: str | None,
    gpu_indices: list[int] | None = None,
) -> EnergySampler | None:
    """Select and return an energy measurement sampler.

    This is the primary API for sampler selection.

    Selection rules:
    - ``None``: intentional disable — returns ``None`` immediately, no warnings.
    - Specific name (``"nvml"``, ``"zeus"``, ``"codecarbon"``): instantiate that
      sampler; raise ``ConfigError`` with install guidance if unavailable.
    - ``"auto"``: probe in priority order (Zeus > NVML > CodeCarbon); return the
      first available sampler, or ``None`` if nothing is available (CPU-only machine).

    Args:
        explicit: Sampler name, ``"auto"``, or ``None`` to disable energy measurement.
        gpu_indices: GPU device indices to monitor. Defaults to [0] when None.
            Forwarded to NVMLSampler and ZeusSampler constructors.

    Returns:
        An EnergySampler instance, or ``None`` if measurement is unavailable/disabled.

    Raises:
        ConfigError: When an explicitly requested sampler is not available.
    """
    # Intentional disable — null in YAML maps to Python None
    if explicit is None:
        return None

    if explicit == "auto":
        return _auto_select(gpu_indices=gpu_indices)

    # Explicit sampler requested
    sampler = _instantiate(explicit, gpu_indices=gpu_indices)
    if not sampler.is_available():
        guidance = _INSTALL_GUIDANCE.get(explicit, f"pip install llenergymeasure[{explicit}]")
        raise ConfigError(
            f"Energy sampler '{explicit}' is not available on this system.\n"
            f"Install with: {guidance}"
        )
    return sampler


def _auto_select(gpu_indices: list[int] | None = None) -> EnergySampler | None:
    """Auto-select best available sampler: Zeus > NVML > CodeCarbon > None."""
    # Zeus — preferred: hardware energy register accuracy
    if importlib.util.find_spec("zeus") is not None:
        sampler = ZeusSampler(gpu_indices=gpu_indices)
        if sampler.is_available():
            return sampler

    # NVML — always available on GPU machines (nvidia-ml-py is a base dep)
    nvml_sampler = NVMLSampler(gpu_indices=gpu_indices)
    if nvml_sampler.is_available():
        return nvml_sampler

    # CodeCarbon — software fallback (no gpu_indices: CodeCarbon handles its own GPU detection)
    if importlib.util.find_spec("codecarbon") is not None:
        from llenergymeasure.energy.codecarbon import CodeCarbonSampler

        cc_sampler = CodeCarbonSampler()
        if cc_sampler.is_available():
            return cc_sampler

    # CPU-only or no energy measurement available
    return None


def _instantiate(name: str, gpu_indices: list[int] | None = None) -> EnergySampler:
    """Instantiate a named sampler.

    Args:
        name: One of ``"nvml"``, ``"zeus"``, ``"codecarbon"``.
        gpu_indices: GPU device indices to monitor. Forwarded to NVML/Zeus constructors.

    Returns:
        Sampler instance (not yet checked for availability).

    Raises:
        ConfigError: If the name is not a known sampler.
    """
    if name == "nvml":
        return NVMLSampler(gpu_indices=gpu_indices)
    if name == "zeus":
        return ZeusSampler(gpu_indices=gpu_indices)
    if name == "codecarbon":
        from llenergymeasure.energy.codecarbon import CodeCarbonSampler

        return CodeCarbonSampler()

    known = ", ".join(["nvml", "zeus", "codecarbon", "auto"])
    raise ConfigError(
        f"Unknown energy sampler: '{name}'. Valid options are: {known}, or null to disable."
    )


__all__ = [
    "CodeCarbonData",
    "CodeCarbonSampler",
    "EnergyMeasurement",
    "EnergySampler",
    "NVMLSampler",
    "ZeusSampler",
    "select_energy_sampler",
    "warm_up",
]
