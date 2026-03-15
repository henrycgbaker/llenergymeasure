"""Energy measurement backends for llenergymeasure.

Provides a plugin registry (legacy) and the v2.0 ``select_energy_backend()``
auto-selection function.

Backend priority for auto-selection: Zeus > NVML > CodeCarbon > None.

Usage (v2.0 API)::

    from llenergymeasure.core.energy_backends import select_energy_backend

    # Auto-select best available backend
    backend = select_energy_backend("auto")

    # Explicit backend (raises ConfigError if unavailable)
    backend = select_energy_backend("zeus")

    # Intentional disable — returns None, no warnings
    backend = select_energy_backend(None)

    if backend is not None:
        tracker = backend.start_tracking()
        # ... run inference ...
        measurement = backend.stop_tracking(tracker)

Legacy registry usage::

    backend = get_backend("codecarbon")

"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from llenergymeasure.core.energy_backends.base import EnergyBackend
from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement, NVMLBackend
from llenergymeasure.core.energy_backends.zeus import ZeusBackend
from llenergymeasure.exceptions import ConfigError, ConfigurationError

if TYPE_CHECKING:
    from llenergymeasure.core.energy_backends.codecarbon import (
        CodeCarbonBackend as CodeCarbonBackend,
    )
    from llenergymeasure.core.energy_backends.codecarbon import (
        CodeCarbonData as CodeCarbonData,
    )
    from llenergymeasure.core.energy_backends.codecarbon import (
        warm_up as warm_up,
    )

# ---------------------------------------------------------------------------
# Legacy plugin registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[EnergyBackend]] = {}


def register_backend(name: str, backend_cls: type[EnergyBackend]) -> None:
    """Register an energy backend.

    Args:
        name: Unique name for the backend.
        backend_cls: Backend class implementing EnergyBackend protocol.

    Raises:
        ValueError: If name is already registered.
    """
    if name in _BACKENDS:
        raise ValueError(f"Backend '{name}' is already registered")
    _BACKENDS[name] = backend_cls


def get_backend(name: str, **kwargs: Any) -> EnergyBackend:
    """Get an instance of a registered backend.

    Args:
        name: Name of the registered backend.
        **kwargs: Arguments to pass to the backend constructor.

    Returns:
        Instance of the requested backend.

    Raises:
        ConfigurationError: If backend name is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys()) or "(none)"
        raise ConfigurationError(f"Unknown backend: '{name}'. Available: {available}")
    return _BACKENDS[name](**kwargs)  # type: ignore[return-value]


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        List of registered backend names.
    """
    return list(_BACKENDS.keys())


def clear_backends() -> None:
    """Clear all registered backends.

    Primarily for testing purposes.
    """
    _BACKENDS.clear()


def _register_default_backends() -> None:
    """Register built-in backends in the legacy registry."""
    register_backend("nvml", NVMLBackend)
    # CodeCarbon requires torch — register lazily only if available
    if importlib.util.find_spec("codecarbon") is not None:
        from llenergymeasure.core.energy_backends.codecarbon import CodeCarbonBackend

        register_backend("codecarbon", CodeCarbonBackend)


# Auto-register on import
_register_default_backends()


# ---------------------------------------------------------------------------
# v2.0 auto-selection API
# ---------------------------------------------------------------------------

_INSTALL_GUIDANCE: dict[str, str] = {
    "zeus": "pip install 'llenergymeasure[zeus]'",
    "nvml": "pip install 'llenergymeasure'  # nvidia-ml-py is a base dependency",
    "codecarbon": "pip install 'llenergymeasure[codecarbon]'",
}


def select_energy_backend(
    explicit: str | None,
    gpu_indices: list[int] | None = None,
) -> EnergyBackend | None:
    """Select and return an energy measurement backend.

    This is the primary v2.0 API for backend selection.

    Selection rules:
    - ``None``: intentional disable — returns ``None`` immediately, no warnings.
    - Specific name (``"nvml"``, ``"zeus"``, ``"codecarbon"``): instantiate that
      backend; raise ``ConfigError`` with install guidance if unavailable.
    - ``"auto"``: probe in priority order (Zeus > NVML > CodeCarbon); return the
      first available backend, or ``None`` if nothing is available (CPU-only machine).

    Args:
        explicit: Backend name, ``"auto"``, or ``None`` to disable energy measurement.
        gpu_indices: GPU device indices to monitor. Defaults to [0] when None.
            Forwarded to NVMLBackend and ZeusBackend constructors.

    Returns:
        An EnergyBackend instance, or ``None`` if measurement is unavailable/disabled.

    Raises:
        ConfigError: When an explicitly requested backend is not available.
    """
    # Intentional disable — null in YAML maps to Python None
    if explicit is None:
        return None

    if explicit == "auto":
        return _auto_select(gpu_indices=gpu_indices)

    # Explicit backend requested
    backend = _instantiate(explicit, gpu_indices=gpu_indices)
    if not backend.is_available():
        guidance = _INSTALL_GUIDANCE.get(explicit, f"pip install llenergymeasure[{explicit}]")
        raise ConfigError(
            f"Energy backend '{explicit}' is not available on this system.\n"
            f"Install with: {guidance}"
        )
    return backend


def _auto_select(gpu_indices: list[int] | None = None) -> EnergyBackend | None:
    """Auto-select best available backend: Zeus > NVML > CodeCarbon > None."""
    # Zeus — preferred: hardware energy register accuracy
    if importlib.util.find_spec("zeus") is not None:
        backend = ZeusBackend(gpu_indices=gpu_indices)
        if backend.is_available():
            return backend

    # NVML — always available on GPU machines (nvidia-ml-py is a base dep)
    nvml_backend = NVMLBackend(gpu_indices=gpu_indices)
    if nvml_backend.is_available():
        return nvml_backend

    # CodeCarbon — software fallback (no gpu_indices: CodeCarbon handles its own GPU detection)
    if importlib.util.find_spec("codecarbon") is not None:
        from llenergymeasure.core.energy_backends.codecarbon import CodeCarbonBackend

        cc_backend = CodeCarbonBackend()
        if cc_backend.is_available():
            return cc_backend

    # CPU-only or no energy measurement available
    return None


def _instantiate(name: str, gpu_indices: list[int] | None = None) -> EnergyBackend:
    """Instantiate a named backend.

    Args:
        name: One of ``"nvml"``, ``"zeus"``, ``"codecarbon"``.
        gpu_indices: GPU device indices to monitor. Forwarded to NVML/Zeus constructors.

    Returns:
        Backend instance (not yet checked for availability).

    Raises:
        ConfigError: If the name is not a known backend.
    """
    if name == "nvml":
        return NVMLBackend(gpu_indices=gpu_indices)
    if name == "zeus":
        return ZeusBackend(gpu_indices=gpu_indices)
    if name == "codecarbon":
        from llenergymeasure.core.energy_backends.codecarbon import CodeCarbonBackend

        return CodeCarbonBackend()

    known = ", ".join(["nvml", "zeus", "codecarbon", "auto"])
    raise ConfigError(
        f"Unknown energy backend: '{name}'. Valid options are: {known}, or null to disable."
    )


__all__ = [
    "CodeCarbonBackend",
    "CodeCarbonData",
    "EnergyBackend",
    "EnergyMeasurement",
    "NVMLBackend",
    "ZeusBackend",
    "clear_backends",
    "get_backend",
    "list_backends",
    "register_backend",
    "select_energy_backend",
    "warm_up",
]
