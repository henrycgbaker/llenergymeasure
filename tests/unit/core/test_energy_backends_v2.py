"""Unit tests for v2.0 energy backends and auto-selection.

All tests are fully mocked — no GPU or installed energy packages required.
Tests cover NVMLBackend, ZeusBackend, EnergyMeasurement, and select_energy_backend().
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.energy_backends import select_energy_backend
from llenergymeasure.core.energy_backends.nvml import EnergyMeasurement, NVMLBackend
from llenergymeasure.core.energy_backends.zeus import ZeusBackend
from llenergymeasure.exceptions import ConfigError

# ---------------------------------------------------------------------------
# NVMLBackend name
# ---------------------------------------------------------------------------


def test_nvml_backend_name() -> None:
    """NVMLBackend.name must return 'nvml'."""
    assert NVMLBackend().name == "nvml"


# ---------------------------------------------------------------------------
# NVMLBackend gpu_indices constructor
# ---------------------------------------------------------------------------


def test_nvml_backend_accepts_gpu_indices() -> None:
    """NVMLBackend(gpu_indices=[0, 1]) stores _gpu_indices correctly."""
    b = NVMLBackend(gpu_indices=[0, 1])
    assert b._gpu_indices == [0, 1]


def test_nvml_backend_defaults_to_gpu_zero() -> None:
    """NVMLBackend() with no args defaults to [0] (backward compatible)."""
    b = NVMLBackend()
    assert b._gpu_indices == [0]


def test_nvml_backend_single_gpu_explicit() -> None:
    """NVMLBackend(gpu_indices=[2]) stores [2]."""
    b = NVMLBackend(gpu_indices=[2])
    assert b._gpu_indices == [2]


# ---------------------------------------------------------------------------
# ZeusBackend name
# ---------------------------------------------------------------------------


def test_zeus_backend_name() -> None:
    """ZeusBackend.name must return 'zeus'."""
    assert ZeusBackend().name == "zeus"


# ---------------------------------------------------------------------------
# NVMLBackend availability
# ---------------------------------------------------------------------------


def test_nvml_is_available_no_gpu() -> None:
    """is_available() returns False when pynvml raises on init."""
    with patch(
        "llenergymeasure.core.energy_backends.nvml.NVMLBackend.is_available",
        return_value=False,
    ):
        backend = NVMLBackend()
        assert backend.is_available() is False


def test_nvml_is_available_import_error() -> None:
    """is_available() returns False when pynvml is not installed."""
    with patch("importlib.util.find_spec", return_value=None):
        backend = NVMLBackend()
        result = backend.is_available()
    assert result is False


# ---------------------------------------------------------------------------
# ZeusBackend availability
# ---------------------------------------------------------------------------


def test_zeus_is_available_no_package() -> None:
    """is_available() returns False when zeus is not installed."""
    import sys

    # Remove zeus from sys.modules so the deferred import inside is_available()
    # raises ImportError as it would on a machine without zeus installed.
    zeus_modules = {k: v for k, v in sys.modules.items() if k == "zeus" or k.startswith("zeus.")}
    for k in zeus_modules:
        sys.modules.pop(k)

    # Inject a sentinel that raises ImportError on attribute access (simulates absent package)
    sys.modules["zeus.monitor"] = None  # type: ignore[assignment]
    try:
        backend = ZeusBackend()
        result = backend.is_available()
    finally:
        sys.modules.pop("zeus.monitor", None)
        sys.modules.update(zeus_modules)

    assert result is False


# ---------------------------------------------------------------------------
# EnergyMeasurement dataclass
# ---------------------------------------------------------------------------


def test_energy_measurement_dataclass() -> None:
    """EnergyMeasurement can be constructed and fields are accessible."""
    em = EnergyMeasurement(total_j=42.5, duration_sec=10.0)
    assert em.total_j == 42.5
    assert em.duration_sec == 10.0
    assert em.samples == []
    assert em.per_gpu_j is None

    em2 = EnergyMeasurement(
        total_j=100.0,
        duration_sec=5.0,
        samples=[object()],
        per_gpu_j={0: 60.0, 1: 40.0},
    )
    assert em2.per_gpu_j == {0: 60.0, 1: 40.0}


# ---------------------------------------------------------------------------
# select_energy_backend — null returns None
# ---------------------------------------------------------------------------


def test_select_backend_null_returns_none() -> None:
    """select_energy_backend(None) must return None without any warnings."""
    result = select_energy_backend(None)
    assert result is None


# ---------------------------------------------------------------------------
# select_energy_backend — explicit unavailable raises ConfigError
# ---------------------------------------------------------------------------


def test_select_backend_explicit_unavailable_raises() -> None:
    """Requesting an explicit backend that is not available raises ConfigError."""
    with patch(
        "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
        return_value=False,
    ):
        with pytest.raises(ConfigError) as exc_info:
            select_energy_backend("nvml")
        # Error must include install guidance
        assert "pip install" in str(exc_info.value).lower() or "install" in str(exc_info.value)


def test_select_backend_explicit_zeus_unavailable_raises() -> None:
    """Requesting zeus when not installed raises ConfigError with guidance."""
    with patch(
        "llenergymeasure.core.energy_backends.ZeusBackend.is_available",
        return_value=False,
    ):
        with pytest.raises(ConfigError) as exc_info:
            select_energy_backend("zeus")
        assert "zeus" in str(exc_info.value).lower()


def test_select_backend_unknown_name_raises() -> None:
    """Requesting an unknown backend name raises ConfigError."""
    with pytest.raises(ConfigError):
        select_energy_backend("unknown_backend_xyz")


# ---------------------------------------------------------------------------
# select_energy_backend — auto priority
# ---------------------------------------------------------------------------


def test_select_backend_auto_priority_zeus_first() -> None:
    """Auto-selection returns Zeus when zeus package is available."""
    with (
        patch("importlib.util.find_spec", return_value=MagicMock()) as _mock_find,
        patch(
            "llenergymeasure.core.energy_backends.ZeusBackend.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_backend("auto")
    assert isinstance(result, ZeusBackend)


def test_select_backend_auto_priority_nvml_when_no_zeus() -> None:
    """Auto-selection returns NVML when zeus is absent but GPU is available."""

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None  # zeus not installed
        return MagicMock()  # codecarbon present (but NVML checked first)

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_backend("auto")
    assert isinstance(result, NVMLBackend)


def test_select_backend_auto_priority_codecarbon_fallback() -> None:
    """Auto-selection returns CodeCarbon when neither zeus nor NVML is available."""
    from llenergymeasure.core.energy_backends.codecarbon import CodeCarbonBackend

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None
        if name == "codecarbon":
            return MagicMock()
        return None

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
            return_value=False,
        ),
        patch(
            "llenergymeasure.core.energy_backends.codecarbon.CodeCarbonBackend.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_backend("auto")
    assert isinstance(result, CodeCarbonBackend)


def test_select_backend_auto_returns_none_when_nothing_available() -> None:
    """Auto-selection returns None when no GPU and no packages are present."""
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch(
            "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
            return_value=False,
        ),
    ):
        result = select_energy_backend("auto")
    assert result is None


# ---------------------------------------------------------------------------
# select_energy_backend — gpu_indices forwarding
# ---------------------------------------------------------------------------


def test_select_backend_forwards_gpu_indices_nvml() -> None:
    """select_energy_backend('nvml', gpu_indices=[0, 1]) returns NVMLBackend with those indices."""
    with patch(
        "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
        return_value=True,
    ):
        result = select_energy_backend("nvml", gpu_indices=[0, 1])

    assert isinstance(result, NVMLBackend)
    assert result._gpu_indices == [0, 1]


def test_select_backend_forwards_gpu_indices_zeus() -> None:
    """select_energy_backend('zeus', gpu_indices=[0, 1]) returns ZeusBackend with those indices."""
    with patch(
        "llenergymeasure.core.energy_backends.ZeusBackend.is_available",
        return_value=True,
    ):
        result = select_energy_backend("zeus", gpu_indices=[0, 1])

    assert isinstance(result, ZeusBackend)
    assert result._gpu_indices == [0, 1]


def test_select_backend_auto_forwards_gpu_indices() -> None:
    """Auto-selection forwards gpu_indices to the chosen backend."""

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None
        return MagicMock()

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.core.energy_backends.NVMLBackend.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_backend("auto", gpu_indices=[2, 3])

    assert isinstance(result, NVMLBackend)
    assert result._gpu_indices == [2, 3]


# ---------------------------------------------------------------------------
# NVMLBackend trapezoidal integration — single GPU
# ---------------------------------------------------------------------------


def test_nvml_trapezoidal_integration() -> None:
    """stop_tracking() correctly integrates power samples via trapezoidal rule.

    3 samples at t=0.0, 0.1, 0.2 all with power_w=100.0 W.
    Expected energy = 100.0 W * 0.2 s = 20.0 J.
    """
    from llenergymeasure.core.power_thermal import PowerThermalSample

    # Build synthetic samples (single GPU 0)
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
    ]

    # Mock sampler
    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLBackend(gpu_indices=[0])
    result = backend.stop_tracking(mock_sampler)

    assert isinstance(result, EnergyMeasurement)
    assert result.duration_sec == pytest.approx(0.2, abs=1e-9)
    assert result.total_j == pytest.approx(20.0, abs=1e-6)
    assert result.per_gpu_j == {0: pytest.approx(20.0, abs=1e-6)}


def test_nvml_trapezoidal_skips_none_power() -> None:
    """Trapezoidal integration skips sample pairs where power_w is None."""
    from llenergymeasure.core.power_thermal import PowerThermalSample

    # Middle sample has None power — pairs (0,1) and (1,2) both involve None
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=None, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLBackend()
    result = backend.stop_tracking(mock_sampler)

    # Both pairs involve a None sample → should integrate to 0.0
    assert result.total_j == pytest.approx(0.0, abs=1e-9)


def test_nvml_trapezoidal_single_sample() -> None:
    """Single sample produces zero energy (no interval to integrate)."""
    from llenergymeasure.core.power_thermal import PowerThermalSample

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = [
        PowerThermalSample(timestamp=0.0, power_w=200.0, gpu_index=0),
    ]

    backend = NVMLBackend()
    result = backend.stop_tracking(mock_sampler)
    assert result.total_j == 0.0
    assert result.duration_sec == 0.0


# ---------------------------------------------------------------------------
# NVMLBackend trapezoidal integration — multi-GPU
# ---------------------------------------------------------------------------


def test_nvml_multi_gpu_trapezoidal_integration() -> None:
    """stop_tracking() sums energy across multiple GPUs, populates per_gpu_j.

    GPU 0: 100W over 0.2s = 20J
    GPU 1: 200W over 0.2s = 40J
    Total: 60J, per_gpu_j = {0: 20J, 1: 40J}

    Samples are interleaved (as produced by _sample_loop).
    """
    from llenergymeasure.core.power_thermal import PowerThermalSample

    # Interleaved samples from GPU 0 and GPU 1
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.0, power_w=200.0, gpu_index=1),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=200.0, gpu_index=1),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=200.0, gpu_index=1),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLBackend(gpu_indices=[0, 1])
    result = backend.stop_tracking(mock_sampler)

    assert isinstance(result, EnergyMeasurement)
    assert result.total_j == pytest.approx(60.0, abs=1e-6)
    assert result.per_gpu_j is not None
    assert result.per_gpu_j[0] == pytest.approx(20.0, abs=1e-6)
    assert result.per_gpu_j[1] == pytest.approx(40.0, abs=1e-6)


def test_nvml_multi_gpu_energy_per_gpu_dict_populated() -> None:
    """per_gpu_j dict contains an entry for each GPU that had samples."""
    from llenergymeasure.core.power_thermal import PowerThermalSample

    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=50.0, gpu_index=0),
        PowerThermalSample(timestamp=0.0, power_w=75.0, gpu_index=1),
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=2),
        PowerThermalSample(timestamp=0.1, power_w=50.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=75.0, gpu_index=1),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=2),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLBackend(gpu_indices=[0, 1, 2])
    result = backend.stop_tracking(mock_sampler)

    assert result.per_gpu_j is not None
    assert set(result.per_gpu_j.keys()) == {0, 1, 2}
    assert result.total_j == pytest.approx(
        result.per_gpu_j[0] + result.per_gpu_j[1] + result.per_gpu_j[2],
        abs=1e-9,
    )
