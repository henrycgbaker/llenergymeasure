"""Protocol conformance tests for EnergyBackend.

Tests that FakeEnergyBackend satisfies the runtime_checkable EnergyBackend
Protocol interface defined in llenergymeasure.energy.base.

INF-10 compliance: no unittest.mock.patch on internal modules. Fakes are
injected via constructor args and satisfy isinstance() checks at runtime.
"""

from __future__ import annotations

from llenergymeasure.energy.base import EnergyBackend
from tests.fakes import FakeEnergyBackend

# ---------------------------------------------------------------------------
# Protocol isinstance checks (structural conformance)
# ---------------------------------------------------------------------------


def test_fake_energy_backend_satisfies_protocol():
    """FakeEnergyBackend satisfies isinstance(EnergyBackend) at runtime."""
    fake = FakeEnergyBackend()
    assert isinstance(fake, EnergyBackend)


# ---------------------------------------------------------------------------
# FakeEnergyBackend behaviour
# ---------------------------------------------------------------------------


def test_fake_energy_backend_lifecycle():
    """start_tracking() then stop_tracking() returns an EnergyMeasurement."""
    from llenergymeasure.energy.nvml import EnergyMeasurement

    fake = FakeEnergyBackend(total_j=50.0, duration_sec=10.0)
    tracker = fake.start_tracking()
    measurement = fake.stop_tracking(tracker)

    assert isinstance(measurement, EnergyMeasurement)
    assert measurement.total_j == 50.0
    assert measurement.duration_sec == 10.0


def test_fake_energy_backend_is_available():
    """FakeEnergyBackend.is_available() always returns True."""
    fake = FakeEnergyBackend()
    assert fake.is_available() is True


def test_fake_energy_backend_start_tracking_returns_handle():
    """start_tracking() returns a non-None tracker handle."""
    fake = FakeEnergyBackend()
    tracker = fake.start_tracking()
    assert tracker is not None
