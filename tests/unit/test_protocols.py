"""Protocol conformance tests for EnergySampler and EnginePlugin.

Tests that FakeEnergySampler satisfies the runtime_checkable EnergySampler
Protocol interface defined in llenergymeasure.energy.base, and that the
EnginePlugin protocol has the expected methods.

INF-10 compliance: no unittest.mock.patch on internal modules. Fakes are
injected via constructor args and satisfy isinstance() checks at runtime.
"""

from __future__ import annotations

from llenergymeasure.energy.base import EnergySampler
from llenergymeasure.engines.protocol import EnginePlugin
from tests.fakes import FakeEnergySampler

# ---------------------------------------------------------------------------
# Protocol isinstance checks (structural conformance)
# ---------------------------------------------------------------------------


def test_fake_energy_sampler_satisfies_protocol():
    """FakeEnergySampler satisfies isinstance(EnergySampler) at runtime."""
    fake = FakeEnergySampler()
    assert isinstance(fake, EnergySampler)


# ---------------------------------------------------------------------------
# FakeEnergySampler behaviour
# ---------------------------------------------------------------------------


def test_fake_energy_sampler_lifecycle():
    """start_tracking() then stop_tracking() returns an EnergyMeasurement."""
    from llenergymeasure.energy.nvml import EnergyMeasurement

    fake = FakeEnergySampler(total_j=50.0, duration_sec=10.0)
    tracker = fake.start_tracking()
    measurement = fake.stop_tracking(tracker)

    assert isinstance(measurement, EnergyMeasurement)
    assert measurement.total_j == 50.0
    assert measurement.duration_sec == 10.0


def test_fake_energy_sampler_is_available():
    """FakeEnergySampler.is_available() always returns True."""
    fake = FakeEnergySampler()
    assert fake.is_available() is True


def test_fake_energy_sampler_start_tracking_returns_handle():
    """start_tracking() returns a non-None tracker handle."""
    fake = FakeEnergySampler()
    tracker = fake.start_tracking()
    assert tracker is not None


# ---------------------------------------------------------------------------
# EnginePlugin protocol checks
# ---------------------------------------------------------------------------


def test_engine_plugin_has_run_warmup_prompt():
    """EnginePlugin protocol includes run_warmup_prompt method."""
    assert "run_warmup_prompt" in dir(EnginePlugin)


def test_engine_plugin_protocol_methods():
    """EnginePlugin protocol has all expected methods."""
    expected_methods = [
        "name",
        "load_model",
        "run_warmup_prompt",
        "run_inference",
        "cleanup",
        "validate_config",
        "check_hardware",
    ]
    for method in expected_methods:
        assert method in dir(EnginePlugin), f"EnginePlugin missing: {method}"
