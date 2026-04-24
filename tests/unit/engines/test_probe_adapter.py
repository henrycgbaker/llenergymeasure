"""Unit tests for ``engines.probe_adapter.build_config_probe``.

The adapter composes hardware errors (from ``EnginePlugin.check_hardware``)
with dormancy observations (from ``ExperimentConfig._apply_vendored_rules``)
into a :class:`ConfigProbe`. M1 leaves effective-params placeholders empty;
they come to life when the M2 introspection walker supplies the surface.
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.engine_configs import TensorRTConfig, TensorRTQuantConfig
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.probe import ConfigProbe, DormantField
from llenergymeasure.engines.probe_adapter import build_config_probe
from llenergymeasure.utils.exceptions import EngineError
from tests.conftest import make_config


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    """Default: NVML reports no GPU. Individual tests can override."""
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: None,
    )


# ---------------------------------------------------------------------------
# Basic composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine", ["transformers", "vllm", "tensorrt"], ids=["transformers", "vllm", "tensorrt"]
)
def test_build_config_probe_returns_probe_per_engine(engine):
    """Each of the three engines yields a ConfigProbe instance."""
    config = make_config(model="test-model", engine=engine)
    probe = build_config_probe(config)
    assert isinstance(probe, ConfigProbe)


def test_unknown_engine_raises_engine_error():
    """``get_engine`` raises ``EngineError`` on an unknown engine name."""
    config = ExperimentConfig(task={"model": "test-model"})
    # Force-override the engine literal without going through Pydantic
    object.__setattr__(config, "engine", "unknown-engine")
    with pytest.raises(EngineError):
        build_config_probe(config)


# ---------------------------------------------------------------------------
# Hardware errors land in ConfigProbe.errors
# ---------------------------------------------------------------------------


def test_hardware_errors_surface_on_probe(monkeypatch):
    """check_hardware errors (SM below floor) appear in ``probe.errors``."""
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: (7, 0),
    )
    config = make_config(model="test-model", engine="tensorrt")
    probe = build_config_probe(config)
    assert not probe.is_valid
    assert any("SM >= 7.5" in e for e in probe.errors)


def test_check_hardware_exception_captured(monkeypatch):
    """Defensive: if check_hardware raises, the exception is captured into errors."""

    def boom(_config):
        raise RuntimeError("NVML crashed")

    # Patch the TRT engine's check_hardware to raise
    from llenergymeasure.engines.tensorrt import TensorRTEngine

    monkeypatch.setattr(TensorRTEngine, "check_hardware", staticmethod(boom))

    config = make_config(model="test-model", engine="tensorrt")
    probe = build_config_probe(config)
    assert len(probe.errors) == 1
    assert "RuntimeError" in probe.errors[0]
    assert "NVML crashed" in probe.errors[0]


# ---------------------------------------------------------------------------
# Dormancy passes through from the generic validator
# ---------------------------------------------------------------------------


def test_dormancy_propagated_from_config(monkeypatch):
    """``config._dormant_observations`` is copied verbatim into probe.dormant_fields."""
    config = make_config(model="test-model", engine="transformers")
    fake = {
        "transformers.sampling.temperature": DormantField(
            declared_value=0.9, effective_value=None, reason="greedy"
        )
    }
    object.__setattr__(config, "_dormant_observations", fake)

    probe = build_config_probe(config)
    assert probe.dormant_fields == fake


def test_missing_dormant_observations_yields_empty(monkeypatch):
    """Adapter does not crash when ``_dormant_observations`` is absent/None."""
    config = make_config(model="test-model", engine="transformers")
    object.__setattr__(config, "_dormant_observations", None)
    probe = build_config_probe(config)
    assert probe.dormant_fields == {}


# ---------------------------------------------------------------------------
# M1 contract: effective-params fields start empty (M2 supply pipeline fills them)
# ---------------------------------------------------------------------------


def test_effective_params_empty_in_m1():
    """observed_engine_params and observed_sampling_params are placeholders in M1.

    Delete/flip when the M2 introspection walker supplies the effective-kwargs surface.
    """
    config = make_config(model="test-model", engine="transformers")
    probe = build_config_probe(config)
    assert probe.observed_engine_params == {}
    assert probe.observed_sampling_params == {}


# ---------------------------------------------------------------------------
# Preflight-substitution invariant: build_config_probe(c).errors ==
# check_hardware(c) when hardware is the only error source.
# ---------------------------------------------------------------------------


def test_probe_errors_equal_check_hardware_for_hardware_only(monkeypatch):
    """Pins the contract preflight relies on (Check 4 substitution)."""
    monkeypatch.setattr(
        "llenergymeasure.device.gpu_info.get_compute_capability",
        lambda gpu_index=0: (8, 0),
    )
    config = make_config(
        model="test-model",
        engine="tensorrt",
        tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(quant_algo="FP8")),
    )

    from llenergymeasure.engines.tensorrt import TensorRTEngine

    probe = build_config_probe(config)
    hardware_errors = TensorRTEngine.check_hardware(config)
    assert probe.errors == hardware_errors
