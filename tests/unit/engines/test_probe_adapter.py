"""Tests for :func:`llenergymeasure.engines.probe_adapter.build_config_probe`.

Covers the adapter's three responsibilities:
1. Read ``_dormant_observations`` from a validated config and surface them
   through ``ConfigProbe.dormant_fields``.
2. Dispatch to the engine plugin's ``check_hardware`` and surface errors
   through ``ConfigProbe.errors``.
3. Stay silent on missing engines / unexpected exceptions (never raise).
"""

from __future__ import annotations

from unittest.mock import patch

from llenergymeasure.config.engine_configs import (
    TransformersConfig,
    TransformersSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig, _reset_rules_loader_cache
from llenergymeasure.engines.probe_adapter import build_config_probe
from llenergymeasure.engines.protocol import ConfigProbe, DormantField


def _make_fake_plugin(errors: list[str] | None = None, raises: Exception | None = None):
    """Stub plugin whose :meth:`check_hardware` returns *errors* (or raises)."""

    class _FakePlugin:
        def check_hardware(self, config):
            if raises is not None:
                raise raises
            return list(errors or [])

    return _FakePlugin()


# ---------------------------------------------------------------------------
# Dormancy pass-through
# ---------------------------------------------------------------------------


def test_adapter_surfaces_dormant_observations_from_config():
    """Dormancy observations populated by the generic validator appear in the probe."""
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(do_sample=False, temperature=0.9),
        ),
    )
    # Substitute the engine plugin with one that returns no hardware errors so
    # the test isolates the dormancy pass-through.
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)

    assert isinstance(probe, ConfigProbe)
    assert probe.dormant_fields
    # Keys are engine-prefixed for preflight display parity.
    assert all(key.startswith("transformers.") for key in probe.dormant_fields)


def test_adapter_keeps_empty_dormant_when_none_observed():
    """A config that triggers no dormant rules produces an empty dormant dict."""
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)
    assert probe.dormant_fields == {}


def test_adapter_tolerates_missing_dormant_attribute():
    """``model_construct`` bypasses validators; probe should still succeed."""
    cfg = ExperimentConfig.model_construct(task={"model": "gpt2"}, engine="transformers")
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)
    assert probe.dormant_fields == {}


# ---------------------------------------------------------------------------
# check_hardware pass-through
# ---------------------------------------------------------------------------


def test_adapter_surfaces_hardware_errors():
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch(
        "llenergymeasure.engines.get_engine",
        return_value=_make_fake_plugin(["SM too low", "FP8 unsupported"]),
    ):
        probe = build_config_probe(cfg)

    assert "SM too low" in probe.errors
    assert "FP8 unsupported" in probe.errors
    assert probe.is_valid is False


def test_adapter_empty_errors_is_valid():
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)
    assert probe.errors == []
    assert probe.is_valid is True


def test_adapter_quiet_on_missing_engine_package():
    """When get_engine raises, the adapter returns an empty-errors probe."""
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")

    def _raise(_name):
        raise ImportError("vllm not installed")

    with patch("llenergymeasure.engines.get_engine", side_effect=_raise):
        probe = build_config_probe(cfg)
    assert probe.errors == []


def test_adapter_captures_unexpected_check_hardware_exception():
    """``check_hardware`` is meant to be pure, but a raise lands as a probe error."""
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch(
        "llenergymeasure.engines.get_engine",
        return_value=_make_fake_plugin(raises=RuntimeError("boom")),
    ):
        probe = build_config_probe(cfg)
    assert any("check_hardware raised unexpectedly" in err for err in probe.errors)


def test_adapter_handles_non_list_check_hardware_return():
    class _BadPlugin:
        def check_hardware(self, config):
            return "not a list"  # type: ignore[return-value]

    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch("llenergymeasure.engines.get_engine", return_value=_BadPlugin()):
        probe = build_config_probe(cfg)
    assert any("non-list" in err for err in probe.errors)


# ---------------------------------------------------------------------------
# Effective params placeholder (populated by 50.3a; empty here)
# ---------------------------------------------------------------------------


def test_adapter_effective_params_are_empty_placeholders():
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)
    assert probe.effective_engine_params == {}
    assert probe.effective_sampling_params == {}


# ---------------------------------------------------------------------------
# DormantField shape preserved across the adapter boundary
# ---------------------------------------------------------------------------


def test_adapter_preserves_dormant_field_values():
    _reset_rules_loader_cache()
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(do_sample=False, top_p=0.95),
        ),
    )
    with patch("llenergymeasure.engines.get_engine", return_value=_make_fake_plugin([])):
        probe = build_config_probe(cfg)

    for obs in probe.dormant_fields.values():
        assert isinstance(obs, DormantField)
