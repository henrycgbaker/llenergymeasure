"""GPU-free tests for ``EnginePlugin.probe_config`` across all 3 engines.

Mirrors the conventions in ``test_engine_protocol.py`` and
``test_tensorrt_engine.py``: mocks engine libraries via ``sys.modules`` and
monkeypatch so the probe runs on any host. No real network, GPU, or engine
import is required.

Coverage:
  A. :class:`ConfigProbe` / :class:`DormantField` / :func:`compute_dormant_fields`
  B. Per-engine T0 dormancy:
     - Transformers: greedy stripping (still present on main)
     - vLLM: no stripping in new model; "no dormancy" is the assertion
     - TRT-LLM: no stripping in new model; "no dormancy" is the assertion
  C. Exception non-propagation: probe never raises
  D. Framework-error capture: T1/T2/T5 exceptions land in probe.errors
  E. Refactoring verification: _build_sampling_params still works after extraction
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from llenergymeasure.config.engine_configs import (
    TensorRTConfig,
    TensorRTSamplingConfig,
    TransformersConfig,
    TransformersSamplingConfig,
    VLLMConfig,
    VLLMSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.probe import ConfigProbe, DormantField
from llenergymeasure.engines._helpers import compute_dormant_fields
from llenergymeasure.engines.tensorrt import TensorRTEngine
from llenergymeasure.engines.transformers import TransformersEngine
from llenergymeasure.engines.vllm import VLLMEngine

# =============================================================================
# A. Dataclass and helper tests
# =============================================================================


def test_config_probe_is_valid_when_no_errors():
    probe = ConfigProbe(effective_engine_params={}, effective_sampling_params={}, dormant_fields={})
    assert probe.is_valid is True


def test_config_probe_invalid_when_errors_present():
    probe = ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields={},
        errors=["boom"],
    )
    assert probe.is_valid is False


def test_config_probe_warnings_do_not_invalidate():
    probe = ConfigProbe(
        effective_engine_params={},
        effective_sampling_params={},
        dormant_fields={},
        warnings=["heads up"],
    )
    assert probe.is_valid is True


def test_dormant_field_defaults():
    f = DormantField(declared_value=0.9, effective_value=None)
    assert f.declared_value == 0.9
    assert f.effective_value is None
    assert f.reason is None


def test_dormant_field_with_reason():
    f = DormantField(declared_value=0.9, effective_value=0.0, reason="greedy")
    assert f.reason == "greedy"


def test_compute_dormant_fields_returns_empty_when_matching():
    assert compute_dormant_fields({"a": 1}, {"a": 1}) == {}


def test_compute_dormant_fields_detects_stripped_field():
    dormant = compute_dormant_fields({"a": 1, "b": 2}, {"a": 1})
    assert set(dormant.keys()) == {"b"}
    assert dormant["b"].declared_value == 2
    assert dormant["b"].effective_value is None


def test_compute_dormant_fields_detects_overridden_field():
    dormant = compute_dormant_fields({"top_k": 0}, {"top_k": -1})
    assert "top_k" in dormant
    assert dormant["top_k"].declared_value == 0
    assert dormant["top_k"].effective_value == -1


def test_compute_dormant_fields_prefix_applied():
    dormant = compute_dormant_fields({"t": 0.9}, {}, prefix="vllm.sampling.")
    assert "vllm.sampling.t" in dormant


def test_compute_dormant_fields_reason_fn_called():
    def reason_fn(key: str, declared: Any, effective: Any | None) -> str | None:
        return f"{key}:{declared}->{effective}"

    dormant = compute_dormant_fields({"t": 0.9}, {}, reason_fn=reason_fn)
    assert dormant["t"].reason == "t:0.9->None"


def test_compute_dormant_fields_ignores_effective_only_keys():
    dormant = compute_dormant_fields({"a": 1}, {"a": 1, "extra": 99})
    assert dormant == {}


# =============================================================================
# B. Per-engine T0 dormancy tests
# =============================================================================


def test_transformers_dormancy_greedy_decoding():
    """Greedy decoding (do_sample=False) strips temperature/top_k/top_p/min_p."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(
                temperature=0.9, top_p=0.95, top_k=40, do_sample=False, min_p=0.1
            )
        ),
    )
    engine = TransformersEngine()
    declared = engine._declared_sampling_params(cfg)
    effective_full = engine._build_generate_kwargs(cfg)
    # Scope effective to the sampling keys (matches the probe's own scoping).
    _SAMPLING_KEYS = {
        "temperature",
        "do_sample",
        "top_k",
        "top_p",
        "min_p",
        "repetition_penalty",
        "min_new_tokens",
    }
    effective = {k: v for k, v in effective_full.items() if k in _SAMPLING_KEYS}

    dormant = compute_dormant_fields(
        declared,
        effective,
        prefix="transformers.sampling.",
        reason_fn=engine._dormancy_reason,
    )

    assert "transformers.sampling.temperature" in dormant
    assert "transformers.sampling.top_k" in dormant
    assert "transformers.sampling.top_p" in dormant
    assert "transformers.sampling.min_p" in dormant
    assert dormant["transformers.sampling.temperature"].reason is not None
    assert "greedy" in dormant["transformers.sampling.temperature"].reason


def test_transformers_no_dormancy_when_sampling_active():
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(
                temperature=0.9, top_p=0.95, top_k=40, do_sample=True
            )
        ),
    )
    engine = TransformersEngine()
    declared = engine._declared_sampling_params(cfg)
    effective_full = engine._build_generate_kwargs(cfg)
    effective = {k: v for k, v in effective_full.items() if k in declared}
    dormant = compute_dormant_fields(declared, effective)
    assert dormant == {}


def test_vllm_no_dormancy_in_new_data_model():
    """Without beam_search, vLLM forwards all sampling fields — no dormancy."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(
            sampling=VLLMSamplingConfig(temperature=0.9, top_p=0.95, top_k=40),
        ),
    )
    engine = VLLMEngine()
    declared = engine._declared_sampling_params(cfg)
    effective = engine._build_sampling_kwargs(cfg)
    dormant = compute_dormant_fields(declared, effective)
    assert dormant == {}


def test_tensorrt_no_dormancy_in_new_data_model():
    """TRT-LLM no longer strips sampling fields in the per-engine data model."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(
            sampling=TensorRTSamplingConfig(temperature=0.0, top_p=0.95, top_k=40),
        ),
    )
    engine = TensorRTEngine()
    declared = engine._declared_sampling_params(cfg)
    effective = engine._build_sampling_kwargs(cfg)
    dormant = compute_dormant_fields(declared, effective)
    # User's explicit values are all forwarded; effective also carries
    # engine-appended random_seed / max_new_tokens but those are effective-only
    # and compute_dormant_fields ignores them.
    assert dormant == {}


# =============================================================================
# C. Exception non-propagation — probe never raises
# =============================================================================


def test_transformers_probe_t0_engine_kwargs_error_captured():
    """When _model_load_kwargs raises, probe returns with errors and empty effective."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    engine = TransformersEngine()

    with patch.object(engine, "_model_load_kwargs", side_effect=RuntimeError("boom")):
        probe = engine.probe_config(cfg)

    assert not probe.is_valid
    assert any("T0 engine kwargs" in err for err in probe.errors)
    assert probe.effective_engine_params == {}
    assert probe.effective_sampling_params == {}


def test_transformers_probe_t0_sampling_kwargs_error_captured():
    """When _build_generate_kwargs raises, probe captures and bails."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    engine = TransformersEngine()

    with patch.object(engine, "_build_generate_kwargs", side_effect=ValueError("bad")):
        probe = engine.probe_config(cfg)

    assert any("T0 sampling kwargs" in err for err in probe.errors)


def test_vllm_probe_t0_engine_kwargs_error_captured():
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="vllm")
    engine = VLLMEngine()

    with patch.object(engine, "_build_llm_kwargs", side_effect=RuntimeError("boom")):
        probe = engine.probe_config(cfg)

    assert not probe.is_valid
    assert any("T0 engine kwargs" in err for err in probe.errors)


def test_tensorrt_probe_t0_engine_kwargs_error_captured():
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt")
    engine = TensorRTEngine()

    with patch.object(engine, "_build_llm_kwargs", side_effect=RuntimeError("boom")):
        probe = engine.probe_config(cfg)

    assert not probe.is_valid
    assert any("T0 engine kwargs" in err for err in probe.errors)


def test_transformers_probe_never_raises_on_t2_failure(monkeypatch):
    """Even if AutoConfig.from_pretrained raises, probe returns cleanly."""
    cfg = ExperimentConfig(task={"model": "definitely-not-a-real-model"}, engine="transformers")
    engine = TransformersEngine()

    import transformers as _tf

    def _raise(*a, **kw):
        raise OSError("no such model")

    monkeypatch.setattr(_tf.AutoConfig, "from_pretrained", _raise)
    monkeypatch.setenv("LLEM_PROBE_META_DEVICE_ENABLED", "0")

    probe = engine.probe_config(cfg)

    assert any("T2 AutoConfig" in err for err in probe.errors)
    assert probe.effective_engine_params != {}


# =============================================================================
# D. Framework-error capture — T1/T2/T5 exceptions land in probe.errors
# =============================================================================


def test_transformers_t1_generation_config_error_captured(monkeypatch):
    """Mock GenerationConfig to raise on construction — verify T1 error capture."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    engine = TransformersEngine()

    import transformers as _tf

    real_gc = _tf.GenerationConfig

    class _BadGenerationConfig(real_gc):  # type: ignore[misc,valid-type]
        def __init__(self, **kwargs):
            raise ValueError("T1 failure")

    monkeypatch.setattr(_tf, "GenerationConfig", _BadGenerationConfig)
    monkeypatch.setattr(_tf.AutoConfig, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setenv("LLEM_PROBE_META_DEVICE_ENABLED", "0")

    probe = engine.probe_config(cfg)

    assert any("T1 GenerationConfig" in err for err in probe.errors)


def test_vllm_t1_sampling_params_error_captured(monkeypatch):
    """Mock vllm.SamplingParams to raise — verify T1 error capture."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(sampling=VLLMSamplingConfig(temperature=0.9)),
    )
    engine = VLLMEngine()

    fake_vllm = types.ModuleType("vllm")

    class _BadSP:
        def __init__(self, **kwargs):
            raise ValueError("invalid sampling combo")

    fake_vllm.SamplingParams = _BadSP  # type: ignore[attr-defined]
    fake_arg_utils = types.ModuleType("vllm.engine.arg_utils")

    class _FakeEngineArgs:
        def __init__(self, **kwargs):
            pass

        def create_engine_config(self):
            return object()

    fake_arg_utils.EngineArgs = _FakeEngineArgs  # type: ignore[attr-defined]
    fake_engine_pkg = types.ModuleType("vllm.engine")

    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.engine": fake_engine_pkg,
            "vllm.engine.arg_utils": fake_arg_utils,
        },
    ):
        probe = engine.probe_config(cfg)

    assert any("T1 SamplingParams" in err for err in probe.errors)


def test_vllm_t2_engine_args_error_captured():
    """EngineArgs construction raising is recorded as a probe error."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="vllm")
    engine = VLLMEngine()

    fake_vllm = types.ModuleType("vllm")

    class _OkSP:
        def __init__(self, **kwargs):
            pass

    fake_vllm.SamplingParams = _OkSP  # type: ignore[attr-defined]

    fake_arg_utils = types.ModuleType("vllm.engine.arg_utils")

    class _EngineArgs:
        def __init__(self, **kwargs):
            pass

        def create_engine_config(self):
            raise ValueError("dtype/quant incompatible")

    fake_arg_utils.EngineArgs = _EngineArgs  # type: ignore[attr-defined]
    fake_engine_pkg = types.ModuleType("vllm.engine")

    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.engine": fake_engine_pkg,
            "vllm.engine.arg_utils": fake_arg_utils,
        },
    ):
        probe = engine.probe_config(cfg)

    assert any("T2 EngineArgs" in err for err in probe.errors)


def test_tensorrt_t5_sm_too_low_captured(monkeypatch):
    """SM 7.0 (pre-Turing) yields a hardware error from the T5 fallback."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt")
    engine = TensorRTEngine()

    monkeypatch.setattr("llenergymeasure.device.gpu_info.get_compute_capability", lambda: (7, 0))

    probe = engine.probe_config(cfg)

    assert any("SM >= 7.5" in err for err in probe.errors)


def test_tensorrt_t5_fp8_on_a100_captured(monkeypatch):
    """FP8 quant on A100 (SM 8.0) yields a hardware error from T5."""
    from llenergymeasure.config.engine_configs import TensorRTQuantConfig

    fake_llmapi = types.ModuleType("tensorrt_llm.llmapi")

    class _Algo:
        FP8 = "FP8"
        INT8 = "INT8"
        W8A16 = "W8A16"
        W4A16_AWQ = "W4A16_AWQ"
        W4A16_GPTQ = "W4A16_GPTQ"

        def __class_getitem__(cls, item):
            return getattr(cls, item)

    class _QC:
        def __init__(self, **kw):
            self.kw = kw

    fake_llmapi.QuantAlgo = _Algo  # type: ignore[attr-defined]
    fake_llmapi.QuantConfig = _QC  # type: ignore[attr-defined]

    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(quant_config=TensorRTQuantConfig(quant_algo="FP8")),
    )
    engine = TensorRTEngine()

    monkeypatch.setattr("llenergymeasure.device.gpu_info.get_compute_capability", lambda: (8, 0))

    with patch.dict(sys.modules, {"tensorrt_llm.llmapi": fake_llmapi}):
        probe = engine.probe_config(cfg)

    assert any("FP8 quantisation requires SM >= 8.9" in err for err in probe.errors)


def test_tensorrt_probe_no_errors_when_no_gpu_and_no_fp8(monkeypatch):
    """With no GPU visible, T5 returns empty; probe is valid when T0 succeeded."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="tensorrt")
    engine = TensorRTEngine()

    monkeypatch.setattr("llenergymeasure.device.gpu_info.get_compute_capability", lambda: None)

    probe = engine.probe_config(cfg)

    assert probe.is_valid, probe.errors
    assert probe.effective_engine_params != {}


# =============================================================================
# E. Refactoring verification — existing behaviour preserved after Step 3
# =============================================================================


def test_vllm_build_sampling_params_still_returns_object():
    """After extraction, VLLMEngine._build_sampling_params still constructs SamplingParams."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(sampling=VLLMSamplingConfig(temperature=0.9, top_p=0.95)),
    )

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = VLLMEngine._build_sampling_params(cfg, _SP)
    assert isinstance(result, _SP)
    assert result.kwargs["temperature"] == 0.9
    assert result.kwargs["top_p"] == 0.95


def test_vllm_build_sampling_kwargs_matches_build_sampling_params_inputs():
    """The extracted kwargs dict is what _build_sampling_params would pass to the class."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm=VLLMConfig(
            sampling=VLLMSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    kwargs = VLLMEngine._build_sampling_kwargs(cfg)

    captured: dict[str, Any] = {}

    class _SP:
        def __init__(self, **kw):
            captured.update(kw)

    VLLMEngine._build_sampling_params(cfg, _SP)
    assert captured == kwargs


def test_tensorrt_build_sampling_params_still_constructs_from_module():
    """After extraction, _build_sampling_params wires kwargs into tensorrt_llm.SamplingParams."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(
            sampling=TensorRTSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    engine = TensorRTEngine()

    captured: dict[str, Any] = {}

    class _FakeSP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_trt = types.ModuleType("tensorrt_llm")
    fake_trt.SamplingParams = _FakeSP  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"tensorrt_llm": fake_trt}):
        engine._build_sampling_params(cfg)

    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.9
    assert captured["top_k"] == 50
    assert "random_seed" in captured


def test_tensorrt_build_sampling_kwargs_matches_params_construction():
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="tensorrt",
        tensorrt=TensorRTConfig(
            sampling=TensorRTSamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        ),
    )
    engine = TensorRTEngine()
    kwargs = engine._build_sampling_kwargs(cfg)

    captured: dict[str, Any] = {}

    class _FakeSP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_trt = types.ModuleType("tensorrt_llm")
    fake_trt.SamplingParams = _FakeSP  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"tensorrt_llm": fake_trt}):
        engine._build_sampling_params(cfg)

    assert captured == kwargs


# =============================================================================
# Protocol method declared on EnginePlugin
# =============================================================================


def test_probe_config_declared_on_plugin_protocol():
    """All three engines have probe_config as part of the plugin surface."""
    for engine in (TransformersEngine(), VLLMEngine(), TensorRTEngine()):
        assert hasattr(engine, "probe_config")
        assert callable(engine.probe_config)


def test_probe_config_returns_config_probe_instance(monkeypatch):
    """probe_config returns a ConfigProbe (even when errors occur)."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    engine = TransformersEngine()

    import transformers as _tf

    monkeypatch.setattr(
        _tf.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("x")),
    )
    monkeypatch.setenv("LLEM_PROBE_META_DEVICE_ENABLED", "0")

    probe = engine.probe_config(cfg)
    assert isinstance(probe, ConfigProbe)


@pytest.mark.parametrize(
    "engine_cls,engine_name",
    [
        (TransformersEngine, "transformers"),
        (VLLMEngine, "vllm"),
        (TensorRTEngine, "tensorrt"),
    ],
)
def test_probe_never_raises_under_fuzzed_build_kwargs(engine_cls, engine_name, monkeypatch):
    """Simulate a broken _build_*_kwargs — probe still returns a ConfigProbe."""
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine=engine_name)
    engine = engine_cls()

    kwargs_attr = "_model_load_kwargs" if engine_name == "transformers" else "_build_llm_kwargs"
    with patch.object(engine, kwargs_attr, side_effect=RuntimeError("fuzz")):
        probe = engine.probe_config(cfg)

    assert isinstance(probe, ConfigProbe)
    assert not probe.is_valid
