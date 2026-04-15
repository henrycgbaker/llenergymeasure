"""GPU-free tests for the EnginePlugin Protocol, factory, and detection.

All tests run without a GPU. torch is imported only for dtype comparison in
test_resolve_torch_dtype (no CUDA calls). Everything else is pure Python.
"""

import importlib.util

import pytest

# =============================================================================
# Protocol satisfaction — TensorRTEngine (BACK-01)
# =============================================================================


def test_tensorrt_engine_satisfies_plugin_protocol():
    """TensorRTEngine must satisfy the EnginePlugin Protocol."""
    from llenergymeasure.engines.protocol import EnginePlugin
    from llenergymeasure.engines.tensorrt import TensorRTEngine

    engine = TensorRTEngine()
    assert isinstance(engine, EnginePlugin)
    assert engine.name == "tensorrt"


# =============================================================================
# Protocol satisfaction
# =============================================================================


def test_pytorch_engine_satisfies_plugin_protocol():
    """TransformersEngine must satisfy the EnginePlugin Protocol."""
    from llenergymeasure.engines.protocol import EnginePlugin
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    assert isinstance(engine, EnginePlugin)
    assert engine.name == "transformers"


# =============================================================================
# Engine factory
# =============================================================================


def test_get_engine_pytorch():
    """get_engine('transformers') returns a TransformersEngine with name 'transformers'."""
    from llenergymeasure.engines import get_engine

    engine = get_engine("transformers")
    assert engine.name == "transformers"


def test_get_engine_tensorrt():
    """get_engine('tensorrt') returns a TensorRTEngine with name 'tensorrt'."""
    from llenergymeasure.engines import get_engine

    engine = get_engine("tensorrt")
    assert engine.name == "tensorrt"


def test_get_engine_unknown_raises_engine_error():
    """get_engine with unknown name raises EngineError."""
    from llenergymeasure.engines import get_engine
    from llenergymeasure.utils.exceptions import EngineError

    with pytest.raises(EngineError, match="Unknown engine"):
        get_engine("nonexistent")


def test_get_engine_unknown_message_contains_engine_name():
    """Error message includes the unknown engine name."""
    from llenergymeasure.engines import get_engine
    from llenergymeasure.utils.exceptions import EngineError

    with pytest.raises(EngineError, match="'badbackend'"):
        get_engine("badbackend")


# =============================================================================
# Engine detection
# =============================================================================


def test_detect_default_engine_returns_pytorch():
    """detect_default_engine returns 'transformers' when transformers is installed."""
    pytest.importorskip("transformers")
    from llenergymeasure.engines import detect_default_engine

    # transformers must be installed in the test environment
    assert importlib.util.find_spec("transformers") is not None, (
        "transformers must be installed for this test to be meaningful"
    )
    assert detect_default_engine() == "transformers"


def test_detect_default_engine_returns_tensorrt_when_only_trt():
    """detect_default_engine returns 'tensorrt' when only tensorrt_llm is installed."""
    from unittest.mock import patch

    import llenergymeasure.engines as engines_mod

    def mock_available(name):
        return name == "tensorrt"  # Only tensorrt is "installed"

    with patch("llenergymeasure.engines.is_engine_available", side_effect=mock_available):
        result = engines_mod.detect_default_engine()
        assert result == "tensorrt"


def test_detect_default_engine_raises_when_no_engines():
    """detect_default_engine raises EngineError when no engines are installed."""
    from unittest.mock import patch

    import llenergymeasure.engines as engines_mod
    from llenergymeasure.utils.exceptions import EngineError

    # Patch is_engine_available so all engine checks return False
    with (
        patch("llenergymeasure.engines.is_engine_available", return_value=False),
        pytest.raises(EngineError, match="No inference engine"),
    ):
        engines_mod.detect_default_engine()


def test_detect_default_engine_error_message_has_install_hint():
    """Error message from detect_default_engine includes install hint."""
    from unittest.mock import patch

    import llenergymeasure.engines as engines_mod
    from llenergymeasure.utils.exceptions import EngineError

    # Patch is_engine_available so all engine checks return False
    with (
        patch("llenergymeasure.engines.is_engine_available", return_value=False),
        pytest.raises(EngineError, match="pip install"),
    ):
        engines_mod.detect_default_engine()


# =============================================================================
# _model_load_kwargs — P0 fix verification (GPU-free)
# =============================================================================


def test_model_load_kwargs_contains_base_keys():
    """_model_load_kwargs always includes torch_dtype, device_map, trust_remote_code."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2")
    kwargs = engine._model_load_kwargs(config)

    assert "torch_dtype" in kwargs
    assert kwargs["device_map"] == "auto"
    # HF default — env var LLEM_TRUST_REMOTE_CODE not set, no typed override
    assert kwargs["trust_remote_code"] is False


def test_model_load_kwargs_trust_remote_code_env_var_opt_in(monkeypatch):
    """LLEM_TRUST_REMOTE_CODE=1 enables trust_remote_code=True at the model load."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    monkeypatch.setenv("LLEM_TRUST_REMOTE_CODE", "1")
    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2")
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["trust_remote_code"] is True


def test_model_load_kwargs_passthrough_kwargs_merged():
    """passthrough_kwargs are merged into model load kwargs (core P0 fix)."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"custom_key": "custom_value"})
    kwargs = engine._model_load_kwargs(config)

    assert "custom_key" in kwargs
    assert kwargs["custom_key"] == "custom_value"


def test_model_load_kwargs_passthrough_can_override_defaults():
    """passthrough_kwargs can override engine defaults (intentional escape hatch)."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    # Override device_map via passthrough
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"device_map": "cpu"})
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["device_map"] == "cpu"


def test_model_load_kwargs_no_passthrough_when_none():
    """No extra keys added when passthrough_kwargs is None."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2")  # passthrough_kwargs=None by default
    kwargs = engine._model_load_kwargs(config)

    expected_keys = {"torch_dtype", "device_map", "trust_remote_code"}
    assert set(kwargs.keys()) == expected_keys


def test_model_load_kwargs_pytorch_config_attn_implementation():
    """TransformersConfig.attn_implementation is included in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="sdpa"),
    )
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


# =============================================================================
# _resolve_attn_implementation — flash_attn availability guard
# =============================================================================


def test_model_load_kwargs_flash_attention_falls_back_when_not_installed():
    """flash_attention_2 falls back to sdpa when flash_attn package is missing."""
    pytest.importorskip("torch")

    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="flash_attention_2"),
    )

    # flash_attn is not installed in the test environment, so the guard
    # catches the ImportError and falls back to sdpa automatically.
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_flash_attention_kept_when_installed():
    """flash_attention_2 is kept when flash_attn package is available."""
    pytest.importorskip("torch")
    import sys
    import types
    from unittest.mock import patch

    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="flash_attention_2"),
    )

    # Simulate flash_attn and flash_attn.bert_padding being importable by
    # temporarily injecting stub modules into sys.modules.
    fake_flash = types.ModuleType("flash_attn")
    fake_bert_padding = types.ModuleType("flash_attn.bert_padding")
    with patch.dict(
        sys.modules,
        {
            "flash_attn": fake_flash,
            "flash_attn.bert_padding": fake_bert_padding,
        },
    ):
        kwargs = engine._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "flash_attention_2"


def test_model_load_kwargs_sdpa_not_affected_by_flash_guard():
    """sdpa attention is passed through without flash_attn availability check."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="sdpa"),
    )
    kwargs = engine._model_load_kwargs(config)

    # sdpa should never be affected by flash_attn availability
    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_eager_not_affected_by_flash_guard():
    """eager attention is passed through without flash_attn availability check."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="eager"),
    )
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "eager"


def test_model_load_kwargs_flash_attention_3_falls_back_when_not_installed():
    """flash_attention_3 also falls back to sdpa when flash_attn is missing."""
    pytest.importorskip("torch")

    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(attn_implementation="flash_attention_3"),
    )

    # flash_attn is not installed in the test environment, so the guard
    # catches the ImportError and falls back to sdpa automatically.
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_pytorch_config_load_in_4bit():
    """TransformersConfig.load_in_4bit=True produces a BitsAndBytesConfig quantization_config."""
    pytest.importorskip("torch")
    from transformers import BitsAndBytesConfig

    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(load_in_4bit=True),
    )
    kwargs = engine._model_load_kwargs(config)

    # v2.0: BitsAndBytesConfig is used (not raw load_in_4bit kwarg)
    assert "quantization_config" in kwargs
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_4bit is True


def test_model_load_kwargs_pytorch_config_load_in_8bit():
    """TransformersConfig.load_in_8bit=True produces a BitsAndBytesConfig quantization_config."""
    pytest.importorskip("torch")
    from transformers import BitsAndBytesConfig

    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(load_in_8bit=True),
    )
    kwargs = engine._model_load_kwargs(config)

    # v2.0: BitsAndBytesConfig is used (not raw load_in_8bit kwarg)
    assert "quantization_config" in kwargs
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_8bit is True


def test_model_load_kwargs_pytorch_config_none_values_not_included():
    """None values from TransformersConfig are NOT included in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        transformers=TransformersConfig(),  # all fields None
    )
    kwargs = engine._model_load_kwargs(config)

    # None-valued fields should not appear
    assert "attn_implementation" not in kwargs
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


def test_model_load_kwargs_no_pytorch_section():
    """When config.transformers is None, only base keys are present."""
    pytest.importorskip("torch")
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2")  # transformers=None by default
    kwargs = engine._model_load_kwargs(config)

    assert "attn_implementation" not in kwargs
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


# =============================================================================
# _resolve_torch_dtype
# =============================================================================


def test_resolve_torch_dtype_fp32():
    """fp32 maps to torch.float32."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.engines.transformers import TransformersEngine

    assert TransformersEngine._resolve_torch_dtype("float32") == torch.float32


def test_resolve_torch_dtype_fp16():
    """fp16 maps to torch.float16."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.engines.transformers import TransformersEngine

    assert TransformersEngine._resolve_torch_dtype("float16") == torch.float16


def test_resolve_torch_dtype_bf16():
    """bf16 maps to torch.bfloat16."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.engines.transformers import TransformersEngine

    assert TransformersEngine._resolve_torch_dtype("bfloat16") == torch.bfloat16


def test_resolve_torch_dtype_unknown_raises():
    """Unknown dtype string raises KeyError."""
    pytest.importorskip("torch")
    from llenergymeasure.engines.transformers import TransformersEngine

    with pytest.raises(KeyError):
        TransformersEngine._resolve_torch_dtype("int8")


# =============================================================================
# _build_generate_kwargs — GPU-free decoder config mapping
# =============================================================================


def test_build_generate_kwargs_defaults():
    """Default decoder config produces expected generate kwargs."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2")
    kwargs = engine._build_generate_kwargs(config)

    assert "do_sample" in kwargs
    assert "temperature" in kwargs
    assert "top_k" in kwargs
    assert "top_p" in kwargs
    assert "repetition_penalty" in kwargs


def test_build_generate_kwargs_greedy_decoding():
    """Greedy decoding (temperature=0) removes sampling params."""
    from llenergymeasure.config.models import DecoderConfig, ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2",
        decoder=DecoderConfig(temperature=0.0, do_sample=False),
    )
    kwargs = engine._build_generate_kwargs(config)

    assert kwargs["do_sample"] is False
    # temperature and top_k removed for greedy decoding
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs


# =============================================================================
# Protocol structural checks
# =============================================================================


def test_engine_plugin_protocol_has_required_methods():
    """EnginePlugin Protocol defines all required methods plus name/version properties and validate_config."""
    from llenergymeasure.engines.transformers import TransformersEngine

    obj = TransformersEngine()
    assert hasattr(obj, "name")
    assert hasattr(obj, "version")
    assert hasattr(obj, "load_model")
    assert hasattr(obj, "run_warmup_prompt")
    assert hasattr(obj, "run_inference")
    assert hasattr(obj, "cleanup")
    assert hasattr(obj, "validate_config")


def test_engine_plugin_protocol_is_runtime_checkable():
    """EnginePlugin is @runtime_checkable (isinstance works)."""
    from llenergymeasure.engines.protocol import EnginePlugin
    from llenergymeasure.engines.transformers import TransformersEngine

    # runtime_checkable means isinstance() works and confirms protocol conformance
    assert isinstance(TransformersEngine(), EnginePlugin) is True


def test_non_conforming_object_fails_plugin_protocol_check():
    """An object without the 4 plugin methods does not satisfy EnginePlugin."""
    from llenergymeasure.engines.protocol import EnginePlugin

    class NotAnEngine:
        pass

    assert not isinstance(NotAnEngine(), EnginePlugin)


# =============================================================================
# validate_config stubs — PyTorch and vLLM return []
# =============================================================================


def test_pytorch_validate_config_returns_empty():
    """TransformersEngine.validate_config returns empty list."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    config = ExperimentConfig(model="gpt2")
    assert TransformersEngine().validate_config(config) == []


def test_vllm_validate_config_returns_empty():
    """VLLMEngine.validate_config returns empty list."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.vllm import VLLMEngine

    config = ExperimentConfig(model="gpt2", engine="vllm")
    assert VLLMEngine().validate_config(config) == []


# =============================================================================
# Error message lists available backends (includes tensorrt)
# =============================================================================


def test_get_engine_unknown_message_lists_tensorrt():
    """Error message from get_engine lists 'tensorrt' as an available engine."""
    from llenergymeasure.engines import get_engine
    from llenergymeasure.utils.exceptions import EngineError

    with pytest.raises(EngineError, match="tensorrt"):
        get_engine("badbackend")


# =============================================================================
# _model_load_kwargs — tensor parallelism (tp_plan / tp_size)
# =============================================================================


def test_model_load_kwargs_tp_plan_forwarded():
    """With TransformersConfig(tp_plan='auto'), kwargs contains tp_plan and no device_map."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2", transformers=TransformersConfig(tp_plan="auto"))
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["tp_plan"] == "auto"
    assert "device_map" not in kwargs


def test_model_load_kwargs_tp_plan_and_tp_size_forwarded():
    """With TransformersConfig(tp_plan='auto', tp_size=4), kwargs contains both and no device_map."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(
        model="gpt2", transformers=TransformersConfig(tp_plan="auto", tp_size=4)
    )
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["tp_plan"] == "auto"
    assert kwargs["tp_size"] == 4
    assert "device_map" not in kwargs


def test_model_load_kwargs_tp_size_without_tp_plan_ignored():
    """With TransformersConfig(tp_size=4) and no tp_plan, tp_size is not in kwargs and device_map defaults."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2", transformers=TransformersConfig(tp_size=4))
    kwargs = engine._model_load_kwargs(config)

    assert "tp_size" not in kwargs
    assert "tp_plan" not in kwargs
    assert kwargs["device_map"] == "auto"


def test_model_load_kwargs_device_map_still_works():
    """TransformersConfig(device_map='cpu') produces device_map='cpu', no tp_plan in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.config.engine_configs import TransformersConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.transformers import TransformersEngine

    engine = TransformersEngine()
    config = ExperimentConfig(model="gpt2", transformers=TransformersConfig(device_map="cpu"))
    kwargs = engine._model_load_kwargs(config)

    assert kwargs["device_map"] == "cpu"
    assert "tp_plan" not in kwargs


# =============================================================================
# Backend version property
# =============================================================================


def test_pytorch_version_returns_torch_version():
    """TransformersEngine.version returns torch.__version__."""
    torch = pytest.importorskip("torch")
    from llenergymeasure.engines.transformers import TransformersEngine

    assert TransformersEngine().version == torch.__version__


def test_vllm_version_returns_string():
    """VLLMEngine.version returns a string (either vllm version or 'unknown')."""
    from llenergymeasure.engines.vllm import VLLMEngine

    version = VLLMEngine().version
    assert isinstance(version, str)
    assert len(version) > 0


def test_tensorrt_version_returns_string():
    """TensorRTEngine.version returns a string (either tensorrt_llm version or 'unknown')."""
    from llenergymeasure.engines.tensorrt import TensorRTEngine

    version = TensorRTEngine().version
    assert isinstance(version, str)
    assert len(version) > 0
    # On hosts without TRT-LLM CUDA libs, import fails and version is "unknown"
    # On containers with TRT-LLM, it returns the actual version string
