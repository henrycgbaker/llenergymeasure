"""GPU-free tests for the BackendPlugin Protocol, factory, and detection.

All tests run without a GPU. torch is imported only for dtype comparison in
test_precision_to_dtype (no CUDA calls). Everything else is pure Python.
"""

import importlib.util

import pytest

# =============================================================================
# Protocol satisfaction — TensorRTBackend (BACK-01)
# =============================================================================


def test_tensorrt_backend_satisfies_plugin_protocol():
    """TensorRTBackend must satisfy the BackendPlugin Protocol."""
    from llenergymeasure.backends.protocol import BackendPlugin
    from llenergymeasure.backends.tensorrt import TensorRTBackend

    backend = TensorRTBackend()
    assert isinstance(backend, BackendPlugin)
    assert backend.name == "tensorrt"


# =============================================================================
# Protocol satisfaction
# =============================================================================


def test_pytorch_backend_satisfies_plugin_protocol():
    """PyTorchBackend must satisfy the BackendPlugin Protocol."""
    from llenergymeasure.backends.protocol import BackendPlugin
    from llenergymeasure.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    assert isinstance(backend, BackendPlugin)
    assert backend.name == "pytorch"


# =============================================================================
# Backend factory
# =============================================================================


def test_get_backend_pytorch():
    """get_backend('pytorch') returns a PyTorchBackend with name 'pytorch'."""
    from llenergymeasure.backends import get_backend

    backend = get_backend("pytorch")
    assert backend.name == "pytorch"


def test_get_backend_tensorrt():
    """get_backend('tensorrt') returns a TensorRTBackend with name 'tensorrt'."""
    from llenergymeasure.backends import get_backend

    backend = get_backend("tensorrt")
    assert backend.name == "tensorrt"


def test_get_backend_unknown_raises_backend_error():
    """get_backend with unknown name raises BackendError."""
    from llenergymeasure.backends import get_backend
    from llenergymeasure.utils.exceptions import BackendError

    with pytest.raises(BackendError, match="Unknown backend"):
        get_backend("nonexistent")


def test_get_backend_unknown_message_contains_backend_name():
    """Error message includes the unknown backend name."""
    from llenergymeasure.backends import get_backend
    from llenergymeasure.utils.exceptions import BackendError

    with pytest.raises(BackendError, match="'badbackend'"):
        get_backend("badbackend")


# =============================================================================
# Backend detection
# =============================================================================


def test_detect_default_backend_returns_pytorch():
    """detect_default_backend returns 'pytorch' when transformers is installed."""
    pytest.importorskip("transformers")
    from llenergymeasure.backends import detect_default_backend

    # transformers must be installed in the test environment
    assert importlib.util.find_spec("transformers") is not None, (
        "transformers must be installed for this test to be meaningful"
    )
    assert detect_default_backend() == "pytorch"


def test_detect_default_backend_returns_tensorrt_when_only_trt():
    """detect_default_backend returns 'tensorrt' when only tensorrt_llm is installed."""
    from unittest.mock import patch

    import llenergymeasure.backends as backends_mod

    def mock_available(name):
        return name == "tensorrt"  # Only tensorrt is "installed"

    with patch("llenergymeasure.backends.is_backend_available", side_effect=mock_available):
        result = backends_mod.detect_default_backend()
        assert result == "tensorrt"


def test_detect_default_backend_raises_when_no_backends():
    """detect_default_backend raises BackendError when no backends are installed."""
    from unittest.mock import patch

    import llenergymeasure.backends as backends_mod
    from llenergymeasure.utils.exceptions import BackendError

    # Patch is_backend_available so all backend checks return False
    with (
        patch("llenergymeasure.backends.is_backend_available", return_value=False),
        pytest.raises(BackendError, match="No inference backend"),
    ):
        backends_mod.detect_default_backend()


def test_detect_default_backend_error_message_has_install_hint():
    """Error message from detect_default_backend includes install hint."""
    from unittest.mock import patch

    import llenergymeasure.backends as backends_mod
    from llenergymeasure.utils.exceptions import BackendError

    # Patch is_backend_available so all backend checks return False
    with (
        patch("llenergymeasure.backends.is_backend_available", return_value=False),
        pytest.raises(BackendError, match="pip install"),
    ):
        backends_mod.detect_default_backend()


# =============================================================================
# _model_load_kwargs — P0 fix verification (GPU-free)
# =============================================================================


def test_model_load_kwargs_contains_base_keys():
    """_model_load_kwargs always includes torch_dtype, device_map, trust_remote_code."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")
    kwargs = backend._model_load_kwargs(config)

    assert "torch_dtype" in kwargs
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is True


def test_model_load_kwargs_passthrough_kwargs_merged():
    """passthrough_kwargs are merged into model load kwargs (core P0 fix)."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"custom_key": "custom_value"})
    kwargs = backend._model_load_kwargs(config)

    assert "custom_key" in kwargs
    assert kwargs["custom_key"] == "custom_value"


def test_model_load_kwargs_passthrough_can_override_defaults():
    """passthrough_kwargs can override backend defaults (intentional escape hatch)."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    # Override device_map via passthrough
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"device_map": "cpu"})
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["device_map"] == "cpu"


def test_model_load_kwargs_no_passthrough_when_none():
    """No extra keys added when passthrough_kwargs is None."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")  # passthrough_kwargs=None by default
    kwargs = backend._model_load_kwargs(config)

    expected_keys = {"torch_dtype", "device_map", "trust_remote_code"}
    assert set(kwargs.keys()) == expected_keys


def test_model_load_kwargs_pytorch_config_attn_implementation():
    """PyTorchConfig.attn_implementation is included in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="sdpa"),
    )
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


# =============================================================================
# _resolve_attn_implementation — flash_attn availability guard
# =============================================================================


def test_model_load_kwargs_flash_attention_falls_back_when_not_installed():
    """flash_attention_2 falls back to sdpa when flash_attn package is missing."""
    pytest.importorskip("torch")

    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="flash_attention_2"),
    )

    # flash_attn is not installed in the test environment, so the guard
    # catches the ImportError and falls back to sdpa automatically.
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_flash_attention_kept_when_installed():
    """flash_attention_2 is kept when flash_attn package is available."""
    pytest.importorskip("torch")
    import sys
    import types
    from unittest.mock import patch

    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="flash_attention_2"),
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
        kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "flash_attention_2"


def test_model_load_kwargs_sdpa_not_affected_by_flash_guard():
    """sdpa attention is passed through without flash_attn availability check."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="sdpa"),
    )
    kwargs = backend._model_load_kwargs(config)

    # sdpa should never be affected by flash_attn availability
    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_eager_not_affected_by_flash_guard():
    """eager attention is passed through without flash_attn availability check."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="eager"),
    )
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "eager"


def test_model_load_kwargs_flash_attention_3_falls_back_when_not_installed():
    """flash_attention_3 also falls back to sdpa when flash_attn is missing."""
    pytest.importorskip("torch")

    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="flash_attention_3"),
    )

    # flash_attn is not installed in the test environment, so the guard
    # catches the ImportError and falls back to sdpa automatically.
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_pytorch_config_load_in_4bit():
    """PyTorchConfig.load_in_4bit=True produces a BitsAndBytesConfig quantization_config."""
    pytest.importorskip("torch")
    from transformers import BitsAndBytesConfig

    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(load_in_4bit=True),
    )
    kwargs = backend._model_load_kwargs(config)

    # v2.0: BitsAndBytesConfig is used (not raw load_in_4bit kwarg)
    assert "quantization_config" in kwargs
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_4bit is True


def test_model_load_kwargs_pytorch_config_load_in_8bit():
    """PyTorchConfig.load_in_8bit=True produces a BitsAndBytesConfig quantization_config."""
    pytest.importorskip("torch")
    from transformers import BitsAndBytesConfig

    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(load_in_8bit=True),
    )
    kwargs = backend._model_load_kwargs(config)

    # v2.0: BitsAndBytesConfig is used (not raw load_in_8bit kwarg)
    assert "quantization_config" in kwargs
    assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
    assert kwargs["quantization_config"].load_in_8bit is True


def test_model_load_kwargs_pytorch_config_none_values_not_included():
    """None values from PyTorchConfig are NOT included in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(),  # all fields None
    )
    kwargs = backend._model_load_kwargs(config)

    # None-valued fields should not appear
    assert "attn_implementation" not in kwargs
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


def test_model_load_kwargs_no_pytorch_section():
    """When config.pytorch is None, only base keys are present."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")  # pytorch=None by default
    kwargs = backend._model_load_kwargs(config)

    assert "attn_implementation" not in kwargs
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


# =============================================================================
# _precision_to_dtype
# =============================================================================


def test_precision_to_dtype_fp32():
    """fp32 maps to torch.float32."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("fp32") == torch.float32


def test_precision_to_dtype_fp16():
    """fp16 maps to torch.float16."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("fp16") == torch.float16


def test_precision_to_dtype_bf16():
    """bf16 maps to torch.bfloat16."""
    torch = pytest.importorskip("torch")

    from llenergymeasure.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("bf16") == torch.bfloat16


def test_precision_to_dtype_unknown_raises():
    """Unknown precision string raises KeyError."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend

    with pytest.raises(KeyError):
        PyTorchBackend._precision_to_dtype("int8")


# =============================================================================
# _build_generate_kwargs — GPU-free decoder config mapping
# =============================================================================


def test_build_generate_kwargs_defaults():
    """Default decoder config produces expected generate kwargs."""
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")
    kwargs = backend._build_generate_kwargs(config)

    assert "do_sample" in kwargs
    assert "temperature" in kwargs
    assert "top_k" in kwargs
    assert "top_p" in kwargs
    assert "repetition_penalty" in kwargs


def test_build_generate_kwargs_greedy_decoding():
    """Greedy decoding (temperature=0) removes sampling params."""
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import DecoderConfig, ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        decoder=DecoderConfig(temperature=0.0, do_sample=False),
    )
    kwargs = backend._build_generate_kwargs(config)

    assert kwargs["do_sample"] is False
    # temperature and top_k removed for greedy decoding
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs


# =============================================================================
# Protocol structural checks
# =============================================================================


def test_backend_plugin_protocol_has_required_methods():
    """BackendPlugin Protocol defines all 5 required methods plus name property and validate_config."""
    from llenergymeasure.backends.pytorch import PyTorchBackend

    obj = PyTorchBackend()
    assert hasattr(obj, "name")
    assert hasattr(obj, "load_model")
    assert hasattr(obj, "warmup")
    assert hasattr(obj, "run_inference")
    assert hasattr(obj, "cleanup")
    assert hasattr(obj, "validate_config")


def test_backend_plugin_protocol_is_runtime_checkable():
    """BackendPlugin is @runtime_checkable (isinstance works)."""
    from llenergymeasure.backends.protocol import BackendPlugin
    from llenergymeasure.backends.pytorch import PyTorchBackend

    # runtime_checkable means isinstance() works and confirms protocol conformance
    assert isinstance(PyTorchBackend(), BackendPlugin) is True


def test_non_conforming_object_fails_plugin_protocol_check():
    """An object without the 4 plugin methods does not satisfy BackendPlugin."""
    from llenergymeasure.backends.protocol import BackendPlugin

    class NotABackend:
        pass

    assert not isinstance(NotABackend(), BackendPlugin)


# =============================================================================
# validate_config stubs — PyTorch and vLLM return []
# =============================================================================


def test_pytorch_validate_config_returns_empty():
    """PyTorchBackend.validate_config returns empty list."""
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    config = ExperimentConfig(model="gpt2")
    assert PyTorchBackend().validate_config(config) == []


def test_vllm_validate_config_returns_empty():
    """VLLMBackend.validate_config returns empty list."""
    from llenergymeasure.backends.vllm import VLLMBackend
    from llenergymeasure.config.models import ExperimentConfig

    config = ExperimentConfig(model="gpt2", backend="vllm")
    assert VLLMBackend().validate_config(config) == []


# =============================================================================
# Error message lists available backends (includes tensorrt)
# =============================================================================


def test_get_backend_unknown_message_lists_tensorrt():
    """Error message from get_backend lists 'tensorrt' as an available backend."""
    from llenergymeasure.backends import get_backend
    from llenergymeasure.utils.exceptions import BackendError

    with pytest.raises(BackendError, match="tensorrt"):
        get_backend("badbackend")


# =============================================================================
# _model_load_kwargs — tensor parallelism (tp_plan / tp_size)
# =============================================================================


def test_model_load_kwargs_tp_plan_forwarded():
    """With PyTorchConfig(tp_plan='auto'), kwargs contains tp_plan and no device_map."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", pytorch=PyTorchConfig(tp_plan="auto"))
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["tp_plan"] == "auto"
    assert "device_map" not in kwargs


def test_model_load_kwargs_tp_plan_and_tp_size_forwarded():
    """With PyTorchConfig(tp_plan='auto', tp_size=4), kwargs contains both and no device_map."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", pytorch=PyTorchConfig(tp_plan="auto", tp_size=4))
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["tp_plan"] == "auto"
    assert kwargs["tp_size"] == 4
    assert "device_map" not in kwargs


def test_model_load_kwargs_tp_size_without_tp_plan_ignored():
    """With PyTorchConfig(tp_size=4) and no tp_plan, tp_size is not in kwargs and device_map defaults."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", pytorch=PyTorchConfig(tp_size=4))
    kwargs = backend._model_load_kwargs(config)

    assert "tp_size" not in kwargs
    assert "tp_plan" not in kwargs
    assert kwargs["device_map"] == "auto"


def test_model_load_kwargs_device_map_still_works():
    """PyTorchConfig(device_map='cpu') produces device_map='cpu', no tp_plan in kwargs."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", pytorch=PyTorchConfig(device_map="cpu"))
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["device_map"] == "cpu"
    assert "tp_plan" not in kwargs
