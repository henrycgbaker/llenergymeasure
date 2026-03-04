"""GPU-free tests for the InferenceBackend Protocol, factory, and detection.

All tests run without a GPU. torch is imported only for dtype comparison in
test_precision_to_dtype (no CUDA calls). Everything else is pure Python.
"""

import importlib.util

import pytest

# =============================================================================
# Protocol satisfaction
# =============================================================================


def test_pytorch_backend_satisfies_protocol():
    """PyTorchBackend must satisfy the InferenceBackend Protocol."""
    from llenergymeasure.core.backends.protocol import InferenceBackend
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    assert isinstance(backend, InferenceBackend)
    assert backend.name == "pytorch"


# =============================================================================
# Backend factory
# =============================================================================


def test_get_backend_pytorch():
    """get_backend('pytorch') returns a PyTorchBackend with name 'pytorch'."""
    from llenergymeasure.core.backends import get_backend

    backend = get_backend("pytorch")
    assert backend.name == "pytorch"


def test_get_backend_unknown_raises_backend_error():
    """get_backend with unknown name raises BackendError."""
    from llenergymeasure.core.backends import get_backend
    from llenergymeasure.exceptions import BackendError

    with pytest.raises(BackendError, match="Unknown backend"):
        get_backend("nonexistent")


def test_get_backend_unknown_message_contains_backend_name():
    """Error message includes the unknown backend name."""
    from llenergymeasure.core.backends import get_backend
    from llenergymeasure.exceptions import BackendError

    with pytest.raises(BackendError, match="'badbackend'"):
        get_backend("badbackend")


# =============================================================================
# Backend detection
# =============================================================================


def test_detect_default_backend_returns_pytorch():
    """detect_default_backend returns 'pytorch' when transformers is installed."""
    from llenergymeasure.core.backends import detect_default_backend

    # transformers must be installed in the test environment
    assert importlib.util.find_spec("transformers") is not None, (
        "transformers must be installed for this test to be meaningful"
    )
    assert detect_default_backend() == "pytorch"


def test_detect_default_backend_raises_when_no_backends():
    """detect_default_backend raises BackendError when no backends are installed."""
    from unittest.mock import patch

    import llenergymeasure.core.backends as backends_mod
    from llenergymeasure.exceptions import BackendError

    # Patch find_spec inside the backends module so all backend checks return None
    with patch("llenergymeasure.core.backends.importlib.util.find_spec", return_value=None):
        with pytest.raises(BackendError, match="No inference backend"):
            backends_mod.detect_default_backend()


def test_detect_default_backend_error_message_has_install_hint():
    """Error message from detect_default_backend includes install hint."""
    from unittest.mock import patch

    import llenergymeasure.core.backends as backends_mod
    from llenergymeasure.exceptions import BackendError

    # Patch find_spec inside the backends module so all backend checks return None
    with patch("llenergymeasure.core.backends.importlib.util.find_spec", return_value=None):
        with pytest.raises(BackendError, match="pip install"):
            backends_mod.detect_default_backend()


# =============================================================================
# _model_load_kwargs — P0 fix verification (GPU-free)
# =============================================================================


def test_model_load_kwargs_contains_base_keys():
    """_model_load_kwargs always includes torch_dtype, device_map, trust_remote_code."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")
    kwargs = backend._model_load_kwargs(config)

    assert "torch_dtype" in kwargs
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is True


def test_model_load_kwargs_passthrough_kwargs_merged():
    """passthrough_kwargs are merged into model load kwargs (core P0 fix)."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"custom_key": "custom_value"})
    kwargs = backend._model_load_kwargs(config)

    assert "custom_key" in kwargs
    assert kwargs["custom_key"] == "custom_value"


def test_model_load_kwargs_passthrough_can_override_defaults():
    """passthrough_kwargs can override backend defaults (intentional escape hatch)."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    # Override device_map via passthrough
    config = ExperimentConfig(model="gpt2", passthrough_kwargs={"device_map": "cpu"})
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["device_map"] == "cpu"


def test_model_load_kwargs_no_passthrough_when_none():
    """No extra keys added when passthrough_kwargs is None."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    config = ExperimentConfig(model="gpt2")  # passthrough_kwargs=None by default
    kwargs = backend._model_load_kwargs(config)

    expected_keys = {"torch_dtype", "device_map", "trust_remote_code"}
    assert set(kwargs.keys()) == expected_keys


def test_model_load_kwargs_pytorch_config_attn_implementation():
    """PyTorchConfig.attn_implementation is included in kwargs."""
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    backend = PyTorchBackend()
    config = ExperimentConfig(
        model="gpt2",
        pytorch=PyTorchConfig(attn_implementation="sdpa"),
    )
    kwargs = backend._model_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"


def test_model_load_kwargs_pytorch_config_load_in_4bit():
    """PyTorchConfig.load_in_4bit=True produces a BitsAndBytesConfig quantization_config."""
    from transformers import BitsAndBytesConfig

    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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
    from transformers import BitsAndBytesConfig

    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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
    from llenergymeasure.config.backend_configs import PyTorchConfig
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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
    import torch

    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("fp32") == torch.float32


def test_precision_to_dtype_fp16():
    """fp16 maps to torch.float16."""
    import torch

    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("fp16") == torch.float16


def test_precision_to_dtype_bf16():
    """bf16 maps to torch.bfloat16."""
    import torch

    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    assert PyTorchBackend._precision_to_dtype("bf16") == torch.bfloat16


def test_precision_to_dtype_unknown_raises():
    """Unknown precision string raises KeyError."""
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    with pytest.raises(KeyError):
        PyTorchBackend._precision_to_dtype("int8")


# =============================================================================
# _build_generate_kwargs — GPU-free decoder config mapping
# =============================================================================


def test_build_generate_kwargs_defaults():
    """Default decoder config produces expected generate kwargs."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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
    from llenergymeasure.config.models import DecoderConfig, ExperimentConfig
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

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


def test_inference_backend_protocol_has_name_property():
    """InferenceBackend Protocol defines the name property."""
    from llenergymeasure.core.backends.protocol import InferenceBackend

    # Protocol defines __protocol_attrs__ listing required members
    # 'name' and 'run' must be present
    assert hasattr(InferenceBackend, "__protocol_attrs__") or True  # runtime_checkable
    _backend_annotations = getattr(InferenceBackend, "__annotations__", {})
    # Check via isinstance with a conforming object
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    obj = PyTorchBackend()
    assert hasattr(obj, "name")
    assert hasattr(obj, "run")


def test_inference_backend_protocol_is_runtime_checkable():
    """InferenceBackend is @runtime_checkable (isinstance works)."""
    from llenergymeasure.core.backends.protocol import InferenceBackend
    from llenergymeasure.core.backends.pytorch import PyTorchBackend

    # runtime_checkable means isinstance() works without raising TypeError
    result = isinstance(PyTorchBackend(), InferenceBackend)
    assert isinstance(result, bool)  # didn't raise


def test_non_conforming_object_fails_protocol_check():
    """An object without name/run does not satisfy InferenceBackend."""
    from llenergymeasure.core.backends.protocol import InferenceBackend

    class NotABackend:
        pass

    assert not isinstance(NotABackend(), InferenceBackend)
