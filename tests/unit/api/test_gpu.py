"""Unit tests for api/_gpu.py — _resolve_gpu_indices.

All tests run GPU-free. pynvml calls are mocked via patch.dict(sys.modules)
or unittest.mock.patch. No real GPU hardware required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.device.gpu_info import _resolve_gpu_indices

# ---------------------------------------------------------------------------
# Default engine: falls through to [0]
# ---------------------------------------------------------------------------


def test_default_config_returns_single_gpu():
    """pytorch engine with no device_map returns [0] (single-GPU default)."""
    config = ExperimentConfig(model="gpt2", engine="transformers")
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_unknown_engine_returns_single_gpu():
    """Unrecognised engine string falls through to [0]."""
    # ExperimentConfig validates engine, so we test by patching attribute
    config = ExperimentConfig(model="gpt2", engine="transformers")
    object.__setattr__(config, "engine", "someother")
    result = _resolve_gpu_indices(config)
    assert result == [0]


# ---------------------------------------------------------------------------
# vLLM engine: uses tensor_parallel_size * pipeline_parallel_size
# ---------------------------------------------------------------------------


def test_vllm_no_engine_config_returns_single_gpu():
    """vLLM with no engine config falls through to [0]."""
    config = ExperimentConfig(model="gpt2", engine="vllm")
    # config.vllm is None by default
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_vllm_single_gpu_returns_single_gpu():
    """vLLM with tp=1, pp=1 returns [0] (total=1 does not trigger multi-gpu)."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    engine = VLLMEngineConfig(tensor_parallel_size=1, pipeline_parallel_size=1)
    vllm_cfg = VLLMConfig(engine=engine)
    config = ExperimentConfig(model="gpt2", engine="vllm", vllm=vllm_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_vllm_tp2_returns_two_gpus():
    """vLLM with tp=2 returns [0, 1]."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    engine = VLLMEngineConfig(tensor_parallel_size=2)
    vllm_cfg = VLLMConfig(engine=engine)
    config = ExperimentConfig(model="gpt2", engine="vllm", vllm=vllm_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0, 1]


def test_vllm_tp2_pp2_returns_four_gpus():
    """vLLM with tp=2, pp=2 returns [0, 1, 2, 3]."""
    from llenergymeasure.config.engine_configs import VLLMConfig, VLLMEngineConfig

    engine = VLLMEngineConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    vllm_cfg = VLLMConfig(engine=engine)
    config = ExperimentConfig(model="gpt2", engine="vllm", vllm=vllm_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0, 1, 2, 3]


def test_vllm_none_engine_vllm_config_returns_single_gpu():
    """vLLM config with engine=None falls through to [0]."""
    from llenergymeasure.config.engine_configs import VLLMConfig

    vllm_cfg = VLLMConfig(engine=None)
    config = ExperimentConfig(model="gpt2", engine="vllm", vllm=vllm_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0]


# ---------------------------------------------------------------------------
# TensorRT engine: uses tp_size
# ---------------------------------------------------------------------------


def test_tensorrt_no_config_returns_single_gpu():
    """tensorrt engine with no TensorRTConfig falls through to [0]."""
    config = ExperimentConfig(model="gpt2", engine="tensorrt")
    # config.tensorrt is None by default
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_tensorrt_tp1_returns_single_gpu():
    """tensorrt with tp_size=1 returns [0]."""
    from llenergymeasure.config.engine_configs import TensorRTConfig

    trt_cfg = TensorRTConfig(tp_size=1)
    config = ExperimentConfig(model="gpt2", engine="tensorrt", tensorrt=trt_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_tensorrt_tp4_returns_four_gpus():
    """tensorrt with tp_size=4 returns [0, 1, 2, 3]."""
    from llenergymeasure.config.engine_configs import TensorRTConfig

    trt_cfg = TensorRTConfig(tp_size=4)
    config = ExperimentConfig(model="gpt2", engine="tensorrt", tensorrt=trt_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0, 1, 2, 3]


def test_tensorrt_none_tp_size_returns_single_gpu():
    """tensorrt with tp_size=None falls through to [0]."""
    from llenergymeasure.config.engine_configs import TensorRTConfig

    trt_cfg = TensorRTConfig(tp_size=None)
    config = ExperimentConfig(model="gpt2", engine="tensorrt", tensorrt=trt_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0]


# ---------------------------------------------------------------------------
# PyTorch engine with device_map: uses pynvml GPU count
# ---------------------------------------------------------------------------


def test_pytorch_device_map_auto_single_gpu():
    """pytorch with device_map='auto' but 1 GPU falls through to [0]."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    pytorch_cfg = TransformersConfig(device_map="auto")
    config = ExperimentConfig(model="gpt2", engine="transformers", transformers=pytorch_cfg)

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_pynvml.nvmlShutdown.return_value = None

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        result = _resolve_gpu_indices(config)

    assert result == [0]


def test_pytorch_device_map_auto_multi_gpu():
    """pytorch with device_map='auto' and 4 GPUs returns [0, 1, 2, 3]."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    pytorch_cfg = TransformersConfig(device_map="auto")
    config = ExperimentConfig(model="gpt2", engine="transformers", transformers=pytorch_cfg)

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 4
    mock_pynvml.nvmlShutdown.return_value = None

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        result = _resolve_gpu_indices(config)

    assert result == [0, 1, 2, 3]


def test_pytorch_device_map_none_returns_single_gpu():
    """pytorch with device_map=None (not set) returns [0]."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    pytorch_cfg = TransformersConfig(device_map=None)
    config = ExperimentConfig(model="gpt2", engine="transformers", transformers=pytorch_cfg)
    result = _resolve_gpu_indices(config)
    assert result == [0]


def test_pytorch_device_map_pynvml_error_falls_through():
    """pytorch device_map with pynvml error falls through to [0]."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    pytorch_cfg = TransformersConfig(device_map="auto")
    config = ExperimentConfig(model="gpt2", engine="transformers", transformers=pytorch_cfg)

    # pynvml raises on init — should fall through silently
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML not available")

    with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
        result = _resolve_gpu_indices(config)

    assert result == [0]


def test_pytorch_device_map_pynvml_import_error_falls_through():
    """pytorch device_map with pynvml absent falls through to [0]."""
    from llenergymeasure.config.engine_configs import TransformersConfig

    pytorch_cfg = TransformersConfig(device_map="auto")
    config = ExperimentConfig(model="gpt2", engine="transformers", transformers=pytorch_cfg)

    # Remove pynvml from sys.modules to simulate it being absent
    pynvml_mod = sys.modules.pop("pynvml", None)
    sys.modules["pynvml"] = None  # type: ignore[assignment]
    try:
        result = _resolve_gpu_indices(config)
    finally:
        sys.modules.pop("pynvml", None)
        if pynvml_mod is not None:
            sys.modules["pynvml"] = pynvml_mod

    assert result == [0]
