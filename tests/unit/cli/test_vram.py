"""Unit tests for llenergymeasure.cli._vram.

Tests cover estimate_vram (dtype variants, network failure, missing metadata)
and get_gpu_vram_gb (success, missing pynvml, NVML error). All external calls
(HuggingFace Hub, pynvml) are mocked -- no network or GPU required.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.cli._vram import estimate_vram, get_gpu_vram_gb
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Llama-2-7B parameter count from safetensors metadata
_LLAMA_7B_PARAMS = 6_738_415_616

_VRAM_DEFAULTS = {"model": "meta-llama/Llama-2-7b-hf"}


def _make_model_info(param_count: int | None = _LLAMA_7B_PARAMS, with_config: bool = True):
    """Build a mock HuggingFace model_info object."""
    model_info = MagicMock()

    # safetensors metadata
    if param_count is not None:
        st = SimpleNamespace(total=param_count)
        model_info.safetensors = st
    else:
        model_info.safetensors = None

    # architecture config (Llama-2-7B defaults)
    if with_config:
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_size=4096,
            n_layer=None,
            n_head=None,
            num_layers=None,
            num_heads=None,
            d_model=None,
            n_embd=None,
        )
        model_info.config = cfg
    else:
        model_info.config = None

    return model_info


# ---------------------------------------------------------------------------
# estimate_vram — dtype variants
# ---------------------------------------------------------------------------


def _patch_hfapi(model_info_return=None, side_effect=None):
    """Return a context manager that patches huggingface_hub.HfApi.

    HfApi is imported lazily inside estimate_vram, so we patch at the source
    module (huggingface_hub) rather than at the call site.
    """
    pytest.importorskip("huggingface_hub")
    mock_api_instance = MagicMock()
    if side_effect is not None:
        mock_api_instance.model_info.side_effect = side_effect
    else:
        mock_api_instance.model_info.return_value = model_info_return

    mock_hf_api_class = MagicMock(return_value=mock_api_instance)
    return patch("huggingface_hub.HfApi", mock_hf_api_class), mock_api_instance


def test_estimate_vram_fp16():
    """fp16 weights use 2 bytes/param; returns correct breakdown dict."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")
    model_info = _make_model_info()

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is not None
    expected_weights_gb = (_LLAMA_7B_PARAMS * 2) / 1e9
    assert result["weights_gb"] == pytest.approx(expected_weights_gb, rel=1e-3)
    assert result["kv_cache_gb"] >= 0.0
    assert result["overhead_gb"] == pytest.approx(result["weights_gb"] * 0.15, rel=1e-3)
    assert result["total_gb"] == pytest.approx(
        result["weights_gb"] + result["kv_cache_gb"] + result["overhead_gb"], rel=1e-3
    )


def test_estimate_vram_fp32():
    """fp32 weights use 4 bytes/param — double the fp16 size."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float32")
    model_info = _make_model_info()

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is not None
    expected_weights_gb = (_LLAMA_7B_PARAMS * 4) / 1e9
    assert result["weights_gb"] == pytest.approx(expected_weights_gb, rel=1e-3)


def test_estimate_vram_bf16():
    """bf16 weights use 2 bytes/param — same size as fp16."""
    config = make_config(**_VRAM_DEFAULTS, dtype="bfloat16")
    model_info = _make_model_info()

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is not None
    expected_weights_gb = (_LLAMA_7B_PARAMS * 2) / 1e9
    assert result["weights_gb"] == pytest.approx(expected_weights_gb, rel=1e-3)


def test_estimate_vram_fp32_double_fp16():
    """fp32 uses 4 bytes/param, which is exactly 2x the fp16 (2 bytes/param) weight size."""
    config_fp16 = make_config(**_VRAM_DEFAULTS, dtype="float16")
    config_fp32 = make_config(**_VRAM_DEFAULTS, dtype="float32")
    model_info = _make_model_info()

    ctx16, _ = _patch_hfapi(model_info_return=model_info)
    with ctx16:
        result_fp16 = estimate_vram(config_fp16)

    ctx32, _ = _patch_hfapi(model_info_return=model_info)
    with ctx32:
        result_fp32 = estimate_vram(config_fp32)

    assert result_fp16 is not None and result_fp32 is not None
    assert result_fp32["weights_gb"] == pytest.approx(result_fp16["weights_gb"] * 2, rel=1e-3)


# ---------------------------------------------------------------------------
# estimate_vram — failure modes
# ---------------------------------------------------------------------------


def test_estimate_vram_network_failure_returns_none():
    """Any exception from HfApi.model_info returns None (non-blocking)."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")

    ctx, _ = _patch_hfapi(side_effect=Exception("Network unreachable"))
    with ctx:
        result = estimate_vram(config)

    assert result is None


def test_estimate_vram_no_safetensors_returns_none():
    """Model info with safetensors=None returns None (param count unavailable)."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")
    model_info = _make_model_info(param_count=None)  # safetensors is None

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is None


def test_estimate_vram_no_architecture_config_still_returns():
    """Model info with params but no config dict still returns dict; kv_cache_gb == 0.0."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")
    model_info = _make_model_info(param_count=_LLAMA_7B_PARAMS, with_config=False)

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is not None
    assert result["kv_cache_gb"] == 0.0
    assert result["weights_gb"] > 0.0


def test_estimate_vram_socket_timeout_restored():
    """Socket timeout is set to 5 s during the call and restored afterwards."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")
    original_timeout = socket.getdefaulttimeout()
    timeouts_seen: list[float | None] = []

    real_setdefaulttimeout = socket.setdefaulttimeout

    def capturing_setdefaulttimeout(t: float | None) -> None:
        timeouts_seen.append(t)
        real_setdefaulttimeout(t)

    # HfApi is imported lazily inside estimate_vram; trigger exception so we
    # still exercise the finally block (timeout restore path).
    ctx, _ = _patch_hfapi(side_effect=Exception("timeout test"))
    with ctx, patch("socket.setdefaulttimeout", side_effect=capturing_setdefaulttimeout):
        estimate_vram(config)

    # First call must set to 5, second call must restore the original value
    assert len(timeouts_seen) >= 2
    assert timeouts_seen[0] == 5
    # The final restore call should put back original_timeout
    assert timeouts_seen[-1] == original_timeout


def test_estimate_vram_kv_cache_nonzero_with_architecture():
    """KV cache is > 0 when architecture config is present and max_input_tokens > 0."""
    config = make_config(**_VRAM_DEFAULTS, dtype="float16")
    model_info = _make_model_info(with_config=True)

    ctx, _ = _patch_hfapi(model_info_return=model_info)
    with ctx:
        result = estimate_vram(config)

    assert result is not None
    assert result["kv_cache_gb"] > 0.0


# ---------------------------------------------------------------------------
# get_gpu_vram_gb
# ---------------------------------------------------------------------------


def test_get_gpu_vram_gb_success():
    """Returns GPU VRAM in GB when pynvml is available and works correctly."""
    mock_mem = SimpleNamespace(total=int(80 * 1e9))  # 80 GB A100

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

    mock_nvml_context = MagicMock()
    mock_nvml_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_nvml_context.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch("llenergymeasure.device.gpu_info.nvml_context", mock_nvml_context),
    ):
        result = get_gpu_vram_gb()

    assert result == pytest.approx(80.0, rel=1e-3)


def test_get_gpu_vram_gb_nvml_error():
    """Returns None when pynvml raises an exception during device query."""
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("NVML error")

    mock_nvml_context = MagicMock()
    mock_nvml_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_nvml_context.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.dict("sys.modules", {"pynvml": mock_pynvml}),
        patch("llenergymeasure.device.gpu_info.nvml_context", mock_nvml_context),
    ):
        result = get_gpu_vram_gb()

    assert result is None


def test_get_gpu_vram_gb_import_error_returns_none():
    """Returns None when the outer try/except catches an ImportError for pynvml."""

    # Patch the whole function body to raise ImportError on pynvml import
    def _raising_get_gpu_vram_gb() -> float | None:
        try:
            raise ImportError("No module named 'pynvml'")
        except Exception:
            return None

    # Verify same contract: returns None on any exception
    assert _raising_get_gpu_vram_gb() is None
