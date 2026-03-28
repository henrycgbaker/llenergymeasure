"""Unit tests for vLLM LoRA engine config fields.

Tests cover:
- enable_lora field
- max_lora_rank constraints
- lora_extra_vocab_size constraints
- lora_dtype Literal validation
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.backend_configs import VLLMEngineConfig

# ---------------------------------------------------------------------------
# enable_lora
# ---------------------------------------------------------------------------


def test_vllm_enable_lora():
    """VLLMEngineConfig accepts enable_lora=True."""
    config = VLLMEngineConfig(enable_lora=True)
    assert config.enable_lora is True


# ---------------------------------------------------------------------------
# max_lora_rank
# ---------------------------------------------------------------------------


def test_vllm_max_lora_rank():
    """VLLMEngineConfig accepts valid max_lora_rank."""
    config = VLLMEngineConfig(max_lora_rank=32)
    assert config.max_lora_rank == 32


def test_vllm_max_lora_rank_zero_rejected():
    """max_lora_rank=0 violates ge=1 constraint."""
    with pytest.raises(ValidationError):
        VLLMEngineConfig(max_lora_rank=0)


# ---------------------------------------------------------------------------
# lora_extra_vocab_size
# ---------------------------------------------------------------------------


def test_vllm_lora_extra_vocab_size():
    """VLLMEngineConfig accepts valid lora_extra_vocab_size."""
    config = VLLMEngineConfig(lora_extra_vocab_size=512)
    assert config.lora_extra_vocab_size == 512


# ---------------------------------------------------------------------------
# lora_dtype
# ---------------------------------------------------------------------------


def test_vllm_lora_dtype():
    """VLLMEngineConfig accepts valid lora_dtype literal values."""
    for dtype in ("float16", "bfloat16", "float32", "auto"):
        config = VLLMEngineConfig(lora_dtype=dtype)
        assert config.lora_dtype == dtype


def test_vllm_lora_dtype_invalid_rejected():
    """Invalid lora_dtype string is rejected."""
    with pytest.raises(ValidationError):
        VLLMEngineConfig(lora_dtype="int8")
