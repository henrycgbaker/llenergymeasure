"""Unit tests for LoRAConfig Pydantic validation.

Tests cover:
- Exactly-one-source validation (adapter_id XOR adapter_path)
- Revision field requires adapter_id
- Default values
- extra=forbid
- Integration with ExperimentConfig
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExperimentConfig, LoRAConfig

# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------


def test_lora_adapter_id_only():
    """LoRAConfig with only adapter_id is valid."""
    lora = LoRAConfig(adapter_id="my-org/my-adapter")
    assert lora.adapter_id == "my-org/my-adapter"
    assert lora.adapter_path is None


def test_lora_adapter_path_only():
    """LoRAConfig with only adapter_path is valid."""
    lora = LoRAConfig(adapter_path="/path/to/adapter")
    assert lora.adapter_path == "/path/to/adapter"
    assert lora.adapter_id is None


def test_lora_both_sources_rejected():
    """LoRAConfig with both adapter_id and adapter_path raises ValidationError."""
    with pytest.raises(ValidationError, match="exactly one"):
        LoRAConfig(adapter_id="my-org/my-adapter", adapter_path="/path/to/adapter")


def test_lora_neither_source_rejected():
    """LoRAConfig with neither adapter_id nor adapter_path raises ValidationError."""
    with pytest.raises(ValidationError, match="exactly one"):
        LoRAConfig()


# ---------------------------------------------------------------------------
# Revision validation
# ---------------------------------------------------------------------------


def test_lora_revision_with_adapter_id():
    """LoRAConfig with revision and adapter_id is valid."""
    lora = LoRAConfig(adapter_id="my-org/my-adapter", revision="v1.0")
    assert lora.revision == "v1.0"
    assert lora.adapter_id == "my-org/my-adapter"


def test_lora_revision_with_adapter_path_rejected():
    """LoRAConfig with revision and adapter_path raises ValidationError."""
    with pytest.raises(ValidationError, match="revision requires adapter_id"):
        LoRAConfig(adapter_path="/path/to/adapter", revision="v1.0")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_lora_merge_weights_default_false():
    """merge_weights defaults to False."""
    lora = LoRAConfig(adapter_id="my-org/my-adapter")
    assert lora.merge_weights is False


# ---------------------------------------------------------------------------
# extra=forbid
# ---------------------------------------------------------------------------


def test_lora_extra_fields_forbidden():
    """Unknown fields raise ValidationError (extra='forbid')."""
    with pytest.raises(ValidationError):
        LoRAConfig(adapter_id="my-org/my-adapter", unknown_field="x")


# ---------------------------------------------------------------------------
# Integration with ExperimentConfig
# ---------------------------------------------------------------------------


def test_lora_in_experiment_config():
    """ExperimentConfig accepts a lora section."""
    config = ExperimentConfig(
        model="gpt2",
        backend="pytorch",
        lora=LoRAConfig(adapter_id="my-org/my-adapter"),
    )
    assert config.lora is not None
    assert config.lora.adapter_id == "my-org/my-adapter"


def test_lora_null_in_experiment_config():
    """lora=None is valid (default)."""
    config = ExperimentConfig(model="gpt2", backend="pytorch")
    assert config.lora is None
