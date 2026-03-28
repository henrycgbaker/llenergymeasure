"""Unit tests for PyTorch backend LoRA adapter integration.

All tests mock peft and transformers - no GPU or model downloads required.
Tests cover:
- load_model with/without LoRA
- LoRA adapter_id vs adapter_path
- merge_weights behaviour
- revision kwarg forwarding
- validate_config LoRA checks
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from llenergymeasure.backends.pytorch import PyTorchBackend
from llenergymeasure.config.models import ExperimentConfig, LoRAConfig

# =============================================================================
# Helpers
# =============================================================================


def _make_config(**overrides) -> ExperimentConfig:
    """Return a valid ExperimentConfig with sensible defaults."""
    defaults: dict = {"model": "gpt2", "backend": "pytorch"}
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_mock_model():
    """Create a mock HuggingFace model."""
    model = MagicMock()
    model.device = "cpu"
    model.eval.return_value = model
    return model


def _make_mock_tokenizer():
    """Create a mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    return tokenizer


def _patch_transformers():
    """Context manager patches for transformers AutoModel and AutoTokenizer."""
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()

    auto_model_patch = patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    )
    auto_tokenizer_patch = patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    return auto_model_patch, auto_tokenizer_patch, mock_model


# =============================================================================
# load_model tests
# =============================================================================


class TestLoadModelWithoutLora:
    """Test that load_model without LoRA does not touch peft."""

    def test_load_model_without_lora(self):
        """When config.lora is None, PeftModel is NOT imported/called."""
        auto_model_p, auto_tok_p, _mock_model = _patch_transformers()

        backend = PyTorchBackend()
        config = _make_config()
        assert config.lora is None

        with auto_model_p, auto_tok_p:
            result_model, result_tokenizer = backend.load_model(config)

        assert result_model is not None
        assert result_tokenizer is not None


class TestLoadModelWithLoraAdapterId:
    """Test load_model with LoRA adapter_id."""

    def test_load_model_with_lora_adapter_id(self):
        """PeftModel.from_pretrained is called with adapter_id."""
        auto_model_p, auto_tok_p, base_model = _patch_transformers()

        peft_model = _make_mock_model()
        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = peft_model

        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_id="my-org/my-adapter"))

        with auto_model_p, auto_tok_p, patch.dict(sys.modules, {"peft": mock_peft_module}):
            _model, _tokenizer = backend.load_model(config)

        mock_peft_module.PeftModel.from_pretrained.assert_called_once_with(
            base_model, "my-org/my-adapter"
        )


class TestLoadModelWithLoraAdapterPath:
    """Test load_model with LoRA adapter_path."""

    def test_load_model_with_lora_adapter_path(self):
        """PeftModel.from_pretrained is called with adapter_path."""
        auto_model_p, auto_tok_p, base_model = _patch_transformers()

        peft_model = _make_mock_model()
        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = peft_model

        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_path="/path/to/adapter"))

        with auto_model_p, auto_tok_p, patch.dict(sys.modules, {"peft": mock_peft_module}):
            _model, _tokenizer = backend.load_model(config)

        mock_peft_module.PeftModel.from_pretrained.assert_called_once_with(
            base_model, "/path/to/adapter"
        )


class TestLoadModelWithLoraMergeWeights:
    """Test merge_weights behaviour."""

    def test_load_model_with_lora_merge_weights(self):
        """merge_and_unload() is called when merge_weights=True."""
        auto_model_p, auto_tok_p, _base_model = _patch_transformers()

        merged_model = _make_mock_model()
        peft_model = _make_mock_model()
        peft_model.merge_and_unload.return_value = merged_model

        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = peft_model

        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_id="my-org/my-adapter", merge_weights=True))

        with auto_model_p, auto_tok_p, patch.dict(sys.modules, {"peft": mock_peft_module}):
            _model, _tokenizer = backend.load_model(config)

        peft_model.merge_and_unload.assert_called_once()

    def test_load_model_with_lora_no_merge(self):
        """merge_and_unload() NOT called when merge_weights=False."""
        auto_model_p, auto_tok_p, _base_model = _patch_transformers()

        peft_model = _make_mock_model()
        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = peft_model

        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_id="my-org/my-adapter", merge_weights=False))

        with auto_model_p, auto_tok_p, patch.dict(sys.modules, {"peft": mock_peft_module}):
            _model, _tokenizer = backend.load_model(config)

        peft_model.merge_and_unload.assert_not_called()


class TestLoadModelWithLoraRevision:
    """Test revision kwarg forwarding."""

    def test_load_model_with_lora_revision(self):
        """revision kwarg is passed to PeftModel.from_pretrained."""
        auto_model_p, auto_tok_p, base_model = _patch_transformers()

        peft_model = _make_mock_model()
        mock_peft_module = MagicMock()
        mock_peft_module.PeftModel.from_pretrained.return_value = peft_model

        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_id="my-org/my-adapter", revision="v1.0"))

        with auto_model_p, auto_tok_p, patch.dict(sys.modules, {"peft": mock_peft_module}):
            _model, _tokenizer = backend.load_model(config)

        mock_peft_module.PeftModel.from_pretrained.assert_called_once_with(
            base_model, "my-org/my-adapter", revision="v1.0"
        )


# =============================================================================
# validate_config tests
# =============================================================================


class TestValidateConfigLora:
    """Test validate_config LoRA checks."""

    def test_validate_config_no_lora(self):
        """validate_config returns empty list when no lora."""
        backend = PyTorchBackend()
        config = _make_config()
        errors = backend.validate_config(config)
        assert errors == []

    def test_validate_config_lora_missing_peft(self):
        """validate_config returns error when peft not importable."""
        backend = PyTorchBackend()
        config = _make_config(lora=LoRAConfig(adapter_id="my-org/my-adapter"))

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "peft":
                raise ImportError("No module named 'peft'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            errors = backend.validate_config(config)

        assert len(errors) >= 1
        assert any("peft" in e for e in errors)

    def test_validate_config_lora_missing_path(self, tmp_path):
        """validate_config returns error when adapter_path doesn't exist."""
        backend = PyTorchBackend()
        nonexistent = str(tmp_path / "nonexistent_adapter")
        config = _make_config(lora=LoRAConfig(adapter_path=nonexistent))

        errors = backend.validate_config(config)
        assert any("does not exist" in e for e in errors)
