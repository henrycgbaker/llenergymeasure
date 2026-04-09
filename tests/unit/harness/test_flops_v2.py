"""Unit tests for FLOPs estimation: PaLM formula and AutoConfig fallback.

Covers:
- _count_non_embedding_params (embedding exclusion, case-insensitive)
- estimate_flops_palm (formula, batch_size, method/confidence/precision fields)
- _count_params_from_config (AutoConfig-based parameter extraction)
- estimate_flops_palm_from_config (config-based FLOPs estimation)
- FlopsResult validation and derived fields on ExperimentResult
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from llenergymeasure.domain.metrics import FlopsResult
from llenergymeasure.harness.flops import (
    _count_non_embedding_params,
    _count_params_from_config,
    estimate_flops_palm,
    estimate_flops_palm_from_config,
)

# =============================================================================
# Mock model helpers
# =============================================================================


class MockParam:
    """Minimal mock for a torch Parameter with a known numel() value."""

    def __init__(self, numel_val: int) -> None:
        self._numel = numel_val

    def numel(self) -> int:
        return self._numel


class MockModel:
    """Minimal mock model with named_parameters() support."""

    def __init__(self, params: list[tuple[str, MockParam]]) -> None:
        self._params = params

    def named_parameters(self):  # type: ignore[return]
        return iter(self._params)

    def parameters(self):  # type: ignore[return]
        return iter(p for _, p in self._params)


def make_model(
    non_embed_count: int = 1_000_000,
    embed_count: int = 500_000,
) -> MockModel:
    """Create a mock model with known non-embedding and embedding param counts."""
    params = [
        ("layer.weight", MockParam(non_embed_count)),
        ("embed_tokens.weight", MockParam(embed_count)),
    ]
    return MockModel(params)


# =============================================================================
# _count_non_embedding_params
# =============================================================================


def test_count_non_embedding_params_excludes_embeddings() -> None:
    """Only non-embedding layers are counted."""
    model = MockModel(
        [
            ("transformer.layer.weight", MockParam(1_000_000)),
            ("embed_tokens.weight", MockParam(500_000)),
            ("model.embed_positions.weight", MockParam(256_000)),
            ("lm_head.weight", MockParam(200_000)),
        ]
    )
    count = _count_non_embedding_params(model)
    assert count == 1_000_000 + 200_000


def test_count_non_embedding_params_all_non_embed() -> None:
    """When there are no embedding layers, all params are counted."""
    model = MockModel(
        [
            ("layer.weight", MockParam(1_000_000)),
            ("fc.bias", MockParam(1_000)),
        ]
    )
    count = _count_non_embedding_params(model)
    assert count == 1_001_000


def test_count_non_embedding_params_all_embed() -> None:
    """When all params are embedding layers, result is 0."""
    model = MockModel(
        [
            ("embed_tokens.weight", MockParam(500_000)),
            ("embed_positions.weight", MockParam(256_000)),
        ]
    )
    count = _count_non_embedding_params(model)
    assert count == 0


def test_count_non_embedding_params_embed_case_insensitive() -> None:
    """'embed' check is case-insensitive."""
    model = MockModel(
        [
            ("EMBED_tokens.weight", MockParam(100_000)),
            ("Embedding.weight", MockParam(50_000)),
            ("dense.weight", MockParam(200_000)),
        ]
    )
    count = _count_non_embedding_params(model)
    assert count == 200_000


# =============================================================================
# estimate_flops_palm — formula correctness
# =============================================================================


def test_estimate_flops_palm_basic() -> None:
    """PaLM formula: value == 2 * non_embed_params * (n_input + n_output)."""
    model = MockModel(
        [
            ("layer.weight", MockParam(1_000_000)),
            ("embed_tokens.weight", MockParam(500_000)),
        ]
    )
    n_input, n_output = 100, 50
    result = estimate_flops_palm(model, n_input, n_output)

    expected = 2 * 1_000_000 * (n_input + n_output)
    assert result.value == float(expected)


def test_estimate_flops_palm_excludes_embeddings() -> None:
    """Embedding params are excluded from PaLM FLOPs calculation."""
    # With embeddings only
    embed_only_model = MockModel([("embed.weight", MockParam(500_000))])
    result_embed = estimate_flops_palm(embed_only_model, 100, 50)
    assert result_embed.value == 0.0

    # With non-embed params
    mixed_model = MockModel(
        [
            ("layer.weight", MockParam(1_000_000)),
            ("embed.weight", MockParam(500_000)),
        ]
    )
    result_mixed = estimate_flops_palm(mixed_model, 100, 50)
    assert result_mixed.value == float(2 * 1_000_000 * 150)


def test_estimate_flops_palm_method_is_palm() -> None:
    """Result method is 'palm_formula'."""
    model = make_model()
    result = estimate_flops_palm(model, 100, 50)
    assert result.method == "palm_formula"


def test_estimate_flops_palm_confidence_is_high() -> None:
    """Result confidence is 'high'."""
    model = make_model()
    result = estimate_flops_palm(model, 100, 50)
    assert result.confidence == "high"


def test_estimate_flops_palm_batch_size() -> None:
    """Batch size multiplier is applied: flops == 2 * params * batch_size * total_tokens."""
    model = MockModel([("layer.weight", MockParam(1_000_000))])
    n_input, n_output, batch_size = 100, 50, 4

    result = estimate_flops_palm(model, n_input, n_output, batch_size=batch_size)
    expected = 2 * 1_000_000 * batch_size * (n_input + n_output)
    assert result.value == float(expected)


def test_estimate_flops_palm_precision_field() -> None:
    """precision field is 'n/a' (precision does not affect FLOPs)."""
    model = make_model()
    result = estimate_flops_palm(model, 100, 50)
    assert result.precision == "n/a"


def test_estimate_flops_palm_notes_contains_formula() -> None:
    """Notes include PaLM formula reference."""
    model = make_model()
    result = estimate_flops_palm(model, 100, 50)
    assert result.notes is not None
    assert "PaLM" in result.notes


def test_estimate_flops_palm_zero_tokens() -> None:
    """Zero tokens produces 0 FLOPs."""
    model = MockModel([("layer.weight", MockParam(1_000_000))])
    result = estimate_flops_palm(model, 0, 0)
    assert result.value == 0.0


# =============================================================================
# FlopsResult — palm_formula literal
# =============================================================================


def test_flops_result_palm_formula_literal() -> None:
    """FlopsResult accepts 'palm_formula' as a valid method."""
    r = FlopsResult(value=1.0, method="palm_formula", confidence="high", precision="n/a")
    assert r.method == "palm_formula"


def test_flops_result_palm_formula_is_valid() -> None:
    """FlopsResult.is_valid returns True for positive PaLM result."""
    r = FlopsResult(value=1e12, method="palm_formula", confidence="high", precision="n/a")
    assert r.is_valid is True


def test_flops_result_legacy_methods_still_valid() -> None:
    """Legacy method literals still accepted (backward compat)."""
    for method in ("calflops", "architecture", "parameter_estimate"):
        r = FlopsResult(value=1.0, method=method, confidence="medium", precision="fp16")
        assert r.method == method


def test_flops_result_invalid_method_rejected() -> None:
    """Unknown method values raise ValidationError."""
    with pytest.raises(ValidationError):
        FlopsResult(
            value=1.0,
            method="unknown_method",  # type: ignore[arg-type]
            confidence="high",
            precision="fp16",
        )


# =============================================================================
# FLOPs derived fields on ExperimentResult
# =============================================================================


def _make_experiment_result(**overrides):
    """Build a minimal valid ExperimentResult for FLOPs derived field tests.

    Delegates to conftest make_result with FLOPs-specific defaults.
    """
    from tests.conftest import make_result

    defaults = {
        "experiment_id": "flops-test-001",
        "total_tokens": 100,
        "total_inference_time_sec": 10.0,
        "avg_tokens_per_second": 10.0,
        "avg_energy_per_token_j": 0.1,
        "total_flops": 0.0,
    }
    defaults.update(overrides)
    return make_result(**defaults)


def test_flops_derived_fields_computed() -> None:
    """flops_per_output_token, flops_per_input_token, flops_per_second are non-None when data available."""
    total_flops = 1e12
    output_tokens = 50
    input_tokens = 50
    inference_time_sec = 10.0

    result = _make_experiment_result(
        total_flops=total_flops,
        flops_per_output_token=total_flops / output_tokens,
        flops_per_input_token=total_flops / input_tokens,
        flops_per_second=total_flops / inference_time_sec,
    )

    assert result.flops_per_output_token is not None
    assert result.flops_per_input_token is not None
    assert result.flops_per_second is not None
    assert result.flops_per_output_token == pytest.approx(total_flops / output_tokens)
    assert result.flops_per_input_token == pytest.approx(total_flops / input_tokens)
    assert result.flops_per_second == pytest.approx(total_flops / inference_time_sec)


def test_flops_derived_fields_none_when_zero() -> None:
    """All FLOPs derived fields are None when total_flops=0."""
    result = _make_experiment_result(total_flops=0.0)

    assert result.flops_per_output_token is None
    assert result.flops_per_input_token is None
    assert result.flops_per_second is None


# =============================================================================
# _count_params_from_config — AutoConfig-based parameter extraction
# =============================================================================


def _make_autoconfig_mock(
    hidden_size: int = 4096,
    num_hidden_layers: int = 32,
    intermediate_size: int | None = None,
) -> MagicMock:
    """Build a mock AutoConfig object with standard architecture fields."""
    cfg = MagicMock()
    cfg.hidden_size = hidden_size
    cfg.num_hidden_layers = num_hidden_layers
    if intermediate_size is not None:
        cfg.intermediate_size = intermediate_size
    else:
        # getattr fallback: MagicMock will return a MagicMock for missing attrs,
        # so we set it explicitly to simulate the fallback default
        del cfg.intermediate_size  # Remove auto-attribute so getattr uses default
    return cfg


def test_count_params_from_config_success() -> None:
    """_count_params_from_config returns correct count with known config values.

    Formula: attn = 4 * h * h * layers; ffn = 2 * h * intermediate * layers
    Total = attn + ffn
    """
    hidden = 64
    layers = 4
    intermediate = 256  # = hidden * 4

    mock_cfg = MagicMock()
    mock_cfg.hidden_size = hidden
    mock_cfg.num_hidden_layers = layers
    mock_cfg.intermediate_size = intermediate

    mock_autoconfig_cls = MagicMock()
    mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

    with patch.dict(sys.modules, {"transformers": MagicMock(AutoConfig=mock_autoconfig_cls)}):
        result = _count_params_from_config("test/model")

    expected_attn = 4 * hidden * hidden * layers  # 4 * 64 * 64 * 4 = 65_536
    expected_ffn = 2 * hidden * intermediate * layers  # 2 * 64 * 256 * 4 = 131_072
    expected = expected_attn + expected_ffn  # 196_608

    assert result == expected, f"Expected {expected}, got {result}"


def test_count_params_from_config_intermediate_size_default() -> None:
    """_count_params_from_config uses hidden * 4 as default for intermediate_size."""
    hidden = 128
    layers = 2

    mock_cfg = MagicMock(spec=["hidden_size", "num_hidden_layers"])
    mock_cfg.hidden_size = hidden
    mock_cfg.num_hidden_layers = layers

    mock_autoconfig_cls = MagicMock()
    mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

    with patch.dict(sys.modules, {"transformers": MagicMock(AutoConfig=mock_autoconfig_cls)}):
        result = _count_params_from_config("test/model")

    intermediate = hidden * 4  # default
    expected = 4 * hidden * hidden * layers + 2 * hidden * intermediate * layers
    assert result == expected


def test_count_params_from_config_failure() -> None:
    """_count_params_from_config returns None when AutoConfig.from_pretrained raises."""
    mock_autoconfig_cls = MagicMock()
    mock_autoconfig_cls.from_pretrained.side_effect = OSError("Model not found")

    with patch.dict(sys.modules, {"transformers": MagicMock(AutoConfig=mock_autoconfig_cls)}):
        result = _count_params_from_config("nonexistent/model")

    assert result is None


def test_count_params_from_config_import_error() -> None:
    """_count_params_from_config returns None when transformers is not importable."""
    with patch.dict(sys.modules, {"transformers": None}):
        result = _count_params_from_config("test/model")

    assert result is None


def test_estimate_flops_palm_from_config_success() -> None:
    """estimate_flops_palm_from_config returns FlopsResult with confidence='medium'."""
    hidden = 64
    layers = 4
    intermediate = 256
    n_input, n_output = 100, 50

    mock_cfg = MagicMock()
    mock_cfg.hidden_size = hidden
    mock_cfg.num_hidden_layers = layers
    mock_cfg.intermediate_size = intermediate

    mock_autoconfig_cls = MagicMock()
    mock_autoconfig_cls.from_pretrained.return_value = mock_cfg

    with patch.dict(sys.modules, {"transformers": MagicMock(AutoConfig=mock_autoconfig_cls)}):
        result = estimate_flops_palm_from_config("test/model", n_input, n_output)

    assert result is not None
    assert result.confidence == "medium"
    assert result.method == "palm_formula"
    assert result.value > 0

    # Verify formula: 2 * n_params * total_tokens
    n_params = 4 * hidden * hidden * layers + 2 * hidden * intermediate * layers
    expected = 2 * n_params * (n_input + n_output)
    assert result.value == float(expected)


def test_estimate_flops_palm_from_config_none_on_failure() -> None:
    """estimate_flops_palm_from_config returns None when config cannot be loaded."""
    mock_autoconfig_cls = MagicMock()
    mock_autoconfig_cls.from_pretrained.side_effect = ValueError("Bad config")

    with patch.dict(sys.modules, {"transformers": MagicMock(AutoConfig=mock_autoconfig_cls)}):
        result = estimate_flops_palm_from_config("bad/model", 100, 50)

    assert result is None
