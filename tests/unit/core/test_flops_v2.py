"""Unit tests for PaLM FLOPs formula and non-embedding param counting (no GPU required)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.core.flops import (
    _count_non_embedding_params,
    estimate_flops,
    estimate_flops_palm,
)
from llenergymeasure.domain.metrics import FlopsResult

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
# Backward compatibility — estimate_flops still exists
# =============================================================================


def test_legacy_estimate_flops_still_works() -> None:
    """estimate_flops (legacy wrapper) is still importable and callable."""
    assert callable(estimate_flops)
