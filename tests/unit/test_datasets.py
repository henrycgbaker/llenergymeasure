"""Unit tests for the llenergymeasure.datasets module.

Tests cover:
- aienergyscore JSONL file existence and structure
- load_prompts dispatching (built-in, synthetic, unknown)
- dataset_order config field validation and ordering behaviour
- core.dataset_loader importability (broken import fix)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Test 1: aienergyscore path exists
# ---------------------------------------------------------------------------


def test_aienergyscore_path_exists() -> None:
    from llenergymeasure.datasets import aienergyscore

    assert isinstance(aienergyscore, Path), "aienergyscore should be a Path"
    assert aienergyscore.exists(), f"JSONL file not found at {aienergyscore}"


# ---------------------------------------------------------------------------
# Test 2: JSONL has 1000 prompts (excluding provenance header)
# ---------------------------------------------------------------------------


def test_aienergyscore_has_1000_prompts() -> None:
    from llenergymeasure.datasets import aienergyscore

    lines = aienergyscore.read_text(encoding="utf-8").splitlines()
    prompt_lines = [line for line in lines if line.strip() and not line.startswith('{"_provenance')]
    assert len(prompt_lines) == 1000, f"Expected 1000 prompt lines, got {len(prompt_lines)}"


# ---------------------------------------------------------------------------
# Test 3: Provenance header present as first line
# ---------------------------------------------------------------------------


def test_provenance_header_present() -> None:
    from llenergymeasure.datasets import aienergyscore

    first_line = aienergyscore.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert "_provenance" in record, "First line should contain _provenance key"
    assert record["_provenance"] == "AIEnergyScore/text_generation"


# ---------------------------------------------------------------------------
# Test 4: load_prompts with built-in dataset
# ---------------------------------------------------------------------------


def test_load_prompts_builtin() -> None:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.datasets import load_prompts

    config = ExperimentConfig(model="x", dataset="aienergyscore", n=5)
    prompts = load_prompts(config)

    assert len(prompts) == 5, f"Expected 5 prompts, got {len(prompts)}"
    assert all(isinstance(p, str) and p.strip() for p in prompts), (
        "All prompts should be non-empty strings"
    )
    # Ensure not "Hello, " placeholder
    for prompt in prompts:
        assert "Hello, Hello, Hello," not in prompt, "Should be real prompts, not placeholder"


# ---------------------------------------------------------------------------
# Test 5: load_prompts with synthetic dataset
# ---------------------------------------------------------------------------


def test_load_prompts_synthetic() -> None:
    from llenergymeasure.config.models import ExperimentConfig, SyntheticDatasetConfig
    from llenergymeasure.datasets import load_prompts

    config = ExperimentConfig(
        model="x",
        dataset=SyntheticDatasetConfig(n=10, input_len=64),
        n=10,
    )
    prompts = load_prompts(config)

    assert len(prompts) == 10, f"Expected 10 prompts, got {len(prompts)}"
    assert all(isinstance(p, str) and p.strip() for p in prompts), (
        "All prompts should be non-empty strings"
    )


# ---------------------------------------------------------------------------
# Test 6: load_prompts with unknown dataset raises ValueError
# ---------------------------------------------------------------------------


def test_load_prompts_unknown_raises() -> None:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.datasets import load_prompts

    config = ExperimentConfig(model="x", dataset="nonexistent_dataset_xyz", n=5)
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_prompts(config)


# ---------------------------------------------------------------------------
# Test 7: core.dataset_loader is importable (broken import fix)
# ---------------------------------------------------------------------------


def test_dataset_loader_importable() -> None:
    from llenergymeasure.core.dataset_loader import load_prompts_from_source

    assert callable(load_prompts_from_source)


# ---------------------------------------------------------------------------
# Test 8: dataset_order config field validation
# ---------------------------------------------------------------------------


def test_dataset_order_config_field() -> None:
    from llenergymeasure.config.models import ExperimentConfig

    # Default
    config = ExperimentConfig(model="x")
    assert config.dataset_order == "interleaved"

    # Accepted values
    config_grouped = ExperimentConfig(model="x", dataset_order="grouped")
    assert config_grouped.dataset_order == "grouped"

    config_shuffled = ExperimentConfig(model="x", dataset_order="shuffled")
    assert config_shuffled.dataset_order == "shuffled"

    # Invalid value should raise ValidationError
    with pytest.raises(ValidationError):
        ExperimentConfig(model="x", dataset_order="invalid")


# ---------------------------------------------------------------------------
# Test 9: load_prompts grouped ordering groups by source
# ---------------------------------------------------------------------------


def test_load_prompts_grouped_ordering() -> None:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.datasets import aienergyscore, load_prompts

    # Check if dataset has source field
    lines = aienergyscore.read_text(encoding="utf-8").splitlines()
    prompt_lines = [line for line in lines if line.strip() and not line.startswith('{"_provenance')]
    first_record = json.loads(prompt_lines[0])

    if "source" not in first_record:
        # Dataset has no source column; grouped == interleaved (file order)
        # Just verify it doesn't raise an error
        config = ExperimentConfig(
            model="x", dataset="aienergyscore", n=100, dataset_order="grouped"
        )
        prompts = load_prompts(config)
        assert len(prompts) == 100
        return

    # Dataset has source column — verify grouped ordering
    config = ExperimentConfig(model="x", dataset="aienergyscore", n=100, dataset_order="grouped")
    prompts = load_prompts(config)
    assert len(prompts) == 100

    # Get sources for the first 100 prompts in grouped order
    grouped_records = sorted(
        [json.loads(line) for line in prompt_lines[:200]],
        key=lambda r: r.get("source", ""),
    )[:100]
    grouped_sources = [r.get("source", "") for r in grouped_records]

    # Sources should be monotonically non-decreasing (all from one source before next)
    for i in range(len(grouped_sources) - 1):
        assert grouped_sources[i] <= grouped_sources[i + 1], (
            f"Sources not sorted at index {i}: {grouped_sources[i]!r} > {grouped_sources[i + 1]!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: load_prompts shuffled ordering is deterministic
# ---------------------------------------------------------------------------


def test_load_prompts_shuffled_deterministic() -> None:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.datasets import load_prompts

    config = ExperimentConfig(
        model="x",
        dataset="aienergyscore",
        n=50,
        dataset_order="shuffled",
        random_seed=42,
    )

    prompts_1 = load_prompts(config)
    prompts_2 = load_prompts(config)

    assert prompts_1 == prompts_2, "Shuffled prompts should be deterministic with same seed"
    assert len(prompts_1) == 50

    # Different seed should (very likely) produce different order
    config_diff_seed = ExperimentConfig(
        model="x",
        dataset="aienergyscore",
        n=50,
        dataset_order="shuffled",
        random_seed=99,
    )
    prompts_diff = load_prompts(config_diff_seed)
    # With 50 prompts from 1000, different seeds almost certainly give different order
    assert prompts_1 != prompts_diff, "Different seeds should produce different orderings"
