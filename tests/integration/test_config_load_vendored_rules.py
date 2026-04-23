"""End-to-end check: construct ExperimentConfig with the real corpus on disk.

Exercises the vendored-rules pipeline without any monkeypatching:
  configs/validation_rules/transformers.yaml -> loader -> generic validator.

These are quick unit-style integration tests — no network, no GPU, no
engine imports. They live in ``tests/integration/`` because they depend on
the on-disk corpus file rather than pure Python objects.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from llenergymeasure.config.engine_configs import (
    TransformersConfig,
    TransformersSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig, _reset_rules_loader_cache
from llenergymeasure.config.warnings import ConfigValidationWarning


@pytest.fixture(autouse=True)
def reset_loader() -> None:
    """Reset the module-level loader before each test for independence."""
    _reset_rules_loader_cache()


def test_minimal_transformers_config_loads_without_errors_or_warnings() -> None:
    """A bare-minimum transformers config must not trigger any vendored rule.

    If this fails, the shipped corpus contains a rule that erroneously matches
    default values — a corpus bug to fix upstream in 50.2a, not a validator
    bug here.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigValidationWarning)
        cfg = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    spurious = [w for w in caught if issubclass(w.category, ConfigValidationWarning)]
    assert spurious == [], f"corpus fires on default config: {[str(w.message) for w in spurious]}"
    assert cfg._dormant_observations == []


def test_flash_attn_float32_rejected_end_to_end() -> None:
    """Corpus rule ``tf_flash_attn_requires_half_precision`` raises on float32."""
    with pytest.raises(ValidationError, match="flash_attention"):
        ExperimentConfig(
            task={"model": "gpt2"},
            engine="transformers",
            transformers=TransformersConfig(
                dtype="float32", attn_implementation="flash_attention_2"
            ),
        )


def test_greedy_do_sample_false_produces_dormant_observations() -> None:
    """``do_sample=False`` + ``temperature=0.9`` triggers dormancy in the corpus."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(do_sample=False, temperature=0.9),
        ),
    )
    observations = cfg._dormant_observations
    assert observations, "expected at least one dormant observation for temperature under greedy"


def test_vllm_without_corpus_does_not_raise() -> None:
    """Engines without a vendored corpus fall through cleanly (missing-file path)."""
    # vLLM corpus doesn't exist on disk as of 50.2c; the loader raises
    # FileNotFoundError which the validator handles.
    cfg = ExperimentConfig(task={"model": "gpt2"}, engine="vllm")
    assert cfg._dormant_observations == []


def test_multiple_sampling_fields_each_generate_observations() -> None:
    """Every dormant sampling field under greedy produces its own observation."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        transformers=TransformersConfig(
            sampling=TransformersSamplingConfig(
                do_sample=False, temperature=0.9, top_p=0.95, top_k=40
            ),
        ),
    )
    observations = cfg._dormant_observations
    assert len(observations) >= 3, (
        f"expected at least 3 dormant observations (temperature, top_p, top_k); "
        f"got {len(observations)}: {[o.reason for o in observations]}"
    )
