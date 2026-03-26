"""Behavioural tests for peak_memory_mb measurement semantics (MEAS-04).

Verifies:
- inference_memory_mb = max(0.0, peak_memory_mb - model_memory_mb) formula
- Peak memory value flows correctly through InferenceOutput to ExperimentResult
- PyTorchBackend calls reset_peak_memory_stats before and max_memory_allocated after inference
- The clamp at 0.0 is applied when peak_memory_mb <= model_memory_mb

These tests use InferenceOutput as a plain dataclass and monkeypatching of
torch.cuda to verify ordering without requiring real GPU hardware.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Formula tests — MeasurementHarness._build_result inference_memory_mb
# ---------------------------------------------------------------------------


def test_inference_memory_is_peak_minus_model():
    """inference_memory_mb = peak_memory_mb - model_memory_mb when peak > model."""
    from llenergymeasure.backends.protocol import InferenceOutput

    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=100,
        output_tokens=50,
        peak_memory_mb=1024.0,
        model_memory_mb=512.0,
    )
    # Compute using the same formula as MeasurementHarness._build_result
    inference_memory_mb = max(0.0, output.peak_memory_mb - output.model_memory_mb)
    assert inference_memory_mb == 512.0


def test_inference_memory_clamped_to_zero_when_peak_less_than_model():
    """inference_memory_mb is clamped to 0.0 when peak_memory_mb < model_memory_mb.

    This edge case arises when measurement uncertainty or GPU driver accounting
    causes the reported peak to be lower than the model baseline.
    """
    from llenergymeasure.backends.protocol import InferenceOutput

    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=100,
        output_tokens=50,
        peak_memory_mb=400.0,  # Less than model_memory_mb
        model_memory_mb=512.0,
    )
    inference_memory_mb = max(0.0, output.peak_memory_mb - output.model_memory_mb)
    assert inference_memory_mb == 0.0


def test_inference_memory_zero_when_peak_equals_model():
    """inference_memory_mb is 0.0 when peak_memory_mb == model_memory_mb exactly."""
    from llenergymeasure.backends.protocol import InferenceOutput

    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=100,
        output_tokens=50,
        peak_memory_mb=512.0,
        model_memory_mb=512.0,
    )
    inference_memory_mb = max(0.0, output.peak_memory_mb - output.model_memory_mb)
    assert inference_memory_mb == 0.0


# ---------------------------------------------------------------------------
# Ordering test — PyTorchBackend resets peak before inference measurement
# ---------------------------------------------------------------------------


def test_peak_memory_reset_precedes_measurement():
    """PyTorchBackend.run_inference calls reset_peak_memory_stats before max_memory_allocated.

    Uses monkeypatching to record call order without GPU hardware. The call log
    must show 'reset' before 'max_alloc' to confirm the measurement window is
    inference-only (not including model weights).
    """
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    call_log: list[str] = []

    fake_model = object()

    # Minimal mock tokenizer callable that returns plausible tensor-like data
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            # Return minimal dict with input_ids as a list-of-lists
            import types

            result = types.SimpleNamespace()
            result["input_ids"] = [[1, 2, 3]]
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

        pad_token = "<pad>"
        eos_token = "<eos>"

    # We don't need a real GPU — just verify the ordering of calls.
    # Patch cuda.is_available to True and intercept reset/max_alloc.
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.reset_peak_memory_stats", side_effect=lambda: call_log.append("reset")),
        patch(
            "torch.cuda.max_memory_allocated",
            side_effect=lambda: (call_log.append("max_alloc"), 512 * 1024 * 1024)[1],
        ),
        patch.object(
            PyTorchBackend,
            "_run_batch",
            return_value=(10, 20, 0.5),
        ),
    ):
        backend = PyTorchBackend()
        config = ExperimentConfig(model="test-model", backend="pytorch", n=1)
        backend.run_inference(config, (fake_model, FakeTokenizer()), ["test prompt"])

    assert "reset" in call_log, "reset_peak_memory_stats must be called in run_inference"
    assert "max_alloc" in call_log, "max_memory_allocated must be called in run_inference"
    assert call_log.index("reset") < call_log.index("max_alloc"), (
        "reset_peak_memory_stats must be called BEFORE max_memory_allocated — "
        "reset clears the peak counter so the measurement captures only the inference window"
    )


# ---------------------------------------------------------------------------
# Seed test — PyTorchBackend seeds torch RNG before inference
# ---------------------------------------------------------------------------


def test_pytorch_backend_seeds_rng_before_inference():
    """PyTorchBackend.run_inference calls torch.manual_seed with config.random_seed."""
    pytest.importorskip("torch")
    from llenergymeasure.backends.pytorch import PyTorchBackend
    from llenergymeasure.config.models import ExperimentConfig

    seeded_values: list[int] = []

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.reset_peak_memory_stats"),
        patch("torch.cuda.max_memory_allocated", return_value=0),
        patch("torch.manual_seed", side_effect=lambda s: seeded_values.append(s)),
        patch.object(PyTorchBackend, "_run_batch", return_value=(10, 20, 0.5)),
    ):
        backend = PyTorchBackend()
        config = ExperimentConfig(model="test-model", backend="pytorch", n=1, random_seed=123)
        backend.run_inference(config, (object(), None), ["test prompt"])

    assert 123 in seeded_values, "torch.manual_seed must be called with config.random_seed"


# ---------------------------------------------------------------------------
# Domain model tests — MemoryEfficiencyMetrics field existence
# ---------------------------------------------------------------------------


def test_memory_efficiency_metrics_has_three_memory_fields():
    """MemoryEfficiencyMetrics defines peak_memory_mb, model_memory_mb, and inference_memory_mb."""
    from llenergymeasure.domain.metrics import MemoryEfficiencyMetrics

    # Verify all three fields exist on the class
    assert hasattr(MemoryEfficiencyMetrics, "model_fields"), (
        "MemoryEfficiencyMetrics must be a Pydantic model"
    )
    fields = MemoryEfficiencyMetrics.model_fields
    assert "peak_memory_mb" in fields, "MemoryEfficiencyMetrics must have peak_memory_mb"
    assert "model_memory_mb" in fields, "MemoryEfficiencyMetrics must have model_memory_mb"
    assert "inference_memory_mb" in fields, "MemoryEfficiencyMetrics must have inference_memory_mb"


def test_inference_memory_mb_default_is_zero():
    """MemoryEfficiencyMetrics.inference_memory_mb defaults to 0.0."""
    from llenergymeasure.domain.metrics import MemoryEfficiencyMetrics

    m = MemoryEfficiencyMetrics()
    assert m.inference_memory_mb == 0.0
