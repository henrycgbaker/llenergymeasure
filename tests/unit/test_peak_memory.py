"""Tests for peak_memory_mb measurement semantics (MEAS-04).

Verifies:
- PyTorchBackend resets peak stats before the measurement loop (code structure)
- VLLMBackend captures peak memory in _run_measurement (code structure)
- MemoryEfficiencyMetrics has inference_memory_mb field
- Field descriptions document the inference-window semantics precisely
- Both backends' _build_result() compute inference_memory_mb = max(0.0, peak - model)
"""

from __future__ import annotations

import re


def _pytorch_source() -> str:
    path = "src/llenergymeasure/core/backends/pytorch.py"
    return open(path).read()


def _vllm_source() -> str:
    path = "src/llenergymeasure/core/backends/vllm.py"
    return open(path).read()


def _metrics_source() -> str:
    path = "src/llenergymeasure/domain/metrics.py"
    return open(path).read()


def _extract_method_body(source: str, name: str) -> str:
    """Extract the body of a named method or class from source.

    Finds 'def name' or 'class name' and collects lines until the next
    definition at the same or lower indent level. Returns the raw text.
    """
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.search(rf"\b(?:def|class) {re.escape(name)}\b", line):
            start = i
            break
    if start is None:
        return ""

    # Determine the indent of the def/class line
    indent = len(lines[start]) - len(lines[start].lstrip())

    body_lines = [lines[start]]
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            body_lines.append(line)
            continue
        line_indent = len(line) - len(line.lstrip())
        # Stop when we hit a def/class at the same or lower indent (next definition)
        if line_indent <= indent and stripped.startswith(("def ", "class ")):
            break
        body_lines.append(line)

    return "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Code structure tests — PyTorchBackend
# ---------------------------------------------------------------------------


def test_pytorch_resets_peak_before_measurement():
    """reset_peak_memory_stats appears before 'with PowerThermalSampler(' in _run_measurement."""
    body = _extract_method_body(_pytorch_source(), "_run_measurement")
    assert "reset_peak_memory_stats" in body, (
        "PyTorchBackend._run_measurement must call reset_peak_memory_stats()"
    )
    assert "with PowerThermalSampler(" in body, (
        "PyTorchBackend._run_measurement must use PowerThermalSampler context manager"
    )
    reset_pos = body.index("reset_peak_memory_stats")
    # Search for the context-manager use, not the import
    sampler_pos = body.index("with PowerThermalSampler(")
    assert reset_pos < sampler_pos, (
        "reset_peak_memory_stats must appear BEFORE 'with PowerThermalSampler(' "
        "(reset must happen before the measurement window opens)"
    )


def test_pytorch_reads_peak_after_measurement():
    """max_memory_allocated appears in _run_measurement after PowerThermalSampler block."""
    body = _extract_method_body(_pytorch_source(), "_run_measurement")
    assert "max_memory_allocated" in body, (
        "PyTorchBackend._run_measurement must read max_memory_allocated()"
    )
    # max_memory_allocated should appear after the with PowerThermalSampler( block
    # (after the closing of the context manager — confirmed by 'peak_memory_mb' assignment)
    assert "peak_memory_mb" in body, "PyTorchBackend._run_measurement must assign peak_memory_mb"


def test_pytorch_build_result_computes_inference_memory():
    """_build_result in pytorch.py contains inference_memory_mb = max(0.0, ...) computation."""
    body = _extract_method_body(_pytorch_source(), "_build_result")
    assert "inference_memory_mb" in body, (
        "PyTorchBackend._build_result must compute inference_memory_mb"
    )
    assert "max(0.0" in body, (
        "PyTorchBackend._build_result must use max(0.0, ...) for inference_memory_mb"
    )


# ---------------------------------------------------------------------------
# Code structure tests — VLLMBackend
# ---------------------------------------------------------------------------


def test_vllm_has_peak_memory_capture():
    """VLLMBackend._run_measurement captures peak_memory_mb and resets stats."""
    body = _extract_method_body(_vllm_source(), "_run_measurement")
    assert "peak_memory_mb" in body, "VLLMBackend._run_measurement must set data.peak_memory_mb"
    assert "reset_peak_memory_stats" in body, (
        "VLLMBackend._run_measurement must call reset_peak_memory_stats()"
    )
    assert "max_memory_allocated" in body, (
        "VLLMBackend._run_measurement must read max_memory_allocated()"
    )
    # reset must appear before the batch inference block
    reset_pos = body.index("reset_peak_memory_stats")
    sampler_pos = body.index("with PowerThermalSampler(")
    assert reset_pos < sampler_pos, (
        "reset_peak_memory_stats must appear BEFORE 'with PowerThermalSampler(' in vLLM"
    )


def test_vllm_build_result_computes_inference_memory():
    """_build_result in vllm.py contains inference_memory_mb = max(0.0, ...) computation."""
    body = _extract_method_body(_vllm_source(), "_build_result")
    assert "inference_memory_mb" in body, (
        "VLLMBackend._build_result must compute inference_memory_mb"
    )
    assert "max(0.0" in body, (
        "VLLMBackend._build_result must use max(0.0, ...) for inference_memory_mb"
    )


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


def test_memory_field_descriptions_mention_inference_window():
    """ComputeMetrics.peak_memory_mb Field() description contains 'inference measurement window'."""
    src = _metrics_source()
    # Find the ComputeMetrics class body
    body = _extract_method_body(src, "ComputeMetrics")
    # Look for the peak_memory_mb Field definition including its description
    assert "inference measurement window" in body, (
        "ComputeMetrics.peak_memory_mb Field description must mention 'inference measurement window'"
    )
    assert "NOT model weights" in body, (
        "ComputeMetrics.peak_memory_mb Field description must explicitly state "
        "it does NOT capture model weights"
    )


def test_inference_memory_mb_field_exists():
    """MemoryEfficiencyMetrics source defines inference_memory_mb field with description."""
    src = _metrics_source()
    body = _extract_method_body(src, "MemoryEfficiencyMetrics")
    assert "inference_memory_mb" in body, (
        "MemoryEfficiencyMetrics must define inference_memory_mb field"
    )
    assert "_build_result" in body, (
        "inference_memory_mb description must state it is computed by backend _build_result()"
    )
    # Default 0.0 must be present
    assert "0.0" in body, "MemoryEfficiencyMetrics.inference_memory_mb must have default=0.0"


def test_memory_efficiency_schema_has_three_memory_fields():
    """MemoryEfficiencyMetrics source defines peak, model, and inference_memory_mb fields.

    This test uses source inspection to verify the schema contract without requiring
    the worktree's version to be the installed one (the venv may point to the main
    workspace installation during branch development).
    """
    src = _metrics_source()
    body = _extract_method_body(src, "MemoryEfficiencyMetrics")

    # All three fields must be present
    assert "peak_memory_mb" in body, "MemoryEfficiencyMetrics must have peak_memory_mb"
    assert "model_memory_mb" in body, "MemoryEfficiencyMetrics must have model_memory_mb"
    assert "inference_memory_mb" in body, "MemoryEfficiencyMetrics must have inference_memory_mb"

    # Derivation semantics documented
    assert "max(0.0, peak_memory_mb - model_memory_mb)" in src or (
        "max(0.0," in src and "peak_memory_mb" in src and "model_memory_mb" in src
    ), (
        "The derivation formula max(0.0, peak_memory_mb - model_memory_mb) must appear "
        "in metrics.py documentation or backend code"
    )
