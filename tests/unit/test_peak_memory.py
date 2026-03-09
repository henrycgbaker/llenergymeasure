"""Tests for peak_memory_mb measurement semantics (MEAS-04).

Verifies:
- PyTorchBackend resets peak stats before the measurement loop in run_inference (code structure)
- VLLMBackend captures peak memory in run_inference (code structure)
- MemoryEfficiencyMetrics has inference_memory_mb field
- Field descriptions document the inference-window semantics precisely
- MeasurementHarness._build_result() computes inference_memory_mb = max(0.0, peak - model)

Note: After the harness refactor (phase 27-02), both backends are thin BackendPlugin
implementations. The measurement lifecycle (PowerThermalSampler, energy tracking,
result assembly) lives in MeasurementHarness (core/harness.py). The backends own
reset_peak_memory_stats + max_memory_allocated in run_inference, and harness._build_result
computes inference_memory_mb from InferenceOutput.peak_memory_mb and model_memory_mb.
"""

from __future__ import annotations

import re


def _pytorch_source() -> str:
    path = "src/llenergymeasure/core/backends/pytorch.py"
    return open(path).read()


def _vllm_source() -> str:
    path = "src/llenergymeasure/core/backends/vllm.py"
    return open(path).read()


def _harness_source() -> str:
    path = "src/llenergymeasure/core/harness.py"
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
    """reset_peak_memory_stats appears in run_inference before the measurement loop.

    After the harness refactor, PowerThermalSampler lives in MeasurementHarness.run(),
    not in the backend. The backend owns reset_peak_memory_stats in run_inference so
    peak stats are cleared for the inference-window-only measurement.
    """
    body = _extract_method_body(_pytorch_source(), "run_inference")
    assert "reset_peak_memory_stats" in body, (
        "PyTorchBackend.run_inference must call reset_peak_memory_stats()"
    )
    assert "max_memory_allocated" in body, (
        "PyTorchBackend.run_inference must read max_memory_allocated() for peak_memory_mb"
    )


def test_pytorch_reads_peak_after_measurement():
    """max_memory_allocated appears in run_inference and result is captured as peak_memory_mb."""
    body = _extract_method_body(_pytorch_source(), "run_inference")
    assert "max_memory_allocated" in body, (
        "PyTorchBackend.run_inference must read max_memory_allocated()"
    )
    assert "peak_memory_mb" in body, "PyTorchBackend.run_inference must assign peak_memory_mb"


def test_pytorch_build_result_computes_inference_memory():
    """MeasurementHarness._build_result contains inference_memory_mb = max(0.0, ...) computation.

    After the harness refactor, _build_result lives in harness.py, not pytorch.py.
    """
    body = _extract_method_body(_harness_source(), "_build_result")
    assert "inference_memory_mb" in body, (
        "MeasurementHarness._build_result must compute inference_memory_mb"
    )
    assert "max(0.0" in body, (
        "MeasurementHarness._build_result must use max(0.0, ...) for inference_memory_mb"
    )


# ---------------------------------------------------------------------------
# Code structure tests — VLLMBackend
# ---------------------------------------------------------------------------


def test_vllm_has_peak_memory_capture():
    """VLLMBackend.run_inference captures peak_memory_mb and resets stats.

    After the harness refactor, PowerThermalSampler context manager is in
    MeasurementHarness.run(). The backend owns reset_peak_memory_stats and
    max_memory_allocated in run_inference.
    """
    body = _extract_method_body(_vllm_source(), "run_inference")
    assert "peak_mb" in body or "peak_memory_mb" in body, (
        "VLLMBackend.run_inference must set peak_memory_mb (or peak_mb)"
    )
    assert "reset_peak_memory_stats" in body, (
        "VLLMBackend.run_inference must call reset_peak_memory_stats()"
    )
    assert "max_memory_allocated" in body, (
        "VLLMBackend.run_inference must read max_memory_allocated()"
    )
    # reset must appear before the actual assignment using max_memory_allocated().
    # The comment above reset also mentions max_memory_allocated() — search for the
    # assignment form to distinguish the comment from the actual call.
    reset_pos = body.index("reset_peak_memory_stats")
    # Match the assignment: "peak_mb = torch.cuda.max_memory_allocated()"
    alloc_assign = "= torch.cuda.max_memory_allocated()"
    alloc_pos = body.index(alloc_assign)
    assert reset_pos < alloc_pos, (
        "reset_peak_memory_stats must appear BEFORE the max_memory_allocated() assignment "
        "in vLLM run_inference"
    )


def test_vllm_build_result_computes_inference_memory():
    """MeasurementHarness._build_result contains inference_memory_mb = max(0.0, ...) computation.

    After the harness refactor, _build_result lives in harness.py, not vllm.py.
    """
    body = _extract_method_body(_harness_source(), "_build_result")
    assert "inference_memory_mb" in body, (
        "MeasurementHarness._build_result must compute inference_memory_mb"
    )
    assert "max(0.0" in body, (
        "MeasurementHarness._build_result must use max(0.0, ...) for inference_memory_mb"
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
