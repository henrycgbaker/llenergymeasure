"""Unit tests for GPU memory residual check.

All tests mock pynvml — no GPU hardware required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pynvml_mock(used_bytes: int, total_bytes: int = 80 * 1024 * 1024 * 1024) -> MagicMock:
    """Return a mock pynvml module with a configurable memory response."""
    pynvml = MagicMock()
    mem_info = MagicMock()
    mem_info.used = used_bytes
    mem_info.total = total_bytes
    pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    return pynvml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_state_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """500 MB used (below 1024 MB default threshold) → no warning logged."""
    pynvml_mock = _make_pynvml_mock(used_bytes=500 * 1024 * 1024)

    with (
        patch.dict(sys.modules, {"pynvml": pynvml_mock}),
        caplog.at_level("WARNING", logger="llenergymeasure.study.gpu_memory"),
    ):
        check_gpu_memory_residual()

    assert not caplog.records, f"Expected no log records, got: {caplog.records}"


def test_residual_memory_warning(caplog: pytest.LogCaptureFixture) -> None:
    """2 GB used (above 1024 MB default threshold) → warning with device and MB info."""
    pynvml_mock = _make_pynvml_mock(used_bytes=2048 * 1024 * 1024)

    with (
        patch.dict(sys.modules, {"pynvml": pynvml_mock}),
        caplog.at_level("WARNING", logger="llenergymeasure.study.gpu_memory"),
    ):
        check_gpu_memory_residual()

    assert len(caplog.records) == 1
    msg = caplog.records[0].message
    assert "Residual GPU memory detected" in msg
    assert "2048" in msg


def test_custom_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """200 MB used: warns at threshold=150 MB but not at threshold=250 MB."""
    pynvml_mock = _make_pynvml_mock(used_bytes=200 * 1024 * 1024)

    with (
        patch.dict(sys.modules, {"pynvml": pynvml_mock}),
        caplog.at_level("WARNING", logger="llenergymeasure.study.gpu_memory"),
    ):
        # Below 250 MB threshold → no warning
        check_gpu_memory_residual(threshold_mb=250.0)
    assert not caplog.records, "Expected no warning at threshold=250 MB"

    caplog.clear()

    with (
        patch.dict(sys.modules, {"pynvml": pynvml_mock}),
        caplog.at_level("WARNING", logger="llenergymeasure.study.gpu_memory"),
    ):
        # Above 150 MB threshold → warning
        check_gpu_memory_residual(threshold_mb=150.0)
    assert len(caplog.records) == 1
    assert "Residual GPU memory detected" in caplog.records[0].message


def test_pynvml_not_available(caplog: pytest.LogCaptureFixture) -> None:
    """pynvml unavailable → function returns without error, debug log only."""
    with (
        patch.dict(sys.modules, {"pynvml": None}),
        caplog.at_level("DEBUG", logger="llenergymeasure.study.gpu_memory"),
    ):
        check_gpu_memory_residual()  # must not raise

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, "Expected no warning when pynvml is unavailable"

    debug_msgs = [r.message for r in caplog.records if r.levelname == "DEBUG"]
    assert any("pynvml" in m.lower() or "not available" in m.lower() for m in debug_msgs), (
        f"Expected debug log mentioning pynvml unavailability, got: {debug_msgs}"
    )


def test_nvml_error_graceful(caplog: pytest.LogCaptureFixture) -> None:
    """NVMLError during device query → function returns without error, no warning."""
    pynvml_mock = MagicMock()

    # Raise a simulated NVMLError on handle lookup
    nvml_error = Exception("NVMLError: driver not loaded")
    pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = nvml_error

    with (
        patch.dict(sys.modules, {"pynvml": pynvml_mock}),
        caplog.at_level("WARNING", logger="llenergymeasure.study.gpu_memory"),
    ):
        check_gpu_memory_residual()  # must not raise

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings, "Expected no warning on NVMLError"


def test_nvml_shutdown_called() -> None:
    """nvmlShutdown is always called once (cleanup guarantee)."""
    pynvml_mock = _make_pynvml_mock(used_bytes=50 * 1024 * 1024)

    with patch.dict(sys.modules, {"pynvml": pynvml_mock}):
        check_gpu_memory_residual()

    pynvml_mock.nvmlShutdown.assert_called_once()
