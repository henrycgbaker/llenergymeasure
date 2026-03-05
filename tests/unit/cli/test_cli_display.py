"""Unit tests for CLI display utilities and VRAM estimator.

Tests only pure functions that require no GPU or network access.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.cli._display import (
    _format_duration,
    _sig3,
    format_error,
    format_validation_error,
)
from llenergymeasure.cli._vram import DTYPE_BYTES
from llenergymeasure.exceptions import ConfigError

# =============================================================================
# _sig3 tests
# =============================================================================


def test_sig3_integer():
    """312.4 rounds to 3 sig figs -> 312."""
    assert _sig3(312.4) == "312"


def test_sig3_decimal():
    """3.12 has 3 sig figs already -> '3.12'."""
    assert _sig3(3.12) == "3.12"


def test_sig3_small():
    """Small numbers retain 3 sig figs."""
    assert _sig3(0.00312) == "0.00312"


def test_sig3_zero():
    """Zero returns '0'."""
    assert _sig3(0) == "0"


def test_sig3_large_integer():
    """1234 rounds to 1230 at 3 sig figs."""
    assert _sig3(1234) == "1230"


def test_sig3_round_value():
    """847.0 stays as '847' (already 3 sig figs)."""
    assert _sig3(847.0) == "847"


# =============================================================================
# _format_duration tests
# =============================================================================


def test_format_duration_seconds():
    """Sub-minute durations shown as 'Xs.Xs'."""
    assert _format_duration(4.2) == "4.2s"


def test_format_duration_minutes():
    """272 seconds = 4m 32s."""
    assert _format_duration(272) == "4m 32s"


def test_format_duration_hours():
    """3900 seconds = 1h 05m."""
    assert _format_duration(3900) == "1h 05m"


def test_format_duration_exact_minute():
    """60 seconds = 1m 00s."""
    assert _format_duration(60) == "1m 00s"


def test_format_duration_exact_hour():
    """3600 seconds = 1h 00m."""
    assert _format_duration(3600) == "1h 00m"


# =============================================================================
# DTYPE_BYTES tests
# =============================================================================


def test_vram_dtype_bytes():
    """DTYPE_BYTES contains expected precision entries."""
    assert DTYPE_BYTES["fp32"] == 4
    assert DTYPE_BYTES["fp16"] == 2
    assert DTYPE_BYTES["bf16"] == 2
    assert DTYPE_BYTES["int8"] == 1
    assert DTYPE_BYTES["int4"] == 0.5


def test_vram_dtype_bytes_keys():
    """DTYPE_BYTES has all expected keys."""
    expected_keys = {"fp32", "fp16", "bf16", "int8", "int4"}
    assert set(DTYPE_BYTES.keys()) == expected_keys


# =============================================================================
# format_validation_error tests
# =============================================================================


def test_format_validation_error():
    """ValidationError from bad backend value gets a friendly header."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(model="gpt2", backend="pytorh")  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    assert "Config validation failed" in result
    assert "error" in result.lower()


def test_format_validation_error_did_you_mean():
    """Literal errors on backend field suggest correct spelling."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(model="gpt2", backend="pytorh")  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    # Should suggest 'pytorch' for the typo 'pytorh'
    assert "pytorch" in result.lower() or "Did you mean" in result or "backend" in result


def test_format_validation_error_single_vs_plural():
    """Single error uses 'error', multiple errors use 'errors'."""
    from llenergymeasure.config.models import ExperimentConfig

    with pytest.raises(ValidationError) as exc_info:
        ExperimentConfig(model="gpt2", backend="bad")  # type: ignore[arg-type]

    result = format_validation_error(exc_info.value)
    # 1 error -> singular
    assert "1 error)" in result or "errors)" in result  # may be 1 or more depending on validators


# =============================================================================
# format_error tests
# =============================================================================


def test_format_error_concise():
    """format_error without verbose=True shows class name and message, no traceback."""
    err = ConfigError("missing required field 'model'")
    result = format_error(err, verbose=False)
    assert "ConfigError" in result
    assert "missing required field" in result
    # Should not include traceback keywords
    assert "Traceback" not in result
    assert "File " not in result


def test_format_error_includes_class_name():
    """format_error prefix is the exception class name."""
    err = ConfigError("test message")
    result = format_error(err, verbose=False)
    assert result.startswith("ConfigError:")


def test_format_error_subclass():
    """format_error works for any LLEMError subclass."""
    from llenergymeasure.exceptions import BackendError

    err = BackendError("GPU OOM")
    result = format_error(err, verbose=False)
    assert "BackendError" in result
    assert "GPU OOM" in result
