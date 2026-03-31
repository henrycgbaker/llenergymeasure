"""Tests for _atomic_write() crash-safety and ExecutionConfig new fields."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import ExecutionConfig
from llenergymeasure.results.persistence import _atomic_write

# ---------------------------------------------------------------------------
# _atomic_write tests
# ---------------------------------------------------------------------------


def test_atomic_write_produces_correct_content(tmp_path: Path) -> None:
    """_atomic_write() writes the expected content to the target path."""
    target = tmp_path / "output.json"
    _atomic_write('{"key": "value"}', target)
    assert target.exists()
    assert target.read_text() == '{"key": "value"}'


def test_atomic_write_calls_fsync(tmp_path: Path) -> None:
    """_atomic_write() calls os.fsync before os.replace()."""
    target = tmp_path / "output.json"

    fsync_called = []
    original_fsync = os.fsync

    def tracking_fsync(fd: int) -> None:
        fsync_called.append(fd)
        original_fsync(fd)

    with patch("llenergymeasure.results.persistence.os.fsync", side_effect=tracking_fsync):
        _atomic_write("content", target)

    assert len(fsync_called) == 1, "os.fsync should be called exactly once"


def test_atomic_write_cleans_up_tmp_on_failure(tmp_path: Path) -> None:
    """_atomic_write() removes the temp file when os.replace() raises."""
    target = tmp_path / "output.json"

    with (
        patch("llenergymeasure.results.persistence.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        _atomic_write("content", target)

    # No leftover .tmp files
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected temp files left: {tmp_files}"


def test_atomic_write_overwrites_existing_file(tmp_path: Path) -> None:
    """_atomic_write() replaces an existing file atomically."""
    target = tmp_path / "output.json"
    target.write_text("old content")

    _atomic_write("new content", target)

    assert target.read_text() == "new content"


# ---------------------------------------------------------------------------
# ExecutionConfig new fields tests
# ---------------------------------------------------------------------------


def test_execution_config_max_consecutive_failures_default() -> None:
    """max_consecutive_failures defaults to 10."""
    config = ExecutionConfig()
    assert config.max_consecutive_failures == 10


def test_execution_config_max_consecutive_failures_disabled() -> None:
    """max_consecutive_failures=0 disables circuit breaker."""
    config = ExecutionConfig(max_consecutive_failures=0)
    assert config.max_consecutive_failures == 0


def test_execution_config_max_consecutive_failures_fail_fast() -> None:
    """max_consecutive_failures=1 is fail-fast mode."""
    config = ExecutionConfig(max_consecutive_failures=1)
    assert config.max_consecutive_failures == 1


def test_execution_config_max_consecutive_failures_invalid() -> None:
    """max_consecutive_failures must be >= 0."""
    with pytest.raises(ValidationError):
        ExecutionConfig(max_consecutive_failures=-1)


def test_execution_config_circuit_breaker_cooldown_default() -> None:
    """circuit_breaker_cooldown_seconds defaults to 60.0."""
    config = ExecutionConfig()
    assert config.circuit_breaker_cooldown_seconds == 60.0


def test_execution_config_circuit_breaker_cooldown_zero() -> None:
    """circuit_breaker_cooldown_seconds=0.0 is valid (no cooldown)."""
    config = ExecutionConfig(circuit_breaker_cooldown_seconds=0.0)
    assert config.circuit_breaker_cooldown_seconds == 0.0


def test_execution_config_circuit_breaker_cooldown_invalid() -> None:
    """circuit_breaker_cooldown_seconds must be >= 0.0."""
    with pytest.raises(ValidationError):
        ExecutionConfig(circuit_breaker_cooldown_seconds=-1.0)


def test_execution_config_wall_clock_timeout_default() -> None:
    """wall_clock_timeout_hours defaults to None (no limit)."""
    config = ExecutionConfig()
    assert config.wall_clock_timeout_hours is None


def test_execution_config_wall_clock_timeout_valid() -> None:
    """wall_clock_timeout_hours=24.0 is valid."""
    config = ExecutionConfig(wall_clock_timeout_hours=24.0)
    assert config.wall_clock_timeout_hours == 24.0


def test_execution_config_wall_clock_timeout_zero_invalid() -> None:
    """wall_clock_timeout_hours=0 raises ValidationError (gt=0.0)."""
    with pytest.raises(ValidationError):
        ExecutionConfig(wall_clock_timeout_hours=0.0)


def test_execution_config_wall_clock_timeout_negative_invalid() -> None:
    """wall_clock_timeout_hours < 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        ExecutionConfig(wall_clock_timeout_hours=-1.0)


def test_execution_config_all_new_fields_together() -> None:
    """All three new fields can be set simultaneously."""
    config = ExecutionConfig(
        max_consecutive_failures=5,
        circuit_breaker_cooldown_seconds=30.0,
        wall_clock_timeout_hours=12.0,
    )
    assert config.max_consecutive_failures == 5
    assert config.circuit_breaker_cooldown_seconds == 30.0
    assert config.wall_clock_timeout_hours == 12.0
