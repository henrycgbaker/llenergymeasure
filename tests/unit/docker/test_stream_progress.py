"""Unit tests for entrypoints/container.py — StreamProgressCallback."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

from llenergymeasure.domain.progress import ProgressCallback
from llenergymeasure.entrypoints.container import StreamProgressCallback


def test_stream_callback_satisfies_protocol():
    """StreamProgressCallback satisfies the ProgressCallback protocol."""
    assert isinstance(StreamProgressCallback(), ProgressCallback)


def test_stream_callback_step_start_writes_json():
    """on_step_start writes a JSON line to stdout."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_step_start("model", "Loading model", "gpt2")

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "step_start"
    assert event["step"] == "model"
    assert event["description"] == "Loading model"
    assert event["detail"] == "gpt2"


def test_stream_callback_step_update_writes_json():
    """on_step_update writes a JSON line to stdout."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_step_update("warmup", "25/50 prompts")

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "step_update"
    assert event["step"] == "warmup"
    assert event["detail"] == "25/50 prompts"


def test_stream_callback_step_done_writes_json():
    """on_step_done writes a JSON line with elapsed time."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_step_done("model", 42.3)

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "step_done"
    assert event["step"] == "model"
    assert event["elapsed_sec"] == 42.3
