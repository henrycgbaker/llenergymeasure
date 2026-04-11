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


def test_stream_callback_substep_start_writes_json():
    """on_substep_start writes a ``substep_start`` JSON line so the host
    DockerRunner can forward it to the CLI's live heartbeat renderer."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_substep_start("baseline", "launching separate vllm baseline container")

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "substep_start"
    assert event["step"] == "baseline"
    assert event["text"] == "launching separate vllm baseline container"


def test_stream_callback_substep_done_writes_json_with_optional_fields():
    """on_substep_done emits ``substep_done`` with ``text`` / ``elapsed_sec``
    both optional — the host side reuses the start's text + computed elapsed
    when either is missing (None-safe over the wire)."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_substep_done("baseline", "vllm baseline container ready", 2.5)

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "substep_done"
    assert event["step"] == "baseline"
    assert event["text"] == "vllm baseline container ready"
    assert event["elapsed_sec"] == 2.5


def test_stream_callback_substep_done_null_fields_serialise():
    """on_substep_done with no overrides must round-trip through JSON (None
    values stay as nulls — consumer decodes them as missing)."""
    cb = StreamProgressCallback()
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        cb.on_substep_done("baseline")

    line = mock_stdout.getvalue().strip()
    event = json.loads(line)
    assert event["event"] == "substep_done"
    assert event["step"] == "baseline"
    assert event["text"] is None
    assert event["elapsed_sec"] is None
