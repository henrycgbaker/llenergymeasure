"""Tests for the runtime observation capture wrapper."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.runtime_observations import (
    capture_runtime_observations,
    default_cache_path,
    engine_logger_name,
)


def _mk_config() -> ExperimentConfig:
    return ExperimentConfig(task={"model": "gpt2"}, engine="transformers")


class TestEngineLoggerName:
    def test_transformers_maps_to_transformers(self):
        assert engine_logger_name("transformers") == "transformers"

    def test_vllm_maps_to_vllm(self):
        assert engine_logger_name("vllm") == "vllm"

    def test_tensorrt_maps_to_tensorrt_llm(self):
        assert engine_logger_name("tensorrt") == "tensorrt_llm"

    def test_unknown_engine_returns_as_is(self):
        assert engine_logger_name("custom_engine") == "custom_engine"


class TestDefaultCachePath:
    def test_respects_xdg_cache_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert default_cache_path() == tmp_path / "llenergymeasure" / "runtime_observations.jsonl"

    def test_falls_back_to_home_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        path = default_cache_path()
        assert path.parts[-2:] == ("llenergymeasure", "runtime_observations.jsonl")


class TestCaptureRuntimeObservations:
    def test_appends_success_record(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        with capture_runtime_observations(cfg, cache_path=cache):
            pass
        lines = cache.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["outcome"] == "success"
        assert record["engine"] == "transformers"
        assert record["warnings"] == []
        assert record["logger_records"] == []
        assert record["exception"] is None
        assert "observed_at" in record
        assert "config_hash" in record

    def test_captures_warnings(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        with capture_runtime_observations(cfg, cache_path=cache):
            warnings.warn("test-warning-message", UserWarning, stacklevel=2)
        record = json.loads(cache.read_text().splitlines()[0])
        assert any("test-warning-message" in w for w in record["warnings"])

    def test_captures_engine_logger_records(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        with capture_runtime_observations(cfg, cache_path=cache):
            logging.getLogger("transformers").warning("fake transformers emission 0.001")
        record = json.loads(cache.read_text().splitlines()[0])
        assert len(record["logger_records"]) == 1
        assert record["logger_records"][0]["message"] == "fake transformers emission 0.001"
        assert record["logger_records"][0]["level"] == "WARNING"
        assert record["logger_records"][0]["name"] == "transformers"

    def test_submodule_loggers_propagate_to_engine_logger(self, tmp_path: Path):
        # PoC-D: engine-level logger catches submodule emissions via propagation.
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        with capture_runtime_observations(cfg, cache_path=cache):
            logging.getLogger("transformers.generation.utils").warning("sub-emission")
        record = json.loads(cache.read_text().splitlines()[0])
        assert any(r["message"] == "sub-emission" for r in record["logger_records"])

    def test_exception_outcome_propagates_and_records(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        with (
            pytest.raises(ValueError, match="boom"),
            capture_runtime_observations(cfg, cache_path=cache),
        ):
            raise ValueError("boom")
        record = json.loads(cache.read_text().splitlines()[0])
        assert record["outcome"] == "exception"
        assert record["exception"]["type"] == "ValueError"
        assert record["exception"]["message"] == "boom"
        assert "Traceback" in record["exception"]["traceback"]

    def test_handler_removed_on_success(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        logger = logging.getLogger("transformers")
        prior_handlers = list(logger.handlers)
        with capture_runtime_observations(cfg, cache_path=cache):
            pass
        assert logger.handlers == prior_handlers

    def test_handler_removed_on_exception(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        logger = logging.getLogger("transformers")
        prior_handlers = list(logger.handlers)
        with (
            pytest.raises(RuntimeError),
            capture_runtime_observations(cfg, cache_path=cache),
        ):
            raise RuntimeError("boom")
        assert logger.handlers == prior_handlers

    def test_cache_write_failure_does_not_raise(self, tmp_path: Path, monkeypatch):
        # Simulate a non-writable cache path — write should log and swallow.
        cfg = _mk_config()
        bad_path = tmp_path / "nonexistent" / "cache.jsonl"

        def _fail(self, *_args, **_kwargs):
            raise OSError("disk-full")

        monkeypatch.setattr(Path, "mkdir", _fail)
        # Must not raise — the caller is told the study completed even if
        # observation persistence failed.
        with capture_runtime_observations(cfg, cache_path=bad_path):
            pass

    def test_multiple_runs_append(self, tmp_path: Path):
        cache = tmp_path / "runtime_observations.jsonl"
        cfg = _mk_config()
        for _ in range(3):
            with capture_runtime_observations(cfg, cache_path=cache):
                pass
        assert len(cache.read_text().splitlines()) == 3
