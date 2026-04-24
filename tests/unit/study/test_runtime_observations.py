"""Tests for runtime-observation capture (PR A)."""

from __future__ import annotations

import json
import logging
import multiprocessing
import uuid
import warnings
from pathlib import Path
from typing import Any

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.study.runtime_observations import (
    RUNTIME_OBSERVATIONS_FILENAME,
    SCHEMA_VERSION,
    TRACEBACK_TRUNC_BYTES,
    capture_runtime_observations,
    engine_logger_name,
    write_sentinel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_config(engine: str = "transformers") -> ExperimentConfig:
    return ExperimentConfig(task={"model": "gpt2"}, engine=engine)


def _run_id() -> str:
    return str(uuid.uuid4())


_REQUIRED_KEYS = {
    "schema_version",
    "observed_at",
    "study_run_id",
    "config_hash",
    "cycle",
    "engine",
    "library_version",
    "outcome",
    "warnings",
    "logger_records",
    "exception",
    "exit_reason",
    "exit_code",
}


def _assert_record_shape(record: dict[str, Any]) -> None:
    assert set(record.keys()) == _REQUIRED_KEYS, f"unexpected schema; got {sorted(record.keys())!r}"
    assert record["schema_version"] == SCHEMA_VERSION
    assert isinstance(record["observed_at"], str) and record["observed_at"].endswith("Z")
    assert isinstance(record["study_run_id"], str)
    assert isinstance(record["config_hash"], str)
    assert isinstance(record["cycle"], int)
    assert record["outcome"] in {"success", "exception", "subprocess_died"}
    assert isinstance(record["warnings"], list)
    assert isinstance(record["logger_records"], list)


def _read_records(study_dir: Path) -> list[dict[str, Any]]:
    path = study_dir / RUNTIME_OBSERVATIONS_FILENAME
    assert path.exists(), f"expected {path} to exist"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# engine_logger_name
# ---------------------------------------------------------------------------


class TestEngineLoggerName:
    def test_transformers(self) -> None:
        assert engine_logger_name("transformers") == "transformers"

    def test_vllm(self) -> None:
        assert engine_logger_name("vllm") == "vllm"

    def test_tensorrt_maps_to_tensorrt_llm(self) -> None:
        assert engine_logger_name("tensorrt") == "tensorrt_llm"

    def test_unknown_engine_falls_through(self) -> None:
        assert engine_logger_name("nonexistent") == "nonexistent"

    def test_accepts_engine_enum(self) -> None:
        from llenergymeasure.config.ssot import Engine

        assert engine_logger_name(Engine.TENSORRT) == "tensorrt_llm"


# ---------------------------------------------------------------------------
# Success path — empty record
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_empty_success_record_is_written(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        run_id = _run_id()
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id=run_id,
            cycle=1,
            config_hash="hash-1",
        ):
            pass

        records = _read_records(tmp_path)
        assert len(records) == 1
        rec = records[0]
        _assert_record_shape(rec)
        assert rec["outcome"] == "success"
        assert rec["engine"] == "transformers"
        assert rec["study_run_id"] == run_id
        assert rec["config_hash"] == "hash-1"
        assert rec["cycle"] == 1
        assert rec["warnings"] == []
        assert rec["logger_records"] == []
        assert rec["exception"] is None
        assert rec["exit_reason"] is None
        assert rec["exit_code"] is None

    def test_multiple_experiments_append(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        for cycle in (1, 2, 3):
            with capture_runtime_observations(
                cfg,
                study_dir=tmp_path,
                study_run_id="run-x",
                cycle=cycle,
                config_hash="h",
            ):
                pass
        records = _read_records(tmp_path)
        assert [r["cycle"] for r in records] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Warnings capture
# ---------------------------------------------------------------------------


class TestWarningCapture:
    def test_warning_is_captured_with_category(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            warnings.warn("temperature=0.001 is too low", UserWarning, stacklevel=2)

        rec = _read_records(tmp_path)[0]
        assert len(rec["warnings"]) == 1
        wrec = rec["warnings"][0]
        assert wrec["category"] == "UserWarning"
        assert "temperature=0.001" in wrec["message"]
        assert wrec["message_template"] == "temperature=<NUM> is too low"
        assert wrec["filename"] and isinstance(wrec["filename"], str)
        assert isinstance(wrec["lineno"], int)

    def test_warning_filters_restored(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        before = list(warnings.filters)
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            warnings.warn("inside", UserWarning, stacklevel=2)
        assert list(warnings.filters) == before


# ---------------------------------------------------------------------------
# Logger capture
# ---------------------------------------------------------------------------


class TestLoggerCapture:
    def test_engine_logger_warning_is_captured(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            logging.getLogger("transformers").warning("clamp value=42")

        rec = _read_records(tmp_path)[0]
        assert len(rec["logger_records"]) == 1
        lrec = rec["logger_records"][0]
        assert lrec["level"] == "WARNING"
        assert lrec["logger"] == "transformers"
        assert lrec["message"] == "clamp value=42"
        assert lrec["message_template"] == "clamp value=<NUM>"
        assert isinstance(lrec["lineno"], int)

    def test_submodule_logger_propagates(self, tmp_path: Path) -> None:
        # Engine-level handler catches submodule emissions via propagation.
        cfg = _mk_config()
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            logging.getLogger("transformers.generation.utils").warning("sub-emit")
        rec = _read_records(tmp_path)[0]
        assert any(r["message"] == "sub-emit" for r in rec["logger_records"])

    def test_does_not_mutate_logger_level(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        engine_logger = logging.getLogger("transformers")
        prior_level = engine_logger.level
        prior_handlers = list(engine_logger.handlers)
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            pass
        assert engine_logger.level == prior_level
        assert engine_logger.handlers == prior_handlers

    def test_handler_removed_on_exception(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        engine_logger = logging.getLogger("transformers")
        prior_handlers = list(engine_logger.handlers)
        with (
            pytest.raises(RuntimeError),
            capture_runtime_observations(
                cfg,
                study_dir=tmp_path,
                study_run_id="r",
                cycle=1,
                config_hash="h",
            ),
        ):
            raise RuntimeError("boom")
        assert engine_logger.handlers == prior_handlers


# ---------------------------------------------------------------------------
# Exception capture
# ---------------------------------------------------------------------------


class TestExceptionCapture:
    def test_exception_is_recorded_and_reraised(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        with (
            pytest.raises(ValueError, match="bang"),
            capture_runtime_observations(
                cfg,
                study_dir=tmp_path,
                study_run_id="r",
                cycle=1,
                config_hash="h",
            ),
        ):
            raise ValueError("bang 42")

        rec = _read_records(tmp_path)[0]
        _assert_record_shape(rec)
        assert rec["outcome"] == "exception"
        exc = rec["exception"]
        assert exc is not None
        assert exc["type"] == "ValueError"
        assert exc["message"] == "bang 42"
        assert exc["message_template"] == "bang <NUM>"
        assert "Traceback" in exc["traceback_truncated"]
        assert "ValueError: bang 42" in exc["traceback_truncated"]

    def test_traceback_truncated_to_cap(self, tmp_path: Path) -> None:
        cfg = _mk_config()

        # Build a recursion that blows a huge traceback, trapped by RecursionError.
        def deep(n: int) -> int:
            return deep(n + 1)

        with (
            pytest.raises(RecursionError),
            capture_runtime_observations(
                cfg,
                study_dir=tmp_path,
                study_run_id="r",
                cycle=1,
                config_hash="h",
            ),
        ):
            deep(0)

        rec = _read_records(tmp_path)[0]
        tb = rec["exception"]["traceback_truncated"]
        assert len(tb.encode("utf-8")) <= TRACEBACK_TRUNC_BYTES


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


class TestSentinel:
    def test_write_sentinel_appends_subprocess_died(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        write_sentinel(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=2,
            config_hash="h",
            exit_reason="SIGKILL",
            exit_code=-9,
        )
        rec = _read_records(tmp_path)[0]
        _assert_record_shape(rec)
        assert rec["outcome"] == "subprocess_died"
        assert rec["exit_reason"] == "SIGKILL"
        assert rec["exit_code"] == -9
        assert rec["cycle"] == 2

    def test_sentinel_appends_alongside_success(self, tmp_path: Path) -> None:
        cfg = _mk_config()
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            pass
        write_sentinel(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=2,
            config_hash="h",
            exit_reason="timeout",
            exit_code=None,
        )
        records = _read_records(tmp_path)
        assert [r["outcome"] for r in records] == ["success", "subprocess_died"]

    def test_sentinel_swallows_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _mk_config()

        def _boom(*_a, **_k):
            raise OSError("disk-full")

        monkeypatch.setattr("llenergymeasure.study.runtime_observations.os.open", _boom)
        # Must not raise — feedback loop never fails a study.
        write_sentinel(
            cfg,
            study_dir=tmp_path,
            study_run_id="r",
            cycle=1,
            config_hash="h",
            exit_reason="SIGKILL",
            exit_code=-9,
        )


# ---------------------------------------------------------------------------
# Cache-write resilience
# ---------------------------------------------------------------------------


class TestCacheWriteResilience:
    def test_mkdir_failure_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _mk_config()

        def _fail_mkdir(self, *_a, **_k):
            raise OSError("no perm")

        monkeypatch.setattr(Path, "mkdir", _fail_mkdir)
        # Must not raise even though we can't write the record.
        with capture_runtime_observations(
            cfg,
            study_dir=tmp_path / "subdir",
            study_run_id="r",
            cycle=1,
            config_hash="h",
        ):
            pass


# ---------------------------------------------------------------------------
# Integration: real spawn subprocess
# ---------------------------------------------------------------------------


def _spawn_target(study_dir: str, study_run_id: str, cycle: int, config_hash: str) -> None:
    """Run inside a spawned child — imports everything from scratch.

    Emits one warning and one log record through each channel, then exits.
    """
    import logging as _logging
    import warnings as _warnings

    from llenergymeasure.config.models import ExperimentConfig as _Cfg
    from llenergymeasure.study.runtime_observations import (
        capture_runtime_observations as _cap,
    )

    cfg = _Cfg(task={"model": "gpt2"}, engine="transformers")
    with _cap(
        cfg,
        study_dir=study_dir,
        study_run_id=study_run_id,
        cycle=cycle,
        config_hash=config_hash,
    ):
        _warnings.warn("spawned-warning 1e-5", UserWarning, stacklevel=2)
        _logging.getLogger("transformers").warning("spawned-log 42")


class TestSpawnIntegration:
    def test_spawn_worker_writes_one_record(self, tmp_path: Path) -> None:
        ctx = multiprocessing.get_context("spawn")
        run_id = _run_id()
        p = ctx.Process(
            target=_spawn_target,
            args=(str(tmp_path), run_id, 7, "hash-spawn"),
        )
        p.start()
        p.join(timeout=30)
        assert p.exitcode == 0, f"spawn worker failed with exit={p.exitcode}"

        records = _read_records(tmp_path)
        assert len(records) == 1
        rec = records[0]
        _assert_record_shape(rec)
        assert rec["outcome"] == "success"
        assert rec["study_run_id"] == run_id
        assert rec["cycle"] == 7
        assert rec["config_hash"] == "hash-spawn"
        assert any("spawned-warning" in w["message"] for w in rec["warnings"])
        assert any(
            "spawned-log" in r["message"] and r["logger"] == "transformers"
            for r in rec["logger_records"]
        )
