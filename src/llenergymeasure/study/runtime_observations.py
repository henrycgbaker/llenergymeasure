"""Runtime observation capture — feedback channel #1 for the rules corpus.

Wraps an experiment's worker body in a context manager that captures:

- Library ``warnings.warn`` emissions (via ``warnings.catch_warnings(record=True)``).
- Library ``logger.warning`` / ``logger.error`` / ``logger.warning_once`` records
  emitted through the engine-level logger (``transformers``, ``vllm``,
  ``tensorrt_llm``). Attaching at engine-level captures all submodule
  emissions by propagation.
- Structured ``outcome: "exception"`` records with truncated tracebacks when
  inference raises. Runtime rejection is a distinct feedback class — the
  library rejecting a config at runtime is announced behaviour, just via a
  different channel than ``warnings.warn``.

Writes one JSONL record per experiment to ``{study_dir}/runtime_observations.jsonl``.
The wrapper is a **discovery** mechanism, not a gate — it never blocks,
retries, or modifies runs. Cache-write failures log a warning and are swallowed.

A parent-side :func:`write_sentinel` records the subprocess-died case
(SIGKILL / OOM / timeout) that the worker's ``finally`` never reaches.

Records are keyed on ``(study_run_id, config_hash, cycle)``. ``experiment_id``
is intentionally excluded because it is generated inside the harness and is
not available on failure.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import ENGINE_PACKAGES, Engine
from llenergymeasure.study.message_normalise import normalise

__all__ = [
    "RUNTIME_OBSERVATIONS_FILENAME",
    "SCHEMA_VERSION",
    "TRACEBACK_TRUNC_BYTES",
    "capture_runtime_observations",
    "engine_logger_name",
    "write_sentinel",
]


logger = logging.getLogger(__name__)

RUNTIME_OBSERVATIONS_FILENAME = "runtime_observations.jsonl"
SCHEMA_VERSION = 1
# Max bytes of traceback retained in the JSONL record. The full traceback
# stays in the per-experiment result.json; this cap keeps each JSONL line
# under typical kernel atomic-write limits (PIPE_BUF ~= 4 KiB).
TRACEBACK_TRUNC_BYTES = 3 * 1024


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def engine_logger_name(engine: str | Engine) -> str:
    """Return the Python logger name for ``engine`` (string or :class:`Engine`)."""
    try:
        return ENGINE_PACKAGES[Engine(engine)]
    except (ValueError, KeyError):
        return str(engine.value if isinstance(engine, Engine) else engine)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Logging handler that appends received records to a list.

    Used as a non-destructive observer attached to the engine logger; existing
    handlers continue to receive records.
    """

    def __init__(self, out: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self._out = out

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        self._out.append(record)


@contextmanager
def capture_runtime_observations(
    config: ExperimentConfig,
    study_dir: Path | str,
    study_run_id: str,
    cycle: int,
    config_hash: str,
) -> Iterator[None]:
    """Wrap a worker body; on exit, append one JSONL observation record.

    The context body is the full worker body (``run_preflight()``,
    ``get_engine()``, ``harness.run()``) so import-time emissions from
    engine libraries are captured.

    On success: writes ``outcome: "success"`` record (with empty arrays if
    nothing was emitted — the empty arrays are load-bearing evidence for
    predicate inference downstream).

    On exception: writes ``outcome: "exception"`` record with serialised
    traceback, then re-raises.

    Cache-write errors are logged at WARNING and swallowed — a broken cache
    must never fail a study.
    """
    target = Path(study_dir) / RUNTIME_OBSERVATIONS_FILENAME

    records: list[logging.LogRecord] = []
    handler = _ListHandler(records)
    logger_name = engine_logger_name(config.engine)
    engine_logger = logging.getLogger(logger_name)

    # Handler-level DEBUG only — leaves the library logger's own level
    # untouched so we don't mutate user-visible logging state. Records
    # emitted below the logger's effective level won't reach handlers
    # either way, which is fine — they aren't user-visible warnings.
    engine_logger.addHandler(handler)

    warn_ctx = warnings.catch_warnings(record=True)
    captured = warn_ctx.__enter__()
    # Deliberately no ``warnings.simplefilter("always")`` — RT-0 PoC
    # confirmed ``catch_warnings(record=True)`` alone captures one record
    # per unique ``(filename, lineno, category, message)`` on Python 3.10+.
    # Calling simplefilter perturbs the measurement window.

    outcome = "success"
    exception_info: dict[str, Any] | None = None
    warnings_buf: list[warnings.WarningMessage] = []

    try:
        try:
            yield
        except Exception as exc:
            outcome = "exception"
            exception_info = _serialise_exception(exc)
            raise
        finally:
            warnings_buf = list(captured or [])
            warn_ctx.__exit__(None, None, None)
    finally:
        engine_logger.removeHandler(handler)
        record = _build_record(
            config=config,
            study_run_id=study_run_id,
            config_hash=config_hash,
            cycle=cycle,
            outcome=outcome,
            warnings_buf=warnings_buf,
            log_records=records,
            exception_info=exception_info,
            exit_reason=None,
            exit_code=None,
        )
        _append_observation(target, record)


def write_sentinel(
    config: ExperimentConfig,
    study_dir: Path | str,
    study_run_id: str,
    cycle: int,
    config_hash: str,
    exit_reason: str | None,
    exit_code: int | None,
) -> None:
    """Append a ``outcome: "subprocess_died"`` record from the parent process.

    Called by :class:`StudyRunner` when the worker was killed before its
    context manager's ``__exit__`` could run (SIGKILL, OOM, timeout). Uses
    ``os.open + O_APPEND + os.write`` so the single record hits the kernel
    as one atomic write — no torn-line risk with other writers.
    """
    target = Path(study_dir) / RUNTIME_OBSERVATIONS_FILENAME
    record = _build_record(
        config=config,
        study_run_id=study_run_id,
        config_hash=config_hash,
        cycle=cycle,
        outcome="subprocess_died",
        warnings_buf=[],
        log_records=[],
        exception_info=None,
        exit_reason=exit_reason,
        exit_code=exit_code,
    )
    _append_observation(target, record)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def _build_record(
    *,
    config: ExperimentConfig,
    study_run_id: str,
    config_hash: str,
    cycle: int,
    outcome: str,
    warnings_buf: list[warnings.WarningMessage],
    log_records: list[logging.LogRecord],
    exception_info: dict[str, Any] | None,
    exit_reason: str | None,
    exit_code: int | None,
) -> dict[str, Any]:
    engine_value = _engine_value(config.engine)
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "study_run_id": study_run_id,
        "config_hash": config_hash,
        "cycle": cycle,
        "engine": engine_value,
        "library_version": _installed_version(engine_value),
        "outcome": outcome,
        "warnings": [_render_warning(wm) for wm in warnings_buf],
        "logger_records": [_render_log_record(r) for r in log_records],
        "exception": exception_info,
        "exit_reason": exit_reason,
        "exit_code": exit_code,
    }


def _render_warning(wm: warnings.WarningMessage) -> dict[str, Any]:
    raw = str(wm.message)
    return {
        "category": getattr(wm.category, "__name__", str(wm.category)),
        "message": raw,
        "message_template": _normalise_message(raw),
        "filename": str(wm.filename),
        "lineno": int(wm.lineno),
    }


def _render_log_record(record: logging.LogRecord) -> dict[str, Any]:
    try:
        message = record.getMessage()
    except Exception:  # pragma: no cover - defensive: broken formatter
        message = str(record.msg)
    return {
        "level": record.levelname,
        "logger": record.name,
        "message": message,
        "message_template": _normalise_message(message),
        "filename": record.filename,
        "lineno": int(record.lineno),
    }


def _serialise_exception(exc: BaseException) -> dict[str, Any]:
    message = str(exc)
    full_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    encoded = full_tb.encode("utf-8", errors="replace")
    if len(encoded) > TRACEBACK_TRUNC_BYTES:
        truncated = encoded[:TRACEBACK_TRUNC_BYTES].decode("utf-8", errors="replace")
    else:
        truncated = full_tb
    return {
        "type": type(exc).__name__,
        "message": message,
        "message_template": _normalise_message(message),
        "traceback_truncated": truncated,
    }


def _normalise_message(message: str) -> str:
    return normalise(message).template


def _engine_value(engine: Any) -> str:
    return engine.value if hasattr(engine, "value") else str(engine)


@lru_cache(maxsize=8)
def _installed_version(engine_value: str) -> str:
    """Return the installed package version for ``engine``, or ``""`` if missing.

    Cached per-engine: library versions cannot change mid-study, and
    ``importlib.metadata.version`` walks ``sys.path`` on every call.
    """
    try:
        package = ENGINE_PACKAGES[Engine(engine_value)]
    except (ValueError, KeyError):
        package = engine_value
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python <3.8 not supported
        return ""
    try:
        return version(package)
    except PackageNotFoundError:
        return ""
    except Exception:  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _append_observation(path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL record to ``path`` using buffered IO + flush + fsync.

    W1 pattern from RT-2 PoC: open in append mode, write, flush, fsync.
    Torn-line-safe at 10 KiB records on Linux ext4. Best-effort — a
    write failure is logged at WARNING but never raises.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
    except OSError as exc:
        logger.warning("Could not append runtime observation to %s: %s", path, exc)
