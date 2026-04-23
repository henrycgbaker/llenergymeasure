"""Runtime observation capture — feedback channel #1 for the rules corpus.

Design: ``.product/designs/config-deduplication-dormancy/runtime-config-validation.md`` §4.7.

Wraps an experiment's inference call in a context manager that captures:

- Library ``warnings.warn`` emissions (via ``warnings.catch_warnings(record=True)``).
- Library ``logger.warning`` / ``logger.error`` / ``logger.warning_once`` records
  emitted through the engine-level logger (``transformers``, ``vllm``,
  ``tensorrt_llm``). PoC-D (2026-04-23) confirmed engine-level capture is the
  correct strategy after the root-logger approach failed on vLLM's
  ``sampling_params.py`` emissions.
- Structured ``outcome: "exception"`` records with serialised tracebacks when
  inference raises. Runtime exceptions are a distinct feedback class — library
  rejecting the config at runtime is announced behaviour, just announced via a
  different channel than ``warnings.warn``.

Writes one JSON line per experiment to the user-visible cache path
(``~/.cache/llenergymeasure/runtime_observations.jsonl`` by default). The
wrapper is a **discovery** mechanism, not a gate — it never blocks, retries,
or modifies runs. Cache failures log a warning and are swallowed.

The consumer (``llem report-gaps --source runtime-warnings``) scans this cache,
classifies messages against the vendored rules corpus, and proposes candidate
YAML entries for emissions not already covered.
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
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import ENGINE_PACKAGES, Engine
from llenergymeasure.domain.experiment import compute_measurement_config_hash

logger = logging.getLogger(__name__)


__all__ = [
    "capture_runtime_observations",
    "default_cache_path",
    "engine_logger_name",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def engine_logger_name(engine: str | Engine) -> str:
    """Return the Python logger name for ``engine``.

    Maps the ``Engine`` enum onto the library's top-level Python package name
    (e.g. ``Engine.TENSORRT`` → ``tensorrt_llm``). Attaching a handler at this
    level captures all module-level ``logger.warning`` / ``logger.error`` emits
    from inside the library during inference.
    """
    if isinstance(engine, Engine):
        return ENGINE_PACKAGES[engine]
    try:
        return ENGINE_PACKAGES[Engine(str(engine))]
    except (ValueError, KeyError):
        return str(engine)


def default_cache_path() -> Path:
    """Return the runtime-observations JSONL path, respecting ``XDG_CACHE_HOME``.

    Falls back to ``~/.cache/llenergymeasure/runtime_observations.jsonl``.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / "llenergymeasure" / "runtime_observations.jsonl"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    """Logging handler that appends each received ``LogRecord`` to a list.

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
    cache_path: Path | None = None,
) -> Iterator[None]:
    """Wrap an experiment run; append an observation record to ``cache_path``.

    The context body is the actual inference call. On exit (success or
    exception) a single JSONL record is appended to ``cache_path``:

    .. code-block:: json

        {
          "observed_at": "2026-04-23T12:34:56Z",
          "engine": "transformers",
          "library_version": "4.56.0",
          "config_hash": "...",
          "outcome": "success" | "exception",
          "warnings": ["..."],
          "logger_records": [{"level": "WARNING", "name": "...", "message": "..."}],
          "exception": {"type": "...", "message": "...", "traceback": "..."} | null
        }

    Exceptions propagate — the wrapper never swallows them. Cache-write errors
    are logged at WARNING and swallowed to preserve the "discovery, not gate"
    contract (a broken cache must never fail a study).
    """
    target = cache_path or default_cache_path()

    warnings_buf: list[warnings.WarningMessage] = []
    records: list[logging.LogRecord] = []
    handler = _ListHandler(records)
    logger_name = engine_logger_name(config.engine)
    engine_logger = logging.getLogger(logger_name)

    outcome = "success"
    exception_info: dict[str, Any] | None = None

    prior_level = engine_logger.level
    if prior_level == logging.NOTSET or prior_level > logging.WARNING:
        # Ensure emissions at WARNING+ reach our handler even when the library
        # hasn't set its own level. Restore on exit.
        engine_logger.setLevel(logging.WARNING)
    engine_logger.addHandler(handler)

    warn_ctx = warnings.catch_warnings(record=True)
    captured = warn_ctx.__enter__()
    warnings.simplefilter("always")

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
        engine_logger.setLevel(prior_level)
        _append_observation(
            target,
            _build_record(config, outcome, warnings_buf, records, exception_info),
        )


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def _build_record(
    config: ExperimentConfig,
    outcome: str,
    warnings_buf: list[warnings.WarningMessage],
    records: list[logging.LogRecord],
    exception_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": _engine_value(config.engine),
        "library_version": _installed_version(config.engine),
        "config_hash": compute_measurement_config_hash(config),
        "outcome": outcome,
        "warnings": [str(wm.message) for wm in warnings_buf],
        "logger_records": [_render_record(r) for r in records],
        "exception": exception_info,
    }


def _render_record(record: logging.LogRecord) -> dict[str, Any]:
    try:
        message = record.getMessage()
    except Exception:  # pragma: no cover - defensive: broken formatter
        message = str(record.msg)
    return {
        "level": record.levelname,
        "name": record.name,
        "message": message,
    }


def _serialise_exception(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _engine_value(engine: Any) -> str:
    return engine.value if hasattr(engine, "value") else str(engine)


def _installed_version(engine: str | Engine) -> str:
    """Return the installed library version for ``engine``, or ``""`` if missing."""
    package = engine_logger_name(engine)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python <3.8 not supported
        return ""
    try:
        return version(package)
    except (PackageNotFoundError, Exception):
        return ""


def _append_observation(path: Path, record: dict[str, Any]) -> None:
    """Atomically append one JSONL record to ``path``.

    Best-effort — a cache-write failure is logged at WARNING but never raises.
    This is load-bearing: the wrapper is a discovery mechanism, so a broken
    cache must not abort a study.
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
