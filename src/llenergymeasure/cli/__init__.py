"""Command-line interface for llem."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

# Load `.env` from the user's CWD before any other imports read env vars
# (e.g. logging setup reads LLEM_LOG_LEVEL). Anchoring to Path.cwd() rather
# than dotenv's default walk-up-from-__file__ is deliberate: once llem is
# pip-installed the module sits under site-packages and the walk-up never
# reaches the user's project directory. override=False so shell env wins.
load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

import typer  # noqa: E402

from llenergymeasure._version import __version__  # noqa: E402
from llenergymeasure.config.ssot import ENV_LOG_LEVEL  # noqa: E402

app = typer.Typer(
    name="llem",
    help="LLM inference efficiency measurement framework",
    add_completion=False,
)


class _ShortNameFilter(logging.Filter):
    """Inject ``short_name`` into log records (last 2 dot-separated parts of name)."""

    def filter(self, record: logging.LogRecord) -> bool:
        parts = record.name.rsplit(".", 2)
        record.short_name = ".".join(parts[-2:]) if len(parts) >= 2 else record.name  # type: ignore[attr-defined]
        return True


def _setup_logging(verbose: int = 0) -> None:
    """Configure the ``llenergymeasure`` logger hierarchy.

    Precedence: CLI flag > ``LLEM_LOG_LEVEL`` env var > default (WARNING).

    Args:
        verbose: Verbosity count from CLI ``-v`` flag.
            0 = WARNING, 1 = INFO, 2+ = DEBUG.
    """
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        env_level = os.environ.get(ENV_LOG_LEVEL, "").upper()
        level = getattr(logging, env_level, logging.WARNING)

    root_logger = logging.getLogger("llenergymeasure")
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(short_name)s:%(lineno)d] %(message)s"
        )
        handler.setFormatter(fmt)
        handler.addFilter(_ShortNameFilter())
        root_logger.addHandler(handler)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        print(f"llem v{__version__}")
        raise typer.Exit()


@app.callback()  # type: ignore[misc]
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """LLM inference efficiency measurement framework."""


# Register commands — deferred imports inside command functions keep startup fast
from llenergymeasure.cli.run import run as _run_cmd  # noqa: E402

app.command(name="run", help="Run an LLM efficiency experiment")(_run_cmd)

from llenergymeasure.cli.config_cmd import config_command as _config_cmd  # noqa: E402

app.command(name="config", help="Show environment and configuration status")(_config_cmd)

from llenergymeasure.cli.doctor_cmd import doctor_command as _doctor_cmd  # noqa: E402

app.command(
    name="doctor",
    help="Verify Docker images match the host ExperimentConfig schema",
)(_doctor_cmd)

from llenergymeasure.cli.report_gaps import report_gaps_command as _report_gaps_cmd  # noqa: E402

app.command(
    name="report-gaps",
    help="Detect canonicaliser gaps and library drift from past runs",
)(_report_gaps_cmd)

__all__ = ["app"]

if __name__ == "__main__":
    app()
