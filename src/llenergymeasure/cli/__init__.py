"""Command-line interface for llem."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import typer

from llenergymeasure._version import __version__

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
        env_level = os.environ.get("LLEM_LOG_LEVEL", "").upper()
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

__all__ = ["app"]

if __name__ == "__main__":
    app()
