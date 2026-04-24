"""``llem report-gaps`` — propose corpus rules from captured runtime observations.

Thin Typer command that delegates everything interesting to
:mod:`llenergymeasure.api.report_gaps`. CLI-boundary contract: imports only
from ``llenergymeasure.api``; never reaches into ``study``/``harness``/etc.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from llenergymeasure.api import (
    ReportGapsError,
    find_runtime_gaps,
    render_yaml_fragment,
)

__all__ = ["report_gaps_cmd"]


def report_gaps_cmd(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Feedback source to scan. Only 'runtime-warnings' is wired in this release.",
        ),
    ] = "runtime-warnings",
    study_dir: Annotated[
        list[Path] | None,
        typer.Option(
            "--study-dir",
            help="Study directory to scan. Repeat the flag to pass multiple.",
        ),
    ] = None,
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            help="Filter: only propose rules for this engine (transformers/vllm/tensorrt).",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Output path for proposed YAML fragments (one YAML document per gap, separated by '---').",
        ),
    ] = None,
    include_exceptions: Annotated[
        bool,
        typer.Option(
            "--include-exceptions",
            help=(
                "Also propose rules from runtime exceptions. "
                "Disabled by default; exceptions ship through a different review path."
            ),
        ),
    ] = False,
) -> None:
    """Propose rules corpus entries from captured runtime observations."""
    if source != "runtime-warnings":
        raise typer.BadParameter(
            f"Unsupported source {source!r}. Only 'runtime-warnings' is wired in this release.",
            param_hint="--source",
        )
    if not study_dir:
        raise typer.BadParameter(
            "At least one --study-dir is required.",
            param_hint="--study-dir",
        )
    if out is None:
        raise typer.BadParameter(
            "--out PATH is required (YAML fragment output path).",
            param_hint="--out",
        )

    try:
        proposals = find_runtime_gaps(
            study_dirs=list(study_dir),
            engine=engine,
            include_exceptions=include_exceptions,
        )
    except ReportGapsError as exc:
        print(f"llem report-gaps: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from None

    if not proposals:
        print("No gaps found in the scanned study dirs.")
        return

    fragments = [render_yaml_fragment(p) for p in proposals]
    body = "\n---\n".join(fragments)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")

    print(f"{len(proposals)} gap(s) found. Wrote to {out}.")
