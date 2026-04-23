"""CLI subcommand ``llem report-gaps`` — feedback-loop gap reporter.

Thin wrapper around :func:`llenergymeasure.api.report_gaps.run_report_gaps`.
The CLI surface exposes:

- ``--source {runtime-warnings, h3-collisions, both}`` — default ``both``
- ``--engine {transformers, vllm, tensorrt}`` — optional filter
- ``--results-dir PATH`` — root under which to scan sidecars
- ``--cache-path PATH`` — override default ``runtime_observations.jsonl`` path
- ``--out PATH`` — write candidate YAML to file
- ``--open-pr`` — open a draft PR via ``gh`` (never ``--ready-for-review``)
- ``--dry-run`` — print what WOULD be done without writing any files or
  invoking ``gh``
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

__all__ = ["report_gaps_command"]


_SOURCES = ("runtime-warnings", "h3-collisions", "both")
_ENGINES = ("transformers", "vllm", "tensorrt")


def report_gaps_command(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Which feedback channel to scan",
            case_sensitive=False,
        ),
    ] = "both",
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            help="Filter candidates to a single engine",
            case_sensitive=False,
        ),
    ] = None,
    results_dir: Annotated[
        Path | None,
        typer.Option(
            "--results-dir",
            exists=False,
            help="Directory containing result-bundle subdirs to scan (defaults to ./results)",
        ),
    ] = None,
    cache_path: Annotated[
        Path | None,
        typer.Option(
            "--cache-path",
            help=(
                "Runtime-observations JSONL cache to read (defaults to "
                "$XDG_CACHE_HOME/llenergymeasure/runtime_observations.jsonl)"
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write proposed candidate YAML to this path"),
    ] = None,
    open_pr: Annotated[
        bool,
        typer.Option("--open-pr", help="Create a DRAFT PR via `gh pr create`"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print what would be done without writing files or invoking `gh`",
        ),
    ] = False,
) -> None:
    """Scan for canonicaliser gaps and announced library behaviour not in the corpus.

    Returns candidate rule YAML. Default mode prints a summary to stdout;
    ``--out`` writes a YAML file; ``--open-pr`` additionally creates a draft
    PR via ``gh`` with the candidate block appended to the corpus.
    """
    normalised_source = source.lower()
    if normalised_source not in _SOURCES:
        raise typer.BadParameter(
            f"--source must be one of {_SOURCES}; got {source!r}", param_hint="--source"
        )
    if engine is not None and engine.lower() not in _ENGINES:
        raise typer.BadParameter(
            f"--engine must be one of {_ENGINES}; got {engine!r}", param_hint="--engine"
        )

    from llenergymeasure.api.report_gaps import (
        GhNotFoundError,
        RuleCandidate,
        open_draft_pr,
        plan_request,
        render_candidates_yaml,
        run_report_gaps,
    )

    resolved_results = results_dir if results_dir is not None else Path("results")
    report = run_report_gaps(
        source=normalised_source,
        results_dir=resolved_results if normalised_source != "runtime-warnings" else None,
        cache_path=cache_path,
        engine=engine.lower() if engine else None,
    )

    _print_summary(report)

    if out is not None:
        if dry_run:
            typer.echo(f"[dry-run] Would write {len(report.candidates)} candidate(s) to {out}")
        else:
            yaml_body = render_candidates_yaml(report.candidates)
            out.write_text(yaml_body)
            typer.echo(f"Wrote {len(report.candidates)} candidate(s) to {out}")

    if open_pr:
        if dry_run:
            typer.echo("[dry-run] Would open a draft PR via `gh pr create --draft`")
            return
        if not report.candidates:
            typer.echo("No candidates to propose — skipping --open-pr")
            return
        unverified = [c for c in report.candidates if not c.verified]
        if unverified and normalised_source != "runtime-warnings":
            typer.echo(
                f"Refusing --open-pr: {len(unverified)} UNVERIFIED candidate(s) from a "
                "dedup_mode=off study. Re-run with dedup ON or use --out to dump YAML."
            )
            raise typer.Exit(code=2)

        # Group candidates by engine — one draft PR per engine.
        per_engine: dict[str, list[RuleCandidate]] = {}
        for c in report.candidates:
            per_engine.setdefault(c.engine, []).append(c)
        corpus_root = _find_corpus_root()
        try:
            for eng, candidates in per_engine.items():
                req = plan_request(eng, candidates, corpus_root)
                url = open_draft_pr(req)
                typer.echo(f"Opened draft PR for {eng}: {url}")
        except GhNotFoundError as exc:
            typer.echo(f"gh unavailable: {exc}")
            raise typer.Exit(code=3) from exc


def _print_summary(report) -> None:  # type: ignore[no-untyped-def]
    """Render a short summary to stdout, shaped per PLAN Q2 (2026-04-23 amendment)."""
    counts = report.summary_counts
    total = len(report.candidates)
    typer.echo(
        f"Scanned: {report.scanned_sidecars} sidecar(s), "
        f"{report.scanned_observations} runtime observation(s)"
    )
    h3_gap_count = sum(1 for g in report.h3_groups if g.gap_detected)
    typer.echo(
        f"H3 collision groups: {len(report.h3_groups)} total, {h3_gap_count} flagged as gaps"
    )
    typer.echo(
        f"Candidates: {total} ({counts.get('high', 0)} high, "
        f"{counts.get('medium', 0)} medium, {counts.get('low', 0)} low)"
    )
    if report.dedup_off_studies:
        typer.echo(
            f"[!] {report.dedup_off_studies} candidate(s) come from dedup_mode=off studies — "
            "flagged UNVERIFIED; --open-pr will refuse."
        )
    for c in report.candidates:
        flag = "" if c.verified else " [UNVERIFIED]"
        typer.echo(f"  - {c.candidate_id} ({c.confidence}, {c.source}){flag}")


def _find_corpus_root() -> Path:
    """Locate ``configs/validation_rules/`` relative to the installed package.

    Falls back to the CWD's ``configs/validation_rules/`` so invocations from
    a clone of the repo work even when the installed package points to an
    editable install in a sibling directory.
    """
    # Editable-install path — walks up from this module.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "configs" / "validation_rules"
        if candidate.exists():
            return candidate
    return Path.cwd() / "configs" / "validation_rules"
