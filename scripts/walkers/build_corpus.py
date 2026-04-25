"""Canonical corpus builder — orchestrate extractors, merge staging, emit corpus.

The validation-rules pipeline is split into independent extractors (introspection,
AST walker, future runtime-warning miner) that each write to a staging file under
``configs/validation_rules/_staging/{engine}_{name}.yaml``. This module is the
single canonical entry point that runs them, merges their outputs into one
:class:`~llenergymeasure.config.vendored_rules.loader.VendoredRules`-shaped
document, and writes ``configs/validation_rules/{engine}.yaml``.

Why a merger at all
-------------------
Today the corpus is single-writer (introspection only). With the AST walker
landing in parallel, we need a deterministic reconciliation step — same rule
discovered by two independent paths becomes evidence of cross-validation, not
two duplicates. Different fingerprints across paths stay as two rules; the
vendor CI pipeline (``scripts/vendor_rules.py``) runs every rule's
``kwargs_positive`` / ``kwargs_negative`` against the real library and fails
divergent rules — the merger optimises for *recall* and lets CI kill false
positives.

Schema choice for cross-source provenance
-----------------------------------------
Two viable shapes for tagging cross-validation:

    A. ``added_by: list[str]`` — change the existing single-string field to a
       list. Touches the loader, every test that pins the literal, and the
       vendor JSON shape.
    B. ``added_by: <single primary string>`` + ``cross_validated_by: list[str]``
       — additive; primary source unchanged, secondary sources surfaced via a
       new optional field.

This module picks **B**. Rationale: the existing ``AddedBy`` Literal and the
``test_corpus_added_by_values_valid`` invariant treat the field as an enum;
flipping it to ``list[str]`` ripples through every test that constructs a
:class:`Rule` directly. ``cross_validated_by`` is strictly additive — old
rules default to an empty tuple, the loader accepts the new field with its
own enum validation, and the merger writes it only when there's a second
source to cite. Loader changes are minimal: one new field on :class:`Rule`,
one parse step in :func:`_parse_rule`.

Fingerprint
-----------
The dedup key is::

    canonical_serialise({
        "engine": rule.engine,
        "severity": rule.severity,
        "match_fields": rule.match["fields"],
    })

via :func:`llenergymeasure.domain.hashing.canonical_serialise` — same primitive
that powers ``resolved_config_hash``. It normalises floats, sorts keys, and
distinguishes ``None`` from missing. Two rules with the same fingerprint are
the same constraint discovered by two independent paths.

Per-field merge precedence
--------------------------
=================================== =================================
Field                               Wins from
=================================== =================================
``match.fields`` predicate          AST walker (more specific operators)
``message_template``                introspection (real library text)
``observed_messages*``              introspection (real captured emissions)
``kwargs_positive`` / ``negative``  AST walker (derived from conditional)
``walker_source.line_at_scan``      AST walker (real source line)
``walker_confidence``               min of all sources
``references``                      union (let reviewer see all evidence)
``id``                              first source's id is canonical
``added_by``                        primary source (priority order)
``cross_validated_by``              all other sources, sorted
=================================== =================================

If precedence is ambiguous (e.g. both sources are introspection-derived but
disagree on a field), the merger keeps the rule from the higher-priority
source and emits a ``conflict_note`` annotation in the YAML for reviewer
attention. CI will kill whichever variant is wrong on the live library.

Vendor validation gate
----------------------
After the merger produces an in-memory candidate list, this module invokes
:func:`scripts.vendor_rules.vendor_engine` against the candidates, captures
the per-rule divergences, and drops any rule whose declared
``expected_outcome`` doesn't match what the live library actually does. The
canonical YAML therefore contains *only* vendor-validated rules — recall-
first extractors are correct; the missing piece historically was that
divergent candidates leaked into the runtime corpus and fired against valid
configs.

Quarantined rules (those that did diverge) are written to
``_staging/_failed_validation_{engine}.yaml`` along with the per-field
divergences, so a reviewer can inspect what the extractor proposed and why
it was rejected. They are not silently dropped.

Validation requires the engine library to be importable in the current
environment. When it isn't (for instance running the merge step on a
machine without ``transformers`` installed), the validator raises
:class:`scripts.vendor_rules.VendorEngineNotImportable`; the merger treats
this as ``validation_skipped``, leaves all candidates in the canonical
corpus, and prints a warning. ``--skip-validation`` opts out of validation
explicitly for fast local iteration.

CLI
---
::

    python scripts/walkers/build_corpus.py --engine transformers
    # Run extractors -> staging -> merge -> validate -> write canonical corpus

    python scripts/walkers/build_corpus.py --engine transformers --check
    # Re-run (with validation); diff against checked-in corpus; exit 1 on drift.

    python scripts/walkers/build_corpus.py --engine transformers --skip-extract
    # Assume staging files already exist; merge + validate + write.

    python scripts/walkers/build_corpus.py --engine transformers --skip-validation
    # Skip the vendor-validation gate (fast local iteration; CI never sets this).

The ``--engine`` flag defaults to ``transformers`` since that's the only
engine wired today; vLLM / TRT-LLM slot in by adding their own staging
extractors and a new entry in :data:`_ENGINE_EXTRACTORS`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root on sys.path so this script works both as ``python -m
# scripts.walkers.build_corpus`` and as a direct ``python scripts/walkers/build_corpus.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# When this file is run directly (``python scripts/walkers/build_corpus.py``),
# Python prepends ``scripts/walkers/`` to sys.path. That directory contains a
# ``transformers.py`` walker module which would shadow the real ``transformers``
# library on import — vendor validation would then crash with
# ``module 'transformers' has no attribute '__version__'``. Drop the script's
# own directory from sys.path so the upstream library wins. ``-m`` invocation
# doesn't trigger this branch (sys.path[0] is the cwd then).
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

# canonical_serialise lives in domain (Layer 0); safe to depend on from a
# tooling script without breaking import-linter contracts.
from llenergymeasure.domain.hashing import canonical_serialise  # noqa: E402


# ---------------------------------------------------------------------------
# Engine extractor registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Extractor:
    """One staging-file producer for an engine.

    ``module`` is invoked as ``python -m {module} --out {staging_path}`` —
    each subagent's extractor must accept a single ``--out`` argument.

    ``staging_basename`` is the filename written under
    ``configs/validation_rules/_staging/``. Convention:
    ``{engine}_{source_name}.yaml`` so the merger's glob pattern stays
    predictable.
    """

    module: str
    staging_basename: str


# Registry of which extractors produce staging files for each engine.
# Add new (engine, [extractors]) entries to wire vLLM / TRT-LLM walkers in.
_ENGINE_EXTRACTORS: dict[str, tuple[_Extractor, ...]] = {
    "transformers": (
        _Extractor(
            module="scripts.walkers.transformers_ast",
            staging_basename="transformers_ast.yaml",
        ),
        _Extractor(
            module="scripts.walkers.transformers_introspection",
            staging_basename="transformers_introspection.yaml",
        ),
    ),
}


def _staging_dir(corpus_root: Path) -> Path:
    return corpus_root / "_staging"


def _canonical_path(corpus_root: Path, engine: str) -> Path:
    return corpus_root / f"{engine}.yaml"


# ---------------------------------------------------------------------------
# Provenance priority
# ---------------------------------------------------------------------------


# Lower index = higher priority for "primary source" assignment when a rule's
# fingerprint appears in multiple staging files. The remaining sources go into
# ``cross_validated_by`` (sorted alphabetically for stability).
#
# AST walker beats introspection because the AST walker derives match.fields
# and kwargs from structural source — the introspection walker probes the
# real library and trusts the resulting raise/no-raise pattern, which is more
# noise-prone for kwargs-positive synthesis than reading the conditional
# directly. message_template precedence is handled separately (introspection
# wins there because it observes the literal library text).
_PROVENANCE_PRIORITY: tuple[str, ...] = (
    "ast_walker",
    "introspection",
    "manual_seed",
    "runtime_warning",
    "observed_collision",
)


def _provenance_rank(added_by: str) -> int:
    """Lower rank = higher priority. Unknown sources sort to the end."""
    try:
        return _PROVENANCE_PRIORITY.index(added_by)
    except ValueError:
        return len(_PROVENANCE_PRIORITY)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def fingerprint_rule(rule: dict[str, Any]) -> bytes:
    """Return the canonical-serialised fingerprint bytes for a rule dict.

    Two rules with identical fingerprints are the same constraint discovered
    by two independent paths and get merged. The fingerprint uses
    :func:`canonical_serialise` so float jitter, dict-key ordering, and
    NaN don't break dedup.

    The fingerprint deliberately excludes ``id`` (extractors should already
    agree on deterministic ids, but a one-character drift shouldn't break
    cross-validation), ``message_template`` (the real library text differs
    from a structural reconstruction even when the rule is the same), and
    ``walker_source`` (path / line moves on every library bump).
    """
    match = rule.get("match") or {}
    fields = match.get("fields") if isinstance(match, dict) else None
    return canonical_serialise(
        {
            "engine": rule.get("engine"),
            "severity": rule.get("severity"),
            "match_fields": fields or {},
        }
    )


# ---------------------------------------------------------------------------
# Staging extraction
# ---------------------------------------------------------------------------


def run_extractors(engine: str, corpus_root: Path) -> None:
    """Invoke each registered extractor for ``engine`` to produce its staging file.

    Each extractor must accept ``--out <path>`` and write a corpus-shaped
    YAML document. Extractor failures bubble up as :class:`subprocess.CalledProcessError`
    so CI sees them as fatal, matching the walker-landmark contract.
    """
    extractors = _ENGINE_EXTRACTORS.get(engine)
    if not extractors:
        raise ValueError(
            f"No extractors registered for engine={engine!r}. "
            f"Add an entry to _ENGINE_EXTRACTORS in {__file__}."
        )
    staging = _staging_dir(corpus_root)
    staging.mkdir(parents=True, exist_ok=True)
    for extractor in extractors:
        out_path = staging / extractor.staging_basename
        env = os.environ.copy()
        # Make the project's scripts/ package importable inside the subprocess
        # without forcing the caller to set PYTHONPATH.
        env["PYTHONPATH"] = (
            f"{_PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        )
        cmd = [sys.executable, "-m", extractor.module, "--out", str(out_path)]
        print(f"[build_corpus] running {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, env=env, cwd=str(_PROJECT_ROOT))


def discover_staging_files(engine: str, corpus_root: Path) -> list[Path]:
    """Return all staging YAMLs for ``engine``, sorted alphabetically.

    The sort makes merger output deterministic when two staging files have
    the same fingerprint at the same priority rank — first-seen wins, and
    the alphabetical order makes "first-seen" predictable across machines.
    """
    staging = _staging_dir(corpus_root)
    if not staging.is_dir():
        return []
    return sorted(staging.glob(f"{engine}_*.yaml"))


def _load_staging(path: Path) -> dict[str, Any]:
    """Parse a staging YAML file; return the envelope dict (with rules list).

    Raises ``ValueError`` on a missing or malformed envelope rather than
    silently skipping — staging files are produced by trusted extractors,
    so a parse failure is a bug we want to surface.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Staging file {path} is not a YAML mapping")
    if "rules" not in raw or not isinstance(raw["rules"], list):
        raise ValueError(f"Staging file {path} missing 'rules' list")
    return raw


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _walker_confidence_min(*confidences: str | None) -> str | None:
    """Return the lowest walker_confidence among the given values.

    ``low`` < ``medium`` < ``high``. Cross-validation semantically
    *increases* confidence, but if any source flagged the candidate as
    low / medium we keep that flag visible — the reviewer should still
    eyeball it.
    """
    order = ("low", "medium", "high")
    seen = [c for c in confidences if c in order]
    if not seen:
        return None
    return min(seen, key=order.index)


def _union_preserving_order(*lists: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    """Concat lists, dedup while preserving first-seen order.

    Uses ``repr`` as the dedup key for unhashable items (dicts) so a list
    of references containing dict-shaped citations still dedupes.
    """
    ordered: list[Any] = []
    seen_keys: set[Any] = set()
    for lst in lists:
        if not lst:
            continue
        for item in lst:
            key = item if isinstance(item, (str, int, float, bool, type(None))) else repr(item)
            if key not in seen_keys:
                seen_keys.add(key)
                ordered.append(item)
    return ordered


@dataclass
class _MergeResult:
    """Output of merging one fingerprint bucket."""

    rule: dict[str, Any]
    """The merged rule dict (corpus-shape)."""

    sources: tuple[str, ...]
    """All ``added_by`` strings from the source rules in this bucket, sorted."""

    conflict_notes: tuple[str, ...]
    """Per-field conflict annotations for reviewer attention."""


def _merge_bucket(rules: list[dict[str, Any]]) -> _MergeResult:
    """Merge rules sharing one fingerprint into a single corpus entry.

    See module docstring for the per-field precedence table. Conflict
    annotations are accumulated and surfaced on the merged rule so a human
    reviewer can spot ambiguous merges; the vendor CI step will catch
    semantically wrong outputs regardless.
    """
    if len(rules) == 1:
        # Single-source rule — preserve its shape as-is, just normalise an
        # empty cross_validated_by entry away.
        rule = dict(rules[0])
        rule.pop("cross_validated_by", None)
        sources_unique = (str(rule.get("added_by", "manual_seed")),)
        return _MergeResult(rule=rule, sources=sources_unique, conflict_notes=())

    # Sort by provenance priority so the highest-priority source is the
    # "primary" (its fields win where the precedence table doesn't override).
    sorted_rules = sorted(
        rules, key=lambda r: (_provenance_rank(str(r.get("added_by", ""))), r.get("id", ""))
    )
    primary = dict(sorted_rules[0])
    secondaries = sorted_rules[1:]

    # ------------------------------------------------------------------
    # Per-field precedence
    # ------------------------------------------------------------------

    # Pull contributors by source-type so the per-field rules can reach for
    # them by name without re-walking the list.
    by_source = {str(r.get("added_by", "")): dict(r) for r in sorted_rules}
    ast_rule = by_source.get("ast_walker")
    introspect_rule = by_source.get("introspection")

    conflicts: list[str] = []

    # match.fields: AST walker wins if present (more specific operators).
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        ast_fields = ((ast_rule.get("match") or {}).get("fields"))
        primary_fields = ((primary.get("match") or {}).get("fields"))
        if ast_fields and ast_fields != primary_fields:
            primary["match"] = {**(primary.get("match") or {}), "fields": ast_fields}

    # message_template: introspection wins (real library text).
    if introspect_rule is not None and introspect_rule is not sorted_rules[0]:
        intro_msg = introspect_rule.get("message_template")
        primary_msg = primary.get("message_template")
        if intro_msg and intro_msg != primary_msg:
            if primary_msg:
                conflicts.append(
                    f"message_template: introspection text overrode "
                    f"{primary.get('added_by')}'s template"
                )
            primary["message_template"] = intro_msg

    # expected_outcome: introspection's observed_messages / observed_messages_regex
    # win when present; the structural fields (outcome, emission_channel,
    # normalised_fields) are taken from the primary source.
    primary_eo = dict(primary.get("expected_outcome") or {})
    if introspect_rule is not None:
        intro_eo = introspect_rule.get("expected_outcome") or {}
        for key in ("observed_messages", "observed_messages_regex"):
            value = intro_eo.get(key)
            if value:
                primary_eo[key] = value
    primary["expected_outcome"] = primary_eo

    # kwargs_positive / kwargs_negative: AST walker wins.
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        for key in ("kwargs_positive", "kwargs_negative"):
            ast_value = ast_rule.get(key)
            primary_value = primary.get(key)
            if ast_value and ast_value != primary_value:
                primary[key] = ast_value

    # walker_source: AST walker has the real source line.
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        ast_source = ast_rule.get("walker_source")
        if isinstance(ast_source, dict):
            # Compose: prefer AST line_at_scan / path; keep min walker_confidence.
            current = dict(primary.get("walker_source") or {})
            for key in ("path", "method", "line_at_scan"):
                if ast_source.get(key) is not None:
                    current[key] = ast_source[key]
            current["walker_confidence"] = _walker_confidence_min(
                *(
                    str(r.get("walker_source", {}).get("walker_confidence"))
                    for r in sorted_rules
                )
            )
            primary["walker_source"] = current
    else:
        # Even single-AST or single-introspection: take the min confidence
        # across all source rules (cross-validation does not raise confidence;
        # any low-confidence source surfaces).
        current = dict(primary.get("walker_source") or {})
        current["walker_confidence"] = _walker_confidence_min(
            *(
                str(r.get("walker_source", {}).get("walker_confidence"))
                for r in sorted_rules
            )
        )
        primary["walker_source"] = current

    # references: union, preserving first-seen order across all sources.
    all_refs: list[Any] = []
    for r in sorted_rules:
        refs = r.get("references") or []
        all_refs.extend(refs)
    if all_refs:
        primary["references"] = _union_preserving_order(all_refs)

    # id: first source's id is canonical; warn if they differ.
    primary_id = primary.get("id")
    for r in secondaries:
        if r.get("id") != primary_id:
            conflicts.append(
                f"id: {primary.get('added_by')} -> {primary_id!r}; "
                f"{r.get('added_by')} -> {r.get('id')!r}"
            )

    # added_by + cross_validated_by: primary source first; the rest sorted.
    primary_added_by = str(primary.get("added_by", "manual_seed"))
    other_sources = sorted(
        {str(r.get("added_by", "")) for r in secondaries} - {primary_added_by}
    )
    primary["added_by"] = primary_added_by
    if other_sources:
        primary["cross_validated_by"] = other_sources
    else:
        primary.pop("cross_validated_by", None)

    if conflicts:
        primary["conflict_note"] = "; ".join(conflicts)

    sources = tuple(sorted({str(r.get("added_by", "")) for r in sorted_rules}))
    return _MergeResult(rule=primary, sources=sources, conflict_notes=tuple(conflicts))


def merge_staging(
    staging_envelopes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge a list of staging-file dicts into a list of canonical rule dicts.

    Returns ``(rules, envelope)`` where ``rules`` is alphabetically sorted by
    id (so the merger's output is byte-stable for ``--check`` mode), and
    ``envelope`` carries the per-engine schema_version / engine / engine_version
    metadata. When staging envelopes disagree on engine_version, the highest
    version string wins (lex-sorted) and a ``staging_engine_versions`` field
    captures the divergence for review.
    """
    if not staging_envelopes:
        raise ValueError(
            "No staging envelopes provided — at least one extractor must run "
            "before merging. Did the extractors fail to produce output?"
        )

    buckets: dict[bytes, list[dict[str, Any]]] = {}
    for envelope in staging_envelopes:
        for rule in envelope.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            key = fingerprint_rule(rule)
            buckets.setdefault(key, []).append(rule)

    merged_rules: list[dict[str, Any]] = []
    for bucket_rules in buckets.values():
        result = _merge_bucket(bucket_rules)
        merged_rules.append(result.rule)

    # Stability: sort by id so re-running on the same staging produces
    # byte-identical canonical YAML.
    merged_rules.sort(key=lambda r: r.get("id", ""))

    # Build the envelope. schema_version + engine come from the first
    # staging file (extractors all target one schema major). engine_version
    # is taken from the highest staging-reported version — if extractors
    # disagree (one walker pinned to an older release than another), the
    # canonical corpus reflects the newest observed.
    first = staging_envelopes[0]
    engine_versions = sorted({str(s.get("engine_version", "")) for s in staging_envelopes})
    engine_versions = [v for v in engine_versions if v]

    envelope: dict[str, Any] = {
        "schema_version": str(first.get("schema_version", "1.0.0")),
        "engine": str(first.get("engine", "")),
        "engine_version": engine_versions[-1] if engine_versions else "",
    }

    # Surface walker version pin, if any staging file declared one.
    pinned_ranges = sorted(
        {
            str(s.get("walker_pinned_range", ""))
            for s in staging_envelopes
            if s.get("walker_pinned_range")
        }
    )
    if pinned_ranges:
        envelope["walker_pinned_range"] = pinned_ranges[-1]

    # walked_at: max across staging (or env override for reproducibility tests).
    frozen = os.environ.get("LLENERGY_WALKER_FROZEN_AT")
    if frozen:
        envelope["walked_at"] = frozen
    else:
        walked_ats = sorted(
            {str(s.get("walked_at", "")) for s in staging_envelopes if s.get("walked_at")}
        )
        envelope["walked_at"] = walked_ats[-1] if walked_ats else _now_iso()

    if len(engine_versions) > 1:
        envelope["staging_engine_versions"] = engine_versions

    return merged_rules, envelope


def _now_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# Vendor validation gate
# ---------------------------------------------------------------------------


@dataclass
class _ValidationOutcome:
    """Result of running vendor validation on a candidate list."""

    validated_rules: list[dict[str, Any]]
    """Rules whose declared expected_outcome matched live-library behaviour."""

    quarantined: list[dict[str, Any]]
    """Rules dropped because their declared shape diverged from the library.

    Each entry is ``{"rule": <rule dict>, "divergences": [<field, expected,
    observed>, ...]}``.
    """

    skipped: bool
    """True iff validation could not run (engine library not importable)."""

    skip_reason: str | None
    """Human-readable reason validation was skipped, if any."""


def _validate_candidates(
    *,
    engine: str,
    rules: list[dict[str, Any]],
    envelope: dict[str, Any],
    corpus_root: Path,
) -> _ValidationOutcome:
    """Run vendor validation against the merged candidate rule list.

    Writes ``_staging/{engine}_merged_candidates.yaml`` so vendor_rules.py
    can be invoked on a real path (its API is corpus-path-driven), then
    calls :func:`scripts.vendor_rules.vendor_engine` directly. Splits the
    candidate list into validated and quarantined buckets based on the
    returned divergences.

    When the engine library can't be imported, returns a skipped outcome
    that keeps every candidate. The caller decides whether to treat that
    as fatal (CI) or acceptable (local).
    """
    # Late import: vendor_rules pulls in the engine library at import time
    # for its public-API helpers, but the import itself is cheap. Keep this
    # local so build_corpus.py can be import-clean without transformers
    # installed (e.g. for unit tests that exercise the merger only).
    from scripts.vendor_rules import (  # noqa: E402
        VendorEngineNotImportable,
        vendor_engine,
    )

    staging = _staging_dir(corpus_root)
    staging.mkdir(parents=True, exist_ok=True)
    candidates_yaml = staging / f"{engine}_merged_candidates.yaml"
    candidates_json = staging / f"{engine}_merged_candidates.json"

    # Write the same shape vendor_rules.py expects from the canonical YAML.
    candidates_doc: dict[str, Any] = {
        "schema_version": envelope.get("schema_version", "1.0.0"),
        "engine": envelope.get("engine", engine),
        "engine_version": envelope.get("engine_version", ""),
        "rules": [_ordered_rule(r) for r in rules],
    }
    candidates_yaml.write_text(
        yaml.safe_dump(candidates_doc, sort_keys=False, default_flow_style=False, width=100)
    )

    try:
        _envelope, divergences = vendor_engine(
            engine=engine,
            corpus_path=candidates_yaml,
            out_path=candidates_json,
        )
    except VendorEngineNotImportable as exc:
        # Library not installed in this environment — keep every candidate
        # so the canonical corpus reflects the extractors' output. CI runs
        # in the engine container where the library is always importable,
        # so this branch is local-dev only.
        print(
            f"[build_corpus] vendor validation skipped: {exc}. "
            f"All {len(rules)} candidates kept unfiltered.",
            file=sys.stderr,
        )
        return _ValidationOutcome(
            validated_rules=list(rules),
            quarantined=[],
            skipped=True,
            skip_reason=str(exc),
        )

    # Group divergences by rule_id; presence of any divergence quarantines.
    by_id: dict[str, list[dict[str, Any]]] = {}
    for div in divergences:
        by_id.setdefault(div.rule_id, []).append(div.as_dict())

    validated: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for rule in rules:
        rid = rule.get("id", "")
        if rid in by_id:
            quarantined.append({"rule": rule, "divergences": by_id[rid]})
        else:
            validated.append(rule)

    return _ValidationOutcome(
        validated_rules=validated,
        quarantined=quarantined,
        skipped=False,
        skip_reason=None,
    )


def _failed_validation_path(corpus_root: Path, engine: str) -> Path:
    """Sibling-of-staging file recording rules dropped by vendor validation."""
    return _staging_dir(corpus_root) / f"_failed_validation_{engine}.yaml"


def _emit_failed_validation_yaml(
    *,
    engine: str,
    envelope: dict[str, Any],
    quarantined: list[dict[str, Any]],
) -> str:
    """Serialise quarantined rules + their divergences as deterministic YAML.

    Output mirrors the canonical corpus shape (so the same loaders can read
    it for inspection), with the per-rule divergence list attached. Sorted
    by rule id for byte-stable diffs.
    """
    sorted_quarantined = sorted(
        quarantined, key=lambda entry: str(entry.get("rule", {}).get("id", ""))
    )
    doc: dict[str, Any] = {
        "schema_version": envelope.get("schema_version", "1.0.0"),
        "engine": engine,
        "engine_version": envelope.get("engine_version", ""),
        "generated_at": _now_iso(),
        "quarantined_rules": [
            {
                "rule": _ordered_rule(entry["rule"]),
                "divergences": list(entry.get("divergences", [])),
            }
            for entry in sorted_quarantined
        ],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# YAML emission (matches the existing transformers walker's key ordering)
# ---------------------------------------------------------------------------


def _ordered_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Return a rule dict with keys in the conventional corpus order.

    Keeping key order stable across re-runs makes ``--check`` mode reliable.
    Unknown keys (e.g. ``conflict_note``) sort to the end so the canonical
    fields lead.
    """
    canonical_order = [
        "id",
        "engine",
        "library",
        "rule_under_test",
        "severity",
        "native_type",
        "walker_source",
        "match",
        "kwargs_positive",
        "kwargs_negative",
        "expected_outcome",
        "message_template",
        "references",
        "added_by",
        "cross_validated_by",
        "added_at",
    ]
    out: dict[str, Any] = {}
    for key in canonical_order:
        if key in rule:
            out[key] = rule[key]
    # Trailing extras (conflict_note, future fields).
    for key in sorted(rule.keys()):
        if key not in out:
            out[key] = rule[key]
    return out


def emit_yaml(rules: list[dict[str, Any]], envelope: dict[str, Any]) -> str:
    """Serialise the canonical corpus as a deterministic YAML string."""
    doc: dict[str, Any] = {
        "schema_version": envelope.get("schema_version", "1.0.0"),
        "engine": envelope.get("engine", ""),
        "engine_version": envelope.get("engine_version", ""),
    }
    if "walker_pinned_range" in envelope:
        doc["walker_pinned_range"] = envelope["walker_pinned_range"]
    doc["walked_at"] = envelope.get("walked_at", "")
    if "staging_engine_versions" in envelope:
        doc["staging_engine_versions"] = envelope["staging_engine_versions"]
    doc["rules"] = [_ordered_rule(r) for r in rules]
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class _BuildResult:
    """In-memory build output: canonical YAML text plus validation outcome.

    Returned by :func:`build_corpus_text_and_outcome` so callers can write
    the canonical YAML, the failed-validation sidecar, and report counts
    without re-running the merge.
    """

    canonical_text: str
    rules_in_canonical: int
    candidates_merged: int
    outcome: _ValidationOutcome


def build_corpus_text_and_outcome(
    engine: str,
    corpus_root: Path,
    *,
    validate: bool = True,
) -> _BuildResult:
    """Discover staging files, merge, validate, return canonical YAML + outcome.

    When ``validate`` is True (the default), runs vendor validation against
    the merged candidates and filters out rules that diverged from the
    live library. When False, behaves exactly like the pre-validation
    pipeline — the canonical YAML reflects every merged candidate.

    Pure-ish modulo the staging-file read and the vendor library import;
    callable in isolation by pre-populating the staging directory.
    """
    paths = discover_staging_files(engine, corpus_root)
    if not paths:
        raise FileNotFoundError(
            f"No staging files at {_staging_dir(corpus_root)}/{engine}_*.yaml. "
            f"Run extractors first (omit --skip-extract)."
        )
    envelopes = [_load_staging(p) for p in paths]
    rules, envelope = merge_staging(envelopes)
    candidates_count = len(rules)

    if validate:
        outcome = _validate_candidates(
            engine=engine, rules=rules, envelope=envelope, corpus_root=corpus_root
        )
        canonical_rules = outcome.validated_rules
    else:
        outcome = _ValidationOutcome(
            validated_rules=list(rules),
            quarantined=[],
            skipped=True,
            skip_reason="--skip-validation passed",
        )
        canonical_rules = outcome.validated_rules

    text = emit_yaml(canonical_rules, envelope)
    return _BuildResult(
        canonical_text=text,
        rules_in_canonical=len(canonical_rules),
        candidates_merged=candidates_count,
        outcome=outcome,
    )


def build_corpus_text(engine: str, corpus_root: Path, *, validate: bool = True) -> str:
    """Backwards-compatible thin wrapper returning just the canonical YAML text."""
    return build_corpus_text_and_outcome(
        engine, corpus_root, validate=validate
    ).canonical_text


def write_corpus(engine: str, corpus_root: Path, *, validate: bool = True) -> _BuildResult:
    """Build, validate, write the canonical corpus + quarantine sidecar.

    Returns the :class:`_BuildResult` so the CLI can report counts. When
    quarantined rules are present, writes them to
    ``_staging/_failed_validation_{engine}.yaml``; when none, removes any
    stale sidecar so reviewers don't see ghosts from a previous run.
    """
    result = build_corpus_text_and_outcome(engine, corpus_root, validate=validate)
    out_path = _canonical_path(corpus_root, engine)
    out_path.write_text(result.canonical_text)

    failed_path = _failed_validation_path(corpus_root, engine)
    if result.outcome.quarantined:
        # Need the merged envelope to stamp the sidecar with engine_version,
        # which is captured inside build_corpus_text_and_outcome but not on
        # the result. Re-derive it cheaply from the staging headers.
        envelopes = [_load_staging(p) for p in discover_staging_files(engine, corpus_root)]
        _rules, envelope = merge_staging(envelopes)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(
            _emit_failed_validation_yaml(
                engine=engine, envelope=envelope, quarantined=result.outcome.quarantined
            )
        )
    elif failed_path.exists():
        failed_path.unlink()

    return result


def check_drift(engine: str, corpus_root: Path, *, validate: bool = True) -> tuple[int, str]:
    """Re-run merger (with validation); compare against checked-in corpus.

    Returns ``(exit_code, diff_text)``. ``exit_code`` is ``0`` if the
    canonical corpus matches the merger's freshly-built output exactly; ``1``
    on any byte-level drift; ``2`` on missing staging or missing corpus
    (treated as fatal — CI must run the extractors before --check).

    Validation is enabled by default so ``--check`` compares apples-to-apples
    with what ``write_corpus`` would produce. ``validate=False`` matches an
    unvalidated rebuild against the on-disk canonical, which only makes
    sense when the canonical itself was written without validation.
    """
    canonical_path = _canonical_path(corpus_root, engine)
    if not canonical_path.exists():
        return 2, f"Canonical corpus not found at {canonical_path}"
    try:
        rebuilt = build_corpus_text(engine, corpus_root, validate=validate)
    except FileNotFoundError as exc:
        return 2, str(exc)
    actual = canonical_path.read_text()
    if rebuilt == actual:
        return 0, ""
    diff = "".join(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            rebuilt.splitlines(keepends=True),
            fromfile=f"a/{canonical_path.name}",
            tofile=f"b/{canonical_path.name} (rebuilt)",
        )
    )
    return 1, diff


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical validation-rules corpus from staging extractors.",
    )
    parser.add_argument(
        "--engine",
        default="transformers",
        choices=sorted(_ENGINE_EXTRACTORS),
        help="Engine to build the corpus for (default: transformers).",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(_PROJECT_ROOT) / "configs" / "validation_rules",
        help="Root directory for both the canonical corpus and the _staging/ subdir.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip running the extractors; assume staging files already exist.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Don't write the corpus; re-build it and exit non-zero if the "
            "rebuilt corpus differs from the on-disk canonical YAML."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Skip the vendor-validation gate. Default is to validate every "
            "candidate rule against the live engine library and drop divergent "
            "rules before writing the canonical corpus. CI never sets this."
        ),
    )
    args = parser.parse_args(argv)

    corpus_root: Path = args.corpus_root
    validate: bool = not args.skip_validation

    if not args.skip_extract:
        try:
            run_extractors(args.engine, corpus_root)
        except subprocess.CalledProcessError as exc:
            print(f"[build_corpus] extractor failed: {exc}", file=sys.stderr)
            return 3

    if args.check:
        code, diff = check_drift(args.engine, corpus_root, validate=validate)
        if code != 0:
            print(diff, file=sys.stdout)
        return code

    result = write_corpus(args.engine, corpus_root, validate=validate)
    out_path = _canonical_path(corpus_root, args.engine)
    print(f"[build_corpus] wrote {out_path}", file=sys.stderr)
    print(
        f"[build_corpus] {result.candidates_merged} candidates merged, "
        f"{result.rules_in_canonical} validated and kept, "
        f"{len(result.outcome.quarantined)} divergent and quarantined.",
        file=sys.stderr,
    )
    if result.outcome.skipped:
        print(
            f"[build_corpus] validation skipped: {result.outcome.skip_reason}",
            file=sys.stderr,
        )
    if result.outcome.quarantined:
        print(
            f"[build_corpus] quarantined rules written to "
            f"{_failed_validation_path(corpus_root, args.engine)}",
            file=sys.stderr,
        )
        for entry in result.outcome.quarantined[:20]:
            rid = entry["rule"].get("id", "<unknown>")
            div_summary = ", ".join(
                f"{d['field']}(expected={d['expected']!r},observed={d['observed']!r})"
                for d in entry["divergences"][:3]
            )
            print(f"  - {rid}: {div_summary}", file=sys.stderr)
        if len(result.outcome.quarantined) > 20:
            print(
                f"  ... and {len(result.outcome.quarantined) - 20} more.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
