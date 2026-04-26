"""Canonical corpus builder — orchestrate miners, merge staging, emit corpus.

The validation-rules pipeline is split into independent miners (dynamic miner,
static miner, future runtime-warning miner) that each write to a staging file
under ``configs/validation_rules/_staging/{engine}_{name}.yaml``. This module
is the single canonical entry point that runs them, merges their outputs into
one :class:`~llenergymeasure.config.vendored_rules.loader.VendoredRules`-shaped
document, and writes ``configs/validation_rules/{engine}.yaml``.

Pipeline: miners → staging → merge → **vendor-validate** → write canonical corpus.

Vendor validation gate
----------------------
Between merge and canonical-write, every candidate's ``kwargs_positive`` and
``kwargs_negative`` are replayed against the real engine library via
:func:`scripts.vendor_rules.vendor_engine`. Rules whose declared
``expected_outcome`` doesn't match observed library behaviour are quarantined
to ``_staging/_failed_validation_{engine}.yaml`` instead of landing in the
canonical corpus. The merger optimises for *recall*; vendor validation is the
single architectural gate that turns the recall-first candidate list into the
runtime-applied corpus.

Why a merger at all
-------------------
Today the corpus is single-writer (dynamic miner only). With the static miner
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
``match.fields`` predicate          static miner (more specific operators)
``message_template``                dynamic miner (real library text)
``observed_messages*``              dynamic miner (real captured emissions)
``kwargs_positive`` / ``negative``  static miner (derived from conditional)
``miner_source.line_at_scan``       static miner (real source line)
``references``                      union (let reviewer see all evidence)
``id``                              first source's id is canonical
``added_by``                        primary source (priority order)
``cross_validated_by``              all other sources, sorted
=================================== =================================

If precedence is ambiguous (e.g. both sources are dynamic-miner-derived but
disagree on a field), the merger keeps the rule from the higher-priority
source and emits a ``conflict_note`` annotation in the YAML for reviewer
attention.

CLI
---
::

    python scripts/miners/build_corpus.py --engine transformers
    # Run extractors -> staging -> merge -> write canonical corpus

    python scripts/miners/build_corpus.py --engine transformers --check
    # Re-run; diff against checked-in corpus; exit 1 on drift.

    python scripts/miners/build_corpus.py --engine transformers --skip-extract
    # Assume staging files already exist; merge + write.

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
# scripts.miners.build_corpus`` and as a direct ``python scripts/miners/build_corpus.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# When this file is run directly (``python scripts/miners/build_corpus.py``),
# Python prepends ``scripts/miners/`` to sys.path. That directory contains a
# ``transformers_miner.py`` miner module which would shadow the real ``transformers``
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
# Add new (engine, [extractors]) entries to wire vLLM / TRT-LLM miners in.
_ENGINE_EXTRACTORS: dict[str, tuple[_Extractor, ...]] = {
    "transformers": (
        _Extractor(
            module="scripts.miners.transformers_static_miner",
            staging_basename="transformers_static_miner.yaml",
        ),
        _Extractor(
            module="scripts.miners.transformers_dynamic_miner",
            staging_basename="transformers_dynamic_miner.yaml",
        ),
    ),
    # TRT-LLM is static-only by adversarial-review decision #8 — the dynamic
    # constructor probe yields zero raises (TRT-LLM defers all real validation
    # to engine build, opaque from Python). The static miner reads the 0.21.0
    # source tree extracted to /tmp/trt-llm-0.21.0/.
    "tensorrt": (
        _Extractor(
            module="scripts.miners.tensorrt_static_miner",
            staging_basename="tensorrt_static_miner.yaml",
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
# Static miner beats dynamic miner because the static miner derives match.fields
# and kwargs from structural source — the dynamic miner probes the real library
# and trusts the resulting raise/no-raise pattern, which is more noise-prone for
# kwargs-positive synthesis than reading the conditional directly.
# message_template precedence is handled separately (dynamic miner wins there
# because it observes the literal library text).
_PROVENANCE_PRIORITY: tuple[str, ...] = (
    "static_miner",
    "dynamic_miner",
    "pydantic_lift",
    "msgspec_lift",
    "dataclass_lift",
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
    ``miner_source`` (path / line moves on every library bump).
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
    so CI sees them as fatal, matching the miner-landmark contract.
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
        env["PYTHONPATH"] = f"{_PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
            os.pathsep
        )
        cmd = [sys.executable, "-m", extractor.module, "--out", str(out_path)]
        print(f"[build_corpus] running {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, env=env, cwd=str(_PROJECT_ROOT))


def discover_staging_files(engine: str, corpus_root: Path) -> list[Path]:
    """Return all staging YAMLs for ``engine``, sorted alphabetically.

    The sort makes merger output deterministic when two staging files have
    the same fingerprint at the same priority rank — first-seen wins, and
    the alphabetical order makes "first-seen" predictable across machines.

    The merger's own previous output (``{engine}_merged_candidates.yaml``)
    is excluded explicitly: globbing the staging dir would otherwise feed
    the merger's previous run back into itself, masking extractor-side
    fixes (the previous merged file's stale kwargs would dominate the
    re-merge under fingerprint dedup) and giving every successive run
    monotonically older data.
    """
    staging = _staging_dir(corpus_root)
    if not staging.is_dir():
        return []
    merged_self = _MERGED_CANDIDATES_BASENAME.format(engine=engine)
    return sorted(p for p in staging.glob(f"{engine}_*.yaml") if p.name != merged_self)


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
    ast_rule = by_source.get("static_miner")
    introspect_rule = by_source.get("dynamic_miner")

    conflicts: list[str] = []

    # match.fields: static miner wins if present (more specific operators).
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        ast_fields = (ast_rule.get("match") or {}).get("fields")
        primary_fields = (primary.get("match") or {}).get("fields")
        if ast_fields and ast_fields != primary_fields:
            primary["match"] = {**(primary.get("match") or {}), "fields": ast_fields}

    # message_template: dynamic miner wins (real library text).
    if introspect_rule is not None and introspect_rule is not sorted_rules[0]:
        intro_msg = introspect_rule.get("message_template")
        primary_msg = primary.get("message_template")
        if intro_msg and intro_msg != primary_msg:
            if primary_msg:
                conflicts.append(
                    f"message_template: dynamic miner text overrode "
                    f"{primary.get('added_by')}'s template"
                )
            primary["message_template"] = intro_msg

    # expected_outcome: dynamic miner's observed_messages / observed_messages_regex
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

    # kwargs_positive / kwargs_negative: static miner wins.
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        for key in ("kwargs_positive", "kwargs_negative"):
            ast_value = ast_rule.get(key)
            primary_value = primary.get(key)
            if ast_value and ast_value != primary_value:
                primary[key] = ast_value

    # miner_source: static miner has the real source line.
    if ast_rule is not None and ast_rule is not sorted_rules[0]:
        ast_source = ast_rule.get("miner_source")
        if isinstance(ast_source, dict):
            # Compose: prefer static-miner line_at_scan / path / method.
            current = dict(primary.get("miner_source") or {})
            for key in ("path", "method", "line_at_scan"):
                if ast_source.get(key) is not None:
                    current[key] = ast_source[key]
            primary["miner_source"] = current

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
    other_sources = sorted({str(r.get("added_by", "")) for r in secondaries} - {primary_added_by})
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
    # disagree (one miner pinned to an older release than another), the
    # canonical corpus reflects the newest observed.
    first = staging_envelopes[0]
    engine_versions = sorted({str(s.get("engine_version", "")) for s in staging_envelopes})
    engine_versions = [v for v in engine_versions if v]

    envelope: dict[str, Any] = {
        "schema_version": str(first.get("schema_version", "1.0.0")),
        "engine": str(first.get("engine", "")),
        "engine_version": engine_versions[-1] if engine_versions else "",
    }

    # Surface miner version pin, if any staging file declared one.
    pinned_ranges = sorted(
        {
            str(s.get("walker_pinned_range", ""))
            for s in staging_envelopes
            if s.get("walker_pinned_range")
        }
    )
    if pinned_ranges:
        envelope["walker_pinned_range"] = pinned_ranges[-1]

    # mined_at: max across staging (or env override for reproducibility tests).
    frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
    if frozen:
        envelope["mined_at"] = frozen
    else:
        mined_ats = sorted(
            {str(s.get("mined_at", "")) for s in staging_envelopes if s.get("mined_at")}
        )
        envelope["mined_at"] = mined_ats[-1] if mined_ats else _now_iso()

    if len(engine_versions) > 1:
        envelope["staging_engine_versions"] = engine_versions

    return merged_rules, envelope


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Vendor validation gate
# ---------------------------------------------------------------------------


_MERGED_CANDIDATES_BASENAME = "{engine}_merged_candidates.yaml"
"""Staging filename for the unfiltered merger output. Vendor validation reads
this back as a corpus-shaped YAML so :func:`scripts.vendor_rules.vendor_engine`
can replay every candidate's positive/negative kwargs."""

_FAILED_VALIDATION_BASENAME = "_failed_validation_{engine}.yaml"
"""Staging filename for the divergent-rules quarantine.

Schema::

    schema_version: 1.0.0
    engine: <engine>
    engine_version: <version observed during validation>
    generated_at: <ISO-8601 UTC>
    quarantined_rules:
      - rule: <full rule dict>
        divergences:
          - rule_id: <id>
            field: <expected_outcome.* or positive_confirmed/negative_confirmed>
            expected: <value>
            observed: <value>
"""


def _write_merged_candidates(
    corpus_root: Path,
    engine: str,
    rules: list[dict[str, Any]],
    envelope: dict[str, Any],
) -> Path:
    """Write the merger output to ``_staging/{engine}_merged_candidates.yaml``.

    Vendor validation needs a corpus-shaped YAML it can ingest; rather than
    invent a separate transport (in-memory module imports etc.), reuse the
    same on-disk shape :func:`emit_yaml` produces. The file lives under
    ``_staging`` so it's never confused with the canonical corpus.
    """
    staging = _staging_dir(corpus_root)
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / _MERGED_CANDIDATES_BASENAME.format(engine=engine)
    path.write_text(emit_yaml(rules, envelope))
    return path


def _validate_candidates(
    candidates: list[dict[str, Any]],
    engine: str,
    corpus_root: Path,
    envelope: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-run each candidate through the real engine library; filter divergent rules.

    Returns ``(kept, divergent)`` where ``kept`` is the subset whose declared
    ``expected_outcome`` matched observed behaviour AND whose
    ``kwargs_positive`` actually fired AND whose ``kwargs_negative`` actually
    didn't, and ``divergent`` is the rest annotated with the per-field
    diagnostic info from ``vendor_engine``.

    The function writes the candidates to ``_staging/{engine}_merged_candidates.yaml``
    as a side effect — that file is the input to :func:`vendor_engine`. The
    JSON envelope ``vendor_engine`` writes goes to a sibling temp path
    (``_staging/{engine}_vendor.json``) so the canonical
    ``src/llenergymeasure/config/vendored_rules/{engine}.json`` (the
    runtime-loaded sidecar) is never overwritten by this build step.
    """
    # Local import keeps the merger importable in environments that don't have
    # the engine library installed (e.g. CI lint job): only --skip-validation
    # callers need to load vendor_rules.
    from scripts.vendor_rules import vendor_engine

    candidates_path = _write_merged_candidates(corpus_root, engine, candidates, envelope)
    vendor_json_path = _staging_dir(corpus_root) / f"{engine}_vendor.json"

    # vendor_engine returns (envelope, divergences). Divergences carry rule_id,
    # field, expected, observed — exactly the diagnostic we need to surface in
    # the quarantine file.
    _vendor_envelope, divergences = vendor_engine(
        engine=engine,
        corpus_path=candidates_path,
        out_path=vendor_json_path,
    )

    divergence_by_id: dict[str, list[dict[str, Any]]] = {}
    for d in divergences:
        divergence_by_id.setdefault(d.rule_id, []).append(
            {
                "rule_id": d.rule_id,
                "field": d.field,
                "expected": d.expected,
                "observed": d.observed,
            }
        )

    kept: list[dict[str, Any]] = []
    divergent: list[dict[str, Any]] = []
    for rule in candidates:
        rule_id = str(rule.get("id", ""))
        if rule_id in divergence_by_id:
            divergent.append(
                {
                    "rule": rule,
                    "divergences": divergence_by_id[rule_id],
                }
            )
        else:
            kept.append(rule)

    return kept, divergent


def _emit_failed_validation_yaml(
    corpus_root: Path,
    engine: str,
    divergent: list[dict[str, Any]],
    envelope: dict[str, Any],
) -> Path | None:
    """Write the quarantine file for divergent rules.

    Returns the path written, or ``None`` if there are no divergent rules
    (in which case any stale quarantine file is removed so the next reviewer
    isn't misled by leftover state from a previous run).
    """
    staging = _staging_dir(corpus_root)
    path = staging / _FAILED_VALIDATION_BASENAME.format(engine=engine)
    if not divergent:
        if path.exists():
            path.unlink()
        return None
    staging.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": str(envelope.get("schema_version", "1.0.0")),
        "engine": str(envelope.get("engine", engine)),
        "engine_version": str(envelope.get("engine_version", "")),
        "generated_at": _now_iso(),
        "quarantined_rules": [
            {"rule": _ordered_rule(entry["rule"]), "divergences": entry["divergences"]}
            for entry in divergent
        ],
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))
    return path


# ---------------------------------------------------------------------------
# YAML emission (matches the existing transformers miner's key ordering)
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
        "miner_source",
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
    doc["mined_at"] = envelope.get("mined_at", "")
    if "staging_engine_versions" in envelope:
        doc["staging_engine_versions"] = envelope["staging_engine_versions"]
    doc["rules"] = [_ordered_rule(r) for r in rules]
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class _BuildResult:
    """In-memory build output: canonical YAML text plus rule counts.

    Returned by :func:`build_corpus_text_and_outcome` so callers can write
    the canonical YAML and report counts without re-running the merge.
    """

    canonical_text: str
    rules_in_canonical: int
    candidates_merged: int
    rules_quarantined: int = 0
    quarantined_ids: tuple[str, ...] = ()
    validation_skipped: bool = False


def build_corpus_text_and_outcome(
    engine: str,
    corpus_root: Path,
    *,
    skip_validation: bool = False,
) -> _BuildResult:
    """Discover staging files, merge, vendor-validate, return canonical YAML.

    Pure-ish modulo the staging-file read; callable in isolation by
    pre-populating the staging directory. When ``skip_validation`` is true,
    all merged candidates land in the canonical YAML regardless of vendor
    outcomes — useful for fast local iteration but never appropriate in CI.
    """
    paths = discover_staging_files(engine, corpus_root)
    if not paths:
        raise FileNotFoundError(
            f"No staging files at {_staging_dir(corpus_root)}/{engine}_*.yaml. "
            f"Run extractors first (omit --skip-extract)."
        )
    envelopes = [_load_staging(p) for p in paths]
    candidates, envelope = merge_staging(envelopes)
    candidates_count = len(candidates)

    if skip_validation:
        # Still write the merged-candidates staging file so reviewers can
        # inspect the recall-first list; just don't run the vendor gate.
        _write_merged_candidates(corpus_root, engine, candidates, envelope)
        # Drop any stale quarantine file — running --skip-validation with a
        # leftover file from a previous validating run would be misleading.
        stale = _staging_dir(corpus_root) / _FAILED_VALIDATION_BASENAME.format(engine=engine)
        if stale.exists():
            stale.unlink()
        text = emit_yaml(candidates, envelope)
        return _BuildResult(
            canonical_text=text,
            rules_in_canonical=candidates_count,
            candidates_merged=candidates_count,
            rules_quarantined=0,
            quarantined_ids=(),
            validation_skipped=True,
        )

    kept, divergent = _validate_candidates(candidates, engine, corpus_root, envelope)
    _emit_failed_validation_yaml(corpus_root, engine, divergent, envelope)

    quarantined_ids = tuple(sorted(str(entry["rule"].get("id", "")) for entry in divergent))
    text = emit_yaml(kept, envelope)
    return _BuildResult(
        canonical_text=text,
        rules_in_canonical=len(kept),
        candidates_merged=candidates_count,
        rules_quarantined=len(divergent),
        quarantined_ids=quarantined_ids,
        validation_skipped=False,
    )


def build_corpus_text(
    engine: str,
    corpus_root: Path,
    *,
    skip_validation: bool = False,
) -> str:
    """Return the canonical YAML text for ``engine``."""
    return build_corpus_text_and_outcome(
        engine, corpus_root, skip_validation=skip_validation
    ).canonical_text


def write_corpus(
    engine: str,
    corpus_root: Path,
    *,
    skip_validation: bool = False,
) -> _BuildResult:
    """Build and write the canonical corpus.

    Returns the :class:`_BuildResult` so the CLI can report counts.
    """
    result = build_corpus_text_and_outcome(engine, corpus_root, skip_validation=skip_validation)
    out_path = _canonical_path(corpus_root, engine)
    out_path.write_text(result.canonical_text)
    return result


def check_drift(
    engine: str,
    corpus_root: Path,
    *,
    skip_validation: bool = False,
) -> tuple[int, str]:
    """Re-run merger (with vendor validation); compare against checked-in corpus.

    Returns ``(exit_code, diff_text)``. ``exit_code`` is ``0`` if the
    canonical corpus matches the merger's freshly-built output exactly; ``1``
    on any byte-level drift; ``2`` on missing staging or missing corpus
    (treated as fatal — CI must run the extractors before --check).

    Drift detection compares the validated rebuild to the validated canonical
    by default. Without re-running validation, drift would compare the
    validated checked-in corpus against an unvalidated rebuild and surface
    spurious "drift" for every quarantined candidate. ``skip_validation``
    here exists for parity with the build path but should not be used in CI.
    """
    canonical_path = _canonical_path(corpus_root, engine)
    if not canonical_path.exists():
        return 2, f"Canonical corpus not found at {canonical_path}"
    try:
        rebuilt = build_corpus_text(engine, corpus_root, skip_validation=skip_validation)
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
        "--skip-validation",
        action="store_true",
        help=(
            "Skip the vendor-validation gate: write every merged candidate to "
            "the canonical corpus regardless of whether its declared "
            "expected_outcome matches observed library behaviour. Off by "
            "default (CI must always validate); on for fast local iteration."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Don't write the corpus; re-build it and exit non-zero if the "
            "rebuilt corpus differs from the on-disk canonical YAML."
        ),
    )
    args = parser.parse_args(argv)

    corpus_root: Path = args.corpus_root

    if not args.skip_extract:
        try:
            run_extractors(args.engine, corpus_root)
        except subprocess.CalledProcessError as exc:
            print(f"[build_corpus] extractor failed: {exc}", file=sys.stderr)
            return 3

    if args.check:
        code, diff = check_drift(args.engine, corpus_root, skip_validation=args.skip_validation)
        if code != 0:
            print(diff, file=sys.stdout)
        return code

    result = write_corpus(args.engine, corpus_root, skip_validation=args.skip_validation)
    out_path = _canonical_path(corpus_root, args.engine)
    print(f"[build_corpus] wrote {out_path}", file=sys.stderr)
    if result.validation_skipped:
        print(
            f"[build_corpus] {result.candidates_merged} candidates merged; "
            f"vendor validation SKIPPED (use without --skip-validation in CI).",
            file=sys.stderr,
        )
    else:
        print(
            f"[build_corpus] {result.candidates_merged} candidates merged, "
            f"{result.rules_in_canonical} validated and kept, "
            f"{result.rules_quarantined} quarantined.",
            file=sys.stderr,
        )
        if result.rules_quarantined:
            quarantine_path = _staging_dir(corpus_root) / _FAILED_VALIDATION_BASENAME.format(
                engine=args.engine
            )
            print(
                f"[build_corpus] divergent rules written to {quarantine_path}",
                file=sys.stderr,
            )
            for rule_id in result.quarantined_ids[:10]:
                print(f"  - {rule_id}", file=sys.stderr)
            if len(result.quarantined_ids) > 10:
                print(
                    f"  ... and {len(result.quarantined_ids) - 10} more.",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
