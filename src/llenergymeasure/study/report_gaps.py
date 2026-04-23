"""Core logic for ``llem report-gaps`` — feedback-loop gap detection and PR drafting.

Design:
- ``.product/designs/config-deduplication-dormancy/sweep-dedup.md`` §§4, 6
  (H3-collision channel)
- ``.product/designs/config-deduplication-dormancy/runtime-config-validation.md`` §4.7
  (runtime-warnings channel)

Two orthogonal feedback channels:

- ``h3-collisions``: scans results-bundles for H1-dedup-on runs whose
  post-run H3 hashes collide across distinct H1 groups. Any collision is a
  **proven canonicaliser gap** (sweep-dedup.md §4.1 invariant).
- ``runtime-warnings``: scans the user cache of runtime observations from
  :mod:`llenergymeasure.study.runtime_observations`; candidate rules are
  emissions whose normalised message isn't already covered by the corpus.

Both emit :class:`RuleCandidate` entries, deduped across sources and across
multiple sidecars, rendered to a proposed-YAML body. Draft-PR dispatch is
separated into :mod:`llenergymeasure.study._gh_automation` so the dry path
is trivially unit-testable without ``gh`` on PATH.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llenergymeasure.engines.vendored_rules.loader import Rule, VendoredRulesLoader
from llenergymeasure.study.equivalence_groups import (
    EquivalenceGroups,
    PostRunH3Group,
    find_h3_groups,
    load_equivalence_groups,
)
from llenergymeasure.study.message_normalise import NormalisedMessage, normalise

logger = logging.getLogger(__name__)


__all__ = [
    "GapReport",
    "RuleCandidate",
    "generate_report",
    "load_runtime_observations",
    "load_sidecars",
    "render_candidates_yaml",
    "scan_h3_collisions",
    "scan_runtime_warnings",
]


# ---------------------------------------------------------------------------
# Candidate / report types
# ---------------------------------------------------------------------------


@dataclass
class RuleCandidate:
    """Proposed rule entry ready to be rendered into ``configs/validation_rules/{engine}.yaml``."""

    candidate_id: str
    engine: str
    library_version: str
    source: str  # "runtime-warnings" | "h3-collisions"
    severity: str  # "warn" | "error" | "dormant_silent"
    confidence: str  # "high" | "medium" | "low"
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    expected_outcome: dict[str, Any]
    message_template: str | None
    observed_messages_regex: str | None
    evidence: dict[str, Any]
    verified: bool = True

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a YAML-ready dict in the corpus schema, field-order preserved."""
        added_by = "runtime_feedback" if self.source == "runtime-warnings" else "h3_collision"
        entry: dict[str, Any] = {
            "id": self.candidate_id,
            "engine": self.engine,
            "severity": self.severity,
            "native_type": self.evidence.get("native_type", f"{self.engine}.UnknownNativeType"),
            "match": {"engine": self.engine, "fields": self.match_fields},
            "kwargs_positive": self.kwargs_positive,
            "kwargs_negative": self.kwargs_negative,
            "expected_outcome": self.expected_outcome,
            "added_by": added_by,
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        if self.message_template is not None:
            entry["message_template"] = self.message_template
        if self.observed_messages_regex is not None:
            entry["expected_outcome"] = {
                **self.expected_outcome,
                "observed_messages_regex": [self.observed_messages_regex],
            }
        if not self.verified:
            entry["review_notes"] = "UNVERIFIED — dedup was off; confirm distinct collapsed pair."
        return entry


@dataclass
class GapReport:
    """Combined scan output — one per invocation of :func:`generate_report`."""

    candidates: list[RuleCandidate] = field(default_factory=list)
    h3_groups: list[PostRunH3Group] = field(default_factory=list)
    scanned_sidecars: int = 0
    scanned_observations: int = 0
    dedup_off_studies: int = 0
    sources_run: tuple[str, ...] = ()

    @property
    def summary_counts(self) -> dict[str, int]:
        """Return a quick count of candidates keyed by confidence for CLI display."""
        out = {"high": 0, "medium": 0, "low": 0}
        for c in self.candidates:
            out[c.confidence] = out.get(c.confidence, 0) + 1
        return out


# ---------------------------------------------------------------------------
# Sidecar + JSONL IO
# ---------------------------------------------------------------------------


def load_sidecars(results_dir: Path) -> list[dict[str, Any]]:
    """Walk ``results_dir`` recursively for ``config.json`` sidecars with H3 payload.

    Only sidecars that carry ``h3_hash`` are returned — pre-50.3a results
    don't have the field and aren't useful for gap detection.
    """
    sidecars: list[dict[str, Any]] = []
    if not results_dir.exists():
        return sidecars
    for path in results_dir.rglob("config.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable sidecar %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        # Skip sidecars without H3 — only the post-50.3a feedback-loop
        # sidecars are useful to scan.
        if "h3_hash" not in data:
            continue
        data.setdefault("_sidecar_path", str(path))
        sidecars.append(data)
    return sidecars


def load_runtime_observations(cache_path: Path) -> list[dict[str, Any]]:
    """Parse a ``runtime_observations.jsonl`` file into dicts.

    Malformed lines are logged and skipped — the cache is user-visible and
    tolerating partial corruption beats aborting the whole report.
    """
    observations: list[dict[str, Any]] = []
    if not cache_path.exists():
        return observations
    try:
        raw = cache_path.read_text().splitlines()
    except OSError as exc:
        logger.warning("Could not read %s: %s", cache_path, exc)
        return observations
    for line_no, raw_line in enumerate(raw, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            observations.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d in %s: %s", line_no, cache_path, exc)
    return observations


# ---------------------------------------------------------------------------
# Dedup-mode resolution
# ---------------------------------------------------------------------------


def resolve_equivalence_groups(results_dir: Path) -> list[EquivalenceGroups]:
    """Find and load every ``equivalence_groups.json`` under ``results_dir``.

    Each study bundle has its own manifest; returns one :class:`EquivalenceGroups`
    per discovered file. Missing files simply contribute no manifest and the
    caller falls back to per-sidecar heuristics.
    """
    out: list[EquivalenceGroups] = []
    if not results_dir.exists():
        return out
    for path in results_dir.rglob("equivalence_groups.json"):
        try:
            out.append(load_equivalence_groups(path))
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Skipping malformed equivalence_groups.json %s: %s", path, exc)
    return out


# ---------------------------------------------------------------------------
# Source #2: H3-collision scan
# ---------------------------------------------------------------------------


def scan_h3_collisions(
    sidecars: list[dict[str, Any]],
    manifests: list[EquivalenceGroups],
    *,
    engine_filter: str | None = None,
) -> tuple[list[RuleCandidate], list[PostRunH3Group], int]:
    """Detect canonicaliser gaps from H3-collisions across sidecars.

    Returns ``(candidates, groups, dedup_off_study_count)``.

    Dedup-mode resolution: a sidecar whose study manifest has ``dedup_mode="off"``
    produces candidates but they're marked unverified and excluded from
    auto-PR. Sidecars without a resolvable manifest default to verified mode
    (the sweep-dedup.md §4.1 invariant is the conservative default).
    """
    # Walk parents of each sidecar for the nearest equivalence_groups.json.
    sidecar_dedup: dict[str, str] = {}
    for sc in sidecars:
        p = sc.get("_sidecar_path")
        if not p:
            continue
        sidecar_dedup[str(p)] = _dedup_mode_for_path(Path(p))

    # Apply engine filter.
    if engine_filter:
        filtered = [s for s in sidecars if s.get("engine") == engine_filter]
    else:
        filtered = list(sidecars)

    groups = find_h3_groups(filtered)
    gap_groups = [g for g in groups if g.gap_detected]

    dedup_off_count = 0
    candidates: list[RuleCandidate] = []

    # Build index from experiment_id -> sidecar for diff-based candidate synthesis.
    by_exp: dict[str, dict[str, Any]] = {
        str(sc.get("experiment_id")): sc for sc in filtered if sc.get("experiment_id")
    }

    seen: set[str] = set()
    for group in gap_groups:
        members = [by_exp[exp_id] for exp_id in group.member_experiment_ids if exp_id in by_exp]
        if len(members) < 2:
            continue
        # Dedup-mode check: any member from an off-mode study flips the whole group.
        modes = {sidecar_dedup.get(str(m.get("_sidecar_path", "")), "h1") for m in members}
        verified = "off" not in modes
        if not verified:
            dedup_off_count += 1

        candidate = _h3_group_to_candidate(group, members, verified=verified)
        if candidate is None:
            continue
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        candidates.append(candidate)

    return candidates, groups, dedup_off_count


def _dedup_mode_for_path(sidecar_path: Path) -> str:
    """Best-effort resolution of dedup mode for a sidecar.

    Walks parents looking for a directory containing ``equivalence_groups.json``;
    when found, returns that manifest's ``dedup_mode``. Falls back to ``"h1"``
    (the conservative default) when no manifest covers the sidecar.
    """
    for ancestor in [sidecar_path, *sidecar_path.parents]:
        eq_path = ancestor / "equivalence_groups.json"
        if eq_path.exists():
            try:
                loaded = load_equivalence_groups(eq_path)
                return loaded.dedup_mode
            except (OSError, json.JSONDecodeError, ValueError):
                return "h1"
    return "h1"


def _h3_group_to_candidate(
    group: PostRunH3Group,
    members: list[dict[str, Any]],
    *,
    verified: bool,
) -> RuleCandidate | None:
    """Synthesise a candidate rule from an H3-collision group via field-diff.

    Algorithm (§4.9.2 MVP path — simple field-diff, ``needs-generalisation-review``
    applied uniformly):

    1. Compare ``effective_engine_params`` + ``effective_sampling_params``
       of the first two members. Fields identical across all members form the
       *shared effective state* — these go into ``expected_outcome`` as the
       observed normalisation target.
    2. Fields that differ across members form the *trigger* predicate —
       the canonicaliser should have collapsed them and didn't. ``present: true``
       is the safest predicate since we know the library collapsed the field.
    3. Confidence: ``high`` only when exactly one declared-kwargs field
       distinguishes all members; otherwise ``medium`` (multi-field) or
       ``low`` (no clean trigger predicate).
    """
    if len(members) < 2:
        return None

    effective_first = _effective_state(members[0])
    # Fields shared across ALL members' effective state = the shared H3 value.
    shared_effective: dict[str, Any] = {}
    for key, value in effective_first.items():
        if all(_effective_state(m).get(key) == value for m in members[1:]):
            shared_effective[key] = value

    declared_first = _declared_kwargs(members[0])
    diff_fields: dict[str, list[Any]] = {}
    for key in declared_first.keys() | {k for m in members[1:] for k in _declared_kwargs(m)}:
        values = [_declared_kwargs(m).get(key) for m in members]
        unique = {_hashable(v) for v in values}
        if len(unique) > 1:
            diff_fields[key] = values

    if len(diff_fields) == 1:
        confidence = "high"
    elif len(diff_fields) > 1:
        confidence = "medium"
    else:
        confidence = "low"

    match_fields = {key: {"present": True} for key in diff_fields}
    candidate_id = (
        f"{group.engine}_h3_collision_"
        f"{group.h3_hash.replace('sha256:', '')[:12] if group.h3_hash else 'unknown'}"
    )

    evidence = {
        "h3_hash": group.h3_hash,
        "library_version": group.library_version,
        "member_experiment_ids": list(group.member_experiment_ids),
        "member_h1_hashes": list(group.member_h1_hashes),
        "shared_effective_state": shared_effective,
        "declared_value_variants": {k: [str(v) for v in vs] for k, vs in diff_fields.items()},
        "needs_generalisation_review": True,
        "native_type": _infer_native_type(group.engine, members),
    }

    expected_outcome: dict[str, Any] = {
        "outcome": "dormant",
        "emission_channel": "none",
        "normalised_fields": sorted(shared_effective.keys()),
    }

    return RuleCandidate(
        candidate_id=candidate_id,
        engine=group.engine,
        library_version=group.library_version,
        source="h3-collisions",
        severity="dormant_silent",
        confidence=confidence,
        match_fields=match_fields,
        kwargs_positive=_first_declared_kwargs(members[0]),
        kwargs_negative=_first_declared_kwargs(members[1]),
        expected_outcome=expected_outcome,
        message_template=None,
        observed_messages_regex=None,
        evidence=evidence,
        verified=verified,
    )


def _effective_state(sidecar: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in ("effective_engine_params", "effective_sampling_params"):
        val = sidecar.get(key)
        if isinstance(val, dict):
            for k, v in val.items():
                state[f"{key}.{k}"] = v
    return state


def _declared_kwargs(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction of the declared kwargs from a sidecar.

    Real sidecars carry ``effective_*_params`` but not declared kwargs; the
    generic candidate-synthesis path prefers H1/H3 hashes as the axis of
    comparison. For field-diff candidates we approximate declared kwargs by
    using the effective state (adequate for the MVP field-diff inference
    which §4.9.2 flags as ``needs-generalisation-review`` anyway).
    """
    declared = sidecar.get("declared_kwargs")
    if isinstance(declared, dict):
        return dict(declared)
    return _effective_state(sidecar)


def _first_declared_kwargs(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Return a flattened kwargs view for rendering in the proposed YAML."""
    flat: dict[str, Any] = {}
    for fq_key, value in _declared_kwargs(sidecar).items():
        leaf = fq_key.rsplit(".", 1)[-1]
        flat[leaf] = value
    return flat


def _infer_native_type(engine: str, members: Iterable[dict[str, Any]]) -> str:
    for m in members:
        for key in ("native_type_sampling", "native_type_engine"):
            val = m.get(key)
            if isinstance(val, str) and val:
                return val
    return {
        "transformers": "transformers.GenerationConfig",
        "vllm": "vllm.SamplingParams",
        "tensorrt": "tensorrt_llm.LlmArgs",
    }.get(engine, f"{engine}.UnknownNativeType")


def _hashable(value: Any) -> Any:
    """Convert ``value`` into something usable in a set (lists → tuples, dicts → sorted items)."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_hashable(v) for v in value))
    return value


# ---------------------------------------------------------------------------
# Source #1: runtime-warning scan
# ---------------------------------------------------------------------------


def scan_runtime_warnings(
    observations: list[dict[str, Any]],
    loader: VendoredRulesLoader | None = None,
    *,
    engine_filter: str | None = None,
) -> list[RuleCandidate]:
    """Normalise each emission, match against the corpus, emit unmatched candidates.

    Runtime exceptions (outcome == ``"exception"``) are classified as
    ``severity: error`` candidates — §4.7's "third feedback class" extension.
    """
    vendor = loader or VendoredRulesLoader()
    candidates: list[RuleCandidate] = []
    seen: set[str] = set()

    # Per-observation grouping by normalised template + engine.
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        engine = str(obs.get("engine", ""))
        if engine_filter and engine != engine_filter:
            continue
        messages = _extract_emission_messages(obs)
        severity_hint = "error" if obs.get("outcome") == "exception" else "warn"
        for raw_msg, log_level in messages:
            norm = normalise(raw_msg)
            key = (engine, severity_hint, norm.template, log_level)
            buckets[key].append(
                {
                    "observation": obs,
                    "raw_message": raw_msg,
                    "normalised": norm,
                    "log_level": log_level,
                }
            )

    for (engine, severity_hint, template, _log_level), hits in buckets.items():
        if not template.strip():
            continue
        corpus_rules: tuple[Rule, ...]
        try:
            corpus_rules = vendor.load_rules(engine).rules
        except FileNotFoundError:
            corpus_rules = ()
        if _template_matches_corpus(template, corpus_rules):
            continue

        primary = hits[0]
        primary_norm: NormalisedMessage = primary["normalised"]
        library_version = str(primary["observation"].get("library_version", ""))
        candidate_id = _candidate_id_from_template(engine, template)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        severity = "error" if severity_hint == "error" else "warn"
        expected_outcome = {
            "outcome": severity_hint,
            "emission_channel": (
                "runtime_exception" if severity_hint == "error" else "python_warning"
            ),
        }
        candidates.append(
            RuleCandidate(
                candidate_id=candidate_id,
                engine=engine,
                library_version=library_version,
                source="runtime-warnings",
                severity=severity,
                confidence="medium",
                match_fields={},
                kwargs_positive={},
                kwargs_negative={},
                expected_outcome=expected_outcome,
                message_template=primary_norm.template,
                observed_messages_regex=primary_norm.match_regex,
                evidence={
                    "occurrence_count": len(hits),
                    "example_raw_message": primary_norm.original,
                    "config_hashes": sorted(
                        {str(h["observation"].get("config_hash", "")) for h in hits}
                    ),
                    "native_type": _infer_native_type(engine, []),
                },
                verified=True,
            )
        )
    return candidates


def _extract_emission_messages(observation: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(message, log_level)`` pairs for warnings + logger records + exceptions."""
    out: list[tuple[str, str]] = []
    for w in observation.get("warnings", []) or []:
        if isinstance(w, str):
            out.append((w, "WARNING"))
    for rec in observation.get("logger_records", []) or []:
        if not isinstance(rec, dict):
            continue
        msg = rec.get("message")
        if not isinstance(msg, str):
            continue
        out.append((msg, str(rec.get("level", "WARNING"))))
    exc = observation.get("exception")
    if isinstance(exc, dict) and isinstance(exc.get("message"), str):
        out.append((f"{exc.get('type', 'Exception')}: {exc['message']}", "ERROR"))
    return out


def _template_matches_corpus(template: str, rules: Iterable[Rule]) -> bool:
    import re

    for rule in rules:
        observed_regexes = rule.expected_outcome.get("observed_messages_regex") or []
        for pattern in observed_regexes:
            try:
                if re.search(pattern, template):
                    return True
            except re.error:
                continue
        # Strict: normalised template must exactly equal the normalised corpus
        # template. ``in`` is rejected as too fuzzy.
        template_msg = rule.message_template
        if (
            isinstance(template_msg, str)
            and template_msg
            and normalise(template_msg).template == template
        ):
            return True
    return False


def _candidate_id_from_template(engine: str, template: str) -> str:
    import hashlib

    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
    return f"{engine}_runtime_warning_{digest}"


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def generate_report(
    *,
    source: str = "both",
    results_dir: Path | None = None,
    cache_path: Path | None = None,
    engine: str | None = None,
    loader: VendoredRulesLoader | None = None,
) -> GapReport:
    """Drive both scans and return the combined :class:`GapReport`.

    ``source`` is one of ``"runtime-warnings"``, ``"h3-collisions"``, or
    ``"both"``. When the chosen source requires inputs that aren't present
    (no results dir, empty cache), the corresponding scan returns an empty
    list silently — this preserves the CLI's "reports what it finds" contract
    for users who haven't run anything yet.
    """
    report = GapReport(sources_run=(source,))

    if source in ("h3-collisions", "both") and results_dir is not None:
        sidecars = load_sidecars(results_dir)
        manifests = resolve_equivalence_groups(results_dir)
        candidates, groups, dedup_off = scan_h3_collisions(
            sidecars, manifests, engine_filter=engine
        )
        report.scanned_sidecars = len(sidecars)
        report.h3_groups = groups
        report.dedup_off_studies = dedup_off
        report.candidates.extend(candidates)

    if source in ("runtime-warnings", "both") and cache_path is not None:
        observations = load_runtime_observations(cache_path)
        report.scanned_observations = len(observations)
        report.candidates.extend(
            scan_runtime_warnings(observations, loader=loader, engine_filter=engine)
        )

    return report


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------


def render_candidates_yaml(candidates: list[RuleCandidate]) -> str:
    """Render candidates as a YAML ``rules:`` block ready to concatenate into the corpus."""
    import yaml

    entries = [c.to_yaml_dict() for c in candidates]
    if not entries:
        return "rules: []\n"
    return yaml.safe_dump({"rules": entries}, sort_keys=False, default_flow_style=False)
