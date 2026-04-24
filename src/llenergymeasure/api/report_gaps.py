"""``llem report-gaps`` — feedback-loop proposer for the rules corpus.

Reads ``runtime_observations.jsonl`` caches emitted by
:mod:`llenergymeasure.study.runtime_observations` across one or more study
directories, groups captured warnings / log records by their normalised
message template, partitions configs into *fired* (A) and *not fired* (B),
and runs a small predicate-inference pass to propose corpus rules for
templates that the existing rules corpus does not already match.

Design notes:

- **No corpus mutation.** We emit YAML *fragments* to ``--out PATH`` with a
  loud banner comment for a maintainer to splice into the live corpus. The
  live ``configs/validation_rules/{engine}.yaml`` is never touched.
- **Severity is mechanical.** ``severity: warn`` from log-channel emissions
  and ``severity: error`` when ``--include-exceptions`` is set and the
  record is an exception. ``walker_confidence`` is always ``low`` and we
  always flag ``needs_generalisation_review`` when the predicate was narrow
  or missing — the final call belongs to the human reviewer.
- **Round-trip safe.** Every emitted fragment parses through
  :func:`llenergymeasure.config.vendored_rules.loader._parse_rule` without
  error. Required-but-unknown fields (``native_type``,
  ``walker_source.{path,method,line_at_scan}``, ``references``) get
  minimally-acceptable placeholder values plus ``# TODO: human`` markers so
  reviewers know what to fill in.
- **Sentinel filtering.** Records with ``outcome in {"subprocess_died",
  "exception"}`` do not prove "the rule did not fire" — they prove "we don't
  know". They are excluded from the B (not-fired) partition. When
  ``include_exceptions=False`` (default), exception records are also
  excluded from the A partition so the proposer only acts on recorded
  warnings / log-channel emissions.

Source of config kwargs for predicate inference:

    Per-experiment ``_resolution.json`` sidecars (written by the runner
    alongside ``result.json``). Each sidecar maps dotted field paths to
    their effective values; we flatten these into ``dict[str, Any]`` per
    config_hash so the inference algorithm can look for arity-1, arity-2,
    arity-3 distinguishing value-tuples between fired and not-fired configs.

    Experiments whose subprocess died before the sidecar was written have
    no kwargs — they are skipped at partition time regardless of outcome.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import yaml

from llenergymeasure.config.ssot import Engine
from llenergymeasure.config.vendored_rules import (
    VendoredRules,
    VendoredRulesLoader,
)
from llenergymeasure.study.message_normalise import build_template_regex, normalise
from llenergymeasure.study.runtime_observations import RUNTIME_OBSERVATIONS_FILENAME

__all__ = [
    "GapProposal",
    "ReportGapsError",
    "SourceChannel",
    "SupportedEngine",
    "find_runtime_gaps",
    "load_rules_corpus",
    "render_yaml_fragment",
]

logger = logging.getLogger(__name__)

SupportedEngine = Literal["transformers", "vllm", "tensorrt"]
"""Engine literal re-exported for the CLI layer (cli-boundary contract)."""

SourceChannel = Literal["warnings_warn", "logger_warning", "runtime_exception"]
"""Which JSONL-record channel the emission came from.

Kept as a narrow literal so downstream renderers can switch on it without a
try/except. Logger levels INFO/DEBUG are not included — the capture wrapper
only emits log records at WARNING+ and the corpus only cares about
user-visible emissions.
"""


class ReportGapsError(Exception):
    """Raised when ``find_runtime_gaps`` or ``render_yaml_fragment`` fails.

    Thin wrapper so the CLI can surface a helpful message without leaking
    implementation-specific exception types (IO, YAML) to the user.
    """


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapProposal:
    """One proposed corpus rule for a single unmatched normalised template.

    Carries everything ``render_yaml_fragment`` needs to produce a YAML
    fragment that parses through :func:`_parse_rule`. The struct itself is
    pure data — no IO, no side effects.
    """

    normalised_template: str
    source_channel: SourceChannel
    engine: SupportedEngine
    library_version: str
    match_fields: dict[str, Any] | None
    evidence_field_value_distribution: dict[str, dict[str, list[str]]]
    fired_count: int
    not_fired_count: int
    representative_message: str
    walker_confidence: Literal["low"]
    needs_generalisation_review: bool
    severity: Literal["warn", "error"]
    representative_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal record shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Emission:
    """One template-fired-by-a-config observation (in-memory only)."""

    template: str
    representative_message: str
    channel: SourceChannel
    config_hash: str
    engine: SupportedEngine
    library_version: str


@dataclass(frozen=True)
class _Record:
    """Parsed JSONL record with resolved kwargs and derived emission list."""

    config_hash: str
    engine: SupportedEngine
    library_version: str
    outcome: str
    emissions: list[_Emission]
    kwargs: dict[str, Any] | None


_ALLOWED_ENGINES: frozenset[str] = frozenset(e.value for e in Engine)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_rules_corpus(
    engines: list[SupportedEngine] | None = None,
    loader: VendoredRulesLoader | None = None,
) -> dict[str, VendoredRules]:
    """Load the rules corpus for each engine we may emit proposals against.

    Engines missing a YAML corpus are skipped silently — a user scanning
    only transformers studies shouldn't fail because there's no
    ``tensorrt.yaml`` yet.
    """
    loader = loader or VendoredRulesLoader()
    wanted = list(engines) if engines is not None else [e.value for e in Engine]
    out: dict[str, VendoredRules] = {}
    for engine in wanted:
        try:
            out[engine] = loader.load_rules(engine)
        except FileNotFoundError:
            logger.debug("No rules corpus for engine=%s; skipping match lookup.", engine)
    return out


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------


def find_runtime_gaps(
    study_dirs: list[Path],
    rules_corpus: dict[str, VendoredRules] | None = None,
    engine: str | None = None,
    include_exceptions: bool = False,
) -> list[GapProposal]:
    """Scan one or more study directories and return unmatched-template proposals.

    Args:
        study_dirs: Study directories containing
            ``runtime_observations.jsonl``. Missing JSONL files are ignored
            with a log message — a study that produced no observations is a
            valid (if useless) input.
        rules_corpus: Pre-loaded per-engine corpus. When ``None`` the default
            loader is used; engines without a YAML corpus are treated as
            "no existing rules" and every template falls through to a
            proposal.
        engine: If set, only records matching this engine contribute
            proposals. Filter applies before partitioning.
        include_exceptions: If ``True``, records with
            ``outcome == "exception"`` also produce candidate emissions
            (marked ``source_channel="runtime_exception"`` and
            ``severity="error"``). Default ``False`` — exceptions are a
            separate feedback channel with different review semantics and
            the design deliberately keeps this opt-in.

    Returns:
        One :class:`GapProposal` per unique unmatched normalised template.
        Order is stable: sorted by (engine, normalised_template) so
        diffing two runs is meaningful.
    """
    if not study_dirs:
        raise ReportGapsError("No study directories provided. Pass at least one --study-dir.")

    engine_filter = engine
    if engine_filter is not None and engine_filter not in _ALLOWED_ENGINES:
        raise ReportGapsError(
            f"Unsupported engine filter: {engine_filter!r}. "
            f"Expected one of: {sorted(_ALLOWED_ENGINES)}."
        )

    records: list[_Record] = []
    for study_dir in study_dirs:
        records.extend(_load_study(study_dir, engine_filter, include_exceptions))

    if not records:
        return []

    corpus = rules_corpus if rules_corpus is not None else load_rules_corpus()

    # Build unique (engine, template) → emissions map.
    # Key on engine too because the same string emitted by two engines is
    # conceptually two different rules (different native types, different
    # severity conventions).
    emissions_by_key: dict[tuple[str, str], list[_Emission]] = {}
    representative_by_key: dict[tuple[str, str], _Emission] = {}

    for rec in records:
        for emission in rec.emissions:
            key = (emission.engine, emission.template)
            emissions_by_key.setdefault(key, []).append(emission)
            representative_by_key.setdefault(key, emission)

    # Partition records by engine once so the per-template loop stays O(1) per record.
    records_by_engine: dict[str, list[_Record]] = {}
    for rec in records:
        records_by_engine.setdefault(rec.engine, []).append(rec)

    proposals: list[GapProposal] = []
    for (eng, template), emissions in sorted(emissions_by_key.items()):
        corpus_entry = corpus.get(eng)
        if corpus_entry is not None and _template_matched_by_corpus(template, corpus_entry):
            continue

        fired_hashes = {e.config_hash for e in emissions}

        # A partition: records whose config fired this template.
        # B partition: records for configs that did NOT fire this template.
        # Excluded from B regardless of include_exceptions:
        #   - subprocess_died: worker never reached the emission site.
        #   - exception: we don't know whether the template would have been
        #     emitted had the exception not fired.
        # Excluded from A unless include_exceptions: exception records are
        # not suitable evidence for warning-channel proposals.
        fired_configs: list[dict[str, Any]] = []
        not_fired_configs: list[dict[str, Any]] = []

        for rec in records_by_engine.get(eng, []):
            if rec.kwargs is None:
                continue
            if rec.config_hash in fired_hashes:
                fired_configs.append(rec.kwargs)
            else:
                if rec.outcome in {"subprocess_died", "exception"}:
                    continue
                not_fired_configs.append(rec.kwargs)

        match_fields = _infer_predicate(fired_configs, not_fired_configs)
        evidence = _field_value_distribution(fired_configs, not_fired_configs)

        rep = representative_by_key[(eng, template)]
        channel = rep.channel
        severity: Literal["warn", "error"] = "error" if channel == "runtime_exception" else "warn"
        needs_review = match_fields is None or len(match_fields) >= 2

        # Prefer the first fired config's kwargs as kwargs_positive evidence
        # so reviewers see a concrete reproducer.
        representative_kwargs: dict[str, Any] = fired_configs[0] if fired_configs else {}

        proposals.append(
            GapProposal(
                normalised_template=template,
                source_channel=channel,
                engine=eng,  # type: ignore[arg-type]
                library_version=rep.library_version,
                match_fields=match_fields,
                evidence_field_value_distribution=evidence,
                fired_count=len(fired_configs),
                not_fired_count=len(not_fired_configs),
                representative_message=rep.representative_message,
                walker_confidence="low",
                needs_generalisation_review=needs_review,
                severity=severity,
                representative_kwargs=representative_kwargs,
            )
        )

    return proposals


# ---------------------------------------------------------------------------
# JSONL loading + kwargs resolution
# ---------------------------------------------------------------------------


def _load_study(
    study_dir: Path,
    engine_filter: str | None,
    include_exceptions: bool,
) -> list[_Record]:
    """Return parsed records from one study dir; tolerant of missing files."""
    jsonl_path = Path(study_dir) / RUNTIME_OBSERVATIONS_FILENAME
    if not jsonl_path.exists():
        logger.info("No %s in %s; skipping.", RUNTIME_OBSERVATIONS_FILENAME, study_dir)
        return []

    kwargs_by_hash = _load_kwargs_by_hash(Path(study_dir))

    records: list[_Record] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSONL line %d in %s.",
                    line_no,
                    jsonl_path,
                )
                continue
            rec = _parse_record(raw, engine_filter, include_exceptions, kwargs_by_hash)
            if rec is not None:
                records.append(rec)
    return records


def _parse_record(
    raw: dict[str, Any],
    engine_filter: str | None,
    include_exceptions: bool,
    kwargs_by_hash: dict[str, dict[str, Any]],
) -> _Record | None:
    engine_str = str(raw.get("engine", ""))
    if engine_str not in _ALLOWED_ENGINES:
        return None
    if engine_filter is not None and engine_str != engine_filter:
        return None

    outcome = str(raw.get("outcome", ""))
    config_hash = str(raw.get("config_hash", ""))
    library_version = str(raw.get("library_version", ""))

    emissions: list[_Emission] = []

    for warn in raw.get("warnings") or []:
        template = str(warn.get("message_template") or "")
        if not template:
            continue
        emissions.append(
            _Emission(
                template=template,
                representative_message=str(warn.get("message") or template),
                channel="warnings_warn",
                config_hash=config_hash,
                engine=engine_str,  # type: ignore[arg-type]
                library_version=library_version,
            )
        )

    for rec in raw.get("logger_records") or []:
        template = str(rec.get("message_template") or "")
        if not template:
            continue
        emissions.append(
            _Emission(
                template=template,
                representative_message=str(rec.get("message") or template),
                channel="logger_warning",
                config_hash=config_hash,
                engine=engine_str,  # type: ignore[arg-type]
                library_version=library_version,
            )
        )

    if include_exceptions and outcome == "exception":
        exc = raw.get("exception") or {}
        template = str(exc.get("message_template") or "")
        if template:
            emissions.append(
                _Emission(
                    template=template,
                    representative_message=str(exc.get("message") or template),
                    channel="runtime_exception",
                    config_hash=config_hash,
                    engine=engine_str,  # type: ignore[arg-type]
                    library_version=library_version,
                )
            )

    return _Record(
        config_hash=config_hash,
        engine=engine_str,  # type: ignore[arg-type]
        library_version=library_version,
        outcome=outcome,
        emissions=emissions,
        kwargs=kwargs_by_hash.get(config_hash),
    )


def _load_kwargs_by_hash(study_dir: Path) -> dict[str, dict[str, Any]]:
    """Walk experiment subdirs, load ``_resolution.json`` sidecars.

    Directory name ends with the 8-char config hash prefix. We key on the
    *full* hash in the JSONL (not the 8-char prefix) so we need to derive a
    reverse map. Two strategies:

    1. Every ``_resolution.json`` contains the effective config — we hash it
       ourselves to recover the full hash. But that re-implements the
       canonical hasher and couples us to its internals.
    2. Build a prefix-keyed map and accept that collisions within a single
       study would conflate configs. On an 8-char hex prefix that's a ~10^-9
       event per pair at study scale; negligible.

    We use (2) and prefer it — the resolution log's effective dict IS the
    flat, default-stripped view we want for predicate inference. Collision
    risk is documented for reviewers.
    """
    if not study_dir.exists() or not study_dir.is_dir():
        return {}

    by_prefix: dict[str, dict[str, Any]] = {}
    for entry in sorted(study_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Dir format: [{NNN}_]c{cycle}_{slug}_{hash8}
        # Trailing token is always an 8-char hex hash.
        parts = entry.name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        hash_prefix = parts[1]
        if len(hash_prefix) != 8 or not all(c in "0123456789abcdef" for c in hash_prefix):
            continue
        resolution_path = entry / "_resolution.json"
        if not resolution_path.exists():
            continue
        try:
            payload = json.loads(resolution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overrides = payload.get("overrides") or {}
        if not isinstance(overrides, dict):
            continue
        flat: dict[str, Any] = {}
        for key, value in overrides.items():
            if isinstance(value, dict) and "effective" in value:
                flat[str(key)] = value["effective"]
        by_prefix.setdefault(hash_prefix, flat)

    # Expand to a full-hash lookup: any hash whose first 8 chars match.
    # The JSONL record carries the full hash; its prefix is the key we built.
    # Callers look up by full hash, so wrap with a proxy dict via __missing__?
    # Simpler: return a dict whose keys are 8-char prefixes and let the caller
    # slice. But _parse_record passes the full hash. Return a dict keyed on
    # the full hash by propagating the prefix entry for every hash we've seen
    # — but we don't know all hashes yet. Resolve this lazily with a wrapper.
    return _PrefixLookupDict(by_prefix)


class _PrefixLookupDict(dict):  # type: ignore[type-arg]
    """Dict that resolves lookups via a config-hash-prefix-keyed backing map.

    Keeps the caller's ``kwargs_by_hash[full_hash]`` API while the filesystem
    layout only exposes 8-char prefixes.
    """

    def __init__(self, prefix_map: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self._prefix_map = prefix_map

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if not isinstance(key, str) or len(key) < 8:
            return default
        prefix = key[:8]
        return self._prefix_map.get(prefix, default)

    def __getitem__(self, key: str) -> dict[str, Any]:  # type: ignore[override]
        if len(key) < 8:
            raise KeyError(key)
        return self._prefix_map[key[:8]]

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if not isinstance(key, str) or len(key) < 8:
            return False
        return key[:8] in self._prefix_map


# ---------------------------------------------------------------------------
# Corpus template match
# ---------------------------------------------------------------------------


def _template_matched_by_corpus(template: str, corpus: VendoredRules) -> bool:
    """Return True if the corpus already has a rule whose expected_outcome captures ``template``.

    Match strategies (both supported for forward compat):

    - ``observed_messages_regex``: list of anchored regexes written at rule
      authoring time. Preferred because it captures value-specific
      placeholders deliberately.
    - ``observed_messages``: list of concrete emitted messages from the
      vendored JSON overlay. We normalise each entry and compare templates
      directly — a rule whose observed message normalises to the same
      template is definitely firing for the same reason.
    """
    for rule in corpus.rules:
        outcome = rule.expected_outcome
        regexes = outcome.get("observed_messages_regex")
        if isinstance(regexes, list):
            for pat in regexes:
                try:
                    if re.match(str(pat), template):
                        return True
                except re.error:
                    continue
        observed = outcome.get("observed_messages")
        if isinstance(observed, list):
            for sample in observed:
                if not isinstance(sample, str):
                    continue
                sample_template = normalise(sample).template
                if sample_template == template:
                    return True
    return False


# ---------------------------------------------------------------------------
# Predicate inference (arity 1 → 2 → 3 + present:true fallback)
# ---------------------------------------------------------------------------


def _infer_predicate(
    fired: list[dict[str, Any]],
    not_fired: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the smallest field-value dict that distinguishes A from B, or None.

    The search tries arity 1, then 2, then 3 — the first arity whose tuple
    is *uniformly* shared by every config in A and *not* shared by any
    config in B wins.

    ``present:true`` fallback: if no arity-1..3 equality predicate works,
    look for a single field that is set (non-None) in every fired config
    and not set (missing or None) in every not-fired config. This catches
    "any sampling_params present triggers the warning" shapes that
    equality-on-value never sees.
    """
    if not fired:
        return None

    fields = sorted({k for c in fired + not_fired for k in c})
    if not fields:
        return None

    for arity in range(1, 4):
        if arity > len(fields):
            break
        for fset in combinations(fields, arity):
            a_tuples = {tuple(c.get(f) for f in fset) for c in fired}
            if len(a_tuples) != 1:
                continue
            value_tuple = next(iter(a_tuples))
            if any(
                all(b.get(f) == v for f, v in zip(fset, value_tuple, strict=False))
                for b in not_fired
            ):
                continue
            return dict(zip(fset, value_tuple, strict=False))

    # present:true fallback — single-field presence predicate.
    for fname in fields:
        if all(c.get(fname) is not None for c in fired) and all(
            b.get(fname) is None for b in not_fired
        ):
            return {fname: {"present": True}}

    return None


def _field_value_distribution(
    fired: list[dict[str, Any]],
    not_fired: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Return per-field string-ified value sets for the A and B partitions.

    Useful to reviewers when the inferred predicate is missing or narrow:
    the distribution tells you which fields actually vary across the A
    partition and which are constant.
    """
    out: dict[str, dict[str, list[str]]] = {}
    fields = sorted({k for c in fired + not_fired for k in c})
    for f in fields:
        out[f] = {
            "fired": sorted({str(c.get(f)) for c in fired}),
            "not_fired": sorted({str(c.get(f)) for c in not_fired}),
        }
    return out


# ---------------------------------------------------------------------------
# YAML renderer
# ---------------------------------------------------------------------------


_BANNER = (
    "# Rule fragment proposed by 'llem report-gaps'. Review and APPEND to\n"
    "# configs/validation_rules/{engine}.yaml under the 'rules:' key.\n"
    "# ----------------------------------------------------------------------\n"
    "# walker_confidence: low — always, for runtime-derived rules.\n"
    "# needs_generalisation_review: set when the predicate is narrow or\n"
    "# missing. Reviewers must confirm severity, native_type, and predicate\n"
    "# generalisation before merging.\n"
)


def render_yaml_fragment(proposal: GapProposal) -> str:
    """Render one :class:`GapProposal` as a YAML document.

    The output always parses through
    :func:`llenergymeasure.config.vendored_rules.loader._parse_rule` — stub
    fields carry placeholder values that are enum-valid even if
    semantically vacuous, so the round-trip test can assert successful
    parse without the stubs being mistaken for production data.
    """
    template = proposal.normalised_template
    rule_id = _proposed_rule_id(proposal)
    match_regex = build_template_regex(template)
    match_fields: dict[str, Any] = dict(proposal.match_fields) if proposal.match_fields else {}
    # Corpus convention: match.fields is always a dict; empty dict means
    # "predicate not inferred — reviewer to fill in".

    emission_channel = _emission_channel(proposal.source_channel)
    outcome_value = "error" if proposal.severity == "error" else "warn"

    proposal_doc: dict[str, Any] = {
        "id": rule_id,
        "engine": proposal.engine,
        "library": proposal.engine,
        "rule_under_test": (
            "(runtime-derived) Library emitted normalised template; reviewer to confirm semantic."
        ),
        "severity": proposal.severity,
        # Placeholder enum-valid native_type — reviewer to replace.
        # Kept here (rather than omitted) because `_parse_rule` treats
        # native_type as required.
        "native_type": f"{proposal.engine}.<TODO: human — set concrete native type>",
        "walker_source": {
            "path": "<TODO: human — runtime-derived; no AST source>",
            "method": "<TODO: human — no AST source>",
            "line_at_scan": 0,
            "walker_confidence": proposal.walker_confidence,
        },
        "match": {
            "engine": proposal.engine,
            "fields": match_fields,
        },
        "kwargs_positive": dict(proposal.representative_kwargs),
        # kwargs_negative cannot be inferred from observations — reviewer
        # hand-authors a known-clean config. Empty dict is accepted by the
        # loader; we include it so the YAML round-trips.
        "kwargs_negative": {},
        "expected_outcome": {
            "outcome": outcome_value,
            "emission_channel": emission_channel,
            "normalised_fields": [],
            "observed_messages_regex": [match_regex],
            "observed_messages": [proposal.representative_message],
        },
        "message_template": template,
        "references": [
            (
                f"Observed in {proposal.fired_count} configs; "
                f"absent in {proposal.not_fired_count} configs."
            ),
            f"Representative raw message: {proposal.representative_message!r}",
        ],
        "added_by": "runtime_warning",
        "added_at": "<TODO: human — YYYY-MM-DD>",
        "source_channel": proposal.source_channel,
        "needs_generalisation_review": proposal.needs_generalisation_review,
        "evidence_field_value_distribution": proposal.evidence_field_value_distribution,
    }

    body = yaml.safe_dump(
        proposal_doc,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    return _BANNER + body


def _proposed_rule_id(proposal: GapProposal) -> str:
    """Derive a reviewer-friendly rule id from the template.

    Format: ``{engine}_runtime_{slug}`` where ``slug`` is a short,
    stable-ish slice of the template with non-identifier characters
    replaced. Collisions within a study are possible; reviewers rename on
    merge, so we don't bother with a hash suffix here.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", proposal.normalised_template.lower()).strip("_")
    slug = cleaned[:40] or "unnamed"
    return f"{proposal.engine}_runtime_{slug}"


def _emission_channel(source: SourceChannel) -> str:
    """Map the JSONL source_channel to a corpus-schema emission_channel value."""
    if source == "runtime_exception":
        return "runtime_exception"
    if source == "warnings_warn":
        return "warnings_warn"
    return "logger_warning"
