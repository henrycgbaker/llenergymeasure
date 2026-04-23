"""Load, match, and render validation rules from the YAML corpus.

The corpus at ``configs/validation_rules/{engine}.yaml`` is parsed here into
typed :class:`Rule` entries. Each rule carries a match predicate (operators
defined in :func:`evaluate_predicate`) and a message template. The generic
``@model_validator`` in ``config/models.py`` calls :meth:`Rule.try_match` on
every rule for a given engine and emits error/warn/dormant annotations based
on the rule's severity.

Design mirror: this module parallels :mod:`llenergymeasure.config.schema_loader`
from parameter-discovery — same envelope validation
(:class:`UnsupportedSchemaVersionError` on major-version mismatch), same
per-instance caching for test isolation, same lazy load pattern.

Corpus vs vendored JSON:
  The YAML corpus carries each rule's declared ``expected_outcome``. The
  ``config-rules-refresh`` CI pipeline (see ``scripts/vendor_rules.py``)
  runs every rule through the real library and emits ``{engine}.json``
  alongside the package — this JSON captures observed outcomes. When
  present, the loader overlays the vendored observations onto the corpus
  so downstream consumers see CI-validated truth; absent, the loader
  falls back to the YAML corpus so local development without a vendor
  run still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

SUPPORTED_MAJOR_VERSION = 1
"""Major version the loader knows how to parse.

Raised via :class:`UnsupportedSchemaVersionError` on mismatch; the loader
refuses partial reads to avoid silently accepting a future schema shape.
"""


Severity = Literal["dormant", "warn", "error"]
"""Severity tier a rule match produces at validation time.

- ``dormant`` — the config still runs, but a field is silently ignored or
  coerced by the engine. Observable-but-user-invisible; the rule surfaces
  this.
- ``warn`` — the engine announces a suboptimal setting at construct or
  runtime but still proceeds.
- ``error`` — the engine raises and the config cannot run.
"""

Outcome = Literal[
    "dormant_silent",
    "dormant_announced",
    "warn",
    "error",
    "pass",
]
"""What the engine does when the rule's predicate holds.

- ``dormant_silent`` — the engine silently normalises or ignores
  (observable only via ``extract_effective_params`` post-construction).
- ``dormant_announced`` — the engine writes to a ``minor_issues`` dict /
  logger, but the config still runs.
- ``warn`` — the engine calls ``warnings.warn(...)`` or equivalent.
- ``error`` — the engine raises at construct / validate time.
- ``pass`` — the predicate matched but the engine handles it cleanly;
  used for positive-reference rules.
"""

EmissionChannel = Literal[
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "none",
    "runtime_exception",
]
"""How the engine user-visibly signals the issue.

- ``warnings_warn`` — Python ``warnings.warn(...)``.
- ``logger_warning`` / ``logger_warning_once`` — stdlib logger.
- ``minor_issues_dict`` — an internal dict (e.g. HF's ``minor_issues``)
  whose presence is user-observable via strict-mode raise OR log.
- ``none`` — no user-visible emission (silent coercion or raise).
- ``runtime_exception`` — exception raised at engine construct / runtime.

Canonical rule: ``minor_issues_dict`` alone is an internal signal; if HF
composes the dict then emits via ``logger.warning_once``, the
user-visible channel is ``logger_warning_once``. Corpus authors should
record what users see, not the internal staging buffer.
"""

AddedBy = Literal[
    "ast_walker",
    "introspection",
    "manual_seed",
    "runtime_warning",
    "h3_collision",
]
"""Provenance of a rule in the corpus.

Five discovery paths with distinct trust/verifiability profiles:

- ``ast_walker`` — rule extracted by parsing Python source AST
  (used by vLLM / TRT-LLM walkers; CI can re-derive on library bump).
- ``introspection`` — rule extracted via library-API introspection
  (transformers' ``GenerationConfig.validate(strict=True)`` returning
  structured ``minor_issues`` dict; CI can re-derive on library bump).
- ``manual_seed`` — hand-written by a maintainer for cases the walkers
  can't reach (e.g. BNB type rules; not auto-regenerable).
- ``runtime_warning`` — proposed by the feedback loop from captured
  ``logger.warning_once`` emissions (needs human generalisation before
  landing).
- ``h3_collision`` — proposed by the feedback loop from H3-collision
  canonicaliser-gap detection (needs human generalisation before landing).
"""

VALID_SEVERITY: frozenset[str] = frozenset(get_args(Severity))
VALID_OUTCOME: frozenset[str] = frozenset(get_args(Outcome))
VALID_EMISSION_CHANNEL: frozenset[str] = frozenset(get_args(EmissionChannel))
VALID_ADDED_BY: frozenset[str] = frozenset(get_args(AddedBy))


class UnsupportedSchemaVersionError(ValueError):
    """Vendored rules corpus has a schema_version major the loader can't parse."""


class UnknownEnumValueError(ValueError):
    """Rule entry has a closed-enum field value outside the permitted set.

    Covers ``added_by``, ``severity``, ``expected_outcome.outcome``, and
    ``expected_outcome.emission_channel``. Subclassed per field for callers
    that want to distinguish.
    """


class UnknownAddedByError(UnknownEnumValueError):
    """Rule entry has an ``added_by`` value outside :data:`AddedBy`."""


class UnknownSeverityError(UnknownEnumValueError):
    """Rule entry has a ``severity`` value outside :data:`Severity`."""


class UnknownOutcomeError(UnknownEnumValueError):
    """Rule entry has an ``expected_outcome.outcome`` value outside :data:`Outcome`."""


class UnknownEmissionChannelError(UnknownEnumValueError):
    """Rule entry has an ``expected_outcome.emission_channel`` value outside :data:`EmissionChannel`."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleMatch:
    """Result of a rule matching a concrete config.

    ``declared_value`` is the user-set value for the *trigger* field (the first
    non-trivially-predicated field in the match spec). ``effective_value``
    populates only when the rule's ``expected_outcome`` lists the rule as
    ``dormant_silent`` with a ``normalised_fields`` mapping.
    """

    rule: Rule
    declared_value: Any
    effective_value: Any | None = None
    matched_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """One validation rule parsed from the corpus.

    Field names mirror the YAML schema documented in
    ``configs/validation_rules/README.md``. Construction goes through
    :func:`_parse_rule`; tests can instantiate directly for unit coverage.
    """

    id: str
    engine: str
    library: str
    rule_under_test: str
    severity: str
    native_type: str
    match_engine: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    expected_outcome: dict[str, Any]
    message_template: str | None
    walker_source: dict[str, Any]
    references: tuple[str, ...]
    added_by: str
    added_at: str

    def try_match(self, config: Any) -> RuleMatch | None:
        """Return a :class:`RuleMatch` if every predicate in ``match_fields`` holds.

        Field paths are dotted (``"transformers.sampling.temperature"``) and
        resolve against ``config`` attribute-by-attribute, tolerating Pydantic
        models, dataclasses, and plain dicts.

        ``declared_value`` on the returned match is the last field's value —
        corpus convention puts the precondition fields first and the
        *subject* field last (the field the rule is actually about). Users
        see the subject value substituted into message templates.
        """
        matched: dict[str, Any] = {}
        last_value: Any = None
        for path, spec in self.match_fields.items():
            actual = resolve_field_path(config, path)
            if not evaluate_predicate(actual, spec):
                return None
            matched[path] = actual
            last_value = actual
        return RuleMatch(rule=self, declared_value=last_value, matched_fields=matched)

    def render_message(self, match: RuleMatch) -> str:
        """Substitute ``{declared_value}`` / ``{effective_value}`` / ``{rule_id}`` in the template.

        Uses ``str.format`` with permissive defaults — templates that reference
        missing keys fall back to the rule id + raw template rather than
        raising at user-facing time.
        """
        if self.message_template is None:
            return f"[{self.id}] <no message template>"
        try:
            return self.message_template.format(
                declared_value=match.declared_value,
                effective_value=match.effective_value,
                rule_id=self.id,
                **match.matched_fields,
            )
        except (KeyError, IndexError):
            return f"[{self.id}] {self.message_template}"


@dataclass(frozen=True)
class VendoredRules:
    """Parsed corpus for one engine."""

    engine: str
    schema_version: str
    engine_version: str
    rules: tuple[Rule, ...]


# ---------------------------------------------------------------------------
# Predicate engine
# ---------------------------------------------------------------------------


_OPERATOR_HANDLERS: dict[str, Any] = {
    # Comparison operators: all None-safe on the *asymmetric* ones (a missing
    # field doesn't trip ``!=`` / ``not_equal`` / ``<`` / etc). ``==`` and
    # ``equals`` stay as-is — `None == x` evaluates to `False` for any
    # non-None `x`, so they naturally don't fire on None.
    # ``equals`` / ``not_equal`` are word-form aliases of ``==`` / ``!=``
    # and MUST match their symbol forms exactly — corpus authors swap them.
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a is not None and a != b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
    "equals": lambda a, b: a == b,
    "not_equal": lambda a, b: a is not None and a != b,
    # Membership operators: reject non-iterable specs (a string spec would
    # otherwise fall through to substring match — "ab" in "abc" is True,
    # which surprises corpus authors writing {"in": "abc"} thinking "exactly
    # one of these three chars").
    "in": lambda a, b: _require_iterable(b, "in") and a in b,
    "not_in": lambda a, b: _require_iterable(b, "not_in") and a not in b,
    # Presence operators: no None-guard needed (they're the test for None).
    "present": lambda a, _: a is not None,
    "absent": lambda a, _: a is None,
    # Type predicates match by the concrete type's __name__. None-safe (a
    # missing field doesn't trip ``type_is_not``). Spec takes a bare string
    # or a list of strings (any-of); predicate holds if the field's concrete
    # type name matches (resp. does not match) any. See :func:`_type_name`
    # for the type-name format and its known ambiguities.
    "type_is": lambda a, b: a is not None and _type_name(a) in _as_name_set(b),
    "type_is_not": lambda a, b: a is not None and _type_name(a) not in _as_name_set(b),
}


def _type_name(value: Any) -> str:
    """Return the concrete class name of ``value`` — ``type(value).__name__``.

    **Collision limitation:** this is the bare class name without the module
    qualifier, so unrelated libraries that happen to use the same class name
    (e.g. ``torch.dtype`` and ``numpy.dtype``) can't be distinguished with
    ``type_is: "dtype"`` alone. Disambiguate with a companion predicate
    (``present`` + path specificity) or use a bare-Python type (``bool``,
    ``int``, ``str``, ``list``, ``dict``) where collisions don't arise.
    """
    return type(value).__name__


def _as_name_set(spec: Any) -> frozenset[str]:
    """Accept a single type name or an iterable of names; return a frozenset."""
    if isinstance(spec, str):
        return frozenset({spec})
    return frozenset(str(x) for x in spec)


def _require_iterable(b: Any, op_name: str) -> bool:
    """Reject non-iterable specs for ``in`` / ``not_in`` at evaluation time.

    Naked string specs would silently do substring matching, which is not
    what corpus authors mean when they write ``{"in": "abc"}``. Force a
    list/tuple/set by raising on anything else.
    """
    if isinstance(b, (list, tuple, set, frozenset)):
        return True
    raise TypeError(
        f"Operator {op_name!r} requires list/tuple/set spec; got {type(b).__name__}: {b!r}"
    )


def evaluate_predicate(actual: Any, spec: Any) -> bool:
    """Evaluate ``actual`` against the corpus predicate ``spec``.

    ``spec`` shapes:

    - Bare value → equality (``spec == actual``).
    - One-key dict → operator predicate (``{"<": 1}``, ``{"in": ["a", "b"]}``).
    - Multi-key dict → every operator must hold (all predicates AND-combined).

    The last form covers corpus entries like
    ``{present: true, not_equal: 1.0}`` — field must be set AND not default.
    """
    if isinstance(spec, dict):
        if not spec:
            raise ValueError("Empty match predicate dict")
        for op, value in spec.items():
            handler = _OPERATOR_HANDLERS.get(op)
            if handler is None:
                raise ValueError(f"Unknown match operator: {op!r}")
            if not handler(actual, value):
                return False
        return True
    return bool(actual == spec)


# ---------------------------------------------------------------------------
# Field-path resolver
# ---------------------------------------------------------------------------


def resolve_field_path(config: Any, path: str) -> Any:
    """Walk dotted attribute / key path against ``config``.

    Missing attributes return ``None`` rather than raising — the predicate
    engine treats ``None`` as an absent field. Supports nested Pydantic models,
    dataclasses, and plain dicts mixed in any combination.

    **Method collision guard:** ``getattr(pydantic_model, "items")`` returns
    the bound ``.items()`` method, not a field named ``items``. Pydantic ships
    several attribute names (``copy``, ``dict``, ``json``, ``model_copy``,
    ``model_dump``, ``model_fields``, ``items``, ``keys``, ``values``) that
    would collide with field lookups. We check `__dict__` / `model_fields`
    first and only fall back to `getattr` when the key isn't a known field,
    ensuring that a corpus predicate on a real field wins over an accidental
    method match.
    """
    current: Any = config
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        # Pydantic models — use model_fields for the authoritative field set.
        model_fields = getattr(type(current), "model_fields", None)
        if isinstance(model_fields, dict) and part in model_fields:
            current = getattr(current, part, None)
            continue
        # Dataclasses — use __dataclass_fields__ for the authoritative field set.
        dc_fields = getattr(type(current), "__dataclass_fields__", None)
        if isinstance(dc_fields, dict) and part in dc_fields:
            current = getattr(current, part, None)
            continue
        # Fallback: plain objects. Reject callables — they're methods or
        # descriptors, never config field values.
        candidate = getattr(current, part, None)
        current = None if callable(candidate) else candidate
    return current


# ---------------------------------------------------------------------------
# Corpus parsing
# ---------------------------------------------------------------------------


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise UnsupportedSchemaVersionError(
            f"Unparseable schema_version {version!r}; expected semver '1.0.0' form."
        ) from exc


def _parse_rule(raw: dict[str, Any]) -> Rule:
    required = (
        "id",
        "engine",
        "severity",
        "native_type",
        "match",
        "kwargs_positive",
        "kwargs_negative",
        "expected_outcome",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"Rule {raw.get('id', '<unknown>')} missing field: {key}")
    match = raw["match"]
    if not isinstance(match, dict) or "fields" not in match:
        raise ValueError(f"Rule {raw['id']} has malformed match (missing `fields`): {match!r}")
    rule_id = raw["id"]
    severity = str(raw["severity"])
    if severity not in VALID_SEVERITY:
        raise UnknownSeverityError(
            f"Rule {rule_id!r} has severity={severity!r}; must be one of: {sorted(VALID_SEVERITY)}"
        )
    expected_outcome = dict(raw["expected_outcome"])
    outcome = str(expected_outcome.get("outcome", ""))
    if outcome not in VALID_OUTCOME:
        raise UnknownOutcomeError(
            f"Rule {rule_id!r} has expected_outcome.outcome={outcome!r}; "
            f"must be one of: {sorted(VALID_OUTCOME)}"
        )
    emission_channel = str(expected_outcome.get("emission_channel", ""))
    if emission_channel not in VALID_EMISSION_CHANNEL:
        raise UnknownEmissionChannelError(
            f"Rule {rule_id!r} has expected_outcome.emission_channel={emission_channel!r}; "
            f"must be one of: {sorted(VALID_EMISSION_CHANNEL)}"
        )
    added_by = str(raw.get("added_by", "manual_seed"))
    if added_by not in VALID_ADDED_BY:
        raise UnknownAddedByError(
            f"Rule {rule_id!r} has added_by={added_by!r}; must be one of: {sorted(VALID_ADDED_BY)}"
        )
    return Rule(
        id=str(rule_id),
        engine=str(raw["engine"]),
        library=str(raw.get("library", raw["engine"])),
        rule_under_test=str(raw.get("rule_under_test", "")),
        severity=severity,
        native_type=str(raw["native_type"]),
        match_engine=str(match.get("engine", raw["engine"])),
        match_fields=dict(match["fields"]),
        kwargs_positive=dict(raw["kwargs_positive"]),
        kwargs_negative=dict(raw["kwargs_negative"]),
        expected_outcome=expected_outcome,
        message_template=raw.get("message_template"),
        walker_source=dict(raw.get("walker_source") or {}),
        references=tuple(raw.get("references") or ()),
        added_by=added_by,
        added_at=str(raw.get("added_at", "")),
    )


def _parse_envelope(engine: str, raw_text: str) -> VendoredRules:
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Vendored rules for {engine!r} must be a YAML mapping; got {type(data).__name__}"
        )
    schema_version = str(data.get("schema_version", ""))
    if not schema_version:
        raise UnsupportedSchemaVersionError(
            f"Vendored rules for {engine!r} missing schema_version."
        )
    if _major(schema_version) != SUPPORTED_MAJOR_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Vendored rules for {engine!r} has schema_version={schema_version!r}; "
            f"this loader only supports major {SUPPORTED_MAJOR_VERSION}. "
            f"Regenerate the corpus or upgrade the loader."
        )
    raw_rules = data.get("rules") or []
    rules = tuple(_parse_rule(r) for r in raw_rules)
    return VendoredRules(
        engine=engine,
        schema_version=schema_version,
        engine_version=str(data.get("engine_version", "")),
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[4] / "configs" / "validation_rules"
_VENDORED_JSON_PACKAGE = "llenergymeasure.engines.vendored_rules"


class VendoredRulesLoader:
    """Load, cache, and serve :class:`VendoredRules` per engine.

    Per-instance cache (rather than module-level LRU) — tests can instantiate
    a loader and monkeypatch ``corpus_root`` without polluting other tests.

    Load order (picked up automatically):
      1. **YAML corpus** under ``configs/validation_rules/{engine}.yaml`` —
         the maintainer-seeded source of truth; always present in-repo.
      2. **Vendored JSON** shipped beside this module
         (``{engine}.json``) — CI-validated observed behaviour, overlaid
         onto the corpus's rules when present. Written by
         ``scripts/vendor_rules.py`` under the config-rules-refresh CI.
    """

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root: Path = corpus_root or _DEFAULT_CORPUS_ROOT
        self._cache: dict[str, VendoredRules] = {}

    def load_rules(self, engine: str) -> VendoredRules:
        """Return the parsed corpus for ``engine``, parsing once per engine.

        When a CI-validated vendored JSON envelope exists beside this
        module, the loader overlays its observed outcomes onto the
        corpus's rules — downstream consumers see empirically-confirmed
        behaviour rather than the corpus's declared shape alone.
        """
        cached = self._cache.get(engine)
        if cached is not None:
            return cached

        yaml_path = self.corpus_root / f"{engine}.yaml"
        try:
            yaml_text = yaml_path.read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Vendored rules for engine {engine!r} not found at {yaml_path}. "
                f"Run `python -m scripts.walkers.{engine} --out {yaml_path}` to generate."
            ) from exc

        parsed = _parse_envelope(engine, yaml_text)

        vendored_json = _try_load_vendored_json(engine)
        if vendored_json is not None:
            parsed = _overlay_vendored_observations(parsed, vendored_json)

        self._cache[engine] = parsed
        return parsed

    def invalidate(self, engine: str | None = None) -> None:
        """Clear cached rules (all or for one engine)."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)


# ---------------------------------------------------------------------------
# Vendored JSON overlay (config-rules-refresh CI)
# ---------------------------------------------------------------------------


def _try_load_vendored_json(engine: str) -> dict[str, Any] | None:
    """Return parsed vendored-rules JSON for ``engine`` or ``None`` if absent.

    Accessed via :mod:`importlib.resources` so the JSON is picked up
    regardless of install layout (editable checkout vs installed wheel).
    Swallows ``JSONDecodeError`` and rejects unsupported envelope versions
    to avoid breaking startup on a corrupt or future-schema commit-back —
    the vendor CI job will resurface the issue.
    """
    try:
        raw = (resources.files(_VENDORED_JSON_PACKAGE) / f"{engine}.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # Reject unsupported envelope majors — mirror the YAML schema guard so a
    # future-schema JSON can't silently overlay unfamiliar keys.
    envelope_version = str(parsed.get("schema_version", ""))
    if envelope_version and _major(envelope_version) != SUPPORTED_MAJOR_VERSION:
        return None
    return parsed


_OBSERVED_KEY_MAP = {
    "outcome": "observed_outcome",
    "emission_channel": "observed_emission_channel",
    "observed_messages": "observed_messages",
}


def _overlay_vendored_observations(
    parsed: VendoredRules, vendored: dict[str, Any]
) -> VendoredRules:
    """Overlay observed outcomes from a vendored JSON envelope onto the corpus rules.

    The corpus carries the declared shape; the vendored JSON carries what
    CI observed. When JSON is present, the loader writes observed-* keys
    alongside the corpus's declared ``outcome`` / ``emission_channel`` so
    consumers (the generic ``@model_validator``) can act on CI-validated
    truth. The declared fields are left untouched — strict validation in
    :func:`_parse_rule` is not re-exercised against the observed vocabulary
    (which is deliberately wider; see ``scripts/_vendor_common.py``).
    """
    cases = {c["id"]: c for c in vendored.get("cases", []) if isinstance(c, dict) and "id" in c}
    overlaid = tuple(_overlay_rule(rule, cases.get(rule.id)) for rule in parsed.rules)
    return replace(parsed, rules=overlaid)


def _overlay_rule(rule: Rule, observed: dict[str, Any] | None) -> Rule:
    """Merge observed-* fields from a vendor case into a rule's expected_outcome.

    Observed keys are written directly (not via ``setdefault``) so a
    re-applied overlay with updated CI observations replaces a prior
    overlay's values. The corpus declares no ``observed_*`` keys itself,
    so this never clobbers corpus-authored data.
    """
    if observed is None:
        return rule
    merged = dict(rule.expected_outcome)
    for vendored_key, expected_key in _OBSERVED_KEY_MAP.items():
        value = observed.get(vendored_key)
        if value not in (None, [], {}):
            merged[expected_key] = list(value) if isinstance(value, list) else value
    return replace(rule, expected_outcome=merged)
