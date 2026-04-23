"""Load, match, and render validation rules from the YAML corpus.

The corpus at ``configs/validation_rules/{engine}.yaml`` is parsed here into
typed :class:`Rule` entries. Each rule carries a match predicate (operators
defined in :func:`evaluate_predicate`) and a message template. The generic
``@model_validator`` in ``config/models.py`` (landing in phase 50.2c) calls
:meth:`Rule.try_match` on every rule for a given engine and emits
error/warn/dormant annotations based on the rule's severity.

Design mirror: this module parallels :mod:`llenergymeasure.config.schema_loader`
from parameter-discovery — same envelope validation
(:class:`UnsupportedSchemaVersionError` on major-version mismatch), same
per-instance caching for test isolation, same lazy load pattern.

Corpus vs vendored JSON:
  Phase 50.2a seeded the loader against the YAML corpus. Phase 50.2b
  (this phase) adds JSON path consumption: when a vendored JSON envelope
  exists alongside the corpus (produced by ``scripts/vendor_rules.py``),
  the loader prefers it — it is the CI-validated projection of the corpus.
  On missing JSON, the loader falls back to the YAML corpus so local
  development without a vendor run still works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MAJOR_VERSION = 1
"""Major version the loader knows how to parse.

Raised via :class:`UnsupportedSchemaVersionError` on mismatch; the loader
refuses partial reads to avoid silently accepting a future schema shape.
"""


class UnsupportedSchemaVersionError(ValueError):
    """Vendored rules corpus has a schema_version major the loader can't parse."""


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

        ``effective_value`` is populated for ``dormant``/``dormant_silent``
        rules with a ``not_equal`` subject predicate (the canonicaliser's
        normalisation target — see :func:`study.sweep_canonicalise._rule_normalisations`).
        Consumers render "declared X, effective Y" messages without
        re-deriving the target from the rule spec.
        """
        matched: dict[str, Any] = {}
        last_value: Any = None
        last_spec: Any = None
        for path, spec in self.match_fields.items():
            actual = resolve_field_path(config, path)
            if not evaluate_predicate(actual, spec):
                return None
            matched[path] = actual
            last_value = actual
            last_spec = spec
        effective = _derive_effective_value(self, last_spec)
        return RuleMatch(
            rule=self,
            declared_value=last_value,
            effective_value=effective,
            matched_fields=matched,
        )

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
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "present": lambda a, _: a is not None,
    "absent": lambda a, _: a is None,
    "equals": lambda a, b: a == b,
    "not_equal": lambda a, b: a is not None and a != b,
}


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
# Effective-value derivation (for RuleMatch.effective_value)
# ---------------------------------------------------------------------------


def _derive_effective_value(rule: Rule, last_spec: Any) -> Any | None:
    """Return the canonical / effective value the library drives the subject to.

    For dormant rules, the canonicaliser's target is the ``not_equal`` sentinel
    in the subject predicate (see ``study.sweep_canonicalise._rule_normalisations``).
    Rules with an explicit ``normalised_fields`` list in
    ``expected_outcome`` strip to ``None`` by convention — mirror that here.
    Non-dormant rules return ``None``.
    """
    if rule.severity not in ("dormant", "dormant_silent"):
        return None
    normalised = rule.expected_outcome.get("normalised_fields")
    if isinstance(normalised, list) and normalised:
        # Convention: fields listed in normalised_fields strip to None.
        return None
    if isinstance(last_spec, dict) and "not_equal" in last_spec:
        return last_spec["not_equal"]
    return None


# ---------------------------------------------------------------------------
# Field-path resolver
# ---------------------------------------------------------------------------


def resolve_field_path(config: Any, path: str) -> Any:
    """Walk dotted attribute / key path against ``config``.

    Missing attributes return ``None`` rather than raising — the predicate
    engine treats ``None`` as an absent field. Supports nested Pydantic models,
    dataclasses, and plain dicts mixed in any combination.
    """
    current: Any = config
    for part in path.split("."):
        if current is None:
            return None
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
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
    return Rule(
        id=str(raw["id"]),
        engine=str(raw["engine"]),
        library=str(raw.get("library", raw["engine"])),
        rule_under_test=str(raw.get("rule_under_test", "")),
        severity=str(raw["severity"]),
        native_type=str(raw["native_type"]),
        match_engine=str(match.get("engine", raw["engine"])),
        match_fields=dict(match["fields"]),
        kwargs_positive=dict(raw["kwargs_positive"]),
        kwargs_negative=dict(raw["kwargs_negative"]),
        expected_outcome=dict(raw["expected_outcome"]),
        message_template=raw.get("message_template"),
        walker_source=dict(raw.get("walker_source") or {}),
        references=tuple(raw.get("references") or ()),
        added_by=str(raw.get("added_by", "manual_seed")),
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
      1. **Vendored JSON** shipped beside this module
         (``{engine}.json``) — CI-validated observed behaviour, preferred
         whenever present. Written by ``scripts/vendor_rules.py``.
      2. **YAML corpus** under ``configs/validation_rules/{engine}.yaml`` —
         the walker-seeded source of truth; always present in-repo.
    """

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root: Path = corpus_root or _DEFAULT_CORPUS_ROOT
        self._cache: dict[str, VendoredRules] = {}

    def load_rules(self, engine: str) -> VendoredRules:
        """Return the parsed corpus for ``engine``, parsing once per engine.

        Prefers the CI-validated vendored JSON when present; falls back to
        the in-tree YAML corpus otherwise. The JSON envelope carries observed
        outcomes per rule — these populate the returned ``Rule.expected_outcome``
        so downstream consumers see empirically-confirmed behaviour rather
        than the corpus's declared shape.
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


def _try_load_vendored_json(engine: str) -> dict[str, Any] | None:
    """Return parsed vendored-rules JSON for ``engine`` or ``None`` if absent."""
    try:
        raw = (resources.files(_VENDORED_JSON_PACKAGE) / f"{engine}.json").read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed


def _overlay_vendored_observations(
    parsed: VendoredRules, vendored: dict[str, Any]
) -> VendoredRules:
    """Overlay observed outcomes from a vendored JSON envelope onto the corpus rules.

    The corpus carries the declared shape; the vendored JSON carries what CI
    observed. On presence of JSON, the loader surfaces the observed fields so
    consumers (the generic validator in 50.2c) can act on CI-validated truth.
    """
    cases = {c["id"]: c for c in vendored.get("cases", []) if isinstance(c, dict) and "id" in c}
    overlaid = tuple(_overlay_rule(rule, cases.get(rule.id)) for rule in parsed.rules)
    return replace(parsed, rules=overlaid)


_OBSERVED_KEY_MAP = {
    "outcome": "observed_outcome",
    "emission_channel": "observed_emission_channel",
    "observed_messages": "observed_messages",
}


def _overlay_rule(rule: Rule, observed: dict[str, Any] | None) -> Rule:
    """Merge observed-* fields from a vendor case into a rule's expected_outcome."""
    if observed is None:
        return rule
    merged = dict(rule.expected_outcome)
    for vendored_key, expected_key in _OBSERVED_KEY_MAP.items():
        value = observed.get(vendored_key)
        if value not in (None, [], {}):
            merged.setdefault(expected_key, list(value) if isinstance(value, list) else value)
    return replace(rule, expected_outcome=merged)
