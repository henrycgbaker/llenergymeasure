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
  Phase 50.2a (this PR) consumes the YAML corpus directly. Phase 50.2b
  adds the vendor CI pipeline which emits ``{engine}.json`` files alongside
  the corpus; the loader will grow a JSON consumption path then. The JSON
  path is sketched below but disabled — hooks are in place, no user impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    return actual == spec


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


class VendoredRulesLoader:
    """Load, cache, and serve :class:`VendoredRules` per engine.

    Per-instance cache (rather than module-level LRU) — tests can instantiate
    a loader and monkeypatch ``corpus_root`` without polluting other tests.
    """

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root: Path = corpus_root or _DEFAULT_CORPUS_ROOT
        self._cache: dict[str, VendoredRules] = {}

    def load_rules(self, engine: str) -> VendoredRules:
        """Return the parsed corpus for ``engine``, parsing once per engine."""
        cached = self._cache.get(engine)
        if cached is not None:
            return cached
        path = self.corpus_root / f"{engine}.yaml"
        try:
            raw_text = path.read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Vendored rules for engine {engine!r} not found at {path}. "
                f"Run `python -m scripts.walkers.{engine} --out {path}` to generate."
            ) from exc
        parsed = _parse_envelope(engine, raw_text)
        self._cache[engine] = parsed
        return parsed

    def invalidate(self, engine: str | None = None) -> None:
        """Clear cached rules (all or for one engine)."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)
