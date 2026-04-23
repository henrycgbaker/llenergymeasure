"""Vendored validation rules: loader, matcher, predicate engine.

The corpus at ``configs/validation_rules/{engine}.yaml`` is the SSOT for
what runtime config validation tells users about their configs. This package
consumes that corpus and exposes a typed matcher API.

This module does not yet wire into ``config/models.py`` — that wiring lands
in phase 50.2c alongside the generic ``@model_validator``. Landing the
loader contract first lets the vendor CI pipeline (50.2b) and the generic
validator (50.2c) each depend on a stable surface.
"""

from llenergymeasure.engines.vendored_rules.loader import (
    VALID_ADDED_BY,
    VALID_EMISSION_CHANNEL,
    VALID_OUTCOME,
    VALID_SEVERITY,
    AddedBy,
    EmissionChannel,
    Outcome,
    Rule,
    RuleMatch,
    Severity,
    UnknownAddedByError,
    UnknownEmissionChannelError,
    UnknownEnumValueError,
    UnknownOutcomeError,
    UnknownSeverityError,
    UnsupportedSchemaVersionError,
    VendoredRules,
    VendoredRulesLoader,
    evaluate_predicate,
    resolve_field_path,
)

__all__ = [
    "VALID_ADDED_BY",
    "VALID_EMISSION_CHANNEL",
    "VALID_OUTCOME",
    "VALID_SEVERITY",
    "AddedBy",
    "EmissionChannel",
    "Outcome",
    "Rule",
    "RuleMatch",
    "Severity",
    "UnknownAddedByError",
    "UnknownEmissionChannelError",
    "UnknownEnumValueError",
    "UnknownOutcomeError",
    "UnknownSeverityError",
    "UnsupportedSchemaVersionError",
    "VendoredRules",
    "VendoredRulesLoader",
    "evaluate_predicate",
    "resolve_field_path",
]
