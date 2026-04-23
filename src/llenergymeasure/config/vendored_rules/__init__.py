"""Vendored validation rules: loader, matcher, predicate engine.

The corpus at ``configs/validation_rules/{engine}.yaml`` is the SSOT for
what runtime config validation tells users about their configs. This package
consumes that corpus and exposes a typed matcher API used by the generic
``@model_validator`` on :class:`llenergymeasure.config.models.ExperimentConfig`
and (eventually) by the sweep canonicaliser.
"""

from llenergymeasure.config.vendored_rules.loader import (
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
