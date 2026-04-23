"""Transformers library-API introspection walker.

Derives validation rules from HF's own runtime machinery instead of reading
library source. Three extraction paths, all observe the library at walk time:

1. **Mode-gated dormancy auto-enumeration** — for each known trigger class
   (greedy, single-beam), enumerate every public ``GenerationConfig()`` field,
   synthesise a non-default probe value for its Python type, invoke
   ``validate(strict=True)``, and record any field HF reports as dormant in
   the composed raise. Adding a new HF sampling parameter is auto-discovered
   on the next walk — no code edits.

2. **Hardcoded error-class probes** — rules whose violation shape is
   rule-specific (negative integer, out-of-enum string, wrong-type instance)
   can't be discovered by field enumeration alone. The probe list is
   hardcoded, but the message template is still library-authored: each probe
   triggers a construction-time ``ValueError`` whose text we lift verbatim.

3. **Hardcoded validate-time self-triggered dormant probes** — the
   ``pad_token_id < 0`` family (currently one rule) doesn't route via a mode
   flag; its own value triggers the dormancy. Encoded as explicit probes
   rather than extending the trigger enumeration.

Every rule this walker emits carries ``added_by="introspection"``. BNB rules
are out of scope — BNB import touches CUDA, so those stay as ``manual_seed``
entries in :mod:`scripts.walkers.transformers`.

Field defaults, rule message templates, and the per-trigger field list are
all library-observed at walk time. The only encoded knowledge is: (a) two
trigger classes exist (greedy, single-beam), (b) the per-rule violation
shape for error-class probes.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers._base import RuleCandidate, WalkerSource  # noqa: E402

# ---------------------------------------------------------------------------
# Trigger classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DormancyTrigger:
    """One mode-gated dormancy trigger in HF's ``validate()``.

    HF 4.49-4.56 exposes three trigger classes: greedy (``do_sample=False``),
    single-beam (``num_beams=1``), and scalar-output (``return_dict_in_generate=False``).
    Each trigger has an ``isolation_kwargs`` payload — values for the OTHER
    triggers that DON'T activate them — so the auto-enumerator can tell
    which trigger a firing rule belongs to even though the three categories
    partially overlap in HF's default state (``num_beams=1`` and
    ``return_dict_in_generate=False`` are both defaults, so a naive probe
    under ``do_sample=False`` would triple-fire unrelated rules).
    """

    id_prefix: str
    trigger_field: str
    trigger_positive: Any  # value that activates this trigger
    trigger_negative: Any  # value that doesn't
    # Kwargs that deactivate every OTHER trigger. See the class docstring —
    # load-bearing for attributing a firing rule to exactly one trigger class.
    isolation_kwargs: dict[str, Any]
    rule_under_test_template: str  # f-string slot: {field}


_GREEDY_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_greedy_strips_",
    trigger_field="do_sample",
    trigger_positive=False,
    trigger_negative=True,
    isolation_kwargs={"num_beams": 4, "return_dict_in_generate": True},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "do_sample=False and `{field}` is set to a non-default value"
    ),
)


_BEAM_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_single_beam_strips_",
    trigger_field="num_beams",
    trigger_positive=1,
    trigger_negative=4,
    isolation_kwargs={"do_sample": True, "return_dict_in_generate": True},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "num_beams=1 and `{field}` is set"
    ),
)


_RETURN_DICT_TRIGGER = _DormancyTrigger(
    id_prefix="transformers_no_return_dict_strips_",
    trigger_field="return_dict_in_generate",
    trigger_positive=False,
    trigger_negative=True,
    isolation_kwargs={"do_sample": True, "num_beams": 4},
    rule_under_test_template=(
        "GenerationConfig.validate() records dormant `{field}` when "
        "return_dict_in_generate=False and `{field}` is set"
    ),
)


_TRIGGERS: tuple[_DormancyTrigger, ...] = (
    _GREEDY_TRIGGER,
    _BEAM_TRIGGER,
    _RETURN_DICT_TRIGGER,
)


# Fields the auto-enumerator explicitly skips. Categories:
# - Trigger fields themselves (probing a trigger under its own activation
#   produces a nonsense self-triggering rule).
# - Class-instance fields whose probe value would need a specific type
#   (``compile_config``, ``watermarking_config``) — those are covered by
#   ``_ERROR_PROBES`` where relevant.
# - Meta fields and structural fields that don't have dormancy semantics.
_SKIP_FIELDS: frozenset[str] = frozenset(
    {
        "do_sample",
        "num_beams",
        "return_dict_in_generate",
        "compile_config",
        "watermarking_config",
        "generation_kwargs",
        "transformers_version",
    }
)


def _synthesise_probe_value(default: Any) -> Any | None:
    """Return a non-default value whose type matches ``default``.

    ``None`` when the type is too complex to probe mechanically (class
    instances, opaque objects). Those fields either live in ``_SKIP_FIELDS``
    or are caught by the construction-time ``try/except`` inside the
    enumerator and silently skipped.
    """
    # bool is a subclass of int — check bool first.
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    if isinstance(default, float):
        if default in (0.0, 1.0):
            return 0.5
        return round(default + 0.1, 3)
    if default is None:
        # A float probe covers "present" for most HF sampling params. Fields
        # needing a specific type (str enum, int token id) either skip here
        # or trip construction and drop out inside the enumerator.
        return 0.5
    if isinstance(default, list):
        return ["probe"]
    if isinstance(default, str):
        # Likely trips construction for enum fields (caught by the enumerator's
        # try/except); skipping the type entirely would miss any future
        # free-form string dormancy.
        return "__probe__"
    return None


# ---------------------------------------------------------------------------
# Library-message parsing
# ---------------------------------------------------------------------------


_ISSUE_LINE_RE = re.compile(r"^- `([^`]+)`: (.+)$")


def _parse_strict_raise(composed_message: str) -> dict[str, str]:
    """Split HF's ``validate(strict=True)`` raise into ``{field: message}``.

    HF emits::

        GenerationConfig is invalid:
        - `field_a`: <per-field message a>
        - `field_b`: <per-field message b>
        If you're using a pretrained model, ...

    We keep the per-field bodies and drop the header + pretrained-model footer.
    """
    issues: dict[str, str] = {}
    for line in composed_message.splitlines():
        match = _ISSUE_LINE_RE.match(line.strip())
        if match:
            issues[match.group(1)] = match.group(2).strip()
    return issues


def _substitute_declared_value(message: str, probe_value: Any) -> str:
    """Replace the backtick-wrapped probe value in ``message`` with ``{declared_value}``.

    HF quotes every value it interpolates with backticks (``\\`0.9\\```,
    ``\\`True\\```, ``\\`['probe']\\```). We swap that wrapped form for
    ``\\`{declared_value}\\``` so the consumer's ``.format(declared_value=...)``
    interpolates the user's actual value at render time.

    If the probe value doesn't appear wrapped (e.g. ``compile_config`` error
    talks about the Python class, not the value), the message comes back
    unchanged — the consumer treats a placeholder-less template as a literal.
    """
    wrapped = f"`{probe_value}`"
    placeholder = "`{declared_value}`"
    if wrapped in message:
        return message.replace(wrapped, placeholder)
    return message


# ---------------------------------------------------------------------------
# Dormancy auto-enumeration
# ---------------------------------------------------------------------------


def _enumerate_dormancy_candidates() -> list[tuple[_DormancyTrigger, str, Any, Any, str]]:
    """Probe every public ``GenerationConfig`` field under each trigger class.

    Returns ``(trigger, field, default, probe_value, per_field_message)``
    tuples for fields that HF reports as dormant. Filters out: private
    fields, ``_SKIP_FIELDS``, types with no synthesised probe, and fields
    whose construction trips an error under the trigger kwargs.

    Field iteration is ``sorted(vars(baseline).items())`` — byte-stable
    across runs so the corpus output is deterministic.
    """
    from transformers import GenerationConfig  # type: ignore

    baseline = GenerationConfig()
    discovered: list[tuple[_DormancyTrigger, str, Any, Any, str]] = []

    for field_name, default in sorted(vars(baseline).items()):
        if field_name.startswith("_"):
            continue
        if field_name in _SKIP_FIELDS:
            continue
        probe = _synthesise_probe_value(default)
        if probe is None:
            continue

        for trigger in _TRIGGERS:
            # Isolation kwargs deactivate the other two trigger classes, so
            # a firing minor_issue for ``field_name`` is unambiguously
            # attributed to ``trigger``.
            kwargs = {
                **trigger.isolation_kwargs,
                trigger.trigger_field: trigger.trigger_positive,
                field_name: probe,
            }
            try:
                gc = GenerationConfig(**kwargs)
            except Exception:
                # Construction raised: the field's validator rejects our probe.
                # Not a dormancy rule; may be covered by _ERROR_PROBES instead.
                continue

            try:
                gc.validate(strict=True)
            except ValueError as exc:
                issues = _parse_strict_raise(str(exc))
                if field_name in issues:
                    discovered.append((trigger, field_name, default, probe, issues[field_name]))

    return discovered


# ---------------------------------------------------------------------------
# Error-class probes (violation shape is per-rule, so hardcoded)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ErrorProbe:
    """One error-class rule. Violation shape is encoded; message comes from the library."""

    id: str
    rule_under_test: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    # Kwarg whose value appears in the library's error message (for
    # {declared_value} substitution). None when the message references only
    # the type or another field — the template goes out verbatim.
    probed_field: str | None


_ERROR_PROBES: tuple[_ErrorProbe, ...] = (
    _ErrorProbe(
        id="transformers_negative_max_new_tokens",
        rule_under_test="GenerationConfig(max_new_tokens) rejects non-positive values",
        match_fields={"transformers.sampling.max_new_tokens": {"<=": 0}},
        kwargs_positive={"max_new_tokens": -1},
        kwargs_negative={"max_new_tokens": 16},
        probed_field="max_new_tokens",
    ),
    _ErrorProbe(
        id="transformers_invalid_cache_implementation",
        rule_under_test="GenerationConfig rejects unknown cache_implementation strings",
        match_fields={"transformers.sampling.cache_implementation": {"present": True}},
        kwargs_positive={"cache_implementation": "nonsense"},
        kwargs_negative={"cache_implementation": "static"},
        probed_field="cache_implementation",
    ),
    _ErrorProbe(
        id="transformers_invalid_early_stopping",
        rule_under_test="GenerationConfig.early_stopping must be bool or the literal 'never'",
        match_fields={
            "transformers.sampling.early_stopping": {
                "present": True,
                "not_in": [True, False, "never"],
            },
        },
        kwargs_positive={"early_stopping": "sometimes"},
        kwargs_negative={"early_stopping": True},
        probed_field="early_stopping",
    ),
    _ErrorProbe(
        id="transformers_num_return_sequences_exceeds_num_beams",
        rule_under_test="GenerationConfig rejects num_return_sequences > num_beams",
        match_fields={
            "transformers.sampling.num_return_sequences": {">": 1},
            "transformers.sampling.num_beams": {"present": True},
        },
        kwargs_positive={"num_return_sequences": 4, "num_beams": 2},
        kwargs_negative={"num_return_sequences": 2, "num_beams": 4},
        probed_field="num_return_sequences",
    ),
    _ErrorProbe(
        id="transformers_greedy_rejects_num_return_sequences",
        rule_under_test=(
            "Greedy decoding (do_sample=False, num_beams=1) requires num_return_sequences=1"
        ),
        match_fields={
            "transformers.sampling.do_sample": False,
            "transformers.sampling.num_beams": 1,
            "transformers.sampling.num_return_sequences": {">": 1},
        },
        kwargs_positive={"do_sample": False, "num_beams": 1, "num_return_sequences": 3},
        kwargs_negative={"do_sample": False, "num_beams": 1, "num_return_sequences": 1},
        probed_field="num_return_sequences",
    ),
    _ErrorProbe(
        id="transformers_compile_config_type",
        rule_under_test=(
            "GenerationConfig rejects compile_config that is not a CompileConfig instance"
        ),
        match_fields={
            "transformers.sampling.compile_config": {
                "present": True,
                "type_is_not": "CompileConfig",
            },
        },
        kwargs_positive={"compile_config": {"mode": "reduce-overhead"}},
        kwargs_negative={"compile_config": None},
        # HF's error describes the class, not the value — no {declared_value} slot.
        probed_field=None,
    ),
)


# ---------------------------------------------------------------------------
# Validate-time self-triggered dormant probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DormantProbe:
    """A validate-time dormancy rule whose trigger is field-self-driven.

    Not mode-gated (so it's not covered by ``_TRIGGERS``). Currently one
    rule in HF 4.56 (``pad_token_id < 0``); the tuple is present so future
    self-triggered dormancies land as one-line additions.
    """

    id: str
    rule_under_test: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    probed_field: str


_VALIDATE_DORMANT_PROBES: tuple[_DormantProbe, ...] = (
    _DormantProbe(
        id="transformers_negative_pad_token_id",
        rule_under_test="GenerationConfig.validate() records dormant pad_token_id < 0",
        match_fields={"transformers.sampling.pad_token_id": {"<": 0}},
        kwargs_positive={"pad_token_id": -1},
        kwargs_negative={"pad_token_id": 0},
        probed_field="pad_token_id",
    ),
)


# ---------------------------------------------------------------------------
# Rule-candidate factories
# ---------------------------------------------------------------------------


def _default_predicate(default: Any) -> dict[str, Any]:
    """Predicate that fires when the field is set to a non-default value."""
    if default is None:
        return {"present": True}
    return {"present": True, "not_equal": default}


def _find_line(source_file: str, needle: str) -> int:
    """Best-effort ``self.<needle>`` line lookup. ``0`` if not found."""
    try:
        text = Path(source_file).read_text()
    except OSError:
        return 0
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    return 0


def _make_dormancy_candidate(
    trigger: _DormancyTrigger,
    field: str,
    default: Any,
    probe: Any,
    library_message: str,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose a dormancy ``RuleCandidate`` from one library-observed trigger firing."""
    template = _substitute_declared_value(library_message, probe)
    line = _find_line(abs_source_path, f"self.{field}")
    return RuleCandidate(
        id=f"{trigger.id_prefix}{field}",
        engine="transformers",
        library="transformers",
        rule_under_test=trigger.rule_under_test_template.format(field=field),
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields={
            f"transformers.sampling.{trigger.trigger_field}": trigger.trigger_positive,
            f"transformers.sampling.{field}": _default_predicate(default),
        },
        kwargs_positive={trigger.trigger_field: trigger.trigger_positive, field: probe},
        kwargs_negative={trigger.trigger_field: trigger.trigger_negative, field: probe},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "logger_warning_once",
            "normalised_fields": [],
        },
        message_template=template,
        references=[f"transformers.GenerationConfig.validate() (line ~{line})"],
        added_by="introspection",
        added_at=today,
    )


def _make_error_candidate(
    probe: _ErrorProbe,
    library_message: str,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose an error-class ``RuleCandidate`` from a construction raise."""
    if probe.probed_field is not None:
        template = _substitute_declared_value(
            library_message, probe.kwargs_positive[probe.probed_field]
        )
    else:
        template = library_message
    needle_field = probe.probed_field or next(iter(probe.match_fields)).rsplit(".", 1)[-1]
    line = _find_line(abs_source_path, f"self.{needle_field}")
    return RuleCandidate(
        id=probe.id,
        engine="transformers",
        library="transformers",
        rule_under_test=probe.rule_under_test,
        severity="error",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields=probe.match_fields,
        kwargs_positive=probe.kwargs_positive,
        kwargs_negative=probe.kwargs_negative,
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=template,
        references=["transformers.GenerationConfig — observed via construction-time ValueError"],
        added_by="introspection",
        added_at=today,
    )


def _make_validate_dormant_candidate(
    probe: _DormantProbe,
    library_message: str,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose a validate-time self-triggered dormant ``RuleCandidate``."""
    template = _substitute_declared_value(
        library_message, probe.kwargs_positive[probe.probed_field]
    )
    line = _find_line(abs_source_path, f"self.{probe.probed_field}")
    return RuleCandidate(
        id=probe.id,
        engine="transformers",
        library="transformers",
        rule_under_test=probe.rule_under_test,
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields=probe.match_fields,
        kwargs_positive=probe.kwargs_positive,
        kwargs_negative=probe.kwargs_negative,
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "logger_warning_once",
            "normalised_fields": [],
        },
        message_template=template,
        references=[
            "transformers.GenerationConfig.validate() — observed via validate(strict=True)"
        ],
        added_by="introspection",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class IntrospectionProbeDisappeared(RuntimeError):
    """A hardcoded probe stopped firing — library behaviour has drifted.

    Raised when an ``_ERROR_PROBES`` entry's ``kwargs_positive`` no longer
    produces a ``ValueError``, or a ``_VALIDATE_DORMANT_PROBES`` entry no
    longer fires in ``validate(strict=True)``. Semantically equivalent to
    :class:`scripts.walkers._base.WalkerLandmarkMissingError` — walker
    refuses to emit partial output; maintainer reconciles on the next
    library-bump PR.
    """


def walk_generation_config_rules(
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Return all introspection-derived ``RuleCandidate``s for ``GenerationConfig``.

    Composes three sources in deterministic order:

    1. Mode-gated dormancy rules auto-enumerated per trigger class.
    2. Error-class rules from ``_ERROR_PROBES`` observed via construction.
    3. Validate-time self-triggered dormant rules from ``_VALIDATE_DORMANT_PROBES``.

    Raises :class:`IntrospectionProbeDisappeared` if a hardcoded probe stops
    firing — the "silent coverage loss becomes a visible CI failure" contract.
    """
    from transformers import GenerationConfig  # type: ignore

    candidates: list[RuleCandidate] = []

    for trigger, field, default, probe, library_message in _enumerate_dormancy_candidates():
        candidates.append(
            _make_dormancy_candidate(
                trigger,
                field,
                default,
                probe,
                library_message,
                abs_source_path,
                rel_source_path,
                today,
            )
        )

    for eprobe in _ERROR_PROBES:
        try:
            GenerationConfig(**eprobe.kwargs_positive)
        except ValueError as exc:
            library_message = str(exc)
        else:
            raise IntrospectionProbeDisappeared(
                f"Error probe {eprobe.id!r} no longer raises. Library may "
                f"have relaxed or removed the constraint."
            )
        candidates.append(
            _make_error_candidate(eprobe, library_message, abs_source_path, rel_source_path, today)
        )

    for dprobe in _VALIDATE_DORMANT_PROBES:
        gc = GenerationConfig(**dprobe.kwargs_positive)
        try:
            gc.validate(strict=True)
        except ValueError as exc:
            issues = _parse_strict_raise(str(exc))
        else:
            raise IntrospectionProbeDisappeared(
                f"Validate-time dormant probe {dprobe.id!r} no longer fires."
            )
        if dprobe.probed_field not in issues:
            raise IntrospectionProbeDisappeared(
                f"Validate-time dormant probe {dprobe.id!r} fired without "
                f"mentioning {dprobe.probed_field!r}. Shape drift in HF's "
                f"minor_issues output."
            )
        candidates.append(
            _make_validate_dormant_candidate(
                dprobe,
                issues[dprobe.probed_field],
                abs_source_path,
                rel_source_path,
                today,
            )
        )

    return candidates


def discover_dormancy_fields() -> dict[str, set[str]]:
    """Return ``{trigger.id_prefix: {field, ...}}`` — auto-discovered dormancy fields.

    Exposed for the test tier that asserts the hardcoded partition of the
    corpus is a subset of what the live library exposes. Runs the same
    enumeration the walker uses, returns just the field names.
    """
    result: dict[str, set[str]] = {t.id_prefix: set() for t in _TRIGGERS}
    for trigger, field, _default, _probe, _msg in _enumerate_dormancy_candidates():
        result[trigger.id_prefix].add(field)
    return result


# Re-export helpers used by callers (``scripts.walkers.transformers``).
__all__ = [
    "IntrospectionProbeDisappeared",
    "discover_dormancy_fields",
    "walk_generation_config_rules",
]


if __name__ == "__main__":  # pragma: no cover — module is imported, not run.
    print(
        "This module is an internal helper for scripts.walkers.transformers. "
        "Run the parent walker instead:\n\n"
        "  python -m scripts.walkers.transformers --out configs/validation_rules/transformers.yaml",
        file=sys.stderr,
    )
    sys.exit(2)
