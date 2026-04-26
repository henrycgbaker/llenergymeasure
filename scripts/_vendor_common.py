"""Shared utilities for the vendor-rules pipeline.

Factored out of :mod:`scripts.vendor_rules` so that per-engine native-type
runners can live here while the CLI + loop driver stays in ``vendor_rules``.

This module is engine-agnostic. Per-engine behaviour lives behind
:func:`get_native_type_runner`, which dispatches on engine name.

Design contract: the vendor step observes library behaviour concretely. It
never re-interprets the rule's declared shape — if the library behaves
differently from what the corpus claims, CI fails. See
:doc:`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`
§4.3 for the full contract.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Private-field allowlist
# ---------------------------------------------------------------------------
# The vendor state-diff excludes engine-specific bookkeeping fields that would
# pollute the diff with non-deterministic state (commit hashes, cached derived
# flags, per-run tensors). Each engine declares its own allowlist; the default
# covers fields common across engines.

_DEFAULT_PRIVATE_FIELD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "_commit_hash",
        "_from_model_config",
        "_original_object_hash",
        "_all_stop_token_ids",
    }
)

TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST: frozenset[str] = _DEFAULT_PRIVATE_FIELD_ALLOWLIST | frozenset(
    {
        # HF-specific derived fields populated during __post_init__ that do
        # not constitute a user-facing normalisation.
        "_eos_token_tensor",
        "_pad_token_tensor",
        "_bos_token_tensor",
        "transformers_version",
    }
)


# ---------------------------------------------------------------------------
# Observation dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureBuffers:
    """Container for everything a single ``native_type(**kwargs)`` call produced."""

    exception_type: str | None
    exception_message: str | None
    warnings_captured: tuple[str, ...]
    logger_messages: tuple[str, ...]
    observed_state: dict[str, Any] | None
    duration_ms: int


@dataclass
class CaseResult:
    """Per-rule observed outcome, ready for JSON serialisation."""

    id: str
    outcome: str  # see _classify_outcome
    emission_channel: str  # mirrors corpus "emission_channel" tag
    observed_messages: list[str] = field(default_factory=list)
    observed_silent_normalisations: dict[str, dict[str, Any]] = field(default_factory=dict)
    observed_exception: dict[str, str] | None = None
    positive_confirmed: bool = False
    negative_confirmed: bool = False
    duration_ms: int = 0
    skipped_reason: str | None = None


@dataclass
class Divergence:
    """One observed-vs-expected mismatch.

    ``check_failed`` names the gate-soundness check that surfaced this
    divergence (one of ``positive_raises``, ``negative_does_not_raise``,
    ``message_template_match``, ``message_template_too_dynamic``) when the
    divergence came from the soundness checks added per Decision #12 of the
    invariant-miner adversarial review (`.product/designs/adversarial-review-invariant-miner-2026-04-26.md`).
    Pre-existing expected-vs-observed comparisons leave this ``None``.
    """

    rule_id: str
    field: str
    expected: Any
    observed: Any
    check_failed: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_id": self.rule_id,
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
        }
        if self.check_failed is not None:
            out["check_failed"] = self.check_failed
        return out


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def extract_state(
    obj: Any, *, private_allowlist: Iterable[str] = TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST
) -> dict[str, Any]:
    """Uniform dump of an arbitrary config object's public state.

    Handles Pydantic v2 (``model_dump``), dataclasses, ``__slots__`` classes
    and plain ``__dict__`` classes. Private attributes (``_foo``) are dropped
    unless they appear in ``private_allowlist`` — see module docstring for
    why the allowlist exists.
    """
    allowlist = frozenset(private_allowlist)

    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                return {k: v for k, v in dumped.items() if not k.startswith("_") or k in allowlist}
        except Exception:
            pass

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: getattr(obj, f.name)
            for f in dataclasses.fields(obj)
            if not f.name.startswith("_") or f.name in allowlist
        }

    collected: dict[str, Any] = {}
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        for name in slots:
            if (not name.startswith("_") or name in allowlist) and hasattr(obj, name):
                collected[name] = getattr(obj, name)
    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            if not name.startswith("_") or name in allowlist:
                collected.setdefault(name, value)
    return collected


def diff_input_vs_state(
    kwargs: dict[str, Any], observed_state: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Identify silent normalisations — fields the library changed post-construction.

    Returns ``{field: {"declared": <input>, "observed": <state>}}``.
    """
    diffs: dict[str, dict[str, Any]] = {}
    for field_name, declared in kwargs.items():
        if field_name not in observed_state:
            continue
        observed = observed_state[field_name]
        if declared != observed:
            diffs[field_name] = {
                "declared": _jsonable(declared),
                "observed": _jsonable(observed),
            }
    return diffs


# ---------------------------------------------------------------------------
# Capture primitives
# ---------------------------------------------------------------------------


_WARNING_ONCE_SENTINEL = "\x00LLEM_WARNING_ONCE\x00"
"""Prefix injected by :func:`_patch_warning_once` to distinguish
``logger.warning_once`` records from plain ``logger.warning`` at the
stdlib-record level (HF's ``warning_once`` is ``@lru_cache``-wrapped
``self.warning``, identical in the record stream otherwise)."""


def _attach_loggers(
    loggers: Iterable[str],
) -> tuple[logging.Handler, io.StringIO, list[tuple[logging.Logger, int]]]:
    """Attach a StringIO handler to each named logger.

    Returns the handler, its buffer and a list of ``(logger, previous_level)``
    pairs so the caller can restore levels afterwards.
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    restore: list[tuple[logging.Logger, int]] = []
    for name in loggers:
        logger = logging.getLogger(name)
        restore.append((logger, logger.level))
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return handler, buf, restore


def _detach_loggers(handler: logging.Handler, restore: list[tuple[logging.Logger, int]]) -> None:
    for logger, prev in restore:
        logger.removeHandler(handler)
        logger.setLevel(prev)


def _patch_warning_once() -> Callable[[], None]:
    """Install a sentinel-tagging spy over ``Logger.warning_once``.

    Returns a restore callable the caller must run in ``finally``. No-op when
    HF isn't importable (the attribute is attached at ``transformers.utils.logging``
    import time; outside the HF container the method is absent and there is
    nothing to patch).

    HF's ``warning_once`` is ``@functools.lru_cache``-wrapped at the module
    level — the cache survives across ``run_case`` calls in the same process.
    Without clearing it, a dormancy rule that fires its message on rule N
    would silently no-op on rule N+1 reusing the same template, and the
    vendor classifier would observe ``logger_warning`` (the underlying
    ``warning`` channel) instead of ``logger_warning_once`` for every rule
    after the first hit. Clear the cache on every spy installation so each
    rule sees a clean slate.
    """
    original = getattr(logging.Logger, "warning_once", None)
    if original is None:
        return lambda: None

    # Best-effort: clear HF's process-level lru_cache on warning_once / info_once
    # so successive run_case calls in one process don't trip the dedup wrapper.
    # The wrappers live on ``transformers.utils.logging``; if HF isn't importable
    # we already returned no-op above, so this branch is safe.
    try:
        from transformers.utils import logging as _hf_logging  # type: ignore

        for attr in ("warning_once", "info_once"):
            cached = getattr(_hf_logging, attr, None)
            cache_clear = getattr(cached, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    except ImportError:
        pass

    def spy(self: logging.Logger, msg: Any, *args: Any, **kwargs: Any) -> Any:
        tagged = f"{_WARNING_ONCE_SENTINEL}{msg}" if isinstance(msg, str) else msg
        return original(self, tagged, *args, **kwargs)

    logging.Logger.warning_once = spy  # type: ignore[attr-defined]

    def restore() -> None:
        logging.Logger.warning_once = original  # type: ignore[attr-defined]

    return restore


def run_case(
    callable_fn: Callable[[], Any],
    *,
    logger_names: Iterable[str] = (),
    private_allowlist: Iterable[str] = TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
) -> CaptureBuffers:
    """Run ``callable_fn()`` and capture exceptions / warnings / logger output / state.

    ``callable_fn`` is usually ``lambda: native_type(**kwargs)`` or
    ``lambda: native_type(**kwargs).validate(strict=True)``. Returns a
    :class:`CaptureBuffers` regardless of whether the call raised.
    """
    handler, buf, restore = _attach_loggers(logger_names)
    restore_warning_once = _patch_warning_once()
    start = time.perf_counter()
    exc_type: str | None = None
    exc_msg: str | None = None
    obj: Any = None
    captured_warnings: list[warnings.WarningMessage] = []

    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                obj = callable_fn()
            except Exception as exc:
                exc_type = type(exc).__name__
                exc_msg = str(exc)
            # Snapshot inside the catch_warnings scope so warnings captured
            # alongside an exception are preserved (dormant-then-raise paths).
            captured_warnings = list(recorded or [])
    finally:
        restore_warning_once()
        _detach_loggers(handler, restore)
        duration_ms = int((time.perf_counter() - start) * 1000)

    warnings_tuple = tuple(str(w.message) for w in captured_warnings)

    log_messages = _split_log_buffer(buf.getvalue())
    observed_state = (
        extract_state(obj, private_allowlist=private_allowlist) if obj is not None else None
    )

    return CaptureBuffers(
        exception_type=exc_type,
        exception_message=exc_msg,
        warnings_captured=warnings_tuple,
        logger_messages=log_messages,
        observed_state=observed_state,
        duration_ms=duration_ms,
    )


def _split_log_buffer(raw: str) -> tuple[str, ...]:
    """Split buffer text into one entry per record, dropping empty trailing."""
    if not raw:
        return ()
    lines = [line for line in raw.split("\n") if line.strip()]
    return tuple(lines)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def classify_outcome(capture: CaptureBuffers, silent_normalisations: dict[str, Any]) -> str:
    """Given captured behaviour, compute the observed outcome label.

    Preference order:

    1. Exception raised -> ``"error"``
    2. ``warnings.warn`` captured -> ``"warn"``
    3. Logger message captured -> ``"dormant_announced"``
    4. Silent state change detected -> ``"dormant_silent"``
    5. Nothing observed -> ``"no_op"``
    """
    if capture.exception_type is not None:
        return "error"
    if capture.warnings_captured:
        return "warn"
    if capture.logger_messages:
        return "dormant_announced"
    if silent_normalisations:
        return "dormant_silent"
    return "no_op"


def classify_emission_channel(capture: CaptureBuffers) -> str:
    """Return the corpus-compatible ``emission_channel`` tag.

    ``logger_warning_once`` is distinguished from plain ``logger_warning``
    via the sentinel prepended by :func:`_patch_warning_once`. Mixed
    batches (same rule emitting both forms) classify as
    ``logger_warning_once`` — the dedup-wrapped form is the stricter claim
    on user visibility.
    """
    if capture.exception_type is not None:
        return "none"
    if capture.warnings_captured:
        return "warnings_warn"
    if capture.logger_messages:
        if any(_WARNING_ONCE_SENTINEL in m for m in capture.logger_messages):
            return "logger_warning_once"
        return "logger_warning"
    return "none"


def strip_warning_once_sentinel(messages: Iterable[str]) -> tuple[str, ...]:
    """Remove the ``warning_once`` sentinel from captured messages for envelope output.

    Classification (``classify_emission_channel``) needs the sentinel; downstream
    consumers do not. Call this right before serialising observed messages.
    """
    return tuple(m.replace(_WARNING_ONCE_SENTINEL, "") for m in messages)


# ---------------------------------------------------------------------------
# Expected vs observed comparison
# ---------------------------------------------------------------------------


def compare_expected_vs_observed(
    *,
    rule_id: str,
    expected: dict[str, Any],
    observed_outcome: str,
    observed_emission: str,
    silent_normalisations: dict[str, Any],
) -> list[Divergence]:
    """Return the list of expected-vs-observed divergences for one rule.

    Missing/extra fields on either side are *not* treated as divergence —
    only fields present in ``expected`` are checked. This keeps the
    comparison permissive while still catching drift in the tracked fields.
    """
    divergences: list[Divergence] = []
    expected_outcome = expected.get("outcome")
    if expected_outcome and expected_outcome != observed_outcome:
        divergences.append(
            Divergence(
                rule_id=rule_id,
                field="outcome",
                expected=expected_outcome,
                observed=observed_outcome,
            )
        )
    expected_channel = expected.get("emission_channel")
    if expected_channel and expected_channel != observed_emission:
        divergences.append(
            Divergence(
                rule_id=rule_id,
                field="emission_channel",
                expected=expected_channel,
                observed=observed_emission,
            )
        )

    expected_norm_fields = expected.get("normalised_fields") or []
    if expected_norm_fields:
        missing = [f for f in expected_norm_fields if f not in silent_normalisations]
        if missing:
            divergences.append(
                Divergence(
                    rule_id=rule_id,
                    field="normalised_fields",
                    expected=list(expected_norm_fields),
                    observed=sorted(silent_normalisations.keys()),
                )
            )

    return divergences


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Coerce a value so ``json.dumps`` can handle it without ``default=str``."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return str(value)


# ---------------------------------------------------------------------------
# Gate-soundness helpers (Decision #12 of the invariant-miner adversarial review)
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")
"""Matches a single format-string placeholder like ``{declared_value}`` or ``{}``.

We deliberately do NOT match nested braces - Python's format-string grammar
permits them but the corpus's ``message_template`` strings do not use them
(verified by inspection of ``configs/validation_rules/transformers.yaml``).
A non-greedy non-recursive regex is sufficient and simpler to reason about.
"""

_MIN_STATIC_FRAGMENT_LEN = 4
"""Minimum length for a static fragment to be useful for substring matching.

Below this we treat the template as ``too_dynamic`` - a 3-character substring
like "is " or " a " has too high a coincidental-match rate to be load-bearing.
"""


def message_template_to_substring(template: str) -> str:
    """Extract the longest static fragment from a ``message_template``.

    The corpus's ``message_template`` field is a Python format-string with
    placeholders like ``{declared_value}`` filled in at raise time. To
    compare it against a raised exception's ``str()``, we drop placeholders
    and pick the longest contiguous static run as the substring to match.

    The miner sometimes records the AST literal verbatim, so the template
    may arrive wrapped in ``f'...'`` / ``f"..."`` quoting. We strip the
    f-string prefix + trailing quote before extracting fragments so the
    longest-static-run heuristic doesn't pick up the leading ``f'``.

    Returns the empty string if the template has no static content
    longer than :data:`_MIN_STATIC_FRAGMENT_LEN` characters - the caller
    should treat this as ``message_template_too_dynamic`` and skip the
    substring check (recording a divergence so the rule's author knows
    the template is too placeholder-heavy to verify).

    Examples
    --------
    >>> message_template_to_substring("`{flag}` is set to `{value}` but ...")
    '` is set to `'
    >>> message_template_to_substring("Invalid `cache_implementation` ({val}). Choose one of: ...")
    'Invalid `cache_implementation` ('
    >>> message_template_to_substring("{a}{b}")
    ''
    >>> message_template_to_substring("f'Greedy methods do not support {x}.'")
    'Greedy methods do not support '
    """
    if not template:
        return ""
    normalised = _strip_fstring_quoting(template)
    fragments = _PLACEHOLDER_RE.split(normalised)
    longest = max(fragments, key=len, default="")
    if len(longest.strip()) < _MIN_STATIC_FRAGMENT_LEN:
        return ""
    return longest


def _strip_fstring_quoting(template: str) -> str:
    """Strip a leading ``f'`` / ``f"`` and matching trailing quote, if present.

    Some corpus rules record the AST source literal rather than the
    runtime format-string. ``"f'Greedy methods do not support {x}.'"``
    becomes ``"Greedy methods do not support {x}."`` so the placeholder
    splitter can do its job.
    """
    stripped = template.strip()
    if len(stripped) >= 3 and stripped[:2] in ('f"', "f'") and stripped[-1] == stripped[1]:
        return stripped[2:-1]
    return template


def message_matches_template(observed_message: str, template: str) -> tuple[bool, str]:
    """Check whether ``observed_message`` contains the static fragment of ``template``.

    Returns ``(matched, fragment)``. When the template is too dynamic to
    extract a useful fragment, returns ``(False, "")`` and the caller
    should record a ``message_template_too_dynamic`` divergence rather
    than a substring-mismatch one.

    Comparison is case-insensitive - corpus templates and runtime exception
    messages occasionally differ in capitalisation of opening words.
    """
    fragment = message_template_to_substring(template)
    if not fragment:
        return False, ""
    return fragment.lower() in (observed_message or "").lower(), fragment
