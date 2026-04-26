"""Shared infrastructure for per-engine validation-rule walkers.

Walker depth is fixed at 1 (same module, no helper-call tracing). This
module ships the AST primitives and pattern detectors that future
per-engine walkers will compose to extract validation rules from pinned
library source. No concrete walker ships today; they land as independent
PRs per engine.

- :class:`RuleCandidate` — the walker output type, serialised to the YAML
  corpus entry shape in :mod:`llenergymeasure.config.vendored_rules.loader`.
- :class:`WalkerVersionMismatchError`, :class:`WalkerLandmarkMissingError` —
  fail-loud exceptions CI treats as fatal.
- :func:`check_installed_version` — version-envelope guard for each walker.
- AST helpers (:func:`extract_condition_fields`, :func:`resolve_local_assign`,
  etc.) — deterministic, stateless primitives for AST-based walkers.
- Pattern detectors (``ConditionalRaiseDetector``, etc.) — one class per
  known library rule shape; each fires on one ``ast.If`` body at a time.

Tests cover each primitive on synthetic AST fixtures; the per-engine walkers
run against pinned real libraries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from packaging import version as pkg_version
from packaging.specifiers import SpecifierSet

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Severity = Literal["error", "warn", "dormant"]
EmissionChannel = Literal[
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "runtime_exception",
    "none",
]
Confidence = Literal["high", "medium", "low"]

VALID_CONFIDENCE: frozenset[str] = frozenset(get_args(Confidence))
"""Closed enum for :attr:`WalkerSource.walker_confidence` — tests defer
to this constant rather than duplicating the literal set."""


@dataclass
class WalkerSource:
    """Provenance for a walker-extracted rule candidate."""

    path: str
    method: str
    line_at_scan: int
    walker_confidence: Confidence


@dataclass
class RuleCandidate:
    """One extracted rule candidate.

    Serialised verbatim into ``configs/validation_rules/{engine}.yaml`` after
    human review. Field names match the corpus schema so no translation step
    is needed between walker output and corpus entry.
    """

    id: str
    engine: str
    library: str
    rule_under_test: str
    severity: Severity
    native_type: str
    walker_source: WalkerSource
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    expected_outcome: dict[str, Any]
    message_template: str | None
    references: list[str] = field(default_factory=list)
    added_by: str = "ast_walker"
    added_at: str = ""


# ---------------------------------------------------------------------------
# Error types (all inherit from a common base so per-engine walkers can
# raise-or-collect uniformly at CI time)
# ---------------------------------------------------------------------------


class WalkerError(Exception):
    """Base class for structured walker failures."""


class WalkerVersionMismatchError(WalkerError):
    """Raised when the installed library version is outside the walker's pin.

    The walker is pinned to the library version it was authored against; on
    library-bump PRs, Renovate will trip this error and the maintainer updates
    the walker. See runtime-config-validation.md §4.2.
    """

    def __init__(self, library: str, installed: str, expected: SpecifierSet) -> None:
        super().__init__(
            f"Installed {library}=={installed} is outside walker-pinned range "
            f"{expected!s}. Update scripts/extractors/{library}.py "
            f"(bump TESTED_AGAINST_VERSIONS and re-run against the new source)."
        )
        self.library = library
        self.installed = installed
        self.expected = expected


class WalkerLandmarkMissingError(WalkerError):
    """Raised when an expected source landmark (class/method/file) is missing.

    Library refactors (class renamed, method split, file relocated) trip this
    error and the walker refuses to emit partial output. This is load-bearing
    for the "silent coverage loss becomes a visible CI failure" contract.
    """

    def __init__(self, landmark: str, detail: str = "") -> None:
        msg = f"Walker landmark missing: {landmark}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)
        self.landmark = landmark
        self.detail = detail


# ---------------------------------------------------------------------------
# Version pin guard
# ---------------------------------------------------------------------------


def check_installed_version(library: str, installed: str, expected: SpecifierSet) -> None:
    """Raise :class:`WalkerVersionMismatchError` if ``installed`` isn't in ``expected``.

    ``SpecifierSet.contains(..., prereleases=True)`` allows rc / beta tags,
    which is what we want for Renovate-opened PRs that bump to a prerelease
    tag before a stable one exists.
    """
    try:
        parsed = pkg_version.Version(installed)
    except pkg_version.InvalidVersion as exc:
        raise WalkerVersionMismatchError(library, installed, expected) from exc
    if not expected.contains(parsed, prereleases=True):
        raise WalkerVersionMismatchError(library, installed, expected)


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def call_func_path(call: ast.Call) -> list[str] | None:
    """Return dotted path for a ``Call`` node's func, or ``None`` if opaque.

    ``logger.warning(...)`` → ``["logger", "warning"]``.
    ``foo()()`` → ``None`` (not a pure attribute/name chain).
    """
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def first_string_arg(call: ast.Call) -> str | None:
    """First string-like positional argument of a Call, or ``None``.

    Returns the raw string for ``ast.Constant``, ``ast.unparse`` output for
    f-strings (``ast.JoinedStr``) and ``"...".format(...)`` expressions —
    these are the three message-template shapes observed in the 2026-04-22
    AST-scan PoC across transformers / vLLM / TRT-LLM.
    """
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.JoinedStr):
            return ast.unparse(arg)
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "format"
        ):
            return ast.unparse(arg)
    return None


def extract_condition_fields(condition: ast.expr) -> set[str]:
    """Return the set of ``self.<field>`` attribute names referenced in ``condition``.

    Used by the ``condition_references_self`` filter (rule must reference at
    least one public field of the native type).
    """
    fields: set[str] = set()
    for node in ast.walk(condition):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            fields.add(node.attr)
    return fields


def extract_assign_target(stmt: ast.Assign) -> str | None:
    """Return ``"self.<attr>"`` → ``<attr>`` for a single-target self-assignment.

    ``None`` for anything else (tuple unpacking, subscripts, non-self targets).
    """
    if len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def resolve_local_assign(func: ast.FunctionDef, name: str) -> str | None:
    """Find the first ``name = <string-literal>`` inside ``func`` and return the literal.

    Used for HF's ``greedy_wrong_parameter_msg`` pattern — the message template
    is a local variable defined earlier in the same function body.

    **Scope limitation:** only scans top-level statements in ``func.body``.
    Assignments nested in ``if`` / ``try`` / ``with`` / ``for`` blocks are
    not followed. Returns the *first* matching assignment, so a function
    that rebinds the name later will still surface the earliest value —
    fine for message templates that are constant per function call,
    brittle for names the function reassigns. Suits current HF validate()
    shape; if a future library uses branch-local message templates, the
    walker calling site needs a richer resolver.
    """
    for node in func.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (
            isinstance(tgt, ast.Name)
            and tgt.id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def extract_loop_literal_iterable(loop: ast.For) -> list[Any] | None:
    """Return the literal list/tuple a ``for`` loop iterates over, or ``None``.

    ``for arg in [a, b]:`` → ``[a, b]``; ``for arg in self.x:`` → ``None``.
    Enables one parameterised rule per loop when the iterable is AST-static.
    """
    iter_node = loop.iter
    if not isinstance(iter_node, (ast.List, ast.Tuple)):
        return None
    values: list[Any] = []
    for elt in iter_node.elts:
        if isinstance(elt, ast.Constant):
            values.append(elt.value)
        else:
            return None
    return values


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


@dataclass
class DetectedPattern:
    """One detected rule-body statement inside an ``if X: ...`` branch."""

    severity: Severity
    emission_channel: EmissionChannel
    affected_field: str | None
    message_template: str | None
    detail: str


class ConditionalRaiseDetector:
    """``if X: raise SomeException(...)`` — error rule."""

    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Raise) or stmt.exc is None:
            return None
        if isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name):
            exc_type = stmt.exc.func.id
            msg = first_string_arg(stmt.exc)
            return DetectedPattern(
                severity="error",
                emission_channel="none",
                affected_field=None,
                message_template=msg,
                detail=f"raise {exc_type}",
            )
        return None


class ConditionalSelfAssignDetector:
    """``if X: self.A = B`` — silent dormancy rule.

    The affected field is ``A``. Represents the library silently normalising
    the user's declared value — no warning, no error, but the effective state
    differs from the declared state. vLLM epsilon-clamp is the canonical case.
    """

    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Assign):
            return None
        attr = extract_assign_target(stmt)
        if attr is None:
            return None
        rhs = ast.unparse(stmt.value)
        return DetectedPattern(
            severity="dormant",
            emission_channel="none",
            affected_field=attr,
            message_template=None,
            detail=f"self.{attr} = {rhs}",
        )


class ConditionalWarningsWarnDetector:
    """``if X: warnings.warn(...)`` — announced warn rule."""

    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        path = call_func_path(stmt.value)
        if path != ["warnings", "warn"]:
            return None
        return DetectedPattern(
            severity="warn",
            emission_channel="warnings_warn",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="warnings.warn",
        )


class ConditionalLoggerWarningDetector:
    """``if X: logger.warning(...)`` / ``logger.warning_once(...)`` — announced.

    Strictly matches two-element paths (``logger.<method>``). Patterns like
    ``logger.sub.warning(...)`` or ``self.logger.warning(...)`` do NOT match
    — if a real library rule uses a non-top-level logger attribute, the
    walker for that library must supply its own detector.
    """

    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        path = call_func_path(stmt.value)
        if path is None or len(path) != 2 or path[0] != "logger":
            return None
        method = path[-1]
        if method not in {"warning", "warning_once", "error"}:
            return None
        channel: EmissionChannel = (
            "logger_warning_once" if method == "warning_once" else "logger_warning"
        )
        return DetectedPattern(
            severity="warn" if method != "error" else "error",
            emission_channel=channel,
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail=".".join(path),
        )


class MinorIssuesDictAssignDetector:
    """HF-specific: ``if X: minor_issues[key] = msg.format(...)``.

    Represents HF's announced-dormancy pattern — the library composes a
    ``minor_issues`` dict during ``GenerationConfig.validate()`` and later
    emits it via ``logger.warning_once`` (or raises if ``strict=True``).
    """

    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        target = stmt.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            return None
        if target.value.id != "minor_issues":
            return None
        key: str | None = None
        if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
            key = target.slice.value
        msg: str | None = None
        if isinstance(stmt.value, ast.Call):
            msg = ast.unparse(stmt.value)
        elif isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            msg = stmt.value.value
        return DetectedPattern(
            severity="dormant",
            emission_channel="minor_issues_dict",
            affected_field=key,
            message_template=msg,
            detail="minor_issues[key] = msg",
        )


_DEFAULT_DETECTORS: tuple[
    ConditionalRaiseDetector
    | ConditionalSelfAssignDetector
    | ConditionalWarningsWarnDetector
    | ConditionalLoggerWarningDetector
    | MinorIssuesDictAssignDetector,
    ...,
] = (
    ConditionalRaiseDetector(),
    ConditionalSelfAssignDetector(),
    ConditionalWarningsWarnDetector(),
    ConditionalLoggerWarningDetector(),
    MinorIssuesDictAssignDetector(),
)
"""Private default detector bundle — ordered by specificity.

Per-engine walkers are expected to assemble their own tuple from the
individual detector classes, so this constant is not part of the public
API. Ordering rationale for the default: ``raise`` before self-assign
ensures raise+rollback chains are attributed to the raise; ``minor_issues``
before generic ``self.x = y`` ensures HF's dict-assign is picked up
before the fallback self-assign detector.
"""


def default_detectors() -> tuple[
    ConditionalRaiseDetector
    | ConditionalSelfAssignDetector
    | ConditionalWarningsWarnDetector
    | ConditionalLoggerWarningDetector
    | MinorIssuesDictAssignDetector,
    ...,
]:
    """Return the default detector tuple — walkers copy / slice this to taste.

    **Stability contract:** the tuple's *contents* and *ordering* may grow
    or shift as new detectors are added to cover additional libraries.
    Walkers that need behaviour-stable detection across corpus-pipeline
    reruns should capture a local tuple (e.g. select-by-class-name) at
    walker definition time rather than calling this each walk.
    """
    return _DEFAULT_DETECTORS


# ---------------------------------------------------------------------------
# Filters (false-positive guards)
# ---------------------------------------------------------------------------


def filter_condition_references_self(condition: ast.expr, public_fields: frozenset[str]) -> bool:
    """True iff condition references at least one public field via ``self.<field>``.

    Drops argument-dependent rules (``if strict: raise``) and private-state
    rules (``if self._initialized: ...``) that don't constrain user config.

    **Known false-negative:** kwarg-gated rules like HF's
    ``if strict: raise ValueError(...)`` inside ``validate(self, strict=False)``
    get dropped even though they CAN represent real config constraints when
    the kwarg's default changes the behaviour. If a library exposes such a
    rule, the walker for it must supply its own condition-inclusion filter
    (via composition) rather than relying on this one. Pinning this
    behaviour explicitly here so the contract is discoverable at the
    definition site, not only in design-review notes.
    """
    referenced = extract_condition_fields(condition)
    return bool(referenced & public_fields)


def filter_target_is_public_field(pattern: DetectedPattern, public_fields: frozenset[str]) -> bool:
    """For self-assign patterns, ``affected_field`` must be a public field."""
    if pattern.affected_field is None:
        # Not a self-assign — filter doesn't apply (neutral pass).
        return True
    return pattern.affected_field in public_fields


def filter_kwargs_positive_derivable(condition: ast.expr) -> bool:
    """True iff a representative positive kwargs dict can be synthesised.

    Accepts: ``self.field op literal``, ``self.field is None``, ``not isinstance(self.field, T)``,
    boolean combinations of the above, and HF's ``hasattr(self, arg)`` loop pattern.

    Rejects: conditions whose truth depends on opaque function calls against
    external state (``if some_module.flag(): raise``).
    """
    # Heuristic: a condition is derivable if we can find at least one
    # self.<attr> reference AND every Call in the condition is either
    # a builtin predicate (isinstance / hasattr / getattr / len) or a
    # method on self. This matches the empirically-common shapes without
    # claiming universal coverage.
    if not extract_condition_fields(condition):
        # A condition must reference self at all to be useful.
        return False
    safe_call_names = {"isinstance", "hasattr", "getattr", "len", "version"}
    for node in ast.walk(condition):
        if not isinstance(node, ast.Call):
            continue
        path = call_func_path(node)
        if path is None:
            return False
        if path[0] == "self":
            # self.<method>(...) — accept (helper-call; tracer may expand).
            continue
        if path[-1] in safe_call_names:
            continue
        # Opaque external call (e.g., importlib.util.find_spec) — not
        # mechanically derivable into positive kwargs.
        return False
    return True


def score_confidence(pass_count: int) -> Confidence:
    """Map 0-3 filter-pass count to a confidence tier.

    3 passes → ``high``. 2 → ``medium``. 0-1 → ``low``.

    The tier thresholds are deliberately simple: three-hot = high, two-hot =
    medium, else low. No target distribution is enforced — the design's
    earlier 60/25/15 target was descoped pending empirical calibration
    against real walker output. Callers should treat ``medium`` / ``low``
    as "needs human review before landing", not as a quality score.
    """
    if pass_count >= 3:
        return "high"
    if pass_count == 2:
        return "medium"
    return "low"


def candidate_to_dict(candidate: RuleCandidate) -> dict[str, Any]:
    """Serialize a RuleCandidate to the YAML corpus dict shape.

    Used by all per-engine walkers (introspection + AST) to emit staging files.
    Ensures consistent schema across all extractors.
    """
    return {
        "id": candidate.id,
        "engine": candidate.engine,
        "library": candidate.library,
        "rule_under_test": candidate.rule_under_test,
        "severity": candidate.severity,
        "native_type": candidate.native_type,
        "walker_source": {
            "path": candidate.walker_source.path,
            "method": candidate.walker_source.method,
            "line_at_scan": candidate.walker_source.line_at_scan,
            "walker_confidence": candidate.walker_source.walker_confidence,
        },
        "match": {
            "engine": candidate.engine,
            "fields": candidate.match_fields,
        },
        "kwargs_positive": candidate.kwargs_positive,
        "kwargs_negative": candidate.kwargs_negative,
        "expected_outcome": candidate.expected_outcome,
        "message_template": candidate.message_template,
        "references": candidate.references,
        "added_by": candidate.added_by,
        "added_at": candidate.added_at,
    }


# ---------------------------------------------------------------------------
# Class helpers
# ---------------------------------------------------------------------------


def find_class(module: ast.Module, class_name: str) -> ast.ClassDef | None:
    """Return the first ``ClassDef`` named ``class_name`` in ``module``."""
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    """Return the first ``FunctionDef`` named ``method_name`` on ``cls``."""
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None
