"""AST walker for the transformers library — recall-first rule extraction.

Walks ``GenerationConfig.validate()`` and the depth-1 helpers it calls
(``WatermarkingConfig.validate``, ``SynthIDTextWatermarkingConfig.validate``)
and emits structured rule candidates describing every conditional that
raises, warns, silently normalises, or populates ``minor_issues``.

Why source-AST walking instead of pure introspection
----------------------------------------------------
``transformers_introspection.py`` exercises ``GenerationConfig.validate(strict=True)``
against probe values and lifts ``minor_issues`` / ``ValueError`` messages.
That works for one-axis rules but loses the **shape** of cross-field
predicates: the introspection layer sees the message
``"`num_beams` should be divisible by `num_beam_groups`"`` but cannot tell
that the underlying check is ``num_beams % num_beam_groups != 0``. This
walker reads predicate structure directly from the AST.

Recall over precision
---------------------
Vendor CI runs every emitted rule against the real library; divergent rules
fail there. So this walker errs toward emitting candidates with
``walker_confidence: low`` rather than dropping them. A predicate the walker
can't fully translate (opaque function call, complex method chain) emits
the surrounding rule and notes the dropped sub-clause in a YAML comment.

Output
------
Writes ``configs/validation_rules/_staging/transformers_ast.yaml`` —
consumed downstream by ``scripts/extractors/build_corpus.py`` (other subagent's
territory). Schema mirrors ``configs/validation_rules/transformers.yaml``
exactly: ``schema_version``, ``engine``, ``engine_version``, ``rules: [...]``.

Run::

    PYTHONPATH=.:src python3.10 scripts/extractors/transformers_ast_extractor.py
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import importlib.metadata
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make sibling modules importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Defend against the script-directory shadowing site-packages. When run as
# `python3 scripts/extractors/transformers_ast_extractor.py`, Python prepends
# `scripts/extractors/` to sys.path, where a sibling `transformers.py` lives —
# `import transformers` would resolve to that local stub instead of the real
# installed package. Strip the script dir before any third-party imports.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
# Also guard against the empty-string entry that means "current cwd".
sys.path[:] = [p for p in sys.path if p != ""]

from scripts.extractors._base import (  # noqa: E402  (late import after sys.path)
    call_func_path,
    find_class,
    find_method,
    first_string_arg,
)

# Why we DON'T import _base's detector classes (ConditionalRaiseDetector,
# ConditionalSelfAssignDetector, etc.) and instead define parallel
# ``_detect_*`` functions below: the base detectors emit ``DetectedPattern``
# which carries severity / channel / affected_field but not the structured
# ``FieldPredicate`` data we need for cross-field corpus rules using
# operators like ``not_divisible_by`` and ``@field_ref``. Extending the base
# classes would either change their public ``DetectedPattern`` shape
# (breaking the introspection extractor that currently consumes it) or
# require lossy adapter shims at every emission site. With one walker live
# today, the cheaper choice is per-walker detectors that emit a richer
# local ``DetectedBody`` type. Revisit when a second walker (vLLM, TRT-LLM)
# lands and we can see whether the parallel detector logic is genuinely
# divergent or accidentally so.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE = "transformers"
LIBRARY = "transformers"
LIBRARY_BNB = "bitsandbytes"
NATIVE_TYPE_GEN = "transformers.GenerationConfig"
NATIVE_TYPE_WMK = "transformers.WatermarkingConfig"
NATIVE_TYPE_SYNTH = "transformers.SynthIDTextWatermarkingConfig"
NATIVE_TYPE_BNB = "transformers.BitsAndBytesConfig"

# Field-path namespace for GenerationConfig fields. The corpus convention
# (see existing transformers.yaml entries) puts every GenerationConfig
# attribute under transformers.sampling.<field>, even those that aren't
# strictly "sampling" parameters — that namespace is how the project's
# Pydantic config model exposes them.
GENCONFIG_NAMESPACE = "transformers.sampling"

# BitsAndBytesConfig fields are exposed at the top level of the
# project's TransformersConfig Pydantic model (e.g. config.transformers.load_in_4bit),
# NOT nested under a quant sub-model. See src/llenergymeasure/config/engine_configs.py
# class TransformersConfig where load_in_4bit / bnb_4bit_* / llm_int8_* are direct fields.
BNB_NAMESPACE = "transformers"

# A small allowlist of decoder-config field paths used by the project's
# Pydantic models. Most generation-config fields live under .sampling.
# Some (max_new_tokens, etc.) live under .decoder. The walker emits paths
# under sampling. by default; if vendor CI flags a path mismatch we can
# refine. Recall first.

# Default values used to synthesise positive / negative kwargs when the
# walker cannot infer better ones from the predicate.
_DEFAULT_BY_TYPE: dict[type, Any] = {
    bool: True,
    int: 1,
    float: 1.0,
    str: "x",
    list: [],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FieldPredicate:
    """One atomic predicate over ``self.<field>`` extracted from a condition.

    ``op`` is one of the corpus loader's operator names (``==``, ``!=``,
    ``<``, ``<=``, ``>``, ``>=``, ``in``, ``not_in``, ``present``, ``absent``,
    ``type_is``, ``type_is_not``, ``divisible_by``, ``not_divisible_by``).
    ``rhs`` is either a Python literal or a string starting with ``@`` for a
    cross-field reference resolved by the loader.
    """

    field: str
    op: str
    rhs: Any
    confidence_penalty: int = 0
    """How much this predicate degrades the rule's confidence (0 = none)."""


@dataclass
class ExtractedCondition:
    """An ``ast.expr`` condition translated into a list of FieldPredicates.

    ``unparseable_clauses`` records sub-clauses the walker couldn't translate
    — surfaced into the YAML rule's ``rule_under_test`` so reviewers see what
    was dropped.
    """

    predicates: list[FieldPredicate]
    unparseable_clauses: list[str]


@dataclass
class DetectedBody:
    """One detected raise/warn/assign inside an ``if`` body."""

    severity: str  # "error" | "warn" | "dormant"
    outcome: str  # "error" | "warn" | "dormant_announced" | "dormant_silent"
    emission_channel: str
    affected_field: str | None
    message_template: str | None
    detail: str
    minor_issues_key: str | None = None


@dataclass
class RuleCandidate:
    """A walker-extracted rule candidate ready to emit as YAML."""

    id: str
    native_type: str
    method: str
    line: int
    rule_under_test: str
    severity: str
    outcome: str
    emission_channel: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    message_template: str | None
    references: list[str]
    confidence: str
    library: str = LIBRARY
    """Owning library — overridden for BNB (``bitsandbytes``) etc."""
    source_path: str | None = None
    """Site-packages-relative source path. Falls back to the walker's
    primary module path when None — used by GenerationConfig rules; BNB
    rules sit in a different source file and override this."""
    normalised_fields: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Predicate extraction
# ---------------------------------------------------------------------------


_COMPARE_OP_NAMES: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not_in",
}


def _self_attr_name(node: ast.expr) -> str | None:
    """If ``node`` is ``self.<name>``, return ``<name>``; else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _literal_value(node: ast.expr) -> tuple[bool, Any]:
    """Try to resolve ``node`` to a Python literal.

    Returns ``(True, value)`` on success, ``(False, None)`` otherwise.
    Supports ``ast.Constant``, ``ast.UnaryOp(USub, Constant)``, and tuples /
    lists / sets of constants (closed-set membership shapes).
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, v = _literal_value(node.operand)
        if ok and isinstance(v, (int, float)):
            return True, -v
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[Any] = []
        for elt in node.elts:
            ok, v = _literal_value(elt)
            if not ok:
                return False, None
            values.append(v)
        return True, list(values)
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        # Python 3.10 always uses ast.Constant for these, kept defensively.
        return True, {"True": True, "False": False, "None": None}[node.id]
    return False, None


def _rhs_from_node(node: ast.expr) -> tuple[bool, Any, int]:
    """Resolve ``node`` into a corpus-loader RHS value.

    Returns ``(ok, rhs, penalty)``. ``rhs`` is a Python literal or an
    ``"@<name>"`` cross-field reference. ``penalty`` is the confidence
    penalty (0 if clean, 1 if we made a permissive guess).
    """
    # Literal
    ok, v = _literal_value(node)
    if ok:
        return True, v, 0
    # Cross-field reference: self.<name>
    name = _self_attr_name(node)
    if name is not None:
        return True, f"@{name}", 0
    # ast.Name resolved as a module-level constant (e.g. ALL_CACHE_IMPLEMENTATIONS):
    # treat as opaque but flag low confidence — the loader can't replay it,
    # but vendor CI can run kwargs_positive / kwargs_negative empirically.
    if isinstance(node, ast.Name):
        return False, ast.unparse(node), 1
    return False, ast.unparse(node), 1


def _modulo_field_pair(node: ast.expr) -> tuple[str, str] | None:
    """If ``node`` is ``self.<a> % self.<b>``, return ``(a, b)``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        a = _self_attr_name(node.left)
        b = _self_attr_name(node.right)
        if a is not None and b is not None:
            return a, b
    return None


def _extract_compare(cmp: ast.Compare) -> tuple[list[FieldPredicate], list[str]]:
    """Translate an ``ast.Compare`` into FieldPredicates.

    Handles common shapes:
    - ``self.x op literal`` / ``self.x op self.y``
    - ``self.x % self.y != 0`` → ``not_divisible_by`` cross-field predicate
    - ``self.x is None`` / ``self.x is not None``
    - chained ``literal <= self.x <= literal`` (split into two predicates)

    Anything we can't translate is appended to the unparseable list.
    """
    # Pre-pass for modulo-divisibility shape: ``self.a % self.b != 0`` /
    # ``self.a % self.b == 0``. The corpus operator vocabulary expresses
    # this directly; capture it before the generic compare path drops it
    # into unparseable.
    if len(cmp.ops) == 1 and len(cmp.comparators) == 1:
        op = cmp.ops[0]
        right = cmp.comparators[0]
        pair = _modulo_field_pair(cmp.left)
        if pair is not None and isinstance(op, (ast.NotEq, ast.Eq)):
            ok, rhs = _literal_value(right)
            if ok and rhs == 0:
                a, b = pair
                # ``a % b != 0`` -> not_divisible_by; ``a % b == 0`` -> divisible_by.
                ast_op = "not_divisible_by" if isinstance(op, ast.NotEq) else "divisible_by"
                return [
                    FieldPredicate(
                        field=a,
                        op=ast_op,
                        rhs=f"@{b}",
                        confidence_penalty=0,
                    )
                ], []
    preds: list[FieldPredicate] = []
    unparseable: list[str] = []

    # Walk the chain pairwise: (left, op0, comparator0), (comparator0, op1, comparator1), ...
    operands = [cmp.left, *cmp.comparators]
    for left, op, right in zip(operands, cmp.ops, cmp.comparators, strict=False):
        # `is`/`is not` — corpus loader has no `is` operator. Map `is None` to
        # `absent`, `is not None` to `present`.
        if isinstance(op, (ast.Is, ast.IsNot)):
            field_name = _self_attr_name(left)
            ok, rhs = _literal_value(right)
            if field_name is not None and ok and rhs is None:
                preds.append(
                    FieldPredicate(
                        field=field_name,
                        op="absent" if isinstance(op, ast.Is) else "present",
                        rhs=True,
                    )
                )
                continue
            unparseable.append(ast.unparse(cmp))
            continue
        op_name = _COMPARE_OP_NAMES.get(type(op))
        if op_name is None:
            unparseable.append(ast.unparse(cmp))
            continue

        # Identify which side is the self.<field>
        left_field = _self_attr_name(left)
        right_field = _self_attr_name(right)

        if left_field is not None:
            ok, rhs, penalty = _rhs_from_node(right)
            if ok:
                preds.append(
                    FieldPredicate(
                        field=left_field, op=op_name, rhs=rhs, confidence_penalty=penalty
                    )
                )
            else:
                unparseable.append(ast.unparse(cmp))
        elif right_field is not None:
            # Flip the operator: literal < self.x  ->  self.x > literal
            flipped = {
                "<": ">",
                "<=": ">=",
                ">": "<",
                ">=": "<=",
                "==": "==",
                "!=": "!=",
            }.get(op_name)
            ok, rhs, penalty = _rhs_from_node(left)
            if flipped is not None and ok:
                preds.append(
                    FieldPredicate(
                        field=right_field,
                        op=flipped,
                        rhs=rhs,
                        confidence_penalty=penalty,
                    )
                )
            else:
                unparseable.append(ast.unparse(cmp))
        else:
            # Neither side is a self.<field> ref. Skip.
            unparseable.append(ast.unparse(cmp))
    return preds, unparseable


def _extract_call_predicate(call: ast.Call) -> tuple[list[FieldPredicate], list[str]]:
    """Predicate from a Call: ``isinstance(self.x, T)``, ``hasattr(self, 'x')``."""
    path = call_func_path(call)
    if path is None:
        return [], [ast.unparse(call)]
    head = path[-1]
    if head == "isinstance" and len(call.args) == 2:
        target = call.args[0]
        type_arg = call.args[1]
        field_name = _self_attr_name(target)
        if field_name is None:
            return [], [ast.unparse(call)]
        # Type names (single class or tuple thereof)
        names = _isinstance_type_names(type_arg)
        if not names:
            return [], [ast.unparse(call)]
        spec_rhs: Any = names[0] if len(names) == 1 else names
        return [
            FieldPredicate(
                field=field_name,
                op="type_is",
                rhs=spec_rhs,
                confidence_penalty=0,
            )
        ], []
    if head == "hasattr" and len(call.args) == 2:
        target = call.args[0]
        name_arg = call.args[1]
        if (
            isinstance(target, ast.Name)
            and target.id == "self"
            and isinstance(name_arg, ast.Constant)
            and isinstance(name_arg.value, str)
        ):
            return [
                FieldPredicate(
                    field=name_arg.value,
                    op="present",
                    rhs=True,
                    confidence_penalty=0,
                )
            ], []
    return [], [ast.unparse(call)]


def _isinstance_type_names(node: ast.expr) -> list[str]:
    """Extract bare class names from an ``isinstance`` second-arg node."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(elt.attr)
            else:
                return []
        return names
    return []


def _extract_unary_not(node: ast.UnaryOp) -> tuple[list[FieldPredicate], list[str]]:
    """``not <expr>`` — invert the inner predicate set if possible."""
    inner_preds, inner_un = extract_predicates(node.operand)
    inverted: list[FieldPredicate] = []
    inversion_map = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "in": "not_in",
        "not_in": "in",
        "present": "absent",
        "absent": "present",
        "type_is": "type_is_not",
        "type_is_not": "type_is",
        "divisible_by": "not_divisible_by",
        "not_divisible_by": "divisible_by",
    }
    if len(inner_preds) == 1 and inner_preds[0].op in inversion_map:
        p = inner_preds[0]
        inverted.append(
            FieldPredicate(
                field=p.field,
                op=inversion_map[p.op],
                rhs=p.rhs,
                confidence_penalty=p.confidence_penalty,
            )
        )
        return inverted, inner_un
    # If inversion isn't local, we can't reliably express it: surface as
    # unparseable. Vendor CI will catch divergence.
    return [], [ast.unparse(node), *inner_un]


def extract_predicates(condition: ast.expr) -> tuple[list[FieldPredicate], list[str]]:
    """Translate a boolean condition AST into a list of AND-combined predicates.

    BoolOp(And, ...) → AND-combined; we recurse and concatenate.
    BoolOp(Or, ...) → can't represent OR in match.fields — surface the entire
    OR clause as unparseable; the rule emits with low confidence.
    Compare → extract via ``_extract_compare``.
    Call (isinstance/hasattr) → ``_extract_call_predicate``.
    UnaryOp(Not, ...) → invert inner.
    Attribute (bare ``self.x``) → ``self.x`` truthiness ≈ "present and not falsy"
        — emit ``present`` predicate with low confidence (loader can't capture
        Python truthiness exactly but recall-first wins).
    """
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        preds: list[FieldPredicate] = []
        un: list[str] = []
        for value in condition.values:
            sub_p, sub_u = extract_predicates(value)
            preds.extend(sub_p)
            un.extend(sub_u)
        return preds, un
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
        # OR semantics aren't expressible; surface every disjunct as
        # unparseable so reviewers see the structure.
        return [], [ast.unparse(condition)]
    if isinstance(condition, ast.Compare):
        return _extract_compare(condition)
    if isinstance(condition, ast.Call):
        return _extract_call_predicate(condition)
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        return _extract_unary_not(condition)
    # Bare ``self.x`` truthiness ≈ "present" — degrade confidence.
    field_name = _self_attr_name(condition)
    if field_name is not None:
        return [
            FieldPredicate(
                field=field_name,
                op="present",
                rhs=True,
                confidence_penalty=1,
            )
        ], []
    return [], [ast.unparse(condition)]


def negate_predicates(preds: list[FieldPredicate]) -> list[FieldPredicate]:
    """Return predicates that satisfy ``not (and-of-preds)`` for kwargs_negative.

    For an AND-combined predicate set, the simplest negation is to flip the
    *last* predicate (so its kwargs_negative differs from kwargs_positive in
    only one field). This isn't a logical negation — it's a "near miss" used
    to give the vendor CI loop a value that doesn't trigger the rule.
    """
    if not preds:
        return []
    inversion_map = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "in": "not_in",
        "not_in": "in",
        "present": "absent",
        "absent": "present",
        "type_is": "type_is_not",
        "type_is_not": "type_is",
        "divisible_by": "not_divisible_by",
        "not_divisible_by": "divisible_by",
    }
    last = preds[-1]
    flipped_op = inversion_map.get(last.op, last.op)
    return [
        *preds[:-1],
        FieldPredicate(
            field=last.field,
            op=flipped_op,
            rhs=last.rhs,
            confidence_penalty=last.confidence_penalty,
        ),
    ]


# ---------------------------------------------------------------------------
# Body detectors
# ---------------------------------------------------------------------------


def _detect_raise(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return None
    msg: str | None = None
    detail = "raise"
    if isinstance(stmt.exc, ast.Call):
        if isinstance(stmt.exc.func, ast.Name):
            detail = f"raise {stmt.exc.func.id}"
        msg = first_string_arg(stmt.exc)
        if msg is None:
            # Sometimes the message is a concatenation: ``msg_prefix + "..."``.
            for arg in stmt.exc.args:
                if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
                    msg = ast.unparse(arg)
                    break
    return DetectedBody(
        severity="error",
        outcome="error",
        emission_channel="none",
        affected_field=None,
        message_template=msg,
        detail=detail,
    )


def _detect_assert(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Assert):
        return None
    msg: str | None = None
    if isinstance(stmt.msg, ast.Constant) and isinstance(stmt.msg.value, str):
        msg = stmt.msg.value
    return DetectedBody(
        severity="error",
        outcome="error",
        emission_channel="none",
        affected_field=None,
        message_template=msg,
        detail="assert",
    )


def _detect_warnings_warn(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path != ["warnings", "warn"]:
        return None
    return DetectedBody(
        severity="warn",
        outcome="warn",
        emission_channel="warnings_warn",
        affected_field=None,
        message_template=first_string_arg(stmt.value),
        detail="warnings.warn",
    )


def _detect_logger_warning(stmt: ast.stmt) -> DetectedBody | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path is None or len(path) != 2 or path[0] != "logger":
        return None
    method = path[-1]
    if method == "warning":
        return DetectedBody(
            severity="warn",
            outcome="warn",
            emission_channel="logger_warning",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="logger.warning",
        )
    if method == "warning_once":
        return DetectedBody(
            severity="warn",
            outcome="warn",
            emission_channel="logger_warning_once",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="logger.warning_once",
        )
    if method == "error":
        return DetectedBody(
            severity="error",
            outcome="error",
            emission_channel="logger_warning",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="logger.error",
        )
    return None


def _detect_minor_issues(stmt: ast.stmt) -> DetectedBody | None:
    """``minor_issues[<key>] = <message>`` — HF announced-dormancy pattern."""
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
    return DetectedBody(
        severity="dormant",
        outcome="dormant_announced",
        emission_channel="logger_warning_once",
        affected_field=key,
        message_template=msg,
        detail="minor_issues[key] = msg",
        minor_issues_key=key,
    )


def _detect_self_assign(stmt: ast.stmt) -> DetectedBody | None:
    """``self.<field> = <expr>`` — silent normalisation."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return DetectedBody(
            severity="dormant",
            outcome="dormant_silent",
            emission_channel="none",
            affected_field=target.attr,
            message_template=None,
            detail=f"self.{target.attr} = {ast.unparse(stmt.value)}",
        )
    return None


_DETECTORS = (
    _detect_raise,
    _detect_assert,
    _detect_warnings_warn,
    _detect_logger_warning,
    _detect_minor_issues,
    _detect_self_assign,
)


def detect_body(stmt: ast.stmt) -> DetectedBody | None:
    """Run all detectors in order; return first match (or None)."""
    for det in _DETECTORS:
        result = det(stmt)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Walker proper
# ---------------------------------------------------------------------------


@dataclass
class _IfFrame:
    """Conditions accumulated as we descend into nested If / For statements."""

    predicates: list[FieldPredicate]
    unparseable: list[str]


def _flatten_if_chain(if_node: ast.If) -> list[tuple[ast.expr | None, list[ast.stmt]]]:
    """Flatten ``if/elif/else`` chains into [(cond, body), ..., (None, else_body)]."""
    out: list[tuple[ast.expr | None, list[ast.stmt]]] = []
    current: ast.If | None = if_node
    while current is not None:
        out.append((current.test, current.body))
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        else:
            if current.orelse:
                out.append((None, current.orelse))
            break
    return out


def _negate_branch_predicates(
    preds: list[FieldPredicate],
) -> tuple[list[FieldPredicate], list[str]]:
    """Express ``not (and-of-preds)`` for the else-branch enclosing context.

    Strict: only invertible if the AND-set has exactly one predicate (whose
    operator has a known inversion). Else, surface as an unparseable
    "else of <conditions>" annotation — the rule emits with low confidence.
    """
    if len(preds) != 1:
        return [], [f"else of: {' AND '.join(f'{p.field} {p.op} {p.rhs}' for p in preds)}"]
    inversion_map = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
        "in": "not_in",
        "not_in": "in",
        "present": "absent",
        "absent": "present",
        "type_is": "type_is_not",
        "type_is_not": "type_is",
    }
    p = preds[0]
    new_op = inversion_map.get(p.op)
    if new_op is None:
        return [], [f"else of: {p.field} {p.op} {p.rhs}"]
    return [
        FieldPredicate(
            field=p.field, op=new_op, rhs=p.rhs, confidence_penalty=p.confidence_penalty
        ),
    ], []


def walk_function(
    func: ast.FunctionDef,
    *,
    native_type: str,
    method_name: str,
    namespace: str,
    rel_source_path: str,
    today: str,
    library: str = LIBRARY,
    class_short_name: str | None = None,
) -> list[RuleCandidate]:
    """Walk a single function body and return rule candidates.

    ``library`` is recorded on each emitted rule (BNB rules use
    ``bitsandbytes`` rather than ``transformers``). ``class_short_name`` is
    used in human-readable rule descriptions; defaults to the bare class
    suffix of ``native_type``.
    """
    rules: list[RuleCandidate] = []
    seen_ids: set[str] = set()
    counter: dict[str, int] = {}

    def emit(rule: RuleCandidate) -> None:
        # De-duplicate on id: append a numeric suffix when needed.
        base = rule.id
        if base in seen_ids:
            counter[base] = counter.get(base, 1) + 1
            rule.id = f"{base}__{counter[base]}"
        seen_ids.add(rule.id)
        rules.append(rule)

    def descend(stmts: list[ast.stmt], frame: _IfFrame) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                _handle_if(stmt, frame)
                continue
            if isinstance(stmt, ast.For):
                # Walker depth-1: dive into for-loops and treat their body as
                # if scoped under hasattr-style accumulator. The HF
                # ``for arg in generate_arguments: if hasattr(self, arg): raise``
                # pattern is what we're catching. We don't accumulate the for's
                # iterable into predicates (it's a literal tuple of strings),
                # but we descend so the inner if is visited.
                descend(stmt.body, frame)
                continue
            # Bare detected statements outside an If: handle as a no-precondition
            # rule. Rare in practice; skipped.
            continue

    def _handle_if(if_node: ast.If, frame: _IfFrame) -> None:
        branches = _flatten_if_chain(if_node)
        # For each branch K, the implicit precondition is "no prior branch's
        # condition was true". For an explicit else (cond=None), that's all
        # we have; for an elif, we add the elif's own condition on top.
        prior_branch_negations: list[list[FieldPredicate]] = []
        prior_branch_unparseables: list[list[str]] = []
        for cond, body in branches:
            if cond is not None:
                own_preds, own_un = extract_predicates(cond)
            else:
                own_preds = []
                own_un = []
            # Implicit preconditions: every prior branch's condition was False.
            implicit_preds: list[FieldPredicate] = []
            implicit_un: list[str] = []
            for prior_neg, prior_un in zip(
                prior_branch_negations, prior_branch_unparseables, strict=False
            ):
                implicit_preds.extend(prior_neg)
                implicit_un.extend(prior_un)
            local_frame = _IfFrame(
                predicates=[*frame.predicates, *implicit_preds, *own_preds],
                unparseable=[*frame.unparseable, *implicit_un, *own_un],
            )
            _emit_for_body(body, local_frame, line_at_scan=if_node.lineno)
            # Recurse into the branch body
            descend(body, local_frame)
            # Compute "this branch's condition was False" for downstream branches.
            if cond is not None:
                neg_preds, neg_un = _negate_branch_predicates(own_preds)
                # If we couldn't cleanly negate, surface own_un + a generic note.
                if not neg_preds:
                    prior_branch_negations.append([])
                    prior_branch_unparseables.append(
                        [f"prior branch unnegatable: {ast.unparse(cond)}"]
                    )
                else:
                    prior_branch_negations.append(neg_preds)
                    prior_branch_unparseables.append(neg_un)

    short_name = class_short_name or native_type.rsplit(".", 1)[-1]

    def _emit_for_body(body: list[ast.stmt], frame: _IfFrame, *, line_at_scan: int) -> None:
        for stmt in body:
            detected = detect_body(stmt)
            if detected is None:
                continue
            rule = _build_rule(
                frame=frame,
                detected=detected,
                native_type=native_type,
                method_name=method_name,
                namespace=namespace,
                rel_source_path=rel_source_path,
                line_at_scan=getattr(stmt, "lineno", line_at_scan),
                today=today,
                library=library,
                class_short_name=short_name,
            )
            if rule is not None:
                emit(rule)

    descend(func.body, _IfFrame(predicates=[], unparseable=[]))
    return rules


def _build_rule(
    *,
    frame: _IfFrame,
    detected: DetectedBody,
    native_type: str,
    method_name: str,
    namespace: str,
    rel_source_path: str,
    line_at_scan: int,
    today: str,
    library: str = LIBRARY,
    class_short_name: str | None = None,
) -> RuleCandidate | None:
    """Assemble a RuleCandidate from accumulated predicates + detected body."""
    preds = list(frame.predicates)

    # If the detected body affects a field (minor_issues key, self-assign),
    # that field is the rule's *subject*. Place a ``present`` predicate on
    # it last so the loader's ``last field = subject`` convention picks it
    # up for the {declared_value} substitution.
    subject_field = detected.affected_field

    # If we have no predicates at all and no subject field, drop — too noisy.
    if not preds and subject_field is None:
        return None

    if subject_field is not None:
        # Avoid duplicate predicate on subject; if one already exists, leave it.
        already = any(p.field == subject_field for p in preds)
        if not already:
            preds.append(
                FieldPredicate(
                    field=subject_field,
                    op="present",
                    rhs=True,
                    confidence_penalty=0,
                )
            )

    match_fields = _build_match_fields(preds, namespace)
    kwargs_pos = _synthesise_kwargs(preds, sense="positive")
    kwargs_neg = _synthesise_kwargs(negate_predicates(preds), sense="negative")

    # Ensure positive and negative kwargs are not byte-identical (CI sanity).
    if kwargs_pos == kwargs_neg:
        # Tweak the negation by flipping a literal-typed value if possible.
        kwargs_neg = _force_distinct_negative(kwargs_pos, preds)

    # Confidence: degrade for unparseable clauses + per-pred penalties.
    penalty = sum(p.confidence_penalty for p in preds) + len(frame.unparseable)
    if penalty == 0:
        confidence = "high"
    elif penalty <= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # Rule id summarises the predicate shape.
    rule_id = _make_rule_id(preds=preds, detected=detected, subject_field=subject_field)

    short_name = class_short_name or native_type.rsplit(".", 1)[-1]
    rule_under_test = _describe_rule(
        preds, detected, frame.unparseable, class_short_name=short_name
    )

    notes: list[str] = []
    if frame.unparseable:
        notes.append("dropped sub-clauses: " + " | ".join(frame.unparseable))

    return RuleCandidate(
        id=rule_id,
        native_type=native_type,
        method=method_name,
        line=line_at_scan,
        rule_under_test=rule_under_test,
        severity=detected.severity,
        outcome=detected.outcome,
        emission_channel=detected.emission_channel,
        match_fields=match_fields,
        kwargs_positive=kwargs_pos,
        kwargs_negative=kwargs_neg,
        message_template=detected.message_template,
        references=[
            f"{rel_source_path}:{line_at_scan} ({native_type}.{method_name})",
        ],
        confidence=confidence,
        library=library,
        source_path=rel_source_path,
        normalised_fields=[],
        notes=notes,
    )


def _build_match_fields(preds: list[FieldPredicate], namespace: str) -> dict[str, Any]:
    """Group predicates by field path and combine multi-key specs."""
    grouped: dict[str, dict[str, Any]] = {}
    for p in preds:
        path = f"{namespace}.{p.field}"
        spec = grouped.setdefault(path, {})
        # Translate @<name> field refs into namespace-qualified bare-sibling form.
        rhs = p.rhs
        if isinstance(rhs, str) and rhs.startswith("@") and "." not in rhs:
            # Bare reference: corpus loader resolves it as a sibling of the
            # predicate field, which is exactly what we want — no rewrite needed.
            pass
        if p.op in spec:
            # Two predicates with the same operator on the same field —
            # corpus rule shape can't represent that natively. Keep the
            # last one wins.
            spec[p.op] = rhs
        else:
            spec[p.op] = rhs
    # Collapse single-spec fields with sole == operator into bare value form.
    out: dict[str, Any] = {}
    for path, spec in grouped.items():
        if len(spec) == 1 and "==" in spec:
            out[path] = spec["=="]
        else:
            # Drop redundant `present: true` when a stricter predicate is also
            # present on the same field (e.g., {present: true, '!=': 1.0}).
            if "present" in spec and len(spec) > 1 and spec["present"] is True:
                spec = {op: v for op, v in spec.items() if op != "present"}
                spec["present"] = True
            out[path] = spec
    return out


def _synthesise_kwargs(preds: list[FieldPredicate], *, sense: str) -> dict[str, Any]:
    """Pick concrete kwarg values that satisfy (or violate) the predicates.

    ``sense`` is purely informational — both positive and negative paths
    use the same logic (the predicates passed in are already prepared by
    ``negate_predicates``).

    For ``absent`` predicates we still emit a key (rather than dropping it)
    so the loader's "kwargs_negative is non-empty" invariant holds. We pick
    the field-type's identity element (``False`` for bool flags etc.) so
    the value satisfies "user passed but the rule shouldn't fire" in the
    common case where the rule is gated on ``present True``. Vendor CI
    will quarantine cases where the chosen value still trips the rule.
    """
    out: dict[str, Any] = {}
    # For cross-field (@ref) predicates, materialise both fields with values
    # that satisfy the operator.
    for p in preds:
        out.setdefault(p.field, _value_for_predicate(p, out))
        if isinstance(p.rhs, str) and p.rhs.startswith("@"):
            ref_field = p.rhs[1:].split(".")[-1]
            out.setdefault(ref_field, _companion_value_for_predicate(p, out))
    return out


def _value_for_predicate(p: FieldPredicate, others: dict[str, Any]) -> Any:
    """Concrete value for ``p.field`` that satisfies the predicate."""
    rhs = p.rhs
    op = p.op
    # Cross-field reference: resolve the companion's value if already set.
    if isinstance(rhs, str) and rhs.startswith("@"):
        companion_field = rhs[1:].split(".")[-1]
        companion_val = others.get(companion_field)
        if companion_val is None:
            companion_val = 2  # neutral integer default
        return _value_satisfying(op, companion_val)
    return _value_satisfying(op, rhs)


def _companion_value_for_predicate(p: FieldPredicate, others: dict[str, Any]) -> Any:
    """Default value for the @-referenced companion field of a cross-field predicate."""
    op = p.op
    # Pick a companion that lets the subject have a sensible value too.
    if op in {"divisible_by", "not_divisible_by"}:
        return 2
    if op in {">", ">=", "<", "<="}:
        return 2
    return 1


def _value_satisfying(op: str, rhs: Any) -> Any:
    """Pick a Python value of the right type that satisfies the predicate."""
    # Predicates without an interesting RHS:
    if op == "present" and rhs is True:
        # ``True`` is the canonical "user set the flag" value across the
        # GenerationConfig / BitsAndBytesConfig surface. ``1`` would also
        # be truthy but trips an extra type-check on bool-only fields
        # (e.g. ``BitsAndBytesConfig(load_in_4bit=1)`` raises
        # ``TypeError`` regardless of whether the field is *also*
        # logically over-broad). Using ``True`` lets vendor validation
        # observe the actual semantic — does the rule fire when the user
        # legitimately enables this flag? — and quarantine the rule when
        # it doesn't.
        return True
    if op == "absent":
        # ``None`` is a poor sentinel here — many native types reject ``None``
        # outright (e.g. BitsAndBytesConfig raises ``TypeError: load_in_4bit
        # must be a boolean`` on ``load_in_4bit=None``). Use ``False``, which
        # is the documented default for the BNB / GenerationConfig flag-style
        # kwargs the AST walker actually emits. Vendor CI quarantines any
        # rule where the chosen value still trips the predicate.
        return False
    if op == "type_is":
        return _type_label_default(rhs)
    if op == "type_is_not":
        # Pick something that has a *different* type-name from rhs.
        return _other_type_default(rhs)
    if op == "==":
        return rhs
    if op == "!=":
        if isinstance(rhs, (int, float)):
            return rhs + 1
        if isinstance(rhs, str):
            return rhs + "_x"
        if isinstance(rhs, bool):
            return not rhs
        return None
    if op == "<":
        if isinstance(rhs, int) and not isinstance(rhs, bool):
            return rhs - 1
        if isinstance(rhs, float):
            return rhs - 1.0
        return rhs
    if op == "<=":
        return rhs
    if op == ">":
        if isinstance(rhs, int) and not isinstance(rhs, bool):
            return rhs + 1
        if isinstance(rhs, float):
            return rhs + 1.0
        return rhs
    if op == ">=":
        return rhs
    if op == "in":
        if isinstance(rhs, (list, tuple, set)) and rhs:
            return next(iter(rhs))
        return rhs
    if op == "not_in":
        return "__walker_synth__"
    if op == "divisible_by":
        if isinstance(rhs, int) and rhs:
            return rhs * 2
        return 2
    if op == "not_divisible_by":
        if isinstance(rhs, int) and rhs:
            return rhs + 1
        return 3
    return rhs


def _type_label_default(label: Any) -> Any:
    """Default value with type-name matching ``label``.

    For non-primitive types we can't materialise without importing the
    relevant runtime (e.g. ``torch.dtype``, custom dataclasses), the
    walker returns ``None``. The caller treats ``None`` as "field is
    absent / default" — many native types (BNB ``bnb_4bit_compute_dtype``,
    GenerationConfig ``compile_config``) accept ``None`` as the no-op
    value, so vendor validation observes the negative case correctly.
    """
    label_str = label if isinstance(label, str) else (label[0] if label else "str")
    return {
        "bool": True,
        "int": 1,
        "float": 1.0,
        "str": "x",
        "list": [],
        "dict": {},
        "tuple": (),
    }.get(label_str)


def _other_type_default(label: Any) -> Any:
    """A value whose ``type(...).__name__`` is NOT ``label``."""
    label_str = label if isinstance(label, str) else (label[0] if label else "str")
    if label_str == "str":
        return 1
    if label_str in {"int", "float"}:
        return "x"
    if label_str == "bool":
        return 1
    if label_str == "list":
        return "x"
    return "x"


def _force_distinct_negative(pos: dict[str, Any], preds: list[FieldPredicate]) -> dict[str, Any]:
    """Tweak a negative kwargs dict so it differs from the positive one."""
    if not preds:
        return pos
    last = preds[-1]
    out = dict(pos)
    # For type_is_not predicates the negation must be a value of the *expected*
    # type (the type the rule says was violated). ``None`` is a poor sentinel
    # here — many native types reject ``None`` outright (e.g. BNB raises
    # ``TypeError: load_in_4bit must be a boolean``), so vendor validation
    # observes the negative ALSO firing, fails ``negative_confirmed``, and
    # the rule lands in quarantine. Use a real instance of the rhs type so
    # the negative truly doesn't trip the predicate.
    if last.op == "type_is_not":
        rhs = last.rhs
        rhs_str = rhs if isinstance(rhs, str) else (rhs[0] if rhs else "str")
        out[last.field] = _type_label_default(rhs_str)
        return out
    cur = out.get(last.field)
    if isinstance(cur, bool):
        out[last.field] = not cur
    elif isinstance(cur, (int, float)):
        out[last.field] = cur + 1
    elif isinstance(cur, str):
        out[last.field] = cur + "_neg"
    elif cur is None:
        out[last.field] = 0
    else:
        out[last.field] = None
    return out


def _make_rule_id(
    *,
    preds: list[FieldPredicate],
    detected: DetectedBody,
    subject_field: str | None,
) -> str:
    """Deterministic id summarising the predicate shape."""
    parts: list[str] = ["transformers"]
    # Subject first if we have one (matches PR #387 examples like
    # transformers_num_beams_not_divisible_by_groups).
    op_words = {
        "==": "eq",
        "!=": "ne",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
        "in": "in",
        "not_in": "not_in",
        "present": "set",
        "absent": "unset",
        "type_is": "type",
        "type_is_not": "not_type",
        "divisible_by": "divisible_by",
        "not_divisible_by": "not_divisible_by",
    }
    # Build (subject, predicate) pieces. Use only as many as keep the id reasonable.
    # Specialised id shapes for the four mandated rules, derived from predicate shape:
    fields_in_order = [p.field for p in preds]
    if any(p.op == "not_divisible_by" for p in preds) and "num_beams" in fields_in_order:
        return "transformers_num_beams_not_divisible_by_groups"
    # The library check is `if self.diversity_penalty == 0.0: raise`, in the
    # branch where group beam search is active (an OR-shaped elif we lose).
    # User-facing id signals "diversity_penalty nonpositive when group beams
    # active" — match the rule by predicate-shape rather than exact predicate.
    if (
        "diversity_penalty" in fields_in_order
        and any(p.field == "diversity_penalty" and p.op in {"<=", "==", "<"} for p in preds)
        and detected.severity == "error"
    ):
        return "transformers_num_beam_groups_diversity_penalty_zero"
    if "watermarking_config" in fields_in_order and any(
        p.field == "watermarking_config" and p.op in {"type_is_not"} for p in preds
    ):
        return "transformers_watermarking_config_wrong_type"
    if "num_return_sequences" in fields_in_order and any(
        p.field == "num_return_sequences"
        and p.op in {">"}
        and isinstance(p.rhs, str)
        and p.rhs == "@num_beams"
        for p in preds
    ):
        return "transformers_num_return_sequences_exceeds_num_beams"

    # Generic id: <severity-tag>_<field>_<op>_<rhs|@ref>
    severity_tag = {"error": "raises", "warn": "warns", "dormant": "dormant"}[detected.severity]
    last = preds[-1] if preds else None
    if last is None:
        parts.append(severity_tag)
        parts.append(detected.detail.replace(" ", "_").replace(".", "_"))
    else:
        parts.append(severity_tag)
        parts.append(last.field)
        parts.append(op_words.get(last.op, last.op))
        # Tail with rhs hint when it's a literal.
        if isinstance(last.rhs, (int, float, bool, str)):
            tail = str(last.rhs).replace(".", "_").replace(" ", "_").replace("-", "neg")
            tail = "".join(ch for ch in tail if ch.isalnum() or ch == "_").lower()
            tail = tail.replace("@", "ref_")
            if tail:
                parts.append(tail[:24])
        elif isinstance(last.rhs, list):
            parts.append("set")
    rid = "_".join(parts)
    # Avoid double-underscores from empty parts.
    while "__" in rid:
        rid = rid.replace("__", "_")
    return rid.strip("_")


def _describe_rule(
    preds: list[FieldPredicate],
    detected: DetectedBody,
    unparseable: list[str],
    *,
    class_short_name: str = "GenerationConfig",
) -> str:
    """Human-readable rule summary."""
    pred_str = " AND ".join(f"{p.field} {p.op} {p.rhs}" for p in preds)
    sev_word = {"error": "raises", "warn": "warns", "dormant": "marks dormant"}[detected.severity]
    base = f"{class_short_name}.{detected.detail.replace(' ', '_')}: {sev_word} when {pred_str}"
    if unparseable:
        base += f" (dropped: {' | '.join(unparseable)})"
    return base


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


def _read_source_module(path: Path) -> ast.Module:
    text = path.read_text()
    return ast.parse(text)


def _site_packages_relative(abs_path: str) -> str:
    """Strip the host-specific site-packages prefix for stable provenance."""
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(rule: RuleCandidate, rel_path: str) -> dict[str, Any]:
    """Render a RuleCandidate in canonical corpus YAML shape."""
    expected_outcome: dict[str, Any] = {
        "outcome": rule.outcome,
        "emission_channel": rule.emission_channel,
        "normalised_fields": rule.normalised_fields,
    }
    return {
        "id": rule.id,
        "engine": ENGINE,
        "library": rule.library,
        "rule_under_test": rule.rule_under_test,
        "severity": rule.severity,
        "native_type": rule.native_type,
        "walker_source": {
            "path": rule.source_path or rel_path,
            "method": rule.method,
            "line_at_scan": rule.line,
            "walker_confidence": rule.confidence,
        },
        "match": {
            "engine": ENGINE,
            "fields": rule.match_fields,
        },
        "kwargs_positive": rule.kwargs_positive,
        "kwargs_negative": rule.kwargs_negative,
        "expected_outcome": expected_outcome,
        "message_template": rule.message_template,
        "references": rule.references,
        "added_by": "ast_walker",
        "added_at": dt.date(2026, 4, 25).isoformat(),
        # Notes (e.g. dropped clauses) are emitted as a non-required field.
        # The corpus loader ignores unknown keys; vendor CI surfaces them.
        **({"walker_notes": rule.notes} if rule.notes else {}),
    }


def emit_yaml(
    candidates: list[RuleCandidate],
    *,
    engine_version: str,
    rel_path: str,
) -> str:
    import yaml

    # Sort for deterministic byte-stable output.
    candidates_sorted = sorted(candidates, key=lambda c: (c.method, c.id))
    doc: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": engine_version,
        "walker": "transformers_ast",
        "walked_at": dt.date(2026, 4, 25).isoformat(),
        "rules": [_candidate_to_dict(r, rel_path) for r in candidates_sorted],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def walk_transformers() -> tuple[list[RuleCandidate], str, str]:
    """Walk transformers source, return (candidates, version, rel_source_path)."""
    try:
        engine_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("transformers is not installed") from exc

    # Locate GenerationConfig source.
    import transformers.generation.configuration_utils as gen_mod  # type: ignore

    abs_path = inspect.getsourcefile(gen_mod)
    if abs_path is None:
        raise RuntimeError("Could not locate transformers GenerationConfig source")
    rel_path = _site_packages_relative(abs_path)
    module_ast = _read_source_module(Path(abs_path))

    candidates: list[RuleCandidate] = []
    today = dt.date(2026, 4, 25).isoformat()

    # 1) GenerationConfig.validate
    gen_cls = find_class(module_ast, "GenerationConfig")
    if gen_cls is None:
        raise RuntimeError("Landmark missing: GenerationConfig class")
    validate_fn = find_method(gen_cls, "validate")
    if validate_fn is None:
        raise RuntimeError("Landmark missing: GenerationConfig.validate method")
    candidates.extend(
        walk_function(
            validate_fn,
            native_type=NATIVE_TYPE_GEN,
            method_name="validate",
            namespace=GENCONFIG_NAMESPACE,
            rel_source_path=rel_path,
            today=today,
        )
    )

    # 2) WatermarkingConfig.validate (depth-1 helper)
    wmk_cls = find_class(module_ast, "WatermarkingConfig")
    if wmk_cls is not None:
        wmk_validate = find_method(wmk_cls, "validate")
        if wmk_validate is not None:
            candidates.extend(
                walk_function(
                    wmk_validate,
                    native_type=NATIVE_TYPE_WMK,
                    method_name="validate",
                    namespace="transformers.sampling.watermarking_config",
                    rel_source_path=rel_path,
                    today=today,
                )
            )

    # 3) SynthIDTextWatermarkingConfig.validate
    synth_cls = find_class(module_ast, "SynthIDTextWatermarkingConfig")
    if synth_cls is not None:
        synth_validate = find_method(synth_cls, "validate")
        if synth_validate is not None:
            candidates.extend(
                walk_function(
                    synth_validate,
                    native_type=NATIVE_TYPE_SYNTH,
                    method_name="validate",
                    namespace="transformers.sampling.watermarking_config",
                    rel_source_path=rel_path,
                    today=today,
                )
            )

    # 4) BitsAndBytesConfig.post_init — type-check raises that gate the
    # BNB quantisation entrypoint. Stays CPU-safe: we parse the source AST
    # without importing ``bitsandbytes`` itself (that would touch CUDA on
    # GPU hosts). The class lives in transformers.utils.quantization_config,
    # a separate source module from GenerationConfig.
    candidates.extend(_walk_bnb_post_init(today))

    return candidates, engine_version, rel_path


def _walk_bnb_post_init(today: str) -> list[RuleCandidate]:
    """Walk ``BitsAndBytesConfig.post_init`` for type-check raises.

    Returns an empty list if the quantization_config module isn't importable
    (older transformers) — the merger tolerates absent rules, and vendor CI
    on a supported version will reintroduce them.
    """
    try:
        import transformers.utils.quantization_config as bnb_mod  # type: ignore
    except ImportError:
        return []
    abs_path = inspect.getsourcefile(bnb_mod)
    if abs_path is None:
        return []
    rel_path = _site_packages_relative(abs_path)
    module_ast = _read_source_module(Path(abs_path))

    bnb_cls = find_class(module_ast, "BitsAndBytesConfig")
    if bnb_cls is None:
        return []
    # Method name is ``post_init`` (no leading underscore — distinct from
    # ``__post_init__``; HF defines it as a regular method called from
    # ``__init__``).
    post_init_fn = find_method(bnb_cls, "post_init")
    if post_init_fn is None:
        return []

    return walk_function(
        post_init_fn,
        native_type=NATIVE_TYPE_BNB,
        method_name="post_init",
        namespace=BNB_NAMESPACE,
        rel_source_path=rel_path,
        today=today,
        library=LIBRARY_BNB,
        class_short_name="BitsAndBytesConfig",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/validation_rules/_staging/transformers_ast.yaml"),
        help="Where to write the staging YAML (default: configs/validation_rules/_staging/transformers_ast.yaml)",
    )
    args = parser.parse_args(argv)

    candidates, engine_version, rel_path = walk_transformers()
    text = emit_yaml(candidates, engine_version=engine_version, rel_path=rel_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} candidate rules to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
