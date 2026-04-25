"""AST walker for vLLM SamplingParams — structural rule extraction.

Walks ``SamplingParams.__post_init__`` source to detect silent assignments
(dormancy) and warned assignments (announced dormancy) patterns.

Output schema: ``configs/validation_rules/_staging/vllm_ast.yaml``
consumed downstream by ``scripts/walkers/build_corpus.py``.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers._base import (
    RuleCandidate,
    WalkerSource,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE = "vllm"
LIBRARY = "vllm"
NATIVE_TYPE = "vllm.SamplingParams"
NAMESPACE = "vllm.sampling"


def _resolve_source_paths() -> tuple[str, str, str]:
    """Locate vLLM's SamplingParams source on disk.

    Returns ``(version, abs_path, rel_path)`` — the latter rooted at
    ``site-packages/`` for reproducibility.
    """
    try:
        import vllm
        from vllm import SamplingParams

        abs_path = inspect.getsourcefile(SamplingParams) or "<unknown>"
        version = vllm.__version__
    except Exception:
        try:
            from importlib.metadata import version as get_version

            version = get_version("vllm")
        except Exception:
            version = "unknown"
        abs_path = "<unknown>"

    # Relativize: strip site-packages prefix for reproducibility
    marker = "site-packages/"
    idx = abs_path.find(marker)
    rel_path = abs_path[idx + len(marker) :] if idx >= 0 else Path(abs_path).name

    return version, abs_path, rel_path


def _read_sampling_params_source() -> str | None:
    """Read SamplingParams.__post_init__ source via inspect.getsource()."""
    try:
        from vllm import SamplingParams

        return inspect.getsource(SamplingParams.__post_init__)
    except Exception:
        return None


def _extract_ast_rules_from_source(
    source: str,
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Parse source code and extract rules via AST walking."""
    candidates: list[RuleCandidate] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return candidates

    # Find the function definition (should be __post_init__)
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__":
            func_def = node
            break

    if func_def is None:
        return candidates

    # Walk the function body for detected patterns
    line_offset = func_def.lineno  # Line number of the function definition

    for stmt in func_def.body:
        if isinstance(stmt, ast.If):
            # Extract condition fields
            condition_fields = _extract_condition_fields(stmt.test)

            # Check for patterns in the if body
            for body_stmt in stmt.body:
                # Pattern 1: logger.warning(...) inside if condition
                if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call):
                    call = body_stmt.value
                    func_path = _call_func_path(call)
                    if func_path and len(func_path) == 2 and func_path[0] == "logger":
                        method = func_path[1]
                        if method in {"warning", "warning_once", "error"}:
                            msg = _first_string_arg(call)
                            # Extract the affected field from condition
                            for field in condition_fields:
                                candidate = RuleCandidate(
                                    id=f"vllm_ast_warn_{field}",
                                    engine=ENGINE,
                                    library=LIBRARY,
                                    rule_under_test=f"SamplingParams.__post_init__ warns when {field} violates a constraint",
                                    severity="warn",
                                    native_type=NATIVE_TYPE,
                                    walker_source=WalkerSource(
                                        path=rel_source_path,
                                        method="__post_init__",
                                        line_at_scan=body_stmt.lineno + line_offset,
                                        walker_confidence="medium",
                                    ),
                                    match_fields={},
                                    kwargs_positive={},
                                    kwargs_negative={},
                                    expected_outcome={
                                        "outcome": "dormant_announced",
                                        "emission_channel": "logger_warning",
                                        "normalised_fields": [field],
                                    },
                                    message_template=msg,
                                    references=["vllm.SamplingParams.__post_init__()"],
                                    added_by="ast_walker",
                                    added_at=today,
                                )
                                candidates.append(candidate)

                # Pattern 2: self.field = value inside if condition (silent normalization)
                elif isinstance(body_stmt, ast.Assign):
                    attr = _extract_assign_target(body_stmt)
                    if attr and not attr.startswith("_"):
                        # Only emit if the condition actually references self fields
                        if condition_fields:
                            candidate = RuleCandidate(
                                id=f"vllm_ast_silent_{attr}",
                                engine=ENGINE,
                                library=LIBRARY,
                                rule_under_test=f"SamplingParams.__post_init__ silently normalises {attr}",
                                severity="dormant",
                                native_type=NATIVE_TYPE,
                                walker_source=WalkerSource(
                                    path=rel_source_path,
                                    method="__post_init__",
                                    line_at_scan=body_stmt.lineno + line_offset,
                                    walker_confidence="medium",
                                ),
                                match_fields={},
                                kwargs_positive={},
                                kwargs_negative={},
                                expected_outcome={
                                    "outcome": "dormant_silent",
                                    "emission_channel": "none",
                                    "normalised_fields": [attr],
                                },
                                message_template=None,
                                references=["vllm.SamplingParams.__post_init__()"],
                                added_by="ast_walker",
                                added_at=today,
                            )
                            candidates.append(candidate)
                        break

    return candidates


def _extract_condition_fields(condition: ast.expr) -> set[str]:
    """Return the set of ``self.<field>`` attribute names referenced in condition."""
    fields: set[str] = set()
    for node in ast.walk(condition):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            fields.add(node.attr)
    return fields


def _extract_assign_target(stmt: ast.Assign) -> str | None:
    """Return ``<attr>`` for ``self.<attr> = ...``, or None."""
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


def _call_func_path(call: ast.Call) -> list[str] | None:
    """Return dotted path for a Call node's func, or None if opaque."""
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _first_string_arg(call: ast.Call) -> str | None:
    """First string-like positional argument of a Call, or None."""
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


def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
    """Render a RuleCandidate into the YAML corpus entry shape."""
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "rule_under_test": c.rule_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "walker_source": {
            "path": c.walker_source.path,
            "method": c.walker_source.method,
            "line_at_scan": c.walker_source.line_at_scan,
            "walker_confidence": c.walker_source.walker_confidence,
        },
        "match": {
            "engine": c.engine,
            "fields": c.match_fields,
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the AST walker and write the staging YAML."""
    parser = argparse.ArgumentParser(description="vLLM AST walker")
    parser.add_argument("--out", required=True, help="Output staging YAML path")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    version, _abs_source_path, rel_source_path = _resolve_source_paths()
    today = os.environ.get("LLENERGY_WALKER_FROZEN_AT", dt.date.today().isoformat())[:10]

    candidates: list[RuleCandidate] = []

    # Try to read and parse SamplingParams.__post_init__ source
    source = _read_sampling_params_source()
    if source:
        candidates.extend(
            _extract_ast_rules_from_source(
                source,
                rel_source_path,
                today,
            )
        )

    # Stable order: by walker_source.method, then by id
    candidates_sorted = sorted(candidates, key=lambda c: (c.walker_source.method, c.id))

    walked_at = os.environ.get(
        "LLENERGY_WALKER_FROZEN_AT",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )

    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": version,
        "walked_at": walked_at,
        "extractor": "vllm_ast",
        "rules": [_candidate_to_dict(c) for c in candidates_sorted],
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))

    print(
        f"Wrote {len(candidates_sorted)} AST-derived rules to {out_path}",
        file=sys.stderr,
    )
    return 0


__all__ = ["ENGINE", "LIBRARY", "NATIVE_TYPE"]

if __name__ == "__main__":
    raise SystemExit(main())
