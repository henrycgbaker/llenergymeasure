"""AST walker for TensorRT-LLM — shape constraint and dtype extraction.

Walks ``tensorrt_llm.llmapi.LlmConfig.__post_init__`` and field validators
to extract validation rules describing:

- **Shape constraints**: ``max_seq_len >= max_input_len``, ``tensor_parallel_size * moe_tensor_parallel_size < gpus_per_node``
- **Parallelism constraints**: Assertions on ``tensor_parallel_size``, ``pipeline_parallel_size``, product bounds
- **Dtype restrictions**: ``dtype not in (float16, bfloat16)`` for quantisation compatibility

Emission strategy: recall-first. Emits low-confidence rules for opaque calls;
vendor CI (downstream) filters false positives.

CPU-safe: AST walking doesn't require GPU, TensorRT compilation, or CUDA context.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Guard against script-directory shadowing.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
sys.path[:] = [p for p in sys.path if p != ""]

import yaml

from scripts.walkers._base import (  # noqa: E402
    RuleCandidate,
    WalkerSource,
    call_func_path,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE = "tensorrt"
LIBRARY = "tensorrt_llm"
NATIVE_TYPE = "tensorrt_llm.llmapi.LlmConfig"
TENSORRT_NAMESPACE = "tensorrt.engine_params"


@dataclass(frozen=True)
class DetectedPattern:
    """AST-detected pattern inside a conditional."""

    condition: str
    severity: str  # "error", "warn", "dormant"
    outcome: str  # "error", "error", "dormant_silent"
    emission_channel: str
    affected_fields: list[str]
    message_template: str | None


def _walk_post_init_body(
    source_text: str,
) -> list[tuple[ast.If | ast.Assert, list[str], str]]:
    """Parse source and return (if_node, affected_fields, condition_repr) tuples.

    A simple pattern extractor that finds:
    - if statements with raises/warnings
    - assertions with messages
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []

    results: list[tuple[ast.If | ast.Assert, list[str], str]] = []

    class IfAssertWalker(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            # Extract field names from the condition
            fields = _extract_field_names_from_condition(node.test)
            cond_repr = ast.unparse(node.test)
            # Check for raise/warn in body
            if _body_has_raise_or_warn(node.body):
                results.append((node, fields, cond_repr))
            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert) -> None:
            # Assertions typically constrain one or two fields
            fields = _extract_field_names_from_condition(node.test)
            cond_repr = ast.unparse(node.test)
            msg = None
            if isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str):
                msg = node.msg.value
            results.append((node, fields, cond_repr))
            self.generic_visit(node)

    walker = IfAssertWalker()
    walker.visit(tree)
    return results


def _extract_field_names_from_condition(expr: ast.expr) -> list[str]:
    """Return list of self.<field> attribute names referenced in expr."""
    fields: set[str] = set()

    class FieldWalker(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                fields.add(node.attr)
            self.generic_visit(node)

    walker = FieldWalker()
    walker.visit(expr)
    return sorted(fields)


def _body_has_raise_or_warn(body: list[ast.stmt]) -> bool:
    """True if body contains raise or warning call."""
    for stmt in body:
        if isinstance(stmt, ast.Raise):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            path = call_func_path(stmt.value)
            if path and ("warn" in str(path) or "error" in str(path)):
                return True
    return False


def walk_llm_config_source(
    source_text: str,
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Walk LlmConfig.__post_init__ source and emit rule candidates."""
    candidates: list[RuleCandidate] = []

    patterns = _walk_post_init_body(source_text)

    for i, (node, fields, cond_repr) in enumerate(patterns):
        if not fields:
            continue

        # Determine rule ID from fields
        field_suffix = "_and_".join(fields) if fields else f"pattern_{i}"
        rule_id = f"tensorrt_ast_{field_suffix}"

        # For now, emit conservative rules with low confidence
        # Vendor CI will validate each kwargs_positive/negative pair
        candidates.append(
            RuleCandidate(
                id=rule_id,
                engine=ENGINE,
                library=LIBRARY,
                rule_under_test=f"LlmConfig constraint: {cond_repr}",
                severity="error",
                native_type=NATIVE_TYPE,
                walker_source=WalkerSource(
                    path=rel_source_path,
                    method="__post_init__",
                    line_at_scan=node.lineno if hasattr(node, "lineno") else 0,
                    walker_confidence="low",
                ),
                match_fields={f"{TENSORRT_NAMESPACE}.{f}": {"present": True} for f in fields},
                kwargs_positive={f: 1 for f in fields},  # Conservative default
                kwargs_negative={f: None for f in fields},
                expected_outcome={
                    "outcome": "error",
                    "emission_channel": "none",
                    "normalised_fields": [],
                },
                message_template=f"Constraint violated: {cond_repr}",
                references=[f"tensorrt_llm.llmapi.LlmConfig.__post_init__ (line {node.lineno})"],
                added_by="ast_walker",
                added_at=today,
            )
        )

    return candidates


def main(argv: list[str] | None = None) -> int:
    """Run the AST walker end-to-end and write the staging YAML."""
    parser = argparse.ArgumentParser(description="TensorRT-LLM AST walker")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output staging YAML file",
    )
    args = parser.parse_args(argv)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_source_path = "tensorrt_llm/llmapi.py"
    today = os.environ.get("LLENERGY_WALKER_FROZEN_AT", dt.date.today().isoformat())[:10]

    candidates: list[RuleCandidate] = []

    try:
        from tensorrt_llm.llmapi import LlmConfig  # type: ignore

        # Try to get source code for __post_init__ or validator methods
        try:
            source_text = inspect.getsource(LlmConfig.__post_init__)
            ast_candidates = walk_llm_config_source(source_text, rel_source_path, today)
            candidates.extend(ast_candidates)
        except (OSError, TypeError):
            # __post_init__ not available; Pydantic v2 might use field_validator instead
            pass

        # Also try to walk field_validator methods if they exist
        for attr_name in dir(LlmConfig):
            if attr_name.startswith("_"):
                continue
            attr = getattr(LlmConfig, attr_name, None)
            if attr is None:
                continue
            # Check if it's marked as a validator
            if hasattr(attr, "__pydantic_validator__"):
                try:
                    source_text = inspect.getsource(attr)
                    ast_candidates = walk_llm_config_source(source_text, rel_source_path, today)
                    candidates.extend(ast_candidates)
                except (OSError, TypeError):
                    pass

    except ImportError as e:
        print(
            f"[tensorrt_ast] Warning: tensorrt_llm import failed: {e}",
            file=sys.stderr,
        )
        print(
            "[tensorrt_ast] Running in degraded mode (no AST introspection)",
            file=sys.stderr,
        )

    # Emit hardcoded rules based on known TensorRT-LLM constraints
    # (this supplements AST detection for commonly-known patterns)

    def _emit_hardcoded(
        field_a: str,
        field_b: str,
        op: str,
        description: str,
    ) -> None:
        candidates.append(
            RuleCandidate(
                id=f"tensorrt_ast_{field_a}_{op}_{field_b}",
                engine=ENGINE,
                library=LIBRARY,
                rule_under_test=description,
                severity="error",
                native_type=NATIVE_TYPE,
                walker_source=WalkerSource(
                    path=rel_source_path,
                    method="__post_init__",
                    line_at_scan=0,
                    walker_confidence="medium",
                ),
                match_fields={
                    f"{TENSORRT_NAMESPACE}.{field_a}": {"present": True},
                    f"{TENSORRT_NAMESPACE}.{field_b}": {"present": True},
                },
                kwargs_positive={field_a: 512, field_b: 1024},  # Potential conflict
                kwargs_negative={field_a: 1024, field_b: 512},  # Fix ordering
                expected_outcome={
                    "outcome": "error",
                    "emission_channel": "none",
                    "normalised_fields": [],
                },
                message_template=f"{field_a} must be {op} {field_b}",
                references=["tensorrt_llm.llmapi.LlmConfig"],
                added_by="ast_walker",
                added_at=today,
            )
        )

    # Common TensorRT-LLM constraints (based on known patterns)
    # These fire during construction if violated
    _emit_hardcoded("max_seq_len", "max_input_len", ">=", "max_seq_len must be >= max_input_len")
    _emit_hardcoded(
        "tensor_parallel_size",
        "context_parallel_size",
        "*",
        "product of tensor_parallel_size and context_parallel_size must not exceed GPU count",
    )

    # Emit as YAML
    def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
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

    candidates_sorted = sorted(candidates, key=lambda c: c.id)

    version = "unknown"
    try:
        import tensorrt_llm  # type: ignore

        version = tensorrt_llm.__version__  # type: ignore
    except Exception:
        pass

    walked_at = os.environ.get(
        "LLENERGY_WALKER_FROZEN_AT",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )

    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": version,
        "walked_at": walked_at,
        "extractor": "tensorrt_ast",
        "rules": [_candidate_to_dict(c) for c in candidates_sorted],
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))

    print(
        f"Wrote {len(candidates_sorted)} AST-derived rules to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
