"""PoC-A: How much value does depth-2 helper tracing add per engine?

Hypothesis
----------
The design's default of depth=1 helper tracing is extrapolated from vLLM's
greedy-strip case living in `create_engine_config()` (one level out from
`__post_init__`). We don't actually know whether depth=2 would find
additional user-relevant rules in transformers, vLLM, or TRT-LLM.

This PoC runs the walker at depths 1, 2, and 3 and reports the marginal
rule count per depth. Rules found at depth N+1 are manually classifiable
into "user-relevant rule" vs "walker noise".

Decision criteria (pre-committed)
---------------------------------
Per engine:
- Depth-2 yields >3 user-relevant rules not caught at depth-1
    -> Raise WALKER_DEPTH default to 2 for that engine before P1.
- Depth-2 yields 1-3 marginal rules
    -> Manually seed those specific rules as YAML; keep default at 1.
- Depth-2 yields 0 new rules
    -> Confirm depth-1 default for that engine.

Depth-3 results are reported for completeness but are unlikely to justify
raising defaults. If depth-3 is >50% larger than depth-2, investigate.

Run
---
  /usr/bin/python3.10 scripts/probe_walker_depth_poc.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SITE_PACKAGES = Path.home() / ".local/lib/python3.10/site-packages"

# Target methods - entry points where walking starts at depth 0.
# Helper methods in the same module reached via `self.<method>()` are depth 1.
# Helpers called from those helpers are depth 2. And so on.
TARGETS = [
    {
        "library": "transformers",
        "file": SITE_PACKAGES / "transformers/generation/configuration_utils.py",
        "entry_class": "GenerationConfig",
        "entry_method": "validate",
    },
    {
        "library": "vllm",
        "file": SITE_PACKAGES / "vllm/sampling_params.py",
        "entry_class": "SamplingParams",
        "entry_method": "__post_init__",
    },
    {
        "library": "tensorrt_llm",
        "file": SITE_PACKAGES / "tensorrt_llm/llmapi/llm_args.py",
        "entry_class": None,  # decorated validators live at module scope
        "entry_method": None,
        "decorator_match": ["field_validator", "model_validator"],
    },
]


RULE_PATTERNS = Literal["raise", "logger_warning", "warnings_warn", "self_assign", "minor_issues"]


@dataclass(frozen=True)
class Rule:
    source_file: str
    source_line: int
    method_chain: tuple[str, ...]  # e.g., ("validate", "_check_greedy")
    action: str  # raise / logger_warning / warnings_warn / self_assign / minor_issues
    condition_src: str


def _extract_condition(if_node: ast.If) -> str:
    """Dump the if-condition's source. Best-effort."""
    try:
        return ast.unparse(if_node.test)
    except Exception:
        return "<unparseable>"


def _classify_stmt_action(stmt: ast.stmt) -> str | None:
    """Return an action class if this statement is a rule emission."""
    # raise X(...)
    if isinstance(stmt, ast.Raise):
        return "raise"
    if not isinstance(stmt, ast.Expr) and not isinstance(stmt, ast.Assign):
        return None
    # warnings.warn(...) / logger.warning(...) / logger.warning_once(...)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        func = call.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                if func.value.id == "warnings" and func.attr == "warn":
                    return "warnings_warn"
                if func.value.id == "logger" and func.attr in (
                    "warning",
                    "warning_once",
                    "error",
                ):
                    return "logger_warning"
    # self.X = Y  /  minor_issues[k] = msg
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name):
                if tgt.value.id == "self":
                    return "self_assign"
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                if tgt.value.id == "minor_issues":
                    return "minor_issues"
    return None


def _rules_in_if_body(
    if_node: ast.If,
    method_chain: tuple[str, ...],
    source_file: str,
) -> list[Rule]:
    """Extract all direct rule emissions inside an if-block."""
    rules: list[Rule] = []
    cond_src = _extract_condition(if_node)
    for stmt in if_node.body:
        action = _classify_stmt_action(stmt)
        if action is not None:
            rules.append(
                Rule(
                    source_file=source_file,
                    source_line=stmt.lineno,
                    method_chain=method_chain,
                    action=action,
                    condition_src=cond_src,
                )
            )
    return rules


def _helpers_called_in_body(
    body: list[ast.stmt],
    module_methods: dict[str, ast.FunctionDef],
) -> set[str]:
    """Find self.<method>() calls where <method> is defined in this module."""
    called = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "self" and func.attr in module_methods:
                    called.add(func.attr)
            self.generic_visit(node)

    for stmt in body:
        Visitor().visit(stmt)
    return called


def walk_at_depth(
    entry_method: ast.FunctionDef,
    module_methods: dict[str, ast.FunctionDef],
    source_file: str,
    max_depth: int,
) -> list[Rule]:
    """Walk if-guarded rule emissions, following self.<helper>() calls up to max_depth."""
    rules: list[Rule] = []
    visited: set[str] = set()

    def _walk(method: ast.FunctionDef, chain: tuple[str, ...], depth: int) -> None:
        if method.name in visited:
            return
        visited.add(method.name)

        for node in ast.walk(method):
            if isinstance(node, ast.If):
                rules.extend(_rules_in_if_body(node, chain, source_file))

        if depth < max_depth:
            for helper_name in _helpers_called_in_body(method.body, module_methods):
                helper = module_methods.get(helper_name)
                if helper is not None:
                    _walk(helper, chain + (helper_name,), depth + 1)

    _walk(entry_method, (entry_method.name,), depth=0)
    return rules


def _module_methods(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Return a map of method-name -> FunctionDef for all class methods in the module."""
    out: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[item.name] = item
    return out


def _get_entry_method(
    tree: ast.Module, class_name: str | None, method_name: str | None
) -> ast.FunctionDef | None:
    if class_name is None or method_name is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def _get_decorated_methods(tree: ast.Module, decorator_names: list[str]) -> list[ast.FunctionDef]:
    """Return all methods decorated with any of decorator_names."""
    out: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                for dec in item.decorator_list:
                    # @field_validator(...), @model_validator(...), bare forms
                    dec_call = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(dec_call, ast.Name) and dec_call.id in decorator_names:
                        out.append(item)
                        break
    return out


def walk_target(target: dict, max_depth: int) -> list[Rule]:
    src = target["file"].read_text()
    tree = ast.parse(src)
    module_methods = _module_methods(tree)
    source_file = target["file"].name
    all_rules: list[Rule] = []

    if target.get("decorator_match"):
        # TRT-LLM pattern: walk every decorated validator as its own entry
        entries = _get_decorated_methods(tree, target["decorator_match"])
        for entry in entries:
            rules = walk_at_depth(entry, module_methods, source_file, max_depth)
            all_rules.extend(rules)
    else:
        entry = _get_entry_method(tree, target.get("entry_class"), target.get("entry_method"))
        if entry is None:
            return []
        all_rules = walk_at_depth(entry, module_methods, source_file, max_depth)

    return all_rules


def rule_signature(r: Rule) -> tuple[str, str, str]:
    """Collapse rules to a stable signature for set diffs."""
    return (r.method_chain[-1], r.condition_src, r.action)


def main() -> None:
    print("=" * 78)
    print("PoC-A: Walker depth sufficiency per engine")
    print("=" * 78)
    print()

    summary: list[tuple[str, int, int, int]] = []

    for target in TARGETS:
        lib = target["library"]
        print(f"--- {lib} --- ({target['file'].name})")
        if not target["file"].exists():
            print("  SKIP: file not found")
            print()
            continue

        rules_at = {}
        for depth in (1, 2, 3):
            rules_at[depth] = walk_target(target, max_depth=depth)

        d1 = {rule_signature(r) for r in rules_at[1]}
        d2 = {rule_signature(r) for r in rules_at[2]}
        d3 = {rule_signature(r) for r in rules_at[3]}

        print(f"  depth=1: {len(d1)} rules")
        print(f"  depth=2: {len(d2)} rules (+{len(d2 - d1)} vs d1)")
        print(f"  depth=3: {len(d3)} rules (+{len(d3 - d2)} vs d2)")

        new_at_d2 = [r for r in rules_at[2] if rule_signature(r) not in d1]
        if new_at_d2:
            print("  Rules found only at depth>=2 (first 8):")
            for r in new_at_d2[:8]:
                chain = " -> ".join(r.method_chain)
                print(f"    [{r.action}] {chain}:L{r.source_line}")
                print(f"      if {r.condition_src[:90]}")

        summary.append((lib, len(d1), len(d2) - len(d1), len(d3) - len(d2)))
        print()

    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    print(f"{'LIBRARY':<20}  {'d=1':<6}  {'d=2 extra':<10}  {'d=3 extra':<10}  RECOMMENDATION")
    print("-" * 78)
    for lib, d1_count, d2_extra, d3_extra in summary:
        if d2_extra > 3:
            rec = "RAISE WALKER_DEPTH to 2"
        elif d2_extra >= 1:
            rec = "keep depth=1; manual-seed the extras"
        else:
            rec = "confirm depth=1 default"
        print(f"{lib:<20}  {d1_count:<6}  +{d2_extra:<9}  +{d3_extra:<9}  {rec}")
    print()
    print("IMPORTANT: marginal rules at depth>=2 still need human classification")
    print("as 'user-relevant rule' vs 'walker noise' before the recommendation is")
    print("final. The raw rules are printed above for that review.")


if __name__ == "__main__":
    main()
