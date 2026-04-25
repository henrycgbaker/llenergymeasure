"""PoC: AST-walker that extracts validation rules from library source code.

Validates the v2 design proposal's load-bearing assumption: that rules in
HF's GenerationConfig.validate(), vLLM's SamplingParams.__post_init__, and
TRT-LLM's LlmArgs Pydantic validators can be mechanically extracted as
structured data without runtime library invocation.

Targets three specific methods across three libraries and reports, for each:
  - the if-condition that guards the rule
  - the action class (raise / logger.warning / warnings.warn / minor_issues[k]=v /
    self.field = value)
  - the affected field name (if derivable)
  - the message template (if present as a string literal)

Run with system Python 3.10 which has all three libraries installed:
  /usr/bin/python3.10 scripts/probe_ast_scan_poc.py

The point is NOT to be a finished walker. The point is to show the patterns
are extractable. If this works end-to-end, the v2 proposal's AST-scan
mechanism is viable. If it hits structural obstacles, we need a different
approach.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SITE_PACKAGES = Path.home() / ".local/lib/python3.10/site-packages"

# Core engine methods + auxiliary-library rule-hosting methods.
TARGETS = [
    # --- Core engines ---
    {
        "library": "transformers",
        "file": SITE_PACKAGES / "transformers/generation/configuration_utils.py",
        "class": "GenerationConfig",
        "method": "validate",
    },
    {
        "library": "vllm",
        "file": SITE_PACKAGES / "vllm/sampling_params.py",
        "class": "SamplingParams",
        "method": "__post_init__",
    },
    {
        "library": "tensorrt_llm",
        "file": SITE_PACKAGES / "tensorrt_llm/llmapi/llm_args.py",
        "class": None,
        "method": None,
        "decorator_match": ["field_validator", "model_validator"],
    },
    # --- Auxiliary libraries used by the engines ---
    {
        "library": "bitsandbytes (via transformers)",
        "file": SITE_PACKAGES / "transformers/utils/quantization_config.py",
        "class": "BitsAndBytesConfig",
        "method": "post_init",
    },
    {
        "library": "peft.LoraConfig",
        "file": SITE_PACKAGES / "peft/tuners/lora/config.py",
        "class": "LoraConfig",
        "method": "__post_init__",
    },
]


ActionClass = Literal[
    "raise_error",
    "logger_warning",
    "warnings_warn",
    "minor_issues_assign",
    "self_field_assign",
    "other",
]


@dataclass
class ExtractedRule:
    library: str
    source_location: str
    condition: str
    action_class: ActionClass
    affected_field: str | None = None
    message_template: str | None = None
    action_detail: str | None = None
    raw_body_preview: str = ""


def classify_action(stmt: ast.stmt) -> tuple[ActionClass, str | None, str | None, str | None]:
    """Classify a statement inside an `if` body into a rule action.

    Returns (action_class, affected_field, message_template, action_detail).
    """
    # Pattern: raise ValueError(...) / raise TypeError(...) / raise ImportError(...)
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            exc_type = exc.func.id
            msg = _first_string_arg(exc)
            return "raise_error", None, msg, f"raise {exc_type}"

    # Pattern: <expr>, where <expr> is a Call
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        func_path = _call_func_path(call)

        # logger.warning(...) / logger.warning_once(...) / logger.info_once(...)
        if func_path and func_path[0] == "logger":
            return "logger_warning", None, _first_string_arg(call), ".".join(func_path)

        # warnings.warn(...)
        if func_path == ["warnings", "warn"]:
            return "warnings_warn", None, _first_string_arg(call), "warnings.warn"

    # Pattern: minor_issues["key"] = <expr>
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "minor_issues"
        ):
            key = _subscript_key(target)
            msg = _rhs_message(stmt.value)
            return "minor_issues_assign", key, msg, "minor_issues[key] = msg"

        # Pattern: self.field = value
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            field_name = target.attr
            rhs = ast.unparse(stmt.value) if hasattr(ast, "unparse") else "<rhs>"
            return "self_field_assign", field_name, None, f"self.{field_name} = {rhs}"

    return "other", None, None, ast.unparse(stmt)[:80] if hasattr(ast, "unparse") else "<stmt>"


def _call_func_path(call: ast.Call) -> list[str] | None:
    """Resolve a Call's func into a dotted path, e.g. logger.warning → ['logger', 'warning']."""
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _first_string_arg(call: ast.Call) -> str | None:
    """Return the first string-literal argument of a Call, if any."""
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        # Handle f-strings / format strings
        if isinstance(arg, ast.JoinedStr):
            return ast.unparse(arg) if hasattr(ast, "unparse") else "<fstring>"
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "format"
        ):
            # e.g., greedy_wrong_parameter_msg.format(flag_name="temperature", ...)
            return ast.unparse(arg) if hasattr(ast, "unparse") else "<formatted>"
    return None


def _subscript_key(sub: ast.Subscript) -> str | None:
    """Extract the key from a subscript, e.g. minor_issues["temperature"]."""
    slice_node = sub.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _rhs_message(value_node: ast.expr) -> str | None:
    """Extract a message from a minor_issues RHS."""
    if (
        isinstance(value_node, ast.Call)
        and isinstance(value_node.func, ast.Attribute)
        and value_node.func.attr == "format"
    ):
        return ast.unparse(value_node) if hasattr(ast, "unparse") else "<formatted>"
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
        return value_node.value
    return None


def extract_rules_from_method(
    method: ast.FunctionDef, library: str, source_path: Path
) -> list[ExtractedRule]:
    """Walk a method body, extracting rules from all `if` statements."""
    rules: list[ExtractedRule] = []

    for node in ast.walk(method):
        if not isinstance(node, ast.If):
            continue

        # Render the condition
        condition = ast.unparse(node.test) if hasattr(ast, "unparse") else "<condition>"
        source_loc = f"{source_path.name}:{method.name}():L{node.lineno}"

        # Walk the if body looking for rule actions
        for stmt in node.body:
            action_class, field_name, message, detail = classify_action(stmt)

            if action_class == "other":
                continue  # Skip non-rule statements (setup, fallthrough)

            rules.append(
                ExtractedRule(
                    library=library,
                    source_location=source_loc,
                    condition=condition,
                    action_class=action_class,
                    affected_field=field_name,
                    message_template=message,
                    action_detail=detail,
                    raw_body_preview=ast.unparse(stmt)[:100] if hasattr(ast, "unparse") else "",
                )
            )

    return rules


def find_method_in_class(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def find_decorated_methods(tree: ast.Module, decorator_names: list[str]) -> list[ast.FunctionDef]:
    """Find all FunctionDefs decorated with any of the given decorator names."""
    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_name = (
                _call_func_path(dec)[-1]
                if isinstance(dec, ast.Call) and _call_func_path(dec)
                else (dec.id if isinstance(dec, ast.Name) else None)
            )
            if dec_name in decorator_names:
                result.append(node)
                break
    return result


def process_target(target: dict) -> list[ExtractedRule]:
    source_path = Path(target["file"])
    if not source_path.exists():
        print(f"SKIP {target['library']}: {source_path} does not exist", file=sys.stderr)
        return []

    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))

    rules: list[ExtractedRule] = []

    if target.get("class") and target.get("method"):
        method = find_method_in_class(tree, target["class"], target["method"])
        if method is None:
            print(
                f"SKIP {target['library']}: {target['class']}.{target['method']} not found",
                file=sys.stderr,
            )
            return []
        rules.extend(extract_rules_from_method(method, target["library"], source_path))
    elif target.get("decorator_match"):
        methods = find_decorated_methods(tree, target["decorator_match"])
        print(
            f"\n{target['library']}: found {len(methods)} decorated validator methods",
            file=sys.stderr,
        )
        for method in methods[:5]:  # Limit to first 5 for PoC readability
            rules.extend(extract_rules_from_method(method, target["library"], source_path))

    return rules


def format_rule_summary(rules: list[ExtractedRule]) -> str:
    """Produce a YAML-ish summary of extracted rules."""
    lines = []
    by_action: dict[str, int] = {}
    for r in rules:
        by_action[r.action_class] = by_action.get(r.action_class, 0) + 1

    lines.append(f"# Total rules extracted: {len(rules)}")
    lines.append(f"# By action class: {by_action}")
    lines.append("")

    for r in rules:
        lines.append(f"- source: {r.source_location}")
        lines.append(f"  library: {r.library}")
        lines.append(f"  action: {r.action_class}")
        if r.affected_field:
            lines.append(f"  field: {r.affected_field}")
        lines.append(f"  condition: {r.condition!r}")
        if r.message_template:
            msg_preview = r.message_template[:120].replace("\n", " ")
            lines.append(f"  message: {msg_preview!r}")
        if r.action_detail and r.action_detail != r.action_class:
            lines.append(f"  detail: {r.action_detail}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    all_rules: list[ExtractedRule] = []

    for target in TARGETS:
        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"Scanning {target['library']}: {target['file']}", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)
        rules = process_target(target)
        all_rules.extend(rules)
        print(f"  extracted {len(rules)} rule(s)", file=sys.stderr)

    print(format_rule_summary(all_rules))
    print(f"\n{'=' * 80}", file=sys.stderr)
    print(f"TOTAL EXTRACTED: {len(all_rules)} rules", file=sys.stderr)
    print(f"{'=' * 80}", file=sys.stderr)

    # Report verdict
    by_lib: dict[str, int] = {}
    for r in all_rules:
        by_lib[r.library] = by_lib.get(r.library, 0) + 1
    print(f"By library: {by_lib}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
