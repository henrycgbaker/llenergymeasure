"""PoC-H: AST walker — variable resolution and loop-dedup.

Hypothesis
----------
v2 Appendix F.5 listed two "known walker limitations":
  1. Local variable references in message templates aren't resolved.
     HF's validate() has `greedy_wrong_parameter_msg = "..."` followed by
     many `minor_issues[k] = greedy_wrong_parameter_msg.format(...)`.
     Walker currently emits the raw ast.unparse string "greedy_wrong_...".
  2. Loop-wrapped rules produce per-iteration duplicates.
     HF's validate() also has `for arg in generate_arguments: if hasattr: raise`
     producing N rule entries where we want one parameterised rule.

Both were labelled "small polish" with estimates ~30 LoC. This PoC implements
both and measures the actual LoC + output-quality delta.

Decision criteria (pre-committed)
---------------------------------
- Variable resolution: >=80% of "template-reference" messages resolve to
  concrete strings -> polish works; fold into walker for P1.
- Loop dedup: loop-iteration duplicates collapse into 1 parameterised rule
  without losing info -> polish works.
- Combined implementation is <100 LoC -> matches v2 "small polish" estimate.
- Combined implementation is >200 LoC OR produces <50% success on either
  check -> revise walker-LoC estimates in §4.2.

Run
---
  /usr/bin/python3.10 scripts/probe_walker_variable_resolution.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SITE_PACKAGES = Path.home() / ".local/lib/python3.10/site-packages"
TARGET = SITE_PACKAGES / "transformers/generation/configuration_utils.py"


@dataclass
class ResolvedRule:
    line: int
    action: str
    field: str | None
    condition: str
    raw_message: str
    resolved_message: str | None
    is_loop_rule: bool = False
    loop_iterable: str | None = None


def collect_string_assigns(body: list[ast.stmt]) -> dict[str, str]:
    """First pass: collect local variable = 'string literal' assignments."""
    env: dict[str, str] = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    env[var_name] = node.value.value
                elif isinstance(node.value, ast.JoinedStr):
                    # f-strings: best-effort (just grab the constant parts)
                    parts = []
                    for v in node.value.values:
                        if isinstance(v, ast.Constant):
                            parts.append(v.value)
                        else:
                            parts.append("{...}")
                    env[var_name] = "".join(parts)
    return env


def resolve_message(msg_node: ast.expr, env: dict[str, str]) -> str | None:
    """If msg_node is <var>.format(...), return env[var]; else None."""
    if isinstance(msg_node, ast.Call) and isinstance(msg_node.func, ast.Attribute):
        if msg_node.func.attr == "format" and isinstance(msg_node.func.value, ast.Name):
            var = msg_node.func.value.id
            return env.get(var)
    # Bare reference: minor_issues[k] = some_var
    if isinstance(msg_node, ast.Name):
        return env.get(msg_node.id)
    return None


def classify_and_resolve(method: ast.FunctionDef) -> list[ResolvedRule]:
    """Walk method; resolve messages; detect loop rules; classify."""
    env = collect_string_assigns(method.body)
    rules: list[ResolvedRule] = []

    def _process_if(if_node: ast.If, is_loop: bool, loop_iter: str | None) -> None:
        try:
            cond_src = ast.unparse(if_node.test)
        except Exception:
            cond_src = "<unparseable>"
        for stmt in if_node.body:
            # minor_issues[k] = <msg>
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    tgt = stmt.targets[0]
                    if (
                        isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "minor_issues"
                    ):
                        # field name comes from the subscript
                        field_name = None
                        if isinstance(tgt.slice, ast.Constant):
                            field_name = tgt.slice.value
                        try:
                            raw = ast.unparse(stmt.value)
                        except Exception:
                            raw = "<unparseable>"
                        resolved = resolve_message(stmt.value, env)
                        rules.append(
                            ResolvedRule(
                                line=stmt.lineno,
                                action="minor_issues_assign",
                                field=field_name,
                                condition=cond_src,
                                raw_message=raw,
                                resolved_message=resolved,
                                is_loop_rule=is_loop,
                                loop_iterable=loop_iter,
                            )
                        )
            if isinstance(stmt, ast.Raise):
                try:
                    raw = ast.unparse(stmt.exc) if stmt.exc else ""
                except Exception:
                    raw = "<unparseable>"
                rules.append(
                    ResolvedRule(
                        line=stmt.lineno,
                        action="raise",
                        field=None,
                        condition=cond_src,
                        raw_message=raw,
                        resolved_message=None,
                        is_loop_rule=is_loop,
                        loop_iterable=loop_iter,
                    )
                )

    # Walk the body, flagging rules inside `for` loops
    for node in ast.walk(method):
        if isinstance(node, ast.If):
            # Is this if inside a for-loop? Check ancestry.
            parent_loop = None  # Simplification: we do a second scan below
            _process_if(node, False, None)

    # Second pass: identify for-wrapped ifs explicitly
    # Overwrite rules where needed to flag is_loop=True
    loop_flagged: set[int] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.For):
            try:
                iter_src = ast.unparse(node.iter)
            except Exception:
                iter_src = "<unparseable>"
            for inner in ast.walk(node):
                if isinstance(inner, ast.If):
                    for r in rules:
                        if r.line in (inner.body[0].lineno if inner.body else -1,):
                            r.is_loop_rule = True
                            r.loop_iterable = iter_src
                            loop_flagged.add(r.line)

    return rules


def dedup_loop_rules(rules: list[ResolvedRule]) -> list[ResolvedRule]:
    """Collapse rules that share (action, condition, is_loop_rule, loop_iterable)."""
    seen: dict[tuple, ResolvedRule] = {}
    out: list[ResolvedRule] = []
    for r in rules:
        if r.is_loop_rule:
            key = (r.action, r.condition, r.loop_iterable)
            if key not in seen:
                seen[key] = r
                out.append(r)
        else:
            out.append(r)
    return out


def main() -> None:
    print("=" * 78)
    print("PoC-H: Walker variable resolution + loop dedup")
    print("=" * 78)
    print()

    src = TARGET.read_text()
    tree = ast.parse(src)
    validate_method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GenerationConfig":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "validate":
                    validate_method = item
                    break

    if validate_method is None:
        print("ERROR: couldn't find GenerationConfig.validate()")
        return

    print(f"Target: {TARGET.name} :: GenerationConfig.validate()")
    print(f"        (lines {validate_method.lineno}-{validate_method.end_lineno})")
    print()

    rules_raw = classify_and_resolve(validate_method)
    rules_deduped = dedup_loop_rules(rules_raw)

    # --- Variable resolution metrics ---
    template_refs = [
        r for r in rules_raw if "format" in r.raw_message or r.raw_message in {r.raw_message}
    ]
    # Count references to locally-defined string vars
    var_refs = [r for r in rules_raw if ".format(" in r.raw_message or r.raw_message.isidentifier()]
    resolved = [r for r in var_refs if r.resolved_message]
    resolve_pct = (len(resolved) / len(var_refs) * 100) if var_refs else 0.0

    # --- Loop dedup metrics ---
    loop_rules = [r for r in rules_raw if r.is_loop_rule]
    collapse = len(rules_raw) - len(rules_deduped)

    print(f"Raw rules extracted:          {len(rules_raw)}")
    print(f"After loop-dedup:             {len(rules_deduped)}  ({collapse} iterations collapsed)")
    print()
    print(f"Variable-reference messages:  {len(var_refs)}")
    print(f"Successfully resolved:        {len(resolved)}  ({resolve_pct:.1f}%)")
    print()

    print("--- Sample resolved rules (first 5 with resolved messages) ---")
    for r in resolved[:5]:
        print(f"  [{r.action}] field={r.field}  line={r.line}")
        print(f"    cond: {r.condition[:70]}")
        print(f"    msg (resolved): {(r.resolved_message or '')[:110]}")
        print()

    print("--- Loop-rule samples ---")
    for r in loop_rules[:3]:
        print(f"  line={r.line}  loop_iter={r.loop_iterable}")
        print(f"    cond: {r.condition[:90]}")
        print()

    # --- LoC count of this implementation ---
    # Quick self-measurement of the relevant functions
    this_file = Path(__file__).read_text()
    impl_funcs = ["collect_string_assigns", "resolve_message", "dedup_loop_rules"]
    loc_of_impl = 0
    in_func = False
    indent = 0
    for line in this_file.splitlines():
        stripped = line.lstrip()
        if any(line.startswith(f"def {f}(") for f in impl_funcs):
            in_func = True
            indent = len(line) - len(stripped)
            loc_of_impl += 1
            continue
        if in_func:
            if not stripped:
                loc_of_impl += 1
                continue
            current_indent = len(line) - len(stripped)
            if current_indent <= indent and stripped.startswith(("def ", "class ")):
                in_func = False
            else:
                loc_of_impl += 1

    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    print(f"Variable resolution success rate:  {resolve_pct:.1f}% (target >=80%)")
    print(f"Loop-dedup iterations collapsed:   {collapse}")
    print(f"Implementation LoC (rough):        ~{loc_of_impl} lines")
    print("  (v2 §F.5 estimated ~30 LoC per polish; real target <100 LoC combined)")
    print()
    if resolve_pct >= 80 and loc_of_impl < 100:
        print("-> Variable resolution works. LoC within v2's 'small polish' estimate.")
        print("   Fold both polish passes into the P1 walker.")
    elif resolve_pct >= 50:
        print("-> Variable resolution partially works. May need second pass for")
        print("   f-string interpolation. Investigate unresolved cases below.")
    else:
        print("-> Variable resolution struggles with this library's pattern.")
        print("   Revise walker-LoC estimate in §4.2.")

    unresolved = [r for r in var_refs if not r.resolved_message]
    if unresolved:
        print()
        print("Unresolved-message samples (first 3, for debug):")
        for r in unresolved[:3]:
            print(f"  line={r.line}  raw={r.raw_message[:90]}")


if __name__ == "__main__":
    main()
