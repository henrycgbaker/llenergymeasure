"""PoC-L: Cross-module rule-hosting coverage audit.

Hypothesis
----------
Walker TARGETS lists an explicit set of (file, class, method) triples.
Rules living outside this list are invisible unless caught at runtime by the
H3-collision / runtime-warning feedback channels. The canonical motivating
case (vLLM greedy-strip in `create_engine_config()`, not `__post_init__`)
is precisely this class.

Rather than expanding the walker to follow imports or call graphs (expensive
and fragile), this PoC implements a lightweight COVERAGE AUDIT: grep the
library's full source tree for rule-pattern density, flag files not in the
TARGETS list that exceed a threshold. Output is a candidate list for the
maintainer to review and optionally add to TARGETS.

Non-goals
---------
- Resolving cross-module imports (too expensive)
- Following call graphs (way too expensive and fragile)
- Extracting the actual rules from candidate files (that's the walker's job
  once the file is in TARGETS)

This is strictly a "here are other places where rules might live" signal.

Decision criteria (pre-committed)
---------------------------------
- Finds ≥1 candidate per engine that isn't in the current walker TARGETS
  AND contains at least one user-relevant rule (manual classification)
    -> cross-module coverage is a real gap; add the candidate to TARGETS
       or manual-seed the rules; walker ships with coverage audit as a
       companion diagnostic.
- Finds zero candidates OR only noise (deprecation helpers, __repr__,
  internal init dispatch) -> cross-module gap is minimal at current
  library versions; maintain awareness via periodic reruns.

Run
---
  /usr/bin/python3.10 scripts/probe_walker_coverage_audit.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SITE_PACKAGES = Path.home() / ".local/lib/python3.10/site-packages"


# Known walker targets (subset — match the current probe_ast_scan_poc.py TARGETS).
CURRENT_TARGETS = {
    SITE_PACKAGES / "transformers/generation/configuration_utils.py",
    SITE_PACKAGES / "vllm/sampling_params.py",
    SITE_PACKAGES / "tensorrt_llm/llmapi/llm_args.py",
    SITE_PACKAGES / "transformers/utils/quantization_config.py",
    SITE_PACKAGES / "peft/tuners/lora/config.py",
}

# Library root directories for auditing.
LIBRARY_ROOTS = [
    ("transformers", SITE_PACKAGES / "transformers"),
    ("vllm", SITE_PACKAGES / "vllm"),
    ("tensorrt_llm", SITE_PACKAGES / "tensorrt_llm"),
]


@dataclass
class FileAudit:
    path: Path
    n_raise_in_if: int = 0
    n_logger_warning_in_if: int = 0
    n_warnings_warn_in_if: int = 0
    n_self_assign_in_if: int = 0

    @property
    def total(self) -> int:
        return (
            self.n_raise_in_if
            + self.n_logger_warning_in_if
            + self.n_warnings_warn_in_if
            + self.n_self_assign_in_if
        )


def count_rule_patterns(path: Path) -> FileAudit:
    """AST-scan a single file for rule-pattern density."""
    audit = FileAudit(path=path)
    try:
        src = path.read_text()
    except (UnicodeDecodeError, PermissionError, OSError):
        return audit
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return audit

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # We only care about `if` inside a method/function body where the
        # condition references `self.<field>`. Otherwise it's probably
        # argument-based logic (not a config rule).
        has_self_ref = False
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                if sub.value.id == "self":
                    has_self_ref = True
                    break
        if not has_self_ref:
            continue

        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                audit.n_raise_in_if += 1
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                f = stmt.value.func
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                    if f.value.id == "warnings" and f.attr == "warn":
                        audit.n_warnings_warn_in_if += 1
                    if f.value.id == "logger" and f.attr in (
                        "warning",
                        "warning_once",
                        "error",
                    ):
                        audit.n_logger_warning_in_if += 1
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                        if t.value.id == "self":
                            audit.n_self_assign_in_if += 1
    return audit


def audit_library(lib_root: Path) -> list[FileAudit]:
    """Audit all .py files under lib_root, skipping walker targets."""
    out: list[FileAudit] = []
    for py in lib_root.rglob("*.py"):
        if py in CURRENT_TARGETS:
            continue
        if any(part in ("test", "tests", "_test", "__pycache__") for part in py.parts):
            continue
        audit = count_rule_patterns(py)
        if audit.total > 0:
            out.append(audit)
    return sorted(out, key=lambda a: a.total, reverse=True)


def main() -> None:
    print("=" * 78)
    print("PoC-L: Cross-module walker coverage audit")
    print("=" * 78)
    print()
    print("Flags files containing rule-density-pattern matches that are NOT in")
    print("the walker's TARGETS list. These are candidate cross-module rule")
    print("hosts that the walker is currently missing.")
    print()

    THRESHOLD = 3  # minimum rule-pattern count to surface

    for lib_name, lib_root in LIBRARY_ROOTS:
        print(f"--- {lib_name}  ({lib_root.name}) ---")
        if not lib_root.exists():
            print("  library directory not found; skip")
            print()
            continue

        audits = audit_library(lib_root)
        flagged = [a for a in audits if a.total >= THRESHOLD]

        print(f"  Scanned {sum(1 for _ in lib_root.rglob('*.py'))} .py files (test dirs skipped)")
        print(f"  Files with >={THRESHOLD} rule-patterns outside TARGETS: {len(flagged)}")

        for a in flagged[:12]:
            rel = a.path.relative_to(lib_root)
            breakdown = (
                f"raise={a.n_raise_in_if} "
                f"log_warn={a.n_logger_warning_in_if} "
                f"warn={a.n_warnings_warn_in_if} "
                f"self_assign={a.n_self_assign_in_if}"
            )
            print(f"    {a.total:>4}  {str(rel)[:56]:<56}  {breakdown}")

        if len(flagged) > 12:
            print(f"    ... and {len(flagged) - 12} more below threshold")
        print()

    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print()
    print("For each flagged file, maintainer review:")
    print("  - Is this genuinely a rule-hosting module? (read the top N if-")
    print("    guarded raises/logs — are they config-validation rules?)")
    print("  - If yes: add to walker TARGETS list OR seed rules manually in YAML")
    print("  - If no (internal state, deprecation, __repr__ logic): ignore;")
    print("    the pattern density was noise.")
    print()
    print("Run cadence: on every library-bump CI run alongside walker. New")
    print("files appearing above threshold are automatic candidates for review.")
    print("Removing files from the list (e.g., library deletes a module) is")
    print("informational.")


if __name__ == "__main__":
    main()
