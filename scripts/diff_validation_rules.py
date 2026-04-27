#!/usr/bin/env python3
"""Semantic differ for vendored rules JSON envelopes.

Sibling of :mod:`scripts.diff_discovered_schemas` — same contract (safe vs
breaking classification, JSON-on-stdout + Markdown summary), different artifact
kind (rule cases instead of parameter schemas).

Usage::

    python scripts/diff_validation_rules.py <old.json> <new.json> [--out pr-comment.md]

Exit codes:
    0 - identical or rules-safe changes only
    1 - rules-breaking changes detected
    2 - malformed input (missing file, not JSON, etc.)

stdout : JSON with structured diff (mirrors diff_discovered_schemas.py shape)
stderr : human-readable summary
--out  : Markdown summary suitable for a PR comment (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Change categories
# ---------------------------------------------------------------------------

SAFE_KINDS = frozenset(
    {
        "added_rule",
        "severity_relaxed",
        "match_widened",
        "message_template_changed",
        "emission_channel_widened",
    }
)

BREAKING_KINDS = frozenset(
    {
        "removed_rule",
        "severity_escalated",
        "outcome_changed",
        "match_narrowed",
        "emission_channel_changed",
    }
)

# Severity ordering; escalations move from less to more severe.
_SEVERITY_RANK = {
    "skipped_hardware_dependent": 0,
    "no_op": 1,
    "dormant_silent": 2,
    "dormant_announced": 3,
    "warn": 4,
    "error": 5,
}


@dataclass
class Change:
    rule_id: str
    kind: str
    detail: str = ""

    @property
    def severity(self) -> str:
        return "breaking" if self.kind in BREAKING_KINDS else "safe"

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"rule_id": self.rule_id, "kind": self.kind}
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class DiffResult:
    safe: list[Change] = field(default_factory=list)
    breaking: list[Change] = field(default_factory=list)
    metadata_changes: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def is_breaking(self) -> bool:
        return bool(self.breaking)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        for k, v in self.metadata_changes.items():
            parts.append(f"{k}: {v['old']} -> {v['new']}")
        if self.safe:
            parts.append(f"{len(self.safe)} rules-safe change(s)")
        if self.breaking:
            parts.append(f"{len(self.breaking)} rules-BREAKING change(s)")
        if not parts:
            parts.append("No changes")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Diff primitives
# ---------------------------------------------------------------------------


def _index_cases(envelope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in envelope.get("cases", []) if isinstance(c, dict) and "id" in c}


def _compare_case(rule_id: str, old: dict[str, Any], new: dict[str, Any]) -> list[Change]:
    changes: list[Change] = []

    old_outcome = str(old.get("outcome", ""))
    new_outcome = str(new.get("outcome", ""))
    if old_outcome != new_outcome:
        old_rank = _SEVERITY_RANK.get(old_outcome, -1)
        new_rank = _SEVERITY_RANK.get(new_outcome, -1)
        if new_rank > old_rank:
            changes.append(
                Change(
                    rule_id=rule_id,
                    kind="severity_escalated",
                    detail=f"{old_outcome} -> {new_outcome}",
                )
            )
        elif new_rank < old_rank:
            changes.append(
                Change(
                    rule_id=rule_id,
                    kind="severity_relaxed",
                    detail=f"{old_outcome} -> {new_outcome}",
                )
            )
        else:
            changes.append(
                Change(
                    rule_id=rule_id,
                    kind="outcome_changed",
                    detail=f"{old_outcome} -> {new_outcome}",
                )
            )

    old_channel = str(old.get("emission_channel", ""))
    new_channel = str(new.get("emission_channel", ""))
    if old_channel != new_channel:
        if old_channel == "none" and new_channel != "none":
            changes.append(
                Change(
                    rule_id=rule_id,
                    kind="emission_channel_widened",
                    detail=f"{old_channel} -> {new_channel}",
                )
            )
        else:
            changes.append(
                Change(
                    rule_id=rule_id,
                    kind="emission_channel_changed",
                    detail=f"{old_channel} -> {new_channel}",
                )
            )

    old_messages = _normalise_messages(old.get("observed_messages") or [])
    new_messages = _normalise_messages(new.get("observed_messages") or [])
    if old_messages != new_messages:
        changes.append(
            Change(
                rule_id=rule_id,
                kind="message_template_changed",
                detail=(f"{len(old_messages)} -> {len(new_messages)} message(s)"),
            )
        )

    return changes


def _normalise_messages(messages: list[Any]) -> tuple[str, ...]:
    """Reduce messages to a tuple of stripped strings for structural compare."""
    return tuple(sorted(str(m).strip() for m in messages))


def diff_rules(old: dict[str, Any], new: dict[str, Any]) -> DiffResult:
    """Compare two rules envelopes and classify per-rule changes."""
    result = DiffResult()

    for key in ("engine_version", "schema_version", "vendor_commit"):
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            result.metadata_changes[key] = {"old": str(old_val), "new": str(new_val)}

    old_cases = _index_cases(old)
    new_cases = _index_cases(new)
    old_ids = set(old_cases)
    new_ids = set(new_cases)

    for rule_id in sorted(new_ids - old_ids):
        result.safe.append(Change(rule_id=rule_id, kind="added_rule"))

    for rule_id in sorted(old_ids - new_ids):
        result.breaking.append(Change(rule_id=rule_id, kind="removed_rule"))

    for rule_id in sorted(old_ids & new_ids):
        for change in _compare_case(rule_id, old_cases[rule_id], new_cases[rule_id]):
            if change.severity == "safe":
                result.safe.append(change)
            else:
                result.breaking.append(change)

    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(result: DiffResult, *, title: str = "Rules Diff") -> str:
    lines: list[str] = [f"## {title}", ""]
    lines.append(f"**Summary:** {result.summary}")
    lines.append("")

    if result.metadata_changes:
        lines.append("### Envelope metadata")
        lines.append("")
        lines.append("| Field | Old | New |")
        lines.append("| --- | --- | --- |")
        for k, v in result.metadata_changes.items():
            lines.append(f"| `{k}` | `{v['old']}` | `{v['new']}` |")
        lines.append("")

    if result.breaking:
        lines.append("### rules-breaking changes")
        lines.append("")
        lines.append("| Rule ID | Kind | Detail |")
        lines.append("| --- | --- | --- |")
        for change in result.breaking:
            lines.append(f"| `{change.rule_id}` | `{change.kind}` | {change.detail or ''} |")
        lines.append("")

    if result.safe:
        lines.append("### rules-safe changes")
        lines.append("")
        lines.append("| Rule ID | Kind | Detail |")
        lines.append("| --- | --- | --- |")
        for change in result.safe:
            lines.append(f"| `{change.rule_id}` | `{change.kind}` | {change.detail or ''} |")
        lines.append("")

    if not result.metadata_changes and not result.safe and not result.breaking:
        lines.append("_No changes detected._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="Path to old JSON envelope")
    parser.add_argument("new", type=Path, help="Path to new JSON envelope")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If given, write Markdown summary to this path.",
    )
    parser.add_argument(
        "--title",
        default="Rules Diff",
        help="Title used in the Markdown summary.",
    )
    args = parser.parse_args(argv)

    try:
        old = json.loads(args.old.read_text())
        new = json.loads(args.new.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error reading envelopes: {exc}", file=sys.stderr)
        return 2

    result = diff_rules(old, new)

    output = {
        "safe": [c.as_dict() for c in result.safe],
        "breaking": [c.as_dict() for c in result.breaking],
        "metadata_changes": result.metadata_changes,
        "is_breaking": result.is_breaking,
        "summary": result.summary,
    }
    print(json.dumps(output, indent=2))

    print(f"\n{result.summary}", file=sys.stderr)
    if result.breaking:
        print("\nrules-breaking changes:", file=sys.stderr)
        for change in result.breaking:
            detail = f" ({change.detail})" if change.detail else ""
            print(f"  - {change.rule_id}: {change.kind}{detail}", file=sys.stderr)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_markdown(result, title=args.title))

    return 1 if result.is_breaking else 0


if __name__ == "__main__":
    sys.exit(main())
