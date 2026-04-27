#!/usr/bin/env python3
"""Semantic differ for vendored engine schema JSONs.

Classifies changes between two schema versions as safe or breaking.

Usage:
    python scripts/diff_discovered_schemas.py old.json new.json

Exit codes:
    0 = identical or safe changes only
    1 = breaking changes detected
    2 = error (malformed input, missing file, etc.)

stdout: JSON with structured diff
stderr: human-readable summary
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Param sections to diff (metadata fields like discovered_at are excluded
# implicitly by only diffing these sections).
_PARAM_SECTIONS = ("engine_params", "sampling_params")


@dataclass
class Change:
    section: str
    field: str
    kind: str  # added, removed, type_changed, default_changed, description_changed
    severity: str  # safe or breaking
    detail: str = ""


@dataclass
class DiffResult:
    safe: list[Change] = field(default_factory=list)
    breaking: list[Change] = field(default_factory=list)
    metadata_changes: dict[str, dict[str, str]] = field(default_factory=dict)
    is_breaking: bool = False
    summary: str = ""


def _is_type_narrowed(old_type: str, new_type: str) -> bool:
    """Return True if new_type is strictly narrower than old_type."""
    old_parts = {t.strip() for t in old_type.split("|")}
    new_parts = {t.strip() for t in new_type.split("|")}
    # Narrowed if new is a strict subset of old.
    return new_parts < old_parts


def _is_type_widened(old_type: str, new_type: str) -> bool:
    """Return True if new_type is strictly wider than old_type."""
    old_parts = {t.strip() for t in old_type.split("|")}
    new_parts = {t.strip() for t in new_type.split("|")}
    # Widened if old is a strict subset of new.
    return old_parts < new_parts


def diff_schemas(old: dict, new: dict) -> DiffResult:
    """Compare two schema dicts and classify changes."""
    result = DiffResult()

    # Check metadata changes.
    for key in ("engine_version", "schema_version"):
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val != new_val:
            result.metadata_changes[key] = {"old": str(old_val), "new": str(new_val)}

    # Diff param sections.
    for section in _PARAM_SECTIONS:
        old_params = old.get(section, {})
        new_params = new.get(section, {})

        old_keys = set(old_params.keys())
        new_keys = set(new_params.keys())

        # Added fields.
        for key in sorted(new_keys - old_keys):
            change = Change(section=section, field=key, kind="added", severity="safe")
            result.safe.append(change)

        # Removed fields.
        for key in sorted(old_keys - new_keys):
            change = Change(section=section, field=key, kind="removed", severity="breaking")
            result.breaking.append(change)

        # Changed fields.
        for key in sorted(old_keys & new_keys):
            old_field = old_params[key]
            new_field = new_params[key]

            old_type = old_field.get("type", "")
            new_type = new_field.get("type", "")

            if old_type != new_type:
                if _is_type_narrowed(old_type, new_type):
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_narrowed",
                        severity="breaking",
                        detail=f"{old_type} -> {new_type}",
                    )
                    result.breaking.append(change)
                elif _is_type_widened(old_type, new_type):
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_widened",
                        severity="safe",
                        detail=f"{old_type} -> {new_type}",
                    )
                    result.safe.append(change)
                else:
                    change = Change(
                        section=section,
                        field=key,
                        kind="type_changed",
                        severity="breaking",
                        detail=f"{old_type} -> {new_type}",
                    )
                    result.breaking.append(change)

            old_default = old_field.get("default")
            new_default = new_field.get("default")
            if old_default != new_default:
                change = Change(
                    section=section,
                    field=key,
                    kind="default_changed",
                    severity="safe",
                    detail=f"{old_default!r} -> {new_default!r}",
                )
                result.safe.append(change)

            old_desc = old_field.get("description", "")
            new_desc = new_field.get("description", "")
            if old_desc != new_desc and old_desc and new_desc:
                change = Change(
                    section=section,
                    field=key,
                    kind="description_changed",
                    severity="safe",
                )
                result.safe.append(change)

    result.is_breaking = len(result.breaking) > 0

    # Build summary.
    parts = []
    if result.metadata_changes:
        for k, v in result.metadata_changes.items():
            parts.append(f"{k}: {v['old']} -> {v['new']}")
    if result.safe:
        parts.append(f"{len(result.safe)} safe change(s)")
    if result.breaking:
        parts.append(f"{len(result.breaking)} BREAKING change(s)")
    if not parts:
        parts.append("No changes")
    result.summary = "; ".join(parts)

    return result


def _change_to_dict(c: Change) -> dict:
    d = {"section": c.section, "field": c.field, "kind": c.kind}
    if c.detail:
        d["detail"] = c.detail
    return d


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: diff_discovered_schemas.py <old.json> <new.json>", file=sys.stderr)
        return 2

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])

    try:
        old = json.loads(old_path.read_text())
        new = json.loads(new_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading schemas: {e}", file=sys.stderr)
        return 2

    result = diff_schemas(old, new)

    # Structured output to stdout.
    output = {
        "safe": [_change_to_dict(c) for c in result.safe],
        "breaking": [_change_to_dict(c) for c in result.breaking],
        "metadata_changes": result.metadata_changes,
        "is_breaking": result.is_breaking,
        "summary": result.summary,
    }
    print(json.dumps(output, indent=2))

    # Human-readable to stderr.
    print(f"\n{result.summary}", file=sys.stderr)
    if result.breaking:
        print("\nBreaking changes:", file=sys.stderr)
        for c in result.breaking:
            detail = f" ({c.detail})" if c.detail else ""
            print(f"  - [{c.section}] {c.field}: {c.kind}{detail}", file=sys.stderr)

    return 1 if result.is_breaking else 0


if __name__ == "__main__":
    sys.exit(main())
