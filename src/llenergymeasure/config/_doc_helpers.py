"""Shared helpers for rendering Pydantic JSON schema properties as Markdown.

Private module consumed by doc-generation scripts. Not part of the public API.
"""

from __future__ import annotations

from typing import Any


def resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref string to its definition dict."""
    name = ref.split("/")[-1]
    result: dict[str, Any] = defs.get(name, {})
    return result


def type_label(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Return a human-readable type string for a JSON schema property."""
    if "$ref" in prop:
        title: str = resolve_ref(prop["$ref"], defs).get("title", prop["$ref"].split("/")[-1])
        return title

    prop_type = prop.get("type")

    any_of = prop.get("anyOf") or prop.get("allOf")
    if any_of:
        non_null = [p for p in any_of if p.get("type") != "null"]
        parts = [type_label(p, defs) for p in non_null]
        has_null = any(p.get("type") == "null" for p in any_of)
        label = " | ".join(parts)
        return f"{label} | None" if has_null else label

    if "enum" in prop:
        return " | ".join(repr(v) for v in prop["enum"])

    if prop_type == "array":
        items = prop.get("items", {})
        return f"list[{type_label(items, defs)}]"

    if prop_type == "object":
        return "dict"

    return prop_type or "any"


def default_label(prop: dict[str, Any]) -> str:
    """Return a human-readable default value for a JSON schema property."""
    if "default" not in prop:
        return "*(required)*"
    val = prop["default"]
    if val is None:
        return "`null`"
    if isinstance(val, bool):
        return f"`{str(val).lower()}`"
    return f"`{val}`"
