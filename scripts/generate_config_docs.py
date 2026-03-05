#!/usr/bin/env python3
"""Generate configuration reference documentation from ExperimentConfig JSON schema.

Uses Pydantic model_json_schema() to extract the full schema, then renders
a structured Markdown reference grouped by section.

Usage:
    python scripts/generate_config_docs.py
    python scripts/generate_config_docs.py --output docs/study-config-reference.md

Output:
    Markdown to stdout (or --output path). Suitable for inlining into docs/study-config.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref string to its definition dict."""
    name = ref.split("/")[-1]
    return defs.get(name, {})


def _type_label(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Return a human-readable type string for a schema property."""
    if "$ref" in prop:
        return _resolve_ref(prop["$ref"], defs).get("title", prop["$ref"].split("/")[-1])

    prop_type = prop.get("type")

    # anyOf / allOf patterns (Optional[X] → anyOf: [X, null])
    any_of = prop.get("anyOf") or prop.get("allOf")
    if any_of:
        non_null = [p for p in any_of if p.get("type") != "null"]
        parts = [_type_label(p, defs) for p in non_null]
        has_null = any(p.get("type") == "null" for p in any_of)
        label = " | ".join(parts)
        return f"{label} | None" if has_null else label

    if "enum" in prop:
        return " | ".join(repr(v) for v in prop["enum"])

    if prop_type == "array":
        items = prop.get("items", {})
        return f"list[{_type_label(items, defs)}]"

    if prop_type == "object":
        return "dict"

    return prop_type or "any"


def _default_label(prop: dict[str, Any]) -> str:
    """Return a human-readable default value for a schema property."""
    if "default" not in prop:
        return "*(required)*"
    val = prop["default"]
    if val is None:
        return "`null`"
    if isinstance(val, bool):
        return f"`{str(val).lower()}`"
    if isinstance(val, str):
        return f"`{val}`"
    return f"`{val}`"


def _description(prop: dict[str, Any]) -> str:
    return prop.get("description", "")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

_SECTION_ORDER = [
    ("top-level", "Top-Level Fields"),
    ("decoder", "Decoder / Sampling (`decoder:`)"),
    ("warmup", "Warmup (`warmup:`)"),
    ("baseline", "Baseline (`baseline:`)"),
    ("energy", "Energy (`energy:`)"),
    ("pytorch", "PyTorch Backend (`pytorch:`)"),
    ("vllm_engine", "vLLM Engine (`vllm.engine:`)"),
    ("vllm_sampling", "vLLM Sampling (`vllm.sampling:`)"),
    ("vllm_beam_search", "vLLM Beam Search (`vllm.beam_search:`)"),
    ("vllm_attention", "vLLM Attention (`vllm.engine.attention:`)"),
    ("tensorrt", "TensorRT-LLM Backend (`tensorrt:`)"),
]

# Map from JSON schema $defs key to our section key
_DEF_TO_SECTION: dict[str, str] = {
    "DecoderConfig": "decoder",
    "WarmupConfig": "warmup",
    "BaselineConfig": "baseline",
    "EnergyConfig": "energy",
    "PyTorchConfig": "pytorch",
    "VLLMEngineConfig": "vllm_engine",
    "VLLMSamplingConfig": "vllm_sampling",
    "VLLMBeamSearchConfig": "vllm_beam_search",
    "VLLMAttentionConfig": "vllm_attention",
    "TensorRTConfig": "tensorrt",
}


def _render_table(props: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    lines = [
        "| Field | Type | Default | Description |",
        "|-------|------|---------|-------------|",
    ]
    for name, prop in props.items():
        # Resolve $ref to get actual property info
        if "$ref" in prop:
            section_name = prop["$ref"].split("/")[-1]
            # Use field-level description (from ExperimentConfig.Field) not class docstring
            desc = _description(prop)
            lines.append(f"| `{name}` | {section_name} | *(see section)* | {desc} |")
            continue

        # anyOf with $ref (Optional[SubModel])
        any_of = prop.get("anyOf") or []
        ref_in_anyof = next((p for p in any_of if "$ref" in p), None)
        if ref_in_anyof:
            section_name = ref_in_anyof["$ref"].split("/")[-1]
            # Use field-level description (from ExperimentConfig.Field) not class docstring
            desc = _description(prop)
            has_null = any(p.get("type") == "null" for p in any_of)
            type_str = f"{section_name} | None" if has_null else section_name
            default = _default_label(prop)
            lines.append(f"| `{name}` | {type_str} | {default} | {desc} |")
            continue

        type_str = _type_label(prop, defs)
        default = _default_label(prop)
        desc = _description(prop)
        lines.append(f"| `{name}` | {type_str} | {default} | {desc} |")
    return lines


def render_markdown(schema: dict[str, Any]) -> str:
    defs = schema.get("$defs", {})
    top_props = schema.get("properties", {})

    # Build section content
    sections: dict[str, list[str]] = {}

    # Top-level fields
    sections["top-level"] = _render_table(top_props, defs)

    # Sub-model sections from $defs
    for def_name, section_key in _DEF_TO_SECTION.items():
        if def_name in defs:
            def_schema = defs[def_name]
            props = def_schema.get("properties", {})
            if props:
                sections[section_key] = _render_table(props, defs)

    # Render output
    lines: list[str] = [
        "<!-- Auto-generated by scripts/generate_config_docs.py -- do not edit manually -->",
        "",
        "## Configuration Reference",
        "",
        "Full reference for all `ExperimentConfig` fields.",
        "All fields except `model` are optional and have sensible defaults.",
        "",
    ]

    # Table of contents
    lines.append("**Sections:**")
    for section_key, section_title in _SECTION_ORDER:
        if section_key in sections:
            anchor = section_title.lower()
            for ch in " /`.:()`":
                anchor = anchor.replace(ch, "-")
            while "--" in anchor:
                anchor = anchor.replace("--", "-")
            anchor = anchor.strip("-")
            lines.append(f"- [{section_title}](#{anchor})")
    lines.append("")

    # Sections
    for section_key, section_title in _SECTION_ORDER:
        if section_key not in sections:
            continue
        lines.append(f"### {section_title}")
        lines.append("")
        lines.extend(sections[section_key])
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate config reference Markdown")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write output to this file path (default: stdout)",
    )
    args = parser.parse_args()

    from llenergymeasure.config.models import ExperimentConfig

    schema = ExperimentConfig.model_json_schema()
    markdown = render_markdown(schema)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
