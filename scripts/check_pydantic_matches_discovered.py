#!/usr/bin/env python3
"""Check Pydantic engine configs align with discovered schemas.

Detects type drift between llem's hand-authored Pydantic models and the
machine-discovered engine parameter schemas. Catches:
- Pydantic Literal values going stale relative to engine enums
- Type narrowing/widening between Pydantic and discovered
- Pydantic fields with no discovered counterpart (unless whitelisted)

Exit 0: clean alignment. Exit 1: unexplained drift detected.
Structured JSON on stdout; human-readable details on stderr.

Run: python scripts/check_pydantic_matches_discovered.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llenergymeasure.config.introspection import get_engine_params
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.schema_loader import SchemaLoader
from llenergymeasure.config.ssot import Engine

ENGINES = tuple(e.value for e in Engine)

# Pydantic fields intentionally added by llem without an engine counterpart.
# Each entry: (engine, leaf_name) with explanation.
LLEM_NATIVE_FIELDS: set[tuple[str, str]] = {
    # -- transformers --
    # Quantization params surfaced at engine level for consistent interface
    ("transformers", "batch_size"),
    # dtype is HF-native (torch_dtype is a deprecated BC alias). Passed via
    # from_pretrained **kwargs, so signature-based discovery misses it.
    ("transformers", "dtype"),
    ("transformers", "batching_strategy"),
    ("transformers", "load_in_4bit"),
    ("transformers", "load_in_8bit"),
    ("transformers", "bnb_4bit_compute_dtype"),
    ("transformers", "bnb_4bit_quant_type"),
    ("transformers", "bnb_4bit_use_double_quant"),
    # Runtime/compile params not in from_pretrained or GenerationConfig
    ("transformers", "attn_implementation"),
    ("transformers", "torch_compile"),
    ("transformers", "torch_compile_mode"),
    ("transformers", "torch_compile_backend"),
    # Device/memory params
    ("transformers", "device_map"),
    ("transformers", "max_memory"),
    ("transformers", "allow_tf32"),
    ("transformers", "autocast_enabled"),
    ("transformers", "autocast_dtype"),
    ("transformers", "low_cpu_mem_usage"),
    # Parallelism
    ("transformers", "tp_plan"),
    ("transformers", "tp_size"),
    # -- vLLM --
    # Curated fields that map to engine params under different discovery paths
    ("vllm", "quantization_method"),
    ("vllm", "load_format"),
    # Speculative decoding sub-config
    ("vllm", "method"),
    ("vllm", "offload_group_size"),
    ("vllm", "offload_num_in_group"),
    ("vllm", "offload_prefetch_step"),
    ("vllm", "offload_params"),
    ("vllm", "kv_cache_memory_bytes"),
    # Attention sub-config (engine-internal knobs)
    ("vllm", "backend"),
    ("vllm", "flash_attn_version"),
    ("vllm", "flash_attn_max_num_splits_for_cuda_graph"),
    ("vllm", "use_prefill_decode_attention"),
    ("vllm", "use_prefill_query_quantization"),
    ("vllm", "use_cudnn_prefill"),
    ("vllm", "disable_flashinfer_prefill"),
    ("vllm", "disable_flashinfer_q_quantization"),
    ("vllm", "use_trtllm_attention"),
    ("vllm", "use_trtllm_ragged_deepseek_prefill"),
    # Beam search params (llem surfaces from vLLM internals)
    ("vllm", "beam_width"),
    ("vllm", "length_penalty"),
    ("vllm", "early_stopping"),
    # -- TensorRT --
    # Sub-config structure differs from engine API
    ("tensorrt", "max_batch_size"),
    ("tensorrt", "max_input_len"),
    ("tensorrt", "max_seq_len"),
    ("tensorrt", "max_num_tokens"),
    ("tensorrt", "free_gpu_memory_fraction"),
    ("tensorrt", "quant_algo"),
    ("tensorrt", "kv_cache_quant_algo"),
    ("tensorrt", "calib_dataset"),
    ("tensorrt", "calib_num_samples"),
    ("tensorrt", "kv_cache_free_gpu_mem_fraction"),
    ("tensorrt", "enable_block_reuse"),
    ("tensorrt", "max_tokens_in_paged_kv_cache"),
    ("tensorrt", "host_cache_size"),
    ("tensorrt", "top_k"),
    ("tensorrt", "top_p"),
    ("tensorrt", "temperature"),
    ("tensorrt", "repetition_penalty"),
    ("tensorrt", "length_penalty"),
    ("tensorrt", "max_new_tokens"),
    ("tensorrt", "end_id"),
    ("tensorrt", "pad_id"),
    ("tensorrt", "decoding_mode"),
    ("tensorrt", "capacity_scheduling_policy"),
    ("tensorrt", "context_chunking_policy"),
}


# ---------------------------------------------------------------------------
# Type canonicalisation
# ---------------------------------------------------------------------------

_JSON_TO_PYTHON_TYPE = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "str",
    "array": "list",
    "object": "dict",
}


def _canonicalise_discovered_type(type_str: str) -> str:
    """Canonicalise a discovered schema type string."""
    type_str = type_str.strip()

    # Remove | None suffix (llem always wraps in Optional)
    type_str = re.sub(r"\s*\|\s*None\s*$", "", type_str)

    # Handle Literal types - extract and sort values
    literal_match = re.match(r"Literal\[(.+)\]", type_str)
    if literal_match:
        inner = literal_match.group(1)
        values = sorted(v.strip().strip("'\"") for v in inner.split(","))
        return f"Literal[{', '.join(repr(v) for v in values)}]"

    # Normalise compound types (int | str → sorted)
    if "|" in type_str:
        parts = sorted(_JSON_TO_PYTHON_TYPE.get(p.strip(), p.strip()) for p in type_str.split("|"))
        return " | ".join(parts)

    # Normalise single JSON Schema type names to Python
    return _JSON_TO_PYTHON_TYPE.get(type_str, type_str)


def _canonicalise_pydantic_type(prop: dict[str, Any], defs: dict[str, Any]) -> str:
    """Canonicalise a Pydantic JSON schema property type."""
    # Handle anyOf (Optional[X] → anyOf: [X, null])
    any_of = prop.get("anyOf") or prop.get("allOf")
    if any_of:
        non_null = [p for p in any_of if p.get("type") != "null"]
        if len(non_null) == 1:
            return _canonicalise_pydantic_type(non_null[0], defs)
        # Multiple non-null types
        parts = sorted(_canonicalise_pydantic_type(p, defs) for p in non_null)
        return " | ".join(parts)

    # Handle $ref
    if "$ref" in prop:
        ref_name = prop["$ref"].split("/")[-1]
        ref_def = defs.get(ref_name, {})
        if "enum" in ref_def:
            values = sorted(str(v) for v in ref_def["enum"])
            return f"Literal[{', '.join(repr(v) for v in values)}]"
        return ref_name

    # Handle enum (Literal)
    if "enum" in prop:
        values = sorted(str(v) for v in prop["enum"])
        return f"Literal[{', '.join(repr(v) for v in values)}]"

    # Handle array
    if prop.get("type") == "array":
        items = prop.get("items", {})
        inner = _canonicalise_pydantic_type(items, defs)
        return f"list[{inner}]"

    # Base type
    base = prop.get("type", "any")
    return _JSON_TO_PYTHON_TYPE.get(base, base)


def _is_intentional_narrowing(discovered: str, pydantic: str) -> bool:
    """Check if Pydantic intentionally narrows a broad engine type.

    Allowed patterns:
    - str → Literal[...] (curating valid string values)
    - int → Literal[...] (curating valid int values)
    - Complex class type → simpler Pydantic type (e.g. CompilationConfig → dict)
    """
    if pydantic.startswith("Literal["):
        # Simple base type → Literal (str → Literal['a', 'b'])
        if discovered in ("str", "int", "float"):
            return True
        # Compound type containing str → Literal (str | SomeClass → Literal['a', 'b'])
        if "|" in discovered and any(p.strip() == "str" for p in discovered.split("|")):
            return True
    # Complex discovered type (class name) mapped to simple Pydantic type
    return (
        discovered[0].isupper()
        and not discovered.startswith("Literal[")
        and pydantic in ("dict", "str", "list")
    )


# ---------------------------------------------------------------------------
# Schema flattening
# ---------------------------------------------------------------------------


def _get_pydantic_leaves(engine: str, schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Get flattened Pydantic leaves for an engine with their JSON schema props.

    Returns dict mapping leaf_name -> JSON schema property dict.
    """
    defs = schema.get("$defs", {})
    params = get_engine_params(engine)
    result: dict[str, dict[str, Any]] = {}

    # Build a lookup from the JSON schema for detailed type info
    engine_config_names = {
        "transformers": ["TransformersConfig"],
        "vllm": [
            "VLLMEngineConfig",
            "VLLMSamplingConfig",
            "VLLMBeamSearchConfig",
            "VLLMAttentionConfig",
            "VLLMSpeculativeConfig",
        ],
        "tensorrt": [
            "TensorRTConfig",
            "TensorRTQuantConfig",
            "TensorRTKvCacheConfig",
            "TensorRTSamplingConfig",
            "TensorRTSchedulerConfig",
        ],
    }

    # Collect all properties from relevant $defs
    all_props: dict[str, dict[str, Any]] = {}
    for config_name in engine_config_names.get(engine, []):
        if config_name in defs:
            props = defs[config_name].get("properties", {})
            all_props.update(props)

    # Match introspection output to JSON schema props
    for _path, meta in params.items():
        leaf_name = meta["name"]
        result[leaf_name] = all_props.get(leaf_name, {})

    return result


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def check_engine(engine: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    """Check one engine for drift. Returns list of drift records."""
    drifts: list[dict[str, str]] = []
    defs = schema.get("$defs", {})

    loader = SchemaLoader()
    discovered = loader.load_schema(engine)

    # Combine engine_params and sampling_params from discovered
    all_discovered: dict[str, dict[str, Any]] = {}
    all_discovered.update(discovered.engine_params)
    all_discovered.update(discovered.sampling_params)

    # Get Pydantic leaves
    pydantic_leaves = _get_pydantic_leaves(engine, schema)

    # Check Pydantic fields against discovered
    for leaf_name, prop in pydantic_leaves.items():
        if leaf_name in all_discovered:
            # Both sides have it - compare types
            discovered_type = all_discovered[leaf_name].get("type", "")
            if not discovered_type or not prop or discovered_type == "unknown":
                continue

            canon_discovered = _canonicalise_discovered_type(discovered_type)
            canon_pydantic = _canonicalise_pydantic_type(prop, defs)

            if canon_discovered != canon_pydantic:
                # Allow intentional narrowing: engine exposes broad type,
                # llem curates to specific Literal values
                if _is_intentional_narrowing(canon_discovered, canon_pydantic):
                    continue
                drifts.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "kind": "type_mismatch",
                        "discovered": canon_discovered,
                        "pydantic": canon_pydantic,
                    }
                )
        else:
            # Pydantic has it, discovered doesn't
            if (engine, leaf_name) not in LLEM_NATIVE_FIELDS:
                drifts.append(
                    {
                        "engine": engine,
                        "field": leaf_name,
                        "kind": "pydantic_only",
                        "discovered": "(not present)",
                        "pydantic": _canonicalise_pydantic_type(prop, defs) if prop else "unknown",
                    }
                )

    return drifts


def main() -> None:
    schema = ExperimentConfig.model_json_schema()
    all_drifts: list[dict[str, str]] = []

    for engine in ENGINES:
        drifts = check_engine(engine, schema)
        all_drifts.extend(drifts)

        if drifts:
            print(f"\n[{engine}] {len(drifts)} drift(s) detected:", file=sys.stderr)
            for d in drifts:
                print(
                    f"  {d['field']}: {d['kind']} "
                    f"(discovered={d['discovered']}, pydantic={d['pydantic']})",
                    file=sys.stderr,
                )
        else:
            print(f"[{engine}] OK - no drift", file=sys.stderr)

    # Structured output on stdout
    json.dump({"drifts": all_drifts, "total": len(all_drifts)}, sys.stdout, indent=2)
    print(file=sys.stdout)

    sys.exit(1 if all_drifts else 0)


if __name__ == "__main__":
    main()
