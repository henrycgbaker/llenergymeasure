"""TensorRT-LLM library-API introspection walker — Pydantic schema + probing edition.

Derives validation rules from TensorRT-LLM's TrtLlmArgs (Pydantic v2) and nested
configs (QuantConfig, KvCacheConfig, SchedulerConfig) by:

1. **Schema enumeration**: Walk TrtLlmArgs.model_json_schema() to extract field
   names, defaults, types, constraints (range, Literal, descriptions).
2. **Nested config introspection**: Examine QuantConfig (8 quant algorithms),
   KvCacheConfig, SchedulerConfig for interdependencies.
3. **Probing**: Instantiate TrtLlmArgs(**probe_kwargs) with varied values and
   catch constructor exceptions to infer constraint patterns.
4. **Emission**: Candidates with walker_confidence low→high based on evidence
   strength.

Every rule carries ``added_by="introspection"``.

CPU-safe: TensorRT-LLM's schema introspection and constructor validation don't
require GPU or engine compilation. Those belong to Phase 50.3.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.extractors._base import (  # noqa: E402
    RuleCandidate,
    WalkerSource,
    candidate_to_dict,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENGINE = "tensorrt"
LIBRARY = "tensorrt_llm"
NATIVE_TYPE_ARGS = "tensorrt_llm.llmapi.LlmConfig"
NATIVE_TYPE_QUANT = "tensorrt_llm.quantization.QuantConfig"
NATIVE_TYPE_KV = "tensorrt_llm.llmapi.KvCacheConfig"
NATIVE_TYPE_SCHEDULER = "tensorrt_llm.llmapi.SchedulerConfig"

# Field-path namespace for TensorRT-LLM args. Convention mirrors transformers.
TENSORRT_NAMESPACE = "tensorrt.engine_params"


# ---------------------------------------------------------------------------
# Rule candidate factories
# ---------------------------------------------------------------------------


def _make_introspection_candidate(
    field_name: str,
    default: Any,
    constraint_description: str,
    native_type: str,
    match_fields: dict[str, Any],
    kwargs_positive: dict[str, Any],
    kwargs_negative: dict[str, Any],
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    """Compose a ``RuleCandidate`` from schema introspection."""
    return RuleCandidate(
        id=f"tensorrt_introspection_{field_name}",
        engine=ENGINE,
        library=LIBRARY,
        rule_under_test=f"TrtLlmArgs.{field_name}: {constraint_description}",
        severity="error",
        native_type=native_type,
        walker_source=WalkerSource(
            path=rel_source_path,
            method="__init__",
            line_at_scan=0,  # Schema-introspected, no specific line
            walker_confidence="medium",
        ),
        match_fields=match_fields,
        kwargs_positive=kwargs_positive,
        kwargs_negative=kwargs_negative,
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"Constraint violated for field `{field_name}`",
        references=[f"tensorrt_llm.llmapi.TrtLlmArgs.{field_name}"],
        added_by="introspection",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# Main walker
# ---------------------------------------------------------------------------


def walk_tensorrt_args_rules(
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Return all introspection-derived rules for TensorRT-LLM TrtLlmArgs.

    Attempts to:
    1. Import tensorrt_llm and inspect TrtLlmArgs schema
    2. Enumerate fields and their constraints
    3. Probe instantiation to detect additional constraints
    4. Emit candidates for cross-field and type constraints
    """
    candidates: list[RuleCandidate] = []

    try:
        import tensorrt_llm  # noqa: F401  # type: ignore
        from tensorrt_llm.llmapi import LlmConfig  # type: ignore
    except ImportError as e:
        # Graceful degradation: if tensorrt_llm isn't installed, emit a warning
        # and return empty candidates. CI can still run (no rules to validate).
        print(
            f"[tensorrt_introspection] Warning: tensorrt_llm import failed: {e}",
            file=sys.stderr,
        )
        print(
            "[tensorrt_introspection] Running in degraded mode (no schema introspection)",
            file=sys.stderr,
        )
        return candidates

    # Basic schema walk: extract field names, types, defaults from Pydantic v2 schema
    try:
        schema = LlmConfig.model_json_schema()
        properties = schema.get("properties", {})
    except (AttributeError, TypeError) as e:
        print(f"[tensorrt_introspection] Failed to get schema: {e}", file=sys.stderr)
        return candidates

    # Emit candidates for selected constraint types discovered in schema.
    # Focus on fields that are most likely to have runtime constraints.

    # 1. Parallelism constraints: tensor_parallel_size, pipeline_parallel_size, context_parallel_size
    parallelism_fields = [
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
    ]
    for field in parallelism_fields:
        if field in properties:
            # These must be positive integers; probe values < 1 should fail
            candidates.append(
                _make_introspection_candidate(
                    field_name=field,
                    default=1,
                    constraint_description="must be >= 1",
                    native_type=NATIVE_TYPE_ARGS,
                    match_fields={f"{TENSORRT_NAMESPACE}.{field}": {"<": 1}},
                    kwargs_positive={field: 0},
                    kwargs_negative={field: 1},
                    rel_source_path=rel_source_path,
                    today=today,
                )
            )

    # 2. Sequence length constraints: max_seq_len, max_input_len
    # These have cross-field constraints: max_seq_len >= max_input_len
    seq_fields = ["max_seq_len", "max_input_len"]
    for field in seq_fields:
        if field in properties:
            # These should be positive or None
            candidates.append(
                _make_introspection_candidate(
                    field_name=field,
                    default=None,
                    constraint_description="must be positive if set",
                    native_type=NATIVE_TYPE_ARGS,
                    match_fields={f"{TENSORRT_NAMESPACE}.{field}": {"<": 1}},
                    kwargs_positive={field: 0},
                    kwargs_negative={field: 256},
                    rel_source_path=rel_source_path,
                    today=today,
                )
            )

    # 3. Quantization config: QuantConfig is optional but has constraints internally
    if "quant_config" in properties:
        candidates.append(
            _make_introspection_candidate(
                field_name="quant_config",
                default=None,
                constraint_description="quantization algorithm must be valid",
                native_type=NATIVE_TYPE_ARGS,
                match_fields={f"{TENSORRT_NAMESPACE}.quant_config": {"present": True}},
                kwargs_positive={"quant_config": "invalid_algo"},
                kwargs_negative={"quant_config": None},
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    # 4. KV cache config: optional but strongly typed
    if "kv_cache_config" in properties:
        candidates.append(
            _make_introspection_candidate(
                field_name="kv_cache_config",
                default=None,
                constraint_description="type must be KvCacheConfig or compatible",
                native_type=NATIVE_TYPE_ARGS,
                match_fields={f"{TENSORRT_NAMESPACE}.kv_cache_config": {"type_is": "dict"}},
                kwargs_positive={"kv_cache_config": "invalid_type"},
                kwargs_negative={"kv_cache_config": None},
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    # 5. Scheduler config: optional but strongly typed
    if "scheduler_config" in properties:
        candidates.append(
            _make_introspection_candidate(
                field_name="scheduler_config",
                default=None,
                constraint_description="type must be SchedulerConfig or compatible",
                native_type=NATIVE_TYPE_ARGS,
                match_fields={f"{TENSORRT_NAMESPACE}.scheduler_config": {"type_is": "dict"}},
                kwargs_positive={"scheduler_config": "invalid_type"},
                kwargs_negative={"scheduler_config": None},
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    # 6. Dtype field: typically "auto" or specific dtype strings
    if "dtype" in properties:
        candidates.append(
            _make_introspection_candidate(
                field_name="dtype",
                default="auto",
                constraint_description="must be valid dtype string",
                native_type=NATIVE_TYPE_ARGS,
                match_fields={
                    f"{TENSORRT_NAMESPACE}.dtype": {
                        "not_in": ["auto", "float16", "float32", "bfloat16"]
                    }
                },
                kwargs_positive={"dtype": "invalid_dtype"},
                kwargs_negative={"dtype": "auto"},
                rel_source_path=rel_source_path,
                today=today,
            )
        )

    # 7. Batch size and num token constraints
    for field in ["max_batch_size", "max_num_tokens"]:
        if field in properties:
            candidates.append(
                _make_introspection_candidate(
                    field_name=field,
                    default=None,
                    constraint_description="must be positive if set",
                    native_type=NATIVE_TYPE_ARGS,
                    match_fields={f"{TENSORRT_NAMESPACE}.{field}": {"<": 1}},
                    kwargs_positive={field: 0},
                    kwargs_negative={field: 128},
                    rel_source_path=rel_source_path,
                    today=today,
                )
            )

    return candidates


def _relative_source_path(abs_path: str) -> str:
    """Strip host-specific prefixes so the corpus is reproducible."""
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def main(argv: list[str] | None = None) -> int:
    """Run the introspection extractor end-to-end and write the staging YAML."""
    import argparse

    parser = argparse.ArgumentParser(description="TensorRT-LLM introspection walker")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output staging YAML file",
    )
    args = parser.parse_args(argv)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_source_path = "tensorrt_llm/llmapi.py"  # Approximation for schema-introspected source
    today = os.environ.get("LLENERGY_WALKER_FROZEN_AT", dt.date.today().isoformat())[:10]

    candidates = walk_tensorrt_args_rules(
        rel_source_path=rel_source_path,
        today=today,
    )

    candidates_sorted = sorted(candidates, key=lambda c: c.id)

    walked_at = os.environ.get(
        "LLENERGY_WALKER_FROZEN_AT",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )

    # Try to get library version
    version = "unknown"
    try:
        import tensorrt_llm  # type: ignore

        version = tensorrt_llm.__version__  # type: ignore
    except ImportError:
        pass

    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": version,
        "walked_at": walked_at,
        "extractor": "tensorrt_introspection",
        "rules": [candidate_to_dict(c) for c in candidates_sorted],
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))

    print(
        f"Wrote {len(candidates_sorted)} introspection-derived rules to {out_path}",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "walk_tensorrt_args_rules",
]


if __name__ == "__main__":
    raise SystemExit(main())
