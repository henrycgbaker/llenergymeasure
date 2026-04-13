#!/usr/bin/env python3
"""Discover engine parameter schemas from installed engine APIs.

Runs inside an environment where the target engine package is installed
(typically a Docker container). For each engine, introspects the native
Python API surface and writes a JSON schema file with a common envelope.

Expected discovery targets:
    vllm         -> inside vllm/vllm-openai:<tag>
    tensorrt     -> inside nvcr.io/nvidia/tensorrt-llm/release:<tag>
    transformers -> inside llenergymeasure:transformers

Usage:
    python scripts/discover_engine_schemas.py vllm
    python scripts/discover_engine_schemas.py tensorrt
    python scripts/discover_engine_schemas.py transformers
    python scripts/discover_engine_schemas.py --all
    python scripts/discover_engine_schemas.py vllm --output /tmp/vllm.json
    python scripts/discover_engine_schemas.py vllm --image-ref vllm/vllm-openai:v0.7.3

The envelope is versioned separately from the engines (see SCHEMA_VERSION).
Major bumps are breaking and SchemaLoader rejects them. Minor bumps add
envelope keys; downstream loaders are expected to be forward-compatible.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, get_args, get_origin

SCHEMA_VERSION = "1.0.0"

DOCKERFILE_PATHS = {
    "vllm": "docker/Dockerfile.vllm",
    "tensorrt": "docker/Dockerfile.tensorrt",
    "transformers": "docker/Dockerfile.transformers",
}

DEFAULT_OUTPUT_DIR = "src/llenergymeasure/config/discovered_schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _annotation_to_type_str(annotation: Any) -> str:
    """Render a type annotation as a compact readable string.

    Handles None, Optional[X], X | None, Union, Literal, generics, forward
    refs, and inspect.Parameter.empty. Falls back to str(annotation) for
    anything unrecognised so discovery never raises on exotic types.
    """
    if annotation is type(None):
        return "None"
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return "unknown"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        return getattr(annotation, "__name__", str(annotation))

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        parts = [_annotation_to_type_str(a) for a in non_none]
        if has_none:
            parts.append("None")
        return " | ".join(parts)

    origin_str = str(origin)
    if "Literal" in origin_str:
        vals = ", ".join(repr(a) for a in args)
        return f"Literal[{vals}]"

    origin_name = getattr(origin, "__name__", origin_str)
    arg_strs = ", ".join(_annotation_to_type_str(a) for a in args)
    return f"{origin_name}[{arg_strs}]"


def _read_dockerfile_from(dockerfile: Path) -> str:
    """Extract the FROM tag from a Dockerfile, expanding the default ARG value.

    For multi-stage Dockerfiles, prefers the ``AS runtime`` stage (convention
    used by all llenergymeasure Dockerfiles). Falls back to the first FROM
    line that references an external image (not a prior stage name). Only
    default ARG values are substituted — no environment overrides.

    Returns e.g. ``"vllm/vllm-openai:v0.7.3"`` for a Dockerfile with
    ``ARG VLLM_VERSION=v0.7.3`` and ``FROM vllm/vllm-openai:${VLLM_VERSION}``.
    """
    text = dockerfile.read_text()
    arg_defaults: dict[str, str] = {}
    from_lines: list[tuple[str, str | None]] = []  # (ref, stage_alias)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ARG "):
            m = re.match(r"ARG\s+(\w+)(?:=(.+))?", stripped)
            if m:
                arg_defaults[m.group(1)] = (m.group(2) or "").strip()
            continue
        if stripped.startswith("FROM "):
            m = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?", stripped, re.IGNORECASE)
            if m:
                from_lines.append((m.group(1), m.group(2)))

    if not from_lines:
        raise ValueError(f"No FROM directive found in {dockerfile}")

    stage_names = {alias for _, alias in from_lines if alias}

    def _expand(ref: str) -> str:
        return re.sub(
            r"\$\{(\w+)\}",
            lambda match: arg_defaults.get(match.group(1), match.group(0)),
            ref,
        )

    for ref, alias in from_lines:
        if alias == "runtime":
            return _expand(ref)

    for ref, _ in from_lines:
        if ref not in stage_names:
            return _expand(ref)

    # All FROM lines reference prior stages — shouldn't happen in a valid Dockerfile
    return _expand(from_lines[0][0])


def _jsonable(value: Any) -> Any:
    """Coerce a value into something json.dumps can handle without default=str.

    Handles primitives, lists, tuples, dicts, sets, enums, and falls back to
    str(value) for anything else. This keeps the output deterministic and
    free of object repr noise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return str(value)


def _dataclass_fields_to_specs(
    cls: type, *, skip_private: bool = False
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: {type, default}}`` specs from a dataclass.

    Resolves ``default_factory`` by calling it (swallowing errors to ``None``)
    so downstream JSON stays concrete. Types are rendered via
    ``_annotation_to_type_str``.
    """
    specs: dict[str, dict[str, Any]] = {}
    for fld in dataclasses.fields(cls):
        if skip_private and fld.name.startswith("_"):
            continue
        default: Any = None
        if fld.default is not dataclasses.MISSING:
            default = fld.default
        elif fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = fld.default_factory()
            except Exception:
                default = None
        specs[fld.name] = {
            "type": _annotation_to_type_str(fld.type),
            "default": _jsonable(default),
        }
    return specs


def _make_envelope(
    *,
    engine: str,
    engine_version: str,
    engine_commit_sha: str | None,
    image_ref: str,
    base_image_ref: str,
    discovery_method: str,
    discovery_limitations: list[dict[str, Any]],
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": engine_version,
        "engine_commit_sha": engine_commit_sha,
        "image_ref": image_ref,
        "base_image_ref": base_image_ref,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "discovery_method": discovery_method,
        "discovery_limitations": discovery_limitations,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }


# ---------------------------------------------------------------------------
# Per-engine discovery
# ---------------------------------------------------------------------------


def discover_vllm(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM engine and sampling schemas.

    engine_params:   dataclasses.fields(EngineArgs)  (~86 fields)
    sampling_params: msgspec.json.schema(SamplingParams)  (~28 fields)
    """
    import vllm  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []
    engine_params = _dataclass_fields_to_specs(EngineArgs)

    sampling_params: dict[str, Any] = {}
    try:
        import msgspec  # type: ignore[import-not-found]

        raw_schema = msgspec.json.schema(vllm.SamplingParams)
        props = raw_schema.get("properties")
        if not props:
            defs = raw_schema.get("$defs") or raw_schema.get("definitions") or {}
            sp_def = defs.get("SamplingParams") or next(iter(defs.values()), {})
            props = sp_def.get("properties", {}) if isinstance(sp_def, dict) else {}
        for name, spec in (props or {}).items():
            type_repr: Any = spec.get("type", "unknown")
            if isinstance(type_repr, list):
                type_repr = " | ".join(str(t) for t in type_repr)
            sampling_params[name] = {
                "type": type_repr,
                "default": spec.get("default"),
            }
    except Exception as exc:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": [],
                "reason": f"msgspec.json.schema(SamplingParams) failed: {exc!r}",
            }
        )

    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "constraints (e.g. temperature>=0, top_p in (0,1]) live in imperative "
            "_verify_args() and are not introspectable from field metadata",
        }
    )
    limitations.append(
        {
            "section": "engine_params",
            "fields": [],
            "reason": "per-field descriptions unavailable (vLLM EngineArgs has only a class docstring)",
        }
    )

    base_image_ref = _read_dockerfile_from(repo_root / DOCKERFILE_PATHS["vllm"])
    return _make_envelope(
        engine="vllm",
        engine_version=vllm.__version__,
        engine_commit_sha=getattr(vllm, "__commit__", None),
        image_ref=image_ref or base_image_ref,
        base_image_ref=base_image_ref,
        discovery_method="dataclasses.fields(EngineArgs) + msgspec.json.schema(SamplingParams)",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )


def discover_tensorrt(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM engine and sampling schemas.

    Spike (2026-04-13, TRT-LLM 0.21.0 in pristine NGC image):
      - TrtLlmArgs is a Pydantic v2 BaseModel with model_json_schema() (61 fields)
      - LlmArgs is an alias for TrtLlmArgs
      - BuildConfig is NOT Pydantic -> appears as Optional[object] in the schema
      - KvCacheConfig / SchedulerConfig / CalibConfig / BuildCacheConfig are
        Pydantic (fallback path, unused because primary path works)
      - SamplingParams is a dataclass with 47 public fields
      - tensorrt_llm.__commit__ is not exposed (null)

    engine_params:   TrtLlmArgs.model_json_schema() properties (with description + deprecated)
    sampling_params: dataclasses.fields(SamplingParams)
    """
    import tensorrt_llm  # type: ignore[import-not-found]
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []

    raw_schema = TrtLlmArgs.model_json_schema()
    engine_params: dict[str, Any] = {}
    for name, spec in raw_schema.get("properties", {}).items():
        if name.startswith("_"):
            continue
        type_repr: Any = spec.get("type")
        if type_repr is None and "anyOf" in spec:
            parts: list[str] = []
            for sub in spec["anyOf"]:
                if "type" in sub:
                    part = "None" if sub["type"] == "null" else str(sub["type"])
                elif "$ref" in sub:
                    part = str(sub["$ref"]).rsplit("/", 1)[-1]
                else:
                    continue
                if part not in parts:  # dedupe string | string etc.
                    parts.append(part)
            type_repr = " | ".join(parts) if parts else "unknown"
        elif type_repr is None and "$ref" in spec:
            type_repr = str(spec["$ref"]).rsplit("/", 1)[-1]
        if isinstance(type_repr, list):
            type_repr = " | ".join("None" if t == "null" else str(t) for t in type_repr)
        elif type_repr == "null":
            type_repr = "None"
        engine_params[name] = {
            "type": type_repr or "unknown",
            "default": spec.get("default"),
            "description": spec.get("description"),
            "deprecated": spec.get("deprecated", False),
        }

    limitations.append(
        {
            "section": "engine_params",
            "fields": ["build_config"],
            "reason": "BuildConfig is not a Pydantic model; appears as Optional[object] in the schema",
        }
    )

    sampling_params = _dataclass_fields_to_specs(SamplingParams, skip_private=True)

    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "SamplingParams is a dataclass; no per-field descriptions",
        }
    )

    base_image_ref = _read_dockerfile_from(repo_root / DOCKERFILE_PATHS["tensorrt"])
    return _make_envelope(
        engine="tensorrt",
        engine_version=tensorrt_llm.__version__,
        engine_commit_sha=getattr(tensorrt_llm, "__commit__", None),
        image_ref=image_ref or base_image_ref,
        base_image_ref=base_image_ref,
        discovery_method="TrtLlmArgs.model_json_schema() + dataclasses.fields(SamplingParams)",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )


def discover_transformers(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover HuggingFace Transformers schemas.

    engine_params:   best-effort inspect.signature(from_pretrained) scrape;
                     **kwargs are opaque and recorded as a limitation
    sampling_params: GenerationConfig().to_dict() (~69 fields); None defaults
                     get type='unknown' and are listed in discovery_limitations
    """
    import transformers  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        GenerationConfig,
        PreTrainedModel,
    )

    limitations: list[dict[str, Any]] = []
    engine_params: dict[str, Any] = {}
    kwargs_points: list[str] = []

    for cls_name, cls in (
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("PreTrainedModel", PreTrainedModel),
    ):
        try:
            sig = inspect.signature(cls.from_pretrained)
        except (TypeError, ValueError) as exc:
            limitations.append(
                {
                    "section": "engine_params",
                    "fields": [cls_name],
                    "reason": f"inspect.signature({cls_name}.from_pretrained) failed: {exc!r}",
                }
            )
            continue

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "pretrained_model_name_or_path"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                kwargs_points.append(f"{cls_name}.from_pretrained.**{name}")
                continue
            if name in engine_params:
                continue  # prefer AutoModelForCausalLM over PreTrainedModel
            default = None if param.default is inspect.Parameter.empty else param.default
            engine_params[name] = {
                "type": _annotation_to_type_str(param.annotation),
                "default": _jsonable(default),
            }

    if kwargs_points:
        limitations.append(
            {
                "section": "engine_params",
                "fields": kwargs_points,
                "reason": "from_pretrained accepts **kwargs; kwargs are not in the signature "
                "(documented kwargs live in the class docstring only)",
            }
        )

    sampling_params: dict[str, Any] = {}
    none_default_fields: list[str] = []
    gc = GenerationConfig()
    for name, value in gc.to_dict().items():
        if value is None:
            sampling_params[name] = {"type": "unknown", "default": None}
            none_default_fields.append(name)
        elif isinstance(value, (list, tuple)):
            sampling_params[name] = {"type": type(value).__name__, "default": _jsonable(value)}
        elif isinstance(value, dict):
            sampling_params[name] = {"type": "dict", "default": _jsonable(value)}
        else:
            sampling_params[name] = {
                "type": type(value).__name__,
                "default": _jsonable(value),
            }

    if none_default_fields:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": none_default_fields,
                "reason": "GenerationConfig has no type annotations; None defaults yield type='unknown'",
            }
        )

    base_image_ref = _read_dockerfile_from(repo_root / DOCKERFILE_PATHS["transformers"])
    return _make_envelope(
        engine="transformers",
        engine_version=transformers.__version__,
        engine_commit_sha=getattr(transformers, "__commit__", None),
        image_ref=image_ref or base_image_ref,
        base_image_ref=base_image_ref,
        discovery_method="inspect.signature(from_pretrained) + GenerationConfig().to_dict()",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )


DISCOVERY_FUNCTIONS = {
    "vllm": discover_vllm,
    "tensorrt": discover_tensorrt,
    "transformers": discover_transformers,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_output_path(
    *, engine: str, output_arg: Path | None, multi: bool, repo_root: Path
) -> Path:
    if output_arg is None:
        return repo_root / DEFAULT_OUTPUT_DIR / f"{engine}.json"
    if multi or output_arg.is_dir() or output_arg.suffix == "":
        return output_arg / f"{engine}.json"
    return output_arg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover engine parameter schemas and write vendored JSON files."
    )
    parser.add_argument(
        "engines",
        nargs="*",
        choices=list(DISCOVERY_FUNCTIONS),
        default=[],
        help="One or more engines to discover (vllm, tensorrt, transformers). "
        "Omit when using --all.",
    )
    parser.add_argument("--all", action="store_true", help="Discover all known engines.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (single engine) or directory (multiple engines). "
        f"Default: {DEFAULT_OUTPUT_DIR}/<engine>.json relative to repo root.",
    )
    parser.add_argument(
        "--image-ref",
        default=None,
        help="Image reference to record in envelope.image_ref. Defaults to the "
        "Dockerfile FROM tag (also recorded as base_image_ref).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (for Dockerfile lookup). Defaults to the parent of the scripts/ directory.",
    )

    args = parser.parse_args(argv)
    requested: list[str] = list(DISCOVERY_FUNCTIONS) if args.all else args.engines
    if not requested:
        parser.error("Specify at least one engine, or use --all.")

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for engine in requested:
        try:
            envelope = DISCOVERY_FUNCTIONS[engine](repo_root, args.image_ref)
        except ImportError as exc:
            print(f"[{engine}] SKIPPED (not importable): {exc}", file=sys.stderr)
            failed.append((engine, "not importable"))
            continue
        except Exception as exc:
            print(f"[{engine}] FAILED: {exc!r}", file=sys.stderr)
            failed.append((engine, repr(exc)))
            continue

        out_path = _resolve_output_path(
            engine=engine,
            output_arg=args.output,
            multi=len(requested) > 1,
            repo_root=repo_root,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(envelope, indent=2, sort_keys=False, default=_jsonable) + "\n"
        )
        print(
            f"[{engine}] wrote {out_path} "
            f"(version={envelope['engine_version']}, "
            f"engine_params={len(envelope['engine_params'])}, "
            f"sampling_params={len(envelope['sampling_params'])})"
        )
        succeeded.append(engine)

    if not succeeded:
        print(f"\nAll engines failed: {failed}", file=sys.stderr)
        return 1
    if failed:
        print(
            f"\nPartial success: {succeeded} ok, {[e for e, _ in failed]} skipped/failed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
