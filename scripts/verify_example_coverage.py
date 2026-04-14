#!/usr/bin/env python3
"""Verify that configs/example-study-full.yaml covers every typed engine field.

Walk all engine Pydantic model classes, collect every typed field's full YAML
path, then assert each one appears in the showcase config.

Usage:
    python scripts/verify_example_coverage.py
    uv run python scripts/verify_example_coverage.py

Exit codes:
    0  All typed paths present in the YAML.
    1  One or more typed paths missing from the YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without installing the package.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llenergymeasure.config.engine_configs import (  # noqa: E402
    TensorRTConfig,
    TensorRTKvCacheConfig,
    TensorRTQuantConfig,
    TensorRTSamplingConfig,
    TensorRTSchedulerConfig,
    TransformersConfig,
    VLLMAttentionConfig,
    VLLMBeamSearchConfig,
    VLLMEngineConfig,
    VLLMSamplingConfig,
    VLLMSpeculativeConfig,
)

CONFIG_PATH = REPO_ROOT / "configs" / "example-study-full.yaml"

# ---------------------------------------------------------------------------
# Field path collection
# ---------------------------------------------------------------------------


def _is_typed_field(model_class, field_name: str) -> bool:
    """Return True if the field has CurationMetadata (i.e. is a typed field)."""
    field = model_class.model_fields.get(field_name)
    if field is None:
        return False
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return False
    return "curation" in extra


def _collect_typed_paths(prefix: str, model_class) -> list[str]:
    """Recursively collect dot-separated YAML paths for all typed fields."""
    paths: list[str] = []
    for field_name in model_class.model_fields:
        if not _is_typed_field(model_class, field_name):
            continue
        full_path = f"{prefix}.{field_name}" if prefix else field_name
        paths.append(full_path)
    return paths


def collect_all_typed_paths() -> dict[str, list[str]]:
    """Return a dict mapping engine name -> list of expected YAML paths."""
    result: dict[str, list[str]] = {}

    # --- Transformers ---
    result["transformers"] = _collect_typed_paths("transformers", TransformersConfig)

    # --- vLLM ---
    vllm_paths: list[str] = []
    # VLLMEngineConfig fields under vllm.engine.*
    for path in _collect_typed_paths("vllm.engine", VLLMEngineConfig):
        vllm_paths.append(path)
    # VLLMSpeculativeConfig fields under vllm.engine.speculative.*
    for path in _collect_typed_paths("vllm.engine.speculative", VLLMSpeculativeConfig):
        vllm_paths.append(path)
    # VLLMAttentionConfig fields under vllm.engine.attention.*
    for path in _collect_typed_paths("vllm.engine.attention", VLLMAttentionConfig):
        vllm_paths.append(path)
    # VLLMSamplingConfig fields under vllm.sampling.*
    for path in _collect_typed_paths("vllm.sampling", VLLMSamplingConfig):
        vllm_paths.append(path)
    # VLLMBeamSearchConfig fields under vllm.beam_search.*
    for path in _collect_typed_paths("vllm.beam_search", VLLMBeamSearchConfig):
        vllm_paths.append(path)
    result["vllm"] = vllm_paths

    # --- TensorRT ---
    trt_paths: list[str] = []
    # TensorRTConfig top-level fields under tensorrt.*
    for path in _collect_typed_paths("tensorrt", TensorRTConfig):
        trt_paths.append(path)
    # TensorRTQuantConfig fields under tensorrt.quant.*
    for path in _collect_typed_paths("tensorrt.quant", TensorRTQuantConfig):
        trt_paths.append(path)
    # TensorRTKvCacheConfig fields under tensorrt.kv_cache.*
    for path in _collect_typed_paths("tensorrt.kv_cache", TensorRTKvCacheConfig):
        trt_paths.append(path)
    # TensorRTSchedulerConfig fields under tensorrt.scheduler.*
    for path in _collect_typed_paths("tensorrt.scheduler", TensorRTSchedulerConfig):
        trt_paths.append(path)
    # TensorRTSamplingConfig fields under tensorrt.sampling.*
    for path in _collect_typed_paths("tensorrt.sampling", TensorRTSamplingConfig):
        trt_paths.append(path)
    result["tensorrt"] = trt_paths

    return result


# ---------------------------------------------------------------------------
# YAML path collection
# ---------------------------------------------------------------------------


def _collect_yaml_paths(obj: object, prefix: str = "") -> set[str]:
    """Recursively collect all dot-separated key paths from a parsed YAML object."""
    paths: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            paths.add(full_key)
            paths.update(_collect_yaml_paths(v, full_key))
    elif isinstance(obj, list):
        for item in obj:
            paths.update(_collect_yaml_paths(item, prefix))
    return paths


def collect_yaml_paths(config_path: Path) -> set[str]:
    """Load the YAML and return all set paths from all experiment entries."""
    with config_path.open() as f:
        data = yaml.safe_load(f)

    all_paths: set[str] = set()

    # Top-level keys
    all_paths.update(_collect_yaml_paths(data))

    # Walk experiments list for per-experiment engine keys
    experiments = data.get("experiments", [])
    for exp in experiments:
        if isinstance(exp, dict):
            all_paths.update(_collect_yaml_paths(exp))

    return all_paths


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------


def verify_coverage() -> bool:
    """Check every typed path appears in the YAML. Return True if all present."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found: {CONFIG_PATH}", file=sys.stderr)
        return False

    expected_by_engine = collect_all_typed_paths()
    yaml_paths = collect_yaml_paths(CONFIG_PATH)

    all_ok = True
    grand_total = 0
    grand_covered = 0

    print(f"\nCoverage report: {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 70)

    for engine, paths in expected_by_engine.items():
        missing = [p for p in paths if p not in yaml_paths]
        covered = len(paths) - len(missing)
        grand_total += len(paths)
        grand_covered += covered

        status = "PASS" if not missing else "FAIL"
        print(f"\n  [{status}] {engine}: {covered}/{len(paths)} typed fields covered")

        if missing:
            all_ok = False
            for p in sorted(missing):
                print(f"        MISSING: {p}")

    print("\n" + "=" * 70)
    pct = (grand_covered / grand_total * 100) if grand_total else 0.0
    overall = "PASS" if all_ok else "FAIL"
    print(f"  [{overall}] Overall: {grand_covered}/{grand_total} ({pct:.0f}%)")
    print()

    return all_ok


if __name__ == "__main__":
    ok = verify_coverage()
    sys.exit(0 if ok else 1)
