#!/usr/bin/env python3
"""Verify that configs/example-study-full.yaml covers every curated engine field.

Walk all engine Pydantic model classes, collect every curated field's full YAML
path, then assert each one appears in the showcase config.

A curated field is one with CurationMetadata (i.e. a ``curation`` key in its
``json_schema_extra`` dict). Uses :func:`llenergymeasure.config.introspection.is_curated_field`
as the single source of truth for that predicate.

Usage:
    uv run python scripts/verify_example_coverage.py
    python -m scripts.verify_example_coverage  (editable install)

Exit codes:
    0  All curated paths present in the YAML.
    1  One or more curated paths missing, or no curated fields found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from llenergymeasure.config.engine_configs import (
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
from llenergymeasure.config.introspection import is_curated_field

REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "example-study-full.yaml"

# ---------------------------------------------------------------------------
# Expected curated-field paths (engine -> list[str])
# Each tuple is (prefix, model_class) — prefix is the YAML path segment.
# ---------------------------------------------------------------------------

_ENGINE_SEGMENTS: dict[str, list[tuple[str, type]]] = {
    "transformers": [
        ("transformers", TransformersConfig),
    ],
    "vllm": [
        ("vllm.engine", VLLMEngineConfig),
        ("vllm.engine.speculative", VLLMSpeculativeConfig),
        ("vllm.engine.attention", VLLMAttentionConfig),
        ("vllm.sampling", VLLMSamplingConfig),
        ("vllm.beam_search", VLLMBeamSearchConfig),
    ],
    "tensorrt": [
        ("tensorrt", TensorRTConfig),
        ("tensorrt.quant", TensorRTQuantConfig),
        ("tensorrt.kv_cache", TensorRTKvCacheConfig),
        ("tensorrt.scheduler", TensorRTSchedulerConfig),
        ("tensorrt.sampling", TensorRTSamplingConfig),
    ],
}


def collect_expected_paths() -> dict[str, list[str]]:
    """Return engine -> sorted list of expected curated YAML paths."""
    result: dict[str, list[str]] = {}
    for engine, segments in _ENGINE_SEGMENTS.items():
        paths: list[str] = []
        for prefix, model_class in segments:
            for field_name, field_info in model_class.model_fields.items():
                if is_curated_field(field_info):
                    paths.append(f"{prefix}.{field_name}")
        result[engine] = sorted(paths)
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
    """Load the YAML and return all dot-separated paths across all experiment entries.

    Paths are collected relative to each experiment entry so that engine-scoped
    keys like ``transformers.batch_size`` are found regardless of their position
    in the experiments list.
    """
    with config_path.open() as f:
        data = yaml.safe_load(f)

    all_paths: set[str] = set()
    # Walk each experiment entry with an empty prefix so paths are experiment-relative.
    for exp in data.get("experiments", []):
        if isinstance(exp, dict):
            all_paths.update(_collect_yaml_paths(exp))
    return all_paths


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------


def verify_coverage() -> bool:
    """Check every curated path appears in the YAML. Return True if all present."""
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found: {CONFIG_PATH}", file=sys.stderr)
        return False

    expected_by_engine = collect_expected_paths()
    grand_total = sum(len(v) for v in expected_by_engine.values())

    # Guard: if no curated fields found, CurationMetadata is likely missing.
    if grand_total == 0:
        print(
            "ERROR: No curated fields found — is CurationMetadata present in engine_configs.py?",
            file=sys.stderr,
        )
        return False

    yaml_paths = collect_yaml_paths(CONFIG_PATH)

    all_ok = True
    grand_covered = 0

    print(f"\nCoverage report: {CONFIG_PATH.relative_to(REPO_ROOT)}")
    print("=" * 70)

    for engine, paths in expected_by_engine.items():
        missing = [p for p in paths if p not in yaml_paths]
        covered = len(paths) - len(missing)
        grand_covered += covered

        status = "PASS" if not missing else "FAIL"
        print(f"\n  [{status}] {engine}: {covered}/{len(paths)} curated fields covered")

        if missing:
            all_ok = False
            for p in sorted(missing):
                print(f"        MISSING: {p}")

    print("\n" + "=" * 70)
    pct = grand_covered / grand_total * 100
    overall = "PASS" if all_ok else "FAIL"
    print(f"  [{overall}] Overall: {grand_covered}/{grand_total} ({pct:.0f}%)")
    print()

    # TODO: comment-drift check — verify that inline `# curation_reason` YAML
    # comments match CurationMetadata.rationale for each field.
    # Deferred: YAML comment parsing requires ruamel.yaml round-trip mode, which
    # adds a dependency and significant complexity. Track as a follow-up task.

    return all_ok


if __name__ == "__main__":
    ok = verify_coverage()
    sys.exit(0 if ok else 1)
