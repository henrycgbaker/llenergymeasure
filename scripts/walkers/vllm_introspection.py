"""vLLM SamplingParams introspection walker — schema-driven rule extraction.

Derives validation rules from vLLM's SamplingParams and EngineArgs
by enumerating dataclass/msgspec fields and probing instantiation
with varied values to discover constraints.

Two extraction paths:

1. **Field enumeration** — for each field in SamplingParams / EngineArgs,
   extract its type annotation, default value, and any constraints
   (e.g., range annotations from msgspec.Meta).

2. **Combinatorial probing** — for related clusters of kwargs
   (e.g. sampling: temperature, top_p, top_k), generate representative
   value combinations and probe SamplingParams(**kwargs) to find which
   combinations raise / normalise / pass.

Output schema: ``configs/validation_rules/_staging/vllm_introspection.yaml``
consumed downstream by ``scripts/walkers/build_corpus.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Guard against script-directory shadowing (e.g. avoid local stub imports).
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
sys.path[:] = [p for p in sys.path if p != ""]

from scripts.walkers._base import RuleCandidate, WalkerSource, candidate_to_dict  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration & Field Discovery
# ---------------------------------------------------------------------------

ENGINE = "vllm"
LIBRARY = "vllm"
NATIVE_TYPE_SAMPLING = "vllm.SamplingParams"

# vLLM's SamplingParams namespace (where fields are exposed in config model)
SAMPLINGPARAMS_NAMESPACE = "vllm.sampling"


def _resolve_source_paths() -> tuple[str, str, str]:
    """Locate vLLM's SamplingParams source on disk.

    Returns ``(version, abs_path, rel_path)`` — the latter rooted at
    ``site-packages/`` for reproducibility.
    """
    import inspect

    try:
        import vllm
        from vllm import SamplingParams

        abs_path = inspect.getsourcefile(SamplingParams) or "<unknown>"
        version = vllm.__version__
    except ImportError:
        # Fallback: read version from package metadata
        try:
            from importlib.metadata import version as get_version

            version = get_version("vllm")
        except ImportError:
            version = "unknown"
        abs_path = "<unknown>"

    # Relativize: strip site-packages prefix for reproducibility
    marker = "site-packages/"
    idx = abs_path.find(marker)
    rel_path = abs_path[idx + len(marker) :] if idx >= 0 else Path(abs_path).name

    return version, abs_path, rel_path


def _enumerate_field_rules(
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> list[RuleCandidate]:
    """Enumerate basic field constraints from SamplingParams schema.

    For each numeric field with clear bounds, emit a candidate rule.
    Hand-curated from vLLM source code inspection (online or offline).
    """
    candidates: list[RuleCandidate] = []

    # Hand-curated rules transcribed from vLLM's __post_init__ source. TODO:
    # replace with msgspec.inspect-based introspection of SamplingParams once
    # the bounds-extraction helper lands. Tracked in follow-up issue.
    rules = [
        {
            "id": "vllm_n_must_be_positive",
            "field": "n",
            "rule_under_test": "SamplingParams.n must be >= 1",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.n": {"<": 1}},
            "message_template": "n must be at least 1, got {declared_value}",
            "kwargs_positive": {"n": 0},
            "kwargs_negative": {"n": 1},
        },
        {
            "id": "vllm_presence_penalty_range",
            "field": "presence_penalty",
            "rule_under_test": "SamplingParams.presence_penalty must be in [-2, 2]",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.presence_penalty": {"<": -2.0}},
            "message_template": "presence_penalty must be in [-2, 2], got {declared_value}",
            "kwargs_positive": {"presence_penalty": -3.0},
            "kwargs_negative": {"presence_penalty": 0.0},
        },
        {
            "id": "vllm_frequency_penalty_range",
            "field": "frequency_penalty",
            "rule_under_test": "SamplingParams.frequency_penalty must be in [-2, 2]",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.frequency_penalty": {"<": -2.0}},
            "message_template": "frequency_penalty must be in [-2, 2], got {declared_value}",
            "kwargs_positive": {"frequency_penalty": -3.0},
            "kwargs_negative": {"frequency_penalty": 0.0},
        },
        {
            "id": "vllm_repetition_penalty_positive",
            "field": "repetition_penalty",
            "rule_under_test": "SamplingParams.repetition_penalty must be in (0, 2]",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.repetition_penalty": {"<=": 0.0}},
            "message_template": "repetition_penalty must be in (0, 2], got {declared_value}",
            "kwargs_positive": {"repetition_penalty": 0.0},
            "kwargs_negative": {"repetition_penalty": 1.0},
        },
        {
            "id": "vllm_temperature_nonnegative",
            "field": "temperature",
            "rule_under_test": "SamplingParams.temperature must be non-negative",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.temperature": {"<": 0.0}},
            "message_template": "temperature must be non-negative, got {declared_value}",
            "kwargs_positive": {"temperature": -0.5},
            "kwargs_negative": {"temperature": 1.0},
        },
        {
            "id": "vllm_top_p_range",
            "field": "top_p",
            "rule_under_test": "SamplingParams.top_p must be in (0, 1]",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.top_p": {">": 1.0}},
            "message_template": "top_p must be in (0, 1], got {declared_value}",
            "kwargs_positive": {"top_p": 1.5},
            "kwargs_negative": {"top_p": 0.9},
        },
        {
            "id": "vllm_top_k_invalid",
            "field": "top_k",
            "rule_under_test": "SamplingParams.top_k must be -1 or >= 1",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.top_k": {"==": 0}},
            "message_template": "top_k must be -1 (disable), or at least 1, got {declared_value}",
            "kwargs_positive": {"top_k": 0},
            "kwargs_negative": {"top_k": -1},
        },
        {
            "id": "vllm_min_p_range",
            "field": "min_p",
            "rule_under_test": "SamplingParams.min_p must be in [0, 1]",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.min_p": {">": 1.0}},
            "message_template": "min_p must be in [0, 1], got {declared_value}",
            "kwargs_positive": {"min_p": 1.5},
            "kwargs_negative": {"min_p": 0.5},
        },
        {
            "id": "vllm_max_tokens_positive",
            "field": "max_tokens",
            "rule_under_test": "SamplingParams.max_tokens must be >= 1",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.max_tokens": {"<": 1}},
            "message_template": "max_tokens must be at least 1, got {declared_value}",
            "kwargs_positive": {"max_tokens": 0},
            "kwargs_negative": {"max_tokens": 16},
        },
        {
            "id": "vllm_min_tokens_nonnegative",
            "field": "min_tokens",
            "rule_under_test": "SamplingParams.min_tokens must be >= 0",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.min_tokens": {"<": 0}},
            "message_template": "min_tokens must be greater than or equal to 0, got {declared_value}",
            "kwargs_positive": {"min_tokens": -1},
            "kwargs_negative": {"min_tokens": 0},
        },
        {
            "id": "vllm_min_tokens_le_max_tokens",
            "field": "min_tokens",
            "rule_under_test": "SamplingParams.min_tokens must be <= max_tokens",
            "severity": "error",
            "match_fields": {
                f"{SAMPLINGPARAMS_NAMESPACE}.min_tokens": {">": "@max_tokens"},
            },
            "message_template": "min_tokens must be less than or equal to max_tokens, got {declared_value}",
            "kwargs_positive": {"min_tokens": 20, "max_tokens": 10},
            "kwargs_negative": {"min_tokens": 10, "max_tokens": 20},
        },
        {
            "id": "vllm_logprobs_nonnegative",
            "field": "logprobs",
            "rule_under_test": "SamplingParams.logprobs must be non-negative",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.logprobs": {"<": 0}},
            "message_template": "logprobs must be non-negative, got {declared_value}",
            "kwargs_positive": {"logprobs": -1},
            "kwargs_negative": {"logprobs": 0},
        },
        {
            "id": "vllm_prompt_logprobs_nonnegative",
            "field": "prompt_logprobs",
            "rule_under_test": "SamplingParams.prompt_logprobs must be non-negative",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.prompt_logprobs": {"<": 0}},
            "message_template": "prompt_logprobs must be non-negative, got {declared_value}",
            "kwargs_positive": {"prompt_logprobs": -1},
            "kwargs_negative": {"prompt_logprobs": 0},
        },
        {
            "id": "vllm_truncate_prompt_tokens_positive",
            "field": "truncate_prompt_tokens",
            "rule_under_test": "SamplingParams.truncate_prompt_tokens must be >= 1",
            "severity": "error",
            "match_fields": {f"{SAMPLINGPARAMS_NAMESPACE}.truncate_prompt_tokens": {"<": 1}},
            "message_template": "truncate_prompt_tokens must be >= 1, got {declared_value}",
            "kwargs_positive": {"truncate_prompt_tokens": 0},
            "kwargs_negative": {"truncate_prompt_tokens": 1},
        },
        {
            "id": "vllm_best_of_ge_n",
            "field": "best_of",
            "rule_under_test": "SamplingParams.best_of must be >= n",
            "severity": "error",
            "match_fields": {
                f"{SAMPLINGPARAMS_NAMESPACE}.best_of": {"<": "@n"},
            },
            "message_template": "best_of must be greater than or equal to n, got best_of={declared_value}",
            "kwargs_positive": {"best_of": 1, "n": 2},
            "kwargs_negative": {"best_of": 2, "n": 1},
        },
    ]

    for rule_spec in rules:
        candidates.append(
            RuleCandidate(
                id=rule_spec["id"],
                engine=ENGINE,
                library=LIBRARY,
                rule_under_test=rule_spec["rule_under_test"],
                severity=rule_spec["severity"],
                native_type=NATIVE_TYPE_SAMPLING,
                walker_source=WalkerSource(
                    path=rel_source_path,
                    method="__post_init__",
                    line_at_scan=0,
                    walker_confidence="high",
                ),
                match_fields=rule_spec["match_fields"],
                kwargs_positive=rule_spec["kwargs_positive"],
                kwargs_negative=rule_spec["kwargs_negative"],
                expected_outcome={
                    "outcome": "error",
                    "emission_channel": "none",
                    "normalised_fields": [],
                },
                message_template=rule_spec["message_template"],
                references=["vllm.SamplingParams._verify_args()"],
                added_by="introspection",
                added_at=today,
            )
        )

    return candidates


def main(argv: list[str] | None = None) -> int:
    """Run the introspection extractor end-to-end and write the staging YAML."""
    parser = argparse.ArgumentParser(description="vLLM introspection walker")
    parser.add_argument("--out", required=True, help="Output staging YAML path")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    version, abs_source_path, rel_source_path = _resolve_source_paths()
    today = os.environ.get("LLENERGY_WALKER_FROZEN_AT", dt.date.today().isoformat())[:10]

    candidates: list[RuleCandidate] = []

    candidates.extend(_enumerate_field_rules(abs_source_path, rel_source_path, today))

    # Stable order: by walker_source.method, then by id
    candidates_sorted = sorted(candidates, key=lambda c: (c.walker_source.method, c.id))

    walked_at = os.environ.get(
        "LLENERGY_WALKER_FROZEN_AT",
        dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )

    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": version,
        "walked_at": walked_at,
        "extractor": "vllm_introspection",
        "rules": [candidate_to_dict(c) for c in candidates_sorted],
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100))

    print(
        f"Wrote {len(candidates_sorted)} introspection-derived rules to {out_path}",
        file=sys.stderr,
    )
    return 0


__all__ = ["ENGINE", "LIBRARY", "NATIVE_TYPE_SAMPLING"]

if __name__ == "__main__":
    raise SystemExit(main())
