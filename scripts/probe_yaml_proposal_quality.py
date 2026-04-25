#!/usr/bin/env python3
"""PoC-RT-3 — YAML proposal quality on real-shaped warning fixtures.

Question: do the YAML fragments produced by combining the inference algorithm
(PoC-RT-1) with the normalisation pipeline (preview's message_normalise.py) and
the corpus schema actually look mergeable?

Three concrete fixtures exercise the spectrum:

    F1: vLLM epsilon-clamp warning — range predicate that algorithm can't
        infer (fired configs vary in temperature). YAML should still be
        useful: pre-filled metadata + clear "predicate needs hand-writing"
        review note.
    F2: Transformers dormant-temperature warning — equality predicate
        algorithm CAN infer ({do_sample: False}). YAML should be
        merge-with-minor-edits.
    F3: vLLM max_num_batched_tokens ValueError — exception, NOT auto-fed
        back per user reframe. Demonstrates how the YAML for an exception
        looks when forced through the proposer (counter-example).

Pipeline:
    raw_message → normalise → template + regex →
    inference(fired_configs, not_fired_configs) → predicate →
    YAML proposal aligned with corpus schema (configs/validation_rules/transformers.yaml).

Hand-scored on:
    completeness, predicate quality, regex correctness, severity, overall mergability.

Reference: .claude/plans/m3-design-discussion-2026-04-24.md

Written by autonomous overnight PoC run, 2026-04-24.
"""

from __future__ import annotations

import hashlib
import re
from itertools import combinations
from typing import Any

# -----------------------------------------------------------------------------
# Inline normalisation (preview pattern)
# -----------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_0-9])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_0-9])")
_PATH_RE = re.compile(r"(?:(?<=\s)|^)(?:/[^\s:()\[\]]+|[A-Za-z]:\\[^\s:]+)")
_HEX_RE = re.compile(r"\b(?:sha256:)?[0-9a-fA-F]{16,}\b")
_WS_RE = re.compile(r"\s+")


def normalise(msg: str) -> tuple[str, str]:
    """Return (template, match_regex)."""
    n = _ANSI_RE.sub("", msg)
    n = _PATH_RE.sub(" <PATH>", n)
    n = _HEX_RE.sub("<HEX>", n)
    n = _NUMBER_RE.sub("<NUM>", n)
    template = _WS_RE.sub(" ", n).strip()
    # Build regex
    placeholders = {"<NUM>": r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", "<PATH>": r"\S+", "<HEX>": r"\S+"}
    tokens = re.split(r"(<NUM>|<PATH>|<HEX>)", template)
    parts = []
    for t in tokens:
        if t in placeholders:
            parts.append(placeholders[t])
        else:
            parts.append(re.escape(t))
    match_regex = r"\A" + "".join(parts) + r"\Z"
    return template, match_regex


# -----------------------------------------------------------------------------
# Inference algorithm (RT-1 reuse)
# -----------------------------------------------------------------------------


def infer_predicate(
    A: list[dict[str, Any]],
    B: list[dict[str, Any]],
    fields: list[str],
    max_arity: int = 3,
) -> tuple[str, dict | None]:
    if not A:
        return ("none", None)
    for arity in range(1, max_arity + 1):
        for fset in combinations(fields, arity):
            a_tuples = {tuple(a.get(f) for f in fset) for a in A}
            if len(a_tuples) != 1:
                continue
            value_tuple = next(iter(a_tuples))
            if any(all(b.get(f) == v for f, v in zip(fset, value_tuple)) for b in B):
                continue
            return ({1: "high", 2: "medium", 3: "low"}[arity], dict(zip(fset, value_tuple)))
    return ("none", None)


# -----------------------------------------------------------------------------
# YAML proposal renderer
# -----------------------------------------------------------------------------


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def guess_native_type(engine: str, fields: list[str]) -> str:
    """Best-effort native-type guess from engine + observed fields."""
    if engine == "transformers":
        if any(f in {"temperature", "top_p", "top_k", "do_sample", "num_beams"} for f in fields):
            return "transformers.GenerationConfig"
        return "transformers.AutoModel"  # placeholder
    if engine == "vllm":
        if any(f in {"temperature", "top_p", "top_k", "presence_penalty"} for f in fields):
            return "vllm.SamplingParams"
        if any(
            f in {"max_num_batched_tokens", "max_model_len", "tensor_parallel_size"} for f in fields
        ):
            return "vllm.LlmArgs"
        return "vllm.LLM"
    return f"{engine}.UnknownNativeType"


def render_proposal(
    *,
    raw_message: str,
    fixture_id: str,
    engine: str,
    library_version: str,
    emission_channel: str,
    severity_default: str,
    fired_configs: list[dict[str, Any]],
    not_fired_configs: list[dict[str, Any]],
) -> dict:
    """Render a corpus-schema YAML proposal from runtime-warning evidence."""
    template, regex = normalise(raw_message)
    fields = sorted({k for c in fired_configs + not_fired_configs for k in c.keys()})
    confidence, predicate = infer_predicate(fired_configs, not_fired_configs, fields)

    template_hash = hashlib.sha256(template.encode()).hexdigest()[:8]
    rule_id = f"{engine}_runtime_{fixture_id}_{template_hash}"

    # Build match.fields
    match_fields: dict = {}
    if predicate:
        for fname, fval in predicate.items():
            # Use bare field name (real corpus uses dotted paths like
            # "transformers.sampling.do_sample" — left to reviewer to namespace)
            match_fields[fname] = {"equals": fval}
    # else: empty {} — reviewer fills in

    # Pick representative kwargs
    kwargs_positive = dict(fired_configs[0]) if fired_configs else {}
    kwargs_negative = dict(not_fired_configs[0]) if not_fired_configs else {}

    proposal: dict = {
        "id": rule_id,
        "engine": engine,
        "rule_under_test": "(runtime-derived) Library emitted normalised template — reviewer to confirm semantic.",
        "severity": severity_default,
        "native_type": guess_native_type(engine, fields),
        "walker_source": {
            "path": "<runtime-derived; no AST source>",
            "method": "<runtime-derived>",
            "line_at_scan": 0,
            "walker_confidence": "low",
        },
        "match": {
            "engine": engine,
            "fields": match_fields,
        },
        "kwargs_positive": kwargs_positive,
        "kwargs_negative": kwargs_negative,
        "expected_outcome": {
            "outcome": severity_default,
            "emission_channel": emission_channel,
            "normalised_fields": [],
            "observed_messages_regex": [regex],
        },
        "message_template": template,
        "references": [
            f"Runtime capture; raw message sample: {raw_message!r}",
            f"Observed in {len(fired_configs)} configs; absent in {len(not_fired_configs)} configs.",
        ],
        "added_by": "runtime_warning",
        "added_at": "2026-04-24",
        "review_notes": _build_review_notes(
            predicate, confidence, fired_configs, not_fired_configs, severity_default
        ),
    }
    if not predicate:
        proposal["needs_generalisation_review"] = True
        proposal["evidence_field_value_distribution"] = _field_value_distribution(
            fired_configs, not_fired_configs
        )
    elif confidence in ("low", "medium"):
        proposal["needs_generalisation_review"] = True

    return proposal


def _build_review_notes(predicate, confidence, fired, not_fired, severity_default) -> str:
    bits = []
    if predicate is None:
        bits.append(
            "PREDICATE NOT INFERRED. Algorithm could not find a field-value combination "
            f"shared by all {len(fired)} fired configs and absent in all {len(not_fired)} not-fired configs."
        )
        bits.append(
            "Reviewer: hand-write `match.fields` based on evidence_field_value_distribution. "
            "Likely a range or disjunction predicate."
        )
    else:
        bits.append(
            f"Predicate inferred at algorithmic confidence={confidence}. "
            f"Walker confidence: low (always for runtime-derived)."
        )
        bits.append(
            "Reviewer: (1) confirm severity (current default: "
            f"{severity_default!r}); (2) confirm predicate generalises "
            "(equality vs range vs `present:true`); (3) review message template wording."
        )
    return " ".join(bits)


def _field_value_distribution(fired: list[dict], not_fired: list[dict]) -> dict:
    out = {}
    fields = sorted({k for c in fired + not_fired for k in c.keys()})
    for f in fields:
        fired_vals = sorted({str(c.get(f)) for c in fired})
        not_fired_vals = sorted({str(c.get(f)) for c in not_fired})
        out[f] = {"fired": fired_vals, "not_fired": not_fired_vals}
    return out


# -----------------------------------------------------------------------------
# Minimal YAML emitter (preserves ordering, no extra deps)
# -----------------------------------------------------------------------------


def to_yaml(obj: Any, indent: int = 0, _root: bool = True) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1, _root=False))
            else:
                lines.append(f"{pad}{k}: {_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                inner = to_yaml(item, indent + 1, _root=False)
                first_line, *rest = inner.split("\n")
                lines.append(f"{pad}- {first_line.lstrip()}")
                lines.extend(rest)
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        return "\n".join(lines)
    return _scalar(obj)


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#\n[]{},'\"") or s != s.strip():
        return repr(s).replace("\\\\", "\\")
    return s


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def fixture_1_vllm_epsilon_clamp() -> dict:
    """Range predicate algorithm CAN'T infer (varying temperature in fired set)."""
    return {
        "id": "epsilon_clamp",
        "raw_message": "You have set temperature=0.001 which is below the minimum 0.01; clamping to 0.01",
        "engine": "vllm",
        "library_version": "0.17.1",
        "emission_channel": "logger_warning_once",
        "severity_default": "warn",  # mechanically from log level; ideally dormant_silent (S5)
        "fired_configs": [
            {"temperature": 0.001, "top_p": 0.95, "top_k": 50, "do_sample": True},
            {"temperature": 0.005, "top_p": 1.0, "top_k": -1, "do_sample": True},
            {"temperature": 0.0001, "top_p": 0.5, "top_k": 100, "do_sample": True},
            {"temperature": 0.002, "top_p": 0.95, "top_k": -1, "do_sample": True},
            {"temperature": 0.0005, "top_p": 1.0, "top_k": 50, "do_sample": True},
        ],
        "not_fired_configs": [
            {"temperature": 0.01, "top_p": 0.95, "top_k": 50, "do_sample": True},
            {"temperature": 0.5, "top_p": 1.0, "top_k": -1, "do_sample": True},
            {"temperature": 0.7, "top_p": 0.5, "top_k": 100, "do_sample": True},
            {"temperature": 1.0, "top_p": 0.95, "top_k": -1, "do_sample": True},
            {"temperature": 0.3, "top_p": 1.0, "top_k": 50, "do_sample": True},
        ],
    }


def fixture_2_transformers_dormant_temp() -> dict:
    """Equality predicate algorithm CAN infer ({do_sample: False})."""
    return {
        "id": "dormant_temperature",
        "raw_message": "do_sample is set to False. However, temperature is set to 0.7 -- this flag will be ignored.",
        "engine": "transformers",
        "library_version": "4.56.0",
        "emission_channel": "logger_warning_once",
        "severity_default": "warn",  # ideally dormant
        "fired_configs": [
            {"do_sample": False, "temperature": 0.3, "top_p": 0.95, "top_k": 50},
            {"do_sample": False, "temperature": 0.7, "top_p": 1.0, "top_k": -1},
            {"do_sample": False, "temperature": 1.0, "top_p": 0.5, "top_k": 100},
            {"do_sample": False, "temperature": 0.5, "top_p": 0.95, "top_k": 50},
            {"do_sample": False, "temperature": 0.9, "top_p": 1.0, "top_k": -1},
        ],
        "not_fired_configs": [
            {"do_sample": True, "temperature": 0.7, "top_p": 0.95, "top_k": 50},
            {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
            {"do_sample": True, "temperature": 0.3, "top_p": 0.5, "top_k": 100},
            {"do_sample": True, "temperature": 0.5, "top_p": 0.95, "top_k": 50},
            {"do_sample": True, "temperature": 0.9, "top_p": 1.0, "top_k": -1},
        ],
    }


def fixture_3_vllm_exception() -> dict:
    """vLLM ValueError. Exceptions are NOT auto-fed back per user reframe.
    Used as counter-example showing what a forced proposal looks like.
    """
    return {
        "id": "max_num_batched_tokens_error",
        "raw_message": "max_num_batched_tokens (256) must be greater than or equal to max_model_len (4096) when chunked prefill is disabled.",
        "engine": "vllm",
        "library_version": "0.17.1",
        "emission_channel": "runtime_exception",
        "severity_default": "error",
        "fired_configs": [
            {"max_num_batched_tokens": 256, "max_model_len": 4096, "enable_chunked_prefill": False},
            {"max_num_batched_tokens": 512, "max_model_len": 4096, "enable_chunked_prefill": False},
            {"max_num_batched_tokens": 256, "max_model_len": 8192, "enable_chunked_prefill": False},
        ],
        "not_fired_configs": [
            {
                "max_num_batched_tokens": 8192,
                "max_model_len": 4096,
                "enable_chunked_prefill": False,
            },
            {"max_num_batched_tokens": 256, "max_model_len": 4096, "enable_chunked_prefill": True},
            {
                "max_num_batched_tokens": 4096,
                "max_model_len": 4096,
                "enable_chunked_prefill": False,
            },
        ],
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def main() -> int:
    print("PoC-RT-3 — YAML proposal quality")
    fixtures = [
        fixture_1_vllm_epsilon_clamp(),
        fixture_2_transformers_dormant_temp(),
        fixture_3_vllm_exception(),
    ]

    for i, f in enumerate(fixtures, 1):
        print(f"\n{'=' * 76}")
        print(f"Fixture F{i}: {f['id']} ({f['engine']})")
        print("=" * 76)
        print(f"\nRaw message:\n  {f['raw_message']!r}")

        template, regex = normalise(f["raw_message"])
        print(f"\nNormalised template:\n  {template}")
        print(f"\nMatch regex:\n  {regex}")

        proposal = render_proposal(
            raw_message=f["raw_message"],
            fixture_id=f["id"],
            engine=f["engine"],
            library_version=f["library_version"],
            emission_channel=f["emission_channel"],
            severity_default=f["severity_default"],
            fired_configs=f["fired_configs"],
            not_fired_configs=f["not_fired_configs"],
        )
        print("\nProposed YAML fragment:\n")
        yaml_out = to_yaml(proposal)
        # indent the YAML by 2 spaces for visual offset
        print("\n".join(f"  {line}" if line else "" for line in yaml_out.split("\n")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
