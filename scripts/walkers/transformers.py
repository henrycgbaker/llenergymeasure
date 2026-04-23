"""Transformers rules extractor — library-API introspection wrapper.

Per the 2026-04-23 plan amendment and runtime-config-validation.md §4.4,
transformers specifically is extracted via library-API introspection rather
than an AST walker: HF's :meth:`GenerationConfig.validate` populates a
``minor_issues`` dict and (in ``strict=True`` mode) raises a composed
``ValueError`` listing every issue. This exposes the rule surface
programmatically without AST maintenance burden. vLLM and TensorRT-LLM will
use full AST walkers (landing in later per-engine phases).

PoC-J (2026-04-23) confirmed 11/11 greedy+beam dormancy fields and all
error-class rules visible through this API. This module packages that
introspection into a reproducible extractor that emits a corpus-compatible
YAML document.

Usage
-----

::

    python -m scripts.walkers.transformers --out configs/validation_rules/transformers.yaml

The output is intended for human review before merging into the tracked
corpus. The seeded ``configs/validation_rules/transformers.yaml`` in-repo was
produced by this command against the pinned transformers version.

Scope
-----

- :class:`transformers.GenerationConfig` — sampling / greedy / beam rules
  surfaced via ``validate(strict=True)`` and the ``minor_issues`` dict.
- :class:`transformers.BitsAndBytesConfig` — type-check rules extracted via
  targeted probe configs (covers the 10 ``TypeError`` raises from
  ``post_init``).

peft / LoraConfig are explicitly out of scope for P1 (deferred).
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet

# NOTE: the walkers package is a sibling; when run via ``python -m
# scripts.walkers.transformers`` the implicit ``scripts`` package makes this
# work. When run as a script directly, we need the project root on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.walkers._base import (  # noqa: E402  (late import after sys.path)
    RuleCandidate,
    WalkerLandmarkMissingError,
    WalkerSource,
    check_installed_version,
)

TESTED_AGAINST_VERSIONS: SpecifierSet = SpecifierSet(">=4.50,<5.0")
"""Range of transformers versions this walker was authored against.

On mismatch, :func:`check_installed_version` raises
:class:`WalkerVersionMismatchError` and CI fails. When HF 4.x → 5.x ships,
a maintainer re-runs this walker against the new version, reconciles any
landmark / rule-shape changes, and bumps this range.
"""


# ---------------------------------------------------------------------------
# Known rule templates
# ---------------------------------------------------------------------------
# The introspection path surfaces issues by field name and by message; it
# does not expose structured predicates. For each HF field we know about, we
# encode the match predicate here (the walker's equivalent of the AST
# condition-extraction step). This is explicit rather than hidden in a
# library-source regex and stays comparable to the vLLM/TRT-LLM walkers'
# output.


# --- Greedy dormancy (fires when do_sample=False and the field is set) ---
#
# HF emits these as minor_issues entries. Match predicate: do_sample is False
# AND the field is present and not equal to its default. Default values come
# from the library's own GenerationConfig defaults.

_GREEDY_RULES: tuple[tuple[str, Any, Any], ...] = (
    # (field_name, default_value, positive_value)
    ("temperature", 1.0, 0.9),
    ("top_p", 1.0, 0.95),
    ("top_k", 50, 40),
    ("min_p", None, 0.1),
    ("typical_p", 1.0, 0.9),
    ("epsilon_cutoff", 0.0, 0.05),
    ("eta_cutoff", 0.0, 0.05),
)

# --- Single-beam dormancy (fires when num_beams=1 and the field is set) ---

_BEAM_RULES: tuple[tuple[str, Any, Any], ...] = (
    ("early_stopping", False, True),
    ("num_beam_groups", 1, 2),
    ("diversity_penalty", 0.0, 0.5),
    ("length_penalty", 1.0, 2.0),
    ("constraints", None, ["..."]),
)

# --- GenerationConfig.validate() cross-field rules (error + dormant) ---
# Most are error-class (raise ValueError); a few are dormant-class (write to
# HF's `minor_issues` dict, emitted via logger.warning_once). Dormant rules
# set `severity` / `outcome` / `emission_channel` explicitly; errors use
# factory defaults.
_GENERATION_VALIDATE_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "transformers_negative_max_new_tokens",
        "rule_under_test": ("GenerationConfig(max_new_tokens) rejects non-positive values"),
        "match_fields": {
            "transformers.sampling.max_new_tokens": {"<=": 0},
        },
        "kwargs_positive": {"max_new_tokens": -1},
        "kwargs_negative": {"max_new_tokens": 16},
        "message_template": ("`max_new_tokens` must be greater than 0, but is {declared_value}."),
        "source_method": "validate",
    },
    {
        "id": "transformers_invalid_cache_implementation",
        "rule_under_test": ("GenerationConfig rejects unknown cache_implementation strings"),
        "match_fields": {
            "transformers.sampling.cache_implementation": {"present": True},
        },
        "kwargs_positive": {"cache_implementation": "nonsense"},
        "kwargs_negative": {"cache_implementation": "static"},
        "message_template": (
            "Invalid `cache_implementation` ({declared_value}). Choose one from the supported set."
        ),
        "source_method": "validate",
    },
    {
        "id": "transformers_invalid_early_stopping",
        "rule_under_test": ("GenerationConfig.early_stopping must be bool or the literal 'never'"),
        "match_fields": {
            # `present: true` guards against firing on the LlmEnergyMeasure
            # default (None) — HF's GenerationConfig default is False but
            # our ExperimentConfig defaults the field to None, and the
            # predicate runs against our config.
            "transformers.sampling.early_stopping": {
                "present": True,
                "not_in": [True, False, "never"],
            },
        },
        "kwargs_positive": {"early_stopping": "sometimes"},
        "kwargs_negative": {"early_stopping": True},
        "message_template": (
            "`early_stopping` must be a boolean or 'never', but is {declared_value}."
        ),
        "source_method": "validate",
    },
    {
        "id": "transformers_num_return_sequences_exceeds_num_beams",
        "rule_under_test": ("GenerationConfig rejects num_return_sequences > num_beams"),
        "match_fields": {
            "transformers.sampling.num_return_sequences": {">": 1},
            "transformers.sampling.num_beams": {"present": True},
        },
        "kwargs_positive": {"num_return_sequences": 4, "num_beams": 2},
        "kwargs_negative": {"num_return_sequences": 2, "num_beams": 4},
        "message_template": ("`num_return_sequences` ({declared_value}) must be <= `num_beams`."),
        "source_method": "validate",
    },
    {
        "id": "transformers_num_beams_not_divisible_by_groups",
        "rule_under_test": ("Diverse beam search requires num_beams divisible by num_beam_groups"),
        "match_fields": {
            "transformers.sampling.num_beam_groups": {">": 1},
            "transformers.sampling.num_beams": {"present": True},
        },
        "kwargs_positive": {"num_beams": 5, "num_beam_groups": 2},
        "kwargs_negative": {"num_beams": 4, "num_beam_groups": 2},
        "message_template": (
            "`num_beams` ({declared_value}) must be divisible by `num_beam_groups`."
        ),
        "source_method": "validate",
    },
    {
        "id": "transformers_greedy_rejects_num_return_sequences",
        "rule_under_test": (
            "Greedy decoding (do_sample=False, num_beams=1) requires num_return_sequences=1"
        ),
        "match_fields": {
            "transformers.sampling.do_sample": False,
            "transformers.sampling.num_beams": 1,
            "transformers.sampling.num_return_sequences": {">": 1},
        },
        "kwargs_positive": {"do_sample": False, "num_beams": 1, "num_return_sequences": 3},
        "kwargs_negative": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1},
        "message_template": (
            "Greedy methods without beam search do not support "
            "`num_return_sequences` != 1 (got {declared_value})."
        ),
        "source_method": "validate",
    },
    {
        "id": "transformers_diversity_penalty_requires_diverse_beams",
        "rule_under_test": (
            "diversity_penalty is only valid with diverse beam search (num_beam_groups>1)"
        ),
        "match_fields": {
            "transformers.sampling.diversity_penalty": {"present": True, ">": 0.0},
            "transformers.sampling.num_beam_groups": 1,
        },
        "kwargs_positive": {"diversity_penalty": 0.5, "num_beam_groups": 1, "num_beams": 4},
        "kwargs_negative": {"diversity_penalty": 0.5, "num_beam_groups": 2, "num_beams": 4},
        "message_template": (
            "`diversity_penalty` > 0 requires `num_beam_groups` > 1 for diverse beam search."
        ),
        "source_method": "validate",
    },
    {
        # pad_token_id < 0 writes to HF's minor_issues dict, NOT raises.
        # Emission is via logger.warning_once post-collection. Severity is
        # dormant-announcement, not error.
        "id": "transformers_negative_pad_token_id",
        "rule_under_test": "GenerationConfig.validate() records dormant pad_token_id < 0",
        "severity": "dormant",
        "outcome": "dormant_announced",
        "emission_channel": "logger_warning_once",
        "match_fields": {
            "transformers.sampling.pad_token_id": {"<": 0},
        },
        "kwargs_positive": {"pad_token_id": -1},
        "kwargs_negative": {"pad_token_id": 0},
        "message_template": (
            "`pad_token_id` ({declared_value}) should be non-negative. This will cause "
            "errors when batch generating, if there is padding."
        ),
        "source_method": "validate",
    },
    {
        # HF calls `self.watermarking_config.validate()` directly — a raw
        # dict raises AttributeError (not a coded ValueError). We catch it
        # earlier at our config layer with a type predicate, converting an
        # opaque AttributeError into a user-friendly validation error.
        "id": "transformers_watermarking_config_type",
        "rule_under_test": (
            "GenerationConfig.watermarking_config must be a WatermarkingConfig instance"
        ),
        "match_fields": {
            "transformers.sampling.watermarking_config": {
                "present": True,
                "type_is_not": "WatermarkingConfig",
            },
        },
        "kwargs_positive": {"watermarking_config": {"greenlist_ratio": 0.25}},
        "kwargs_negative": {"watermarking_config": None},
        "message_template": (
            "`watermarking_config` must be a WatermarkingConfig instance; "
            "got {declared_value}. Construct a WatermarkingConfig or pass None."
        ),
        "source_method": "validate",
    },
    {
        # HF raises ValueError when compile_config is not an instance of
        # CompileConfig. We mirror with a type predicate for user-friendly
        # pre-construction detection.
        "id": "transformers_compile_config_type",
        "rule_under_test": (
            "GenerationConfig rejects compile_config that is not a CompileConfig instance"
        ),
        "match_fields": {
            "transformers.sampling.compile_config": {
                "present": True,
                "type_is_not": "CompileConfig",
            },
        },
        "kwargs_positive": {"compile_config": {"mode": "reduce-overhead"}},
        "kwargs_negative": {"compile_config": None},
        "message_template": (
            "`compile_config` must be a CompileConfig instance, got {declared_value}."
        ),
        "source_method": "validate",
    },
)

# --- BitsAndBytesConfig type-check rules (post_init) ---
# These are pure type-check raises: if not isinstance(value, T): raise TypeError(...).
# We keep them here rather than running bnb's post_init directly because bnb
# import triggers a CUDA context on GPU-bearing hosts. The walker stays CPU-safe.

_BNB_TYPE_RULES: tuple[tuple[str, str, Any, Any], ...] = (
    # (field, expected_type_label, positive_value, negative_value)
    # type_label matches `type(value).__name__` (strict class-name match
    # used by the loader's `type_is_not` predicate). `torch.dtype`
    # instances have `type(v).__name__ == "dtype"` — not "torch.dtype".
    ("load_in_4bit", "bool", "yes", False),
    ("load_in_8bit", "bool", 1, False),
    ("llm_int8_threshold", "float", "6.0", 6.0),
    ("llm_int8_skip_modules", "list", "head", ["head"]),
    ("llm_int8_enable_fp32_cpu_offload", "bool", "yes", False),
    ("llm_int8_has_fp16_weight", "bool", 0, False),
    ("bnb_4bit_compute_dtype", "dtype", "float16", None),
    ("bnb_4bit_quant_type", "str", 7, "nf4"),
    ("bnb_4bit_use_double_quant", "bool", 1, False),
)


# ---------------------------------------------------------------------------
# Walker entry points
# ---------------------------------------------------------------------------


def _check_landmarks() -> tuple[str, str]:
    """Import transformers, verify the landmarks we rely on exist.

    Returns ``(installed_version, generation_config_path)`` for the envelope.
    """
    try:
        import transformers  # type: ignore
    except ImportError as exc:
        raise WalkerLandmarkMissingError(
            "transformers.__init__", detail="transformers not importable"
        ) from exc

    try:
        from transformers import GenerationConfig  # type: ignore
    except ImportError as exc:
        raise WalkerLandmarkMissingError(
            "transformers.GenerationConfig", detail="symbol not importable"
        ) from exc

    if not hasattr(GenerationConfig, "validate"):
        raise WalkerLandmarkMissingError("GenerationConfig.validate")

    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as exc:
        raise WalkerLandmarkMissingError(
            "transformers.BitsAndBytesConfig", detail="symbol not importable"
        ) from exc

    if not hasattr(BitsAndBytesConfig, "post_init"):
        raise WalkerLandmarkMissingError("BitsAndBytesConfig.post_init")

    source_path = inspect.getsourcefile(GenerationConfig) or "<unknown>"
    return transformers.__version__, source_path


def _line_for_field(source_file: str, needle: str) -> int:
    """Best-effort line lookup for ``self.<needle>`` inside ``source_file``.

    The corpus entry's ``line_at_scan`` is informational only — not used for
    matching — so approximate lookup is acceptable. Returns 0 if not found.
    """
    try:
        text = Path(source_file).read_text()
    except OSError:
        return 0
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    return 0


def _relative_source_path(abs_path: str) -> str:
    """Strip host-specific prefixes so the corpus is reproducible across machines.

    ``/home/alice/.local/lib/python3.10/site-packages/transformers/...``
    → ``transformers/...`` — rooted at ``site-packages/``.
    """
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def _today() -> str:
    return dt.date.today().isoformat()


def _make_greedy_dormancy_rule(
    field: str,
    default: Any,
    positive: Any,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    predicate: dict[str, Any] = (
        {"present": True} if default is None else {"present": True, "not_equal": default}
    )
    line = _line_for_field(abs_source_path, f"self.{field}")
    return RuleCandidate(
        id=f"transformers_greedy_strips_{field}",
        engine="transformers",
        library="transformers",
        rule_under_test=(
            f"GenerationConfig.validate() records dormant `{field}` when "
            f"do_sample=False and `{field}` is set to a non-default value"
        ),
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields={
            "transformers.sampling.do_sample": False,
            f"transformers.sampling.{field}": predicate,
        },
        kwargs_positive={"do_sample": False, field: positive},
        kwargs_negative={"do_sample": True, field: positive},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "logger_warning_once",
            "normalised_fields": [],
        },
        message_template=(
            f"`do_sample=False` is set, so `{field}` ({{declared_value}}) "
            f"has no effect. Remove it or set do_sample=True."
        ),
        references=[f"transformers.GenerationConfig.validate() (line ~{line})"],
        added_by="introspection",
        added_at=today,
    )


def _make_beam_dormancy_rule(
    field: str,
    default: Any,
    positive: Any,
    abs_source_path: str,
    rel_source_path: str,
    today: str,
) -> RuleCandidate:
    predicate: dict[str, Any] = (
        {"present": True} if default is None else {"present": True, "not_equal": default}
    )
    line = _line_for_field(abs_source_path, f"self.{field}")
    return RuleCandidate(
        id=f"transformers_single_beam_strips_{field}",
        engine="transformers",
        library="transformers",
        rule_under_test=(
            f"GenerationConfig.validate() records dormant `{field}` when "
            f"num_beams=1 and `{field}` is set"
        ),
        severity="dormant",
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method="validate",
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields={
            "transformers.sampling.num_beams": 1,
            f"transformers.sampling.{field}": predicate,
        },
        kwargs_positive={"num_beams": 1, field: positive},
        kwargs_negative={"num_beams": 4, field: positive},
        expected_outcome={
            "outcome": "dormant_announced",
            "emission_channel": "logger_warning_once",
            "normalised_fields": [],
        },
        message_template=(
            f"`num_beams=1` is set, so `{field}` ({{declared_value}}) is ignored. "
            f"Set num_beams>1 or remove {field}."
        ),
        references=[f"transformers.GenerationConfig.validate() (line ~{line})"],
        added_by="introspection",
        added_at=today,
    )


def _make_validate_rule(
    spec: dict[str, Any], abs_source_path: str, rel_source_path: str, today: str
) -> RuleCandidate:
    """Factory for GenerationConfig.validate() rules (error OR dormant).

    Severity / outcome / emission_channel can be overridden in the spec dict
    for dormancy rules like ``pad_token_id < 0`` (HF writes to minor_issues,
    which users observe via logger.warning_once). Defaults match the common
    case: severity=error, outcome=error, emission_channel=none.
    """
    probe_field = next(iter(spec["match_fields"]))
    suffix = probe_field.rsplit(".", 1)[-1]
    line = _line_for_field(abs_source_path, f"self.{suffix}")
    severity = spec.get("severity", "error")
    outcome = spec.get("outcome", "error")
    emission_channel = spec.get("emission_channel", "none")
    return RuleCandidate(
        id=spec["id"],
        engine="transformers",
        library="transformers",
        rule_under_test=spec["rule_under_test"],
        severity=severity,
        native_type="transformers.GenerationConfig",
        walker_source=WalkerSource(
            path=rel_source_path,
            method=spec["source_method"],
            line_at_scan=line,
            walker_confidence="high",
        ),
        match_fields=spec["match_fields"],
        kwargs_positive=spec["kwargs_positive"],
        kwargs_negative=spec["kwargs_negative"],
        expected_outcome={
            "outcome": outcome,
            "emission_channel": emission_channel,
            "normalised_fields": [],
        },
        message_template=spec["message_template"],
        references=[
            "transformers.GenerationConfig.validate() — library-API introspection",
        ],
        added_by="introspection",
        added_at=today,
    )


def _make_bnb_type_rule(
    field: str,
    type_label: str,
    positive: Any,
    negative: Any,
    source_path: str,
    today: str,
) -> RuleCandidate:
    """Factory for BitsAndBytesConfig type-check rules.

    These rules surface BNB's `isinstance`-checking `post_init` TypeErrors
    before BNB is actually constructed (BNB import triggers a CUDA context
    on GPU hosts). Predicate uses `type_is_not` — fires only when the field
    is set AND has the wrong concrete type; a valid value (`True` for a
    bool field) does not match.

    Provenance is `manual_seed`: these rules are hand-curated from a
    one-off audit of the BitsAndBytesConfig source, not derived
    programmatically. Re-auditing on BNB library bumps is a maintainer
    task.
    """
    return RuleCandidate(
        id=f"transformers_bnb_{field}_type",
        engine="transformers",
        library="bitsandbytes",
        rule_under_test=(
            f"BitsAndBytesConfig.post_init() rejects non-{type_label} values for `{field}`"
        ),
        severity="error",
        native_type="transformers.BitsAndBytesConfig",
        walker_source=WalkerSource(
            path=source_path,
            method="post_init",
            line_at_scan=0,
            walker_confidence="high",
        ),
        match_fields={
            # `present + type_is_not`: fire only when set AND wrong type.
            # Bare `present: true` (the old predicate) fired on any set
            # value regardless of type — a user setting `load_in_4bit: true`
            # would wrongly trigger `severity=error`.
            f"transformers.quant.{field}": {
                "present": True,
                "type_is_not": type_label,
            },
        },
        kwargs_positive={field: positive},
        kwargs_negative={field: negative},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=(f"`{field}` must be a {type_label}, got {{declared_value}}."),
        references=[
            "transformers.utils.quantization_config.BitsAndBytesConfig.post_init() "
            "— manually audited type-check raises"
        ],
        added_by="manual_seed",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
    """Render a :class:`RuleCandidate` into the YAML corpus entry shape."""
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "rule_under_test": c.rule_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "walker_source": asdict(c.walker_source),
        "match": {
            "engine": c.engine,
            "fields": c.match_fields,
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


def walk(
    *, source_path_override: str | None = None, installed_version_override: str | None = None
) -> tuple[list[RuleCandidate], dict[str, Any]]:
    """Return ``(candidates, envelope_metadata)``.

    ``source_path_override`` and ``installed_version_override`` exist for
    tests: injecting known values lets the extractor unit tests skip the
    transformers import (which is heavy).
    """
    if installed_version_override is None:
        installed_version, abs_source_path = _check_landmarks()
        check_installed_version("transformers", installed_version, TESTED_AGAINST_VERSIONS)
    else:
        installed_version = installed_version_override
        abs_source_path = source_path_override or "<synthetic>"

    # Corpus paths are relative to site-packages so the committed YAML is
    # reproducible across checkouts with different ``~/.local`` roots.
    source_path = _relative_source_path(abs_source_path)

    today = _today()
    candidates: list[RuleCandidate] = []

    for field, default, pos in _GREEDY_RULES:
        candidates.append(
            _make_greedy_dormancy_rule(field, default, pos, abs_source_path, source_path, today)
        )
    for field, default, pos in _BEAM_RULES:
        candidates.append(
            _make_beam_dormancy_rule(field, default, pos, abs_source_path, source_path, today)
        )
    for spec in _GENERATION_VALIDATE_RULES:
        candidates.append(_make_validate_rule(spec, abs_source_path, source_path, today))

    # BitsAndBytesConfig rules — source path is the quantization_config module.
    # We cheaply locate it without importing bnb (which may touch CUDA).
    bnb_source_path = source_path
    try:
        import transformers.utils.quantization_config as _qcfg  # type: ignore

        abs_bnb = inspect.getsourcefile(_qcfg)
        if abs_bnb:
            bnb_source_path = _relative_source_path(abs_bnb)
    except Exception:  # pragma: no cover — import failures are landmark errors
        pass
    for field, type_label, pos, neg in _BNB_TYPE_RULES:
        candidates.append(_make_bnb_type_rule(field, type_label, pos, neg, bnb_source_path, today))

    # Allow tests / reproducibility checks to pin the timestamp.
    frozen = os.environ.get("LLENERGY_WALKER_FROZEN_AT")
    walked_at = frozen if frozen else dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": installed_version,
        "walker_pinned_range": str(TESTED_AGAINST_VERSIONS),
        "walked_at": walked_at,
    }
    return candidates, envelope


def emit_yaml(candidates: list[RuleCandidate], envelope: dict[str, Any]) -> str:
    """Serialise candidates + envelope to a deterministic YAML string.

    Key order is fixed (not alphabetical) for readability: envelope first,
    then candidates in source order. Within each candidate, keys follow the
    corpus schema's documented layout.
    """
    import yaml

    # Sort candidates deterministically for byte-stable output.
    sorted_candidates = sorted(candidates, key=lambda c: (c.walker_source.method, c.id))
    doc = {
        "schema_version": envelope["schema_version"],
        "engine": envelope["engine"],
        "engine_version": envelope["engine_version"],
        "walker_pinned_range": envelope["walker_pinned_range"],
        "walked_at": envelope["walked_at"],
        "rules": [_candidate_to_dict(c) for c in sorted_candidates],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write extracted YAML to this path.",
    )
    args = parser.parse_args(argv)

    candidates, envelope = walk()
    text = emit_yaml(candidates, envelope)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} transformers rules to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
