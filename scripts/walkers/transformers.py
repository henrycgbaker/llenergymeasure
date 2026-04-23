"""Transformers validation-rules walker — landmark-verified orchestrator.

Composes two extraction paths to emit a deterministic rules corpus:

1. **GenerationConfig rules via library-API introspection** — delegated to
   :mod:`scripts.walkers.transformers_introspection`. Every dormancy rule is
   discovered by probing ``GenerationConfig.validate(strict=True)`` against a
   synthesised per-type probe value; every error-class rule's message is
   lifted from the library's own ``ValueError``. Rules emitted:
   ``added_by: introspection``.

2. **BitsAndBytesConfig type-check rules** — hand-curated here. BNB import
   triggers a CUDA context on GPU-bearing hosts, which would make the walker
   unsafe to run CPU-only in CI. Keeping BNB rules as landmark-verified
   manual seed preserves CPU-safety; the justification is the CUDA-context
   risk, not editorial preference. Rules emitted: ``added_by: manual_seed``.

Landmark verification: imports ``transformers`` and confirms
``GenerationConfig``, ``BitsAndBytesConfig``, ``validate`` and ``post_init``
exist. Missing landmark → :class:`WalkerLandmarkMissingError`. Version
envelope: :data:`TESTED_AGAINST_VERSIONS` pins the walker to a known range;
a mismatch raises :class:`WalkerVersionMismatchError` at CI time. Source
paths and line numbers are derived via :func:`inspect.getsourcefile` and a
text grep — informational only, not used for rule matching.

Flash-attention validation (``validate_transformers_flash_attn_dtype``) is
out of scope: its check lives in ``PreTrainedModel._autoset_attn_implementation``,
not ``GenerationConfig.validate`` or ``BitsAndBytesConfig.post_init``, and is
not reachable via this walker's landmarks. Stays hand-written in
``config/models.py`` pending a PreTrainedModel-level walker.

Usage::

    python -m scripts.walkers.transformers --out configs/validation_rules/transformers.yaml

With ``LLENERGY_WALKER_FROZEN_AT`` set for byte-stable reproducibility.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import os
import sys
from dataclasses import asdict
from functools import cache
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
from scripts.walkers.transformers_introspection import (  # noqa: E402
    walk_generation_config_rules,
)

TESTED_AGAINST_VERSIONS: SpecifierSet = SpecifierSet(">=4.49,<4.57")
"""Range of transformers versions this walker has been validated against.

Lower bound tracks the project's own ``transformers>=4.49`` pin in
``pyproject.toml`` so CI's pinned version stays in range. Upper bound
excludes 4.57 because HF 4.57 restructured ``GenerationConfig.validate()`` —
dropped several ``minor_issues`` branches (``num_beam_groups``,
``diversity_penalty``, ``constraints`` single-beam dormancies) and the
watermarking auto-coercion. The introspection path inherits this pin:
on 4.57+, the auto-enumerator would surface a different rule set, and the
diff-rules CI comment would flag the drift for maintainer reconciliation.

On mismatch, :func:`check_installed_version` raises
:class:`WalkerVersionMismatchError` and CI fails.
"""


# ---------------------------------------------------------------------------
# BitsAndBytesConfig type-check rules (kept hand-curated — CPU-safe)
# ---------------------------------------------------------------------------
#
# These rules surface BNB's ``isinstance``-checking ``post_init`` TypeErrors
# before BNB is actually constructed. BNB's ``import`` triggers a CUDA
# context on GPU hosts; the walker stays CPU-safe by not importing it.
# Predicate uses ``type_is_not`` — fires only when the field is set AND has
# the wrong concrete type; a valid value (``True`` for a bool field) does
# not match.
#
# Provenance: ``manual_seed``. Re-auditing on BNB library bumps is a
# maintainer task — until a BNB-side introspection path is wired up (e.g.
# inside a CUDA-bearing container at CI time), the partition stays as-is.

_BNB_TYPE_RULES: tuple[tuple[str, str, Any, Any], ...] = (
    # (field, expected_type_label, positive_value, negative_value)
    # type_label matches ``type(value).__name__`` (strict class-name match
    # used by the loader's ``type_is_not`` predicate). ``torch.dtype``
    # instances have ``type(v).__name__ == "dtype"`` — not "torch.dtype".
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
# Landmark + source-path utilities
# ---------------------------------------------------------------------------


def _check_landmarks() -> tuple[str, str]:
    """Import transformers, verify the landmarks the walker relies on.

    Returns ``(installed_version, generation_config_source_path)``.
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


@cache
def _read_source_lines(source_file: str) -> tuple[str, ...]:
    """Cache parsed source-file lines. BNB path-line lookups hit this cache."""
    try:
        return tuple(Path(source_file).read_text().splitlines())
    except OSError:
        return ()


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


# ---------------------------------------------------------------------------
# BNB rule factory
# ---------------------------------------------------------------------------


def _make_bnb_type_rule(
    field: str,
    type_label: str,
    positive: Any,
    negative: Any,
    source_path: str,
    today: str,
) -> RuleCandidate:
    """Factory for ``BitsAndBytesConfig`` type-check rules."""
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


def walk() -> tuple[list[RuleCandidate], dict[str, Any]]:
    """Return ``(candidates, envelope_metadata)`` — full corpus for this engine.

    Composes the introspection-derived GenerationConfig rules with the
    hand-curated BNB rules. Rules in the returned list carry their own
    ``added_by`` tag; the envelope captures the shared version pin.
    """
    installed_version, abs_source_path = _check_landmarks()
    check_installed_version("transformers", installed_version, TESTED_AGAINST_VERSIONS)

    # Corpus paths are relative to site-packages so the committed YAML is
    # reproducible across checkouts with different ``~/.local`` roots.
    source_path = _relative_source_path(abs_source_path)

    today = _today()
    candidates: list[RuleCandidate] = []

    candidates.extend(
        walk_generation_config_rules(
            abs_source_path=abs_source_path,
            rel_source_path=source_path,
            today=today,
        )
    )

    # BitsAndBytesConfig rules — source path is the quantization_config module.
    # Cheaply locate it without importing bnb (which may touch CUDA).
    bnb_source_path = source_path
    try:
        import transformers.utils.quantization_config as _qcfg  # type: ignore
    except ImportError:  # pragma: no cover — landmark missing, handled upstream
        pass
    else:
        abs_bnb = inspect.getsourcefile(_qcfg)
        if abs_bnb:
            bnb_source_path = _relative_source_path(abs_bnb)
    for field, type_label, pos, neg in _BNB_TYPE_RULES:
        candidates.append(_make_bnb_type_rule(field, type_label, pos, neg, bnb_source_path, today))

    # Allow tests / reproducibility checks to pin the timestamp.
    frozen = os.environ.get("LLENERGY_WALKER_FROZEN_AT")
    walked_at = (
        frozen
        if frozen
        else (dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    )
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
    then candidates sorted by ``(walker_source.method, id)``.
    """
    import yaml

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
