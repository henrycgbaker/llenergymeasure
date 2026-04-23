#!/usr/bin/env python3
"""Run each validation rule through the real library inside its engine container.

The vendor step is the **observe half** of the "observe, don't re-encode"
design in :doc:`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`.
The YAML corpus at ``configs/validation_rules/{engine}.yaml`` declares each
rule's ``expected_outcome``; this script executes the rule through the
library and records what *actually* happened. Divergence between declared and
observed fails CI.

Usage (inside the engine's Docker container)::

    python scripts/vendor_rules.py \\
        --engine transformers \\
        --corpus configs/validation_rules/transformers.yaml \\
        --out src/llenergymeasure/engines/vendored_rules/transformers.json

Exit codes:

    0 - all rules confirmed (positive + negative + expected matches observed)
    1 - one or more divergences; JSON still written
    2 - hard error (corpus malformed, engine not importable, etc.)

The envelope structure mirrors the parameter-discovery envelope in
``src/llenergymeasure/config/discovered_schemas/*.json`` so downstream tooling
and loaders share the same shape. Sibling by design, per §5 of the design doc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure sibling module imports resolve when run via ``python scripts/...``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._vendor_common import (  # noqa: E402  (late import after sys.path)
    TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
    CaptureBuffers,
    CaseResult,
    Divergence,
    classify_emission_channel,
    classify_outcome,
    compare_expected_vs_observed,
    diff_input_vs_state,
    run_case,
    strip_warning_once_sentinel,
)

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class VendorError(Exception):
    """Base class for vendor-step errors."""


class VendorCorpusError(VendorError):
    """Corpus YAML is malformed or missing required fields."""


class VendorEngineNotImportable(VendorError):
    """The engine library is not importable in this environment."""


class VendorDivergenceError(VendorError):
    """One or more rules diverged from their declared expected_outcome."""

    def __init__(self, divergences: list[Divergence]) -> None:
        super().__init__(
            f"{len(divergences)} rule(s) diverged from expected_outcome. "
            "See the vendored JSON 'divergences' array for details."
        )
        self.divergences = divergences


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus(path: Path) -> dict[str, Any]:
    try:
        raw_text = path.read_text()
    except FileNotFoundError as exc:
        raise VendorCorpusError(f"Corpus not found at {path}") from exc
    try:
        import yaml
    except ImportError as exc:
        raise VendorCorpusError("PyYAML not available; cannot parse corpus") from exc
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict) or "rules" not in data:
        raise VendorCorpusError(f"Corpus at {path} must be a mapping with a top-level 'rules' key.")
    return data


# ---------------------------------------------------------------------------
# Per-engine native-type runners
# ---------------------------------------------------------------------------


def _run_transformers(
    native_type: str, kwargs: dict[str, Any], *, strict_validate: bool
) -> CaptureBuffers:
    """Execute one rule's kwargs through the transformers library.

    Handles both ``GenerationConfig`` (uses ``.validate()``) and
    ``BitsAndBytesConfig`` (construction itself raises). Other
    ``transformers.*`` native types are reached via a fallback import.

    ``strict_validate`` routes the GenerationConfig call: ``True`` raises a
    composed ValueError listing every issue (corresponds to corpus
    ``severity=error``); ``False`` emits dormant/announced issues via
    ``logger.warning_once`` (corresponds to corpus ``severity=dormant``). The
    caller picks the mode based on the rule's declared severity.
    """
    logger_names = (
        "transformers",
        "transformers.generation",
        "transformers.generation.configuration_utils",
    )
    if native_type == "transformers.GenerationConfig":
        return run_case(
            lambda: _construct_and_validate_generation_config(kwargs, strict=strict_validate),
            logger_names=logger_names,
            private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
        )
    if native_type == "transformers.BitsAndBytesConfig":
        return run_case(
            lambda: _construct_bitsandbytes_config(kwargs),
            logger_names=logger_names,
            private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
        )
    # Fallback: treat native_type as a dotted import path.
    return run_case(
        lambda: _construct_generic(native_type, kwargs),
        logger_names=logger_names,
        private_allowlist=TRANSFORMERS_PRIVATE_FIELD_ALLOWLIST,
    )


def _construct_and_validate_generation_config(kwargs: dict[str, Any], *, strict: bool) -> Any:
    # Corpus kwargs pass through verbatim — the raw YAML shape IS the rule
    # under test (e.g. compile_config receives a raw dict on purpose).
    from transformers import GenerationConfig  # type: ignore

    gc = GenerationConfig(**kwargs)
    gc.validate(strict=strict)
    return gc


def _construct_bitsandbytes_config(kwargs: dict[str, Any]) -> Any:
    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(**kwargs)


def _construct_generic(native_type: str, kwargs: dict[str, Any]) -> Any:
    module_path, _, class_name = native_type.rpartition(".")
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**kwargs)


_ENGINE_RUNNERS = {
    "transformers": _run_transformers,
}


def get_native_type_runner(engine: str):
    """Return the per-engine dispatcher. Raises if engine unsupported."""
    runner = _ENGINE_RUNNERS.get(engine)
    if runner is None:
        raise VendorError(
            f"No vendor runner registered for engine {engine!r}. "
            f"Known engines: {sorted(_ENGINE_RUNNERS)}"
        )
    return runner


# ---------------------------------------------------------------------------
# Per-rule driver
# ---------------------------------------------------------------------------


def vendor_rule(engine: str, rule: dict[str, Any], *, gpu_mode: str) -> CaseResult:
    """Run one rule's positive + negative kwargs and assemble the case result.

    ``gpu_mode`` is ``"all" | "skip" | "only"`` — hardware-dependent rules
    are skipped unless ``gpu_mode`` permits them.
    """
    rule_id = rule["id"]
    requires_gpu = bool(rule.get("requires_gpu", False))
    hardware_dependent = bool(rule.get("hardware_dependent", False))

    if gpu_mode == "skip" and (requires_gpu or hardware_dependent):
        return CaseResult(
            id=rule_id,
            outcome="skipped_hardware_dependent",
            emission_channel="none",
            skipped_reason="requires_gpu_and_gpu_mode_skip",
        )
    if gpu_mode == "only" and not requires_gpu:
        return CaseResult(
            id=rule_id,
            outcome="skipped_hardware_dependent",
            emission_channel="none",
            skipped_reason="cpu_rule_and_gpu_mode_only",
        )

    native_type = rule["native_type"]
    runner = get_native_type_runner(engine)
    severity = str(rule.get("severity", "")).lower()
    # Per-engine strictness routing: transformers' GenerationConfig has a
    # non-strict path (logger.warning for dormant/announced) and a strict
    # path (composed ValueError for errors). Dispatch by declared severity
    # so the vendor observation matches the corpus's expected outcome shape.
    strict_validate = severity == "error"

    kwargs_positive = dict(rule["kwargs_positive"])
    kwargs_negative = dict(rule["kwargs_negative"])

    pos = runner(native_type, kwargs_positive, strict_validate=strict_validate)
    neg = runner(native_type, kwargs_negative, strict_validate=strict_validate)

    # Silent self-assignments are only meaningful on the positive path and
    # only when construction succeeded.
    silent_normalisations: dict[str, dict[str, Any]] = {}
    if pos.observed_state is not None:
        silent_normalisations = diff_input_vs_state(kwargs_positive, pos.observed_state)

    outcome = classify_outcome(pos, silent_normalisations)
    emission = classify_emission_channel(pos)

    expected = dict(rule.get("expected_outcome") or {})
    positive_confirmed = _positive_confirms(expected, outcome)
    neg_silent = (
        diff_input_vs_state(kwargs_negative, neg.observed_state) if neg.observed_state else {}
    )
    negative_confirmed = _negative_confirms(neg, neg_silent)

    observed_messages = list(pos.warnings_captured) + list(
        strip_warning_once_sentinel(pos.logger_messages)
    )
    observed_exception: dict[str, str] | None = None
    if pos.exception_type is not None:
        observed_exception = {
            "type": pos.exception_type,
            "message": pos.exception_message or "",
        }

    return CaseResult(
        id=rule_id,
        outcome=outcome,
        emission_channel=emission,
        observed_messages=observed_messages,
        observed_silent_normalisations=silent_normalisations,
        observed_exception=observed_exception,
        positive_confirmed=positive_confirmed,
        negative_confirmed=negative_confirmed,
        duration_ms=pos.duration_ms + neg.duration_ms,
    )


_FIRING_OUTCOMES = frozenset({"error", "warn", "dormant_announced", "dormant_silent"})


def _positive_confirms(expected: dict[str, Any], observed_outcome: str) -> bool:
    """True iff the rule fired on the positive kwargs as declared.

    When the corpus declares a specific outcome, positive_confirmed requires
    an exact match. When the corpus leaves ``outcome`` unset, we accept any
    non-``no_op`` observation as confirmation.
    """
    expected_outcome = expected.get("outcome")
    if expected_outcome in _FIRING_OUTCOMES:
        return observed_outcome == expected_outcome
    return observed_outcome != "no_op"


def _negative_confirms(neg: CaptureBuffers, silent_normalisations: dict[str, Any]) -> bool:
    """True iff the rule did NOT fire on the negative kwargs.

    Strict definition — any of exception / warn / logger / silent
    normalisation counts as firing. Catches dead walker entries where
    ``kwargs_negative`` still happens to trip the rule.
    """
    return (
        neg.exception_type is None
        and not neg.warnings_captured
        and not neg.logger_messages
        and not silent_normalisations
    )


# ---------------------------------------------------------------------------
# Envelope assembly
# ---------------------------------------------------------------------------


def assemble_envelope(
    *,
    engine: str,
    engine_version: str,
    image_ref: str,
    base_image_ref: str,
    vendor_commit: str,
    cases: list[CaseResult],
    divergences: list[Divergence],
) -> dict[str, Any]:
    """Build the vendored-rules envelope (parallel to the parameter-discovery envelope)."""
    now = os.environ.get("LLENERGY_VENDOR_FROZEN_AT") or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": engine_version,
        "image_ref": image_ref,
        "base_image_ref": base_image_ref,
        "vendored_at": now,
        "vendor_commit": vendor_commit,
        "cases": [_case_to_dict(c) for c in cases],
        "divergences": [d.as_dict() for d in divergences],
    }


def _case_to_dict(case: CaseResult) -> dict[str, Any]:
    d = asdict(case)
    # Drop nullable optional fields when unset for a quieter envelope.
    for optional_key in ("observed_exception", "skipped_reason"):
        if d.get(optional_key) is None:
            d.pop(optional_key, None)
    return d


# ---------------------------------------------------------------------------
# Main vendor loop
# ---------------------------------------------------------------------------


def vendor_engine(
    *,
    engine: str,
    corpus_path: Path,
    out_path: Path,
    gpu_mode: str = "all",
    image_ref: str | None = None,
    base_image_ref: str | None = None,
    vendor_commit: str = "unknown",
) -> tuple[dict[str, Any], list[Divergence]]:
    """Run the full vendor loop for one engine; write JSON envelope to ``out_path``.

    Returns ``(envelope, divergences)``. Raises :class:`VendorEngineNotImportable`
    if the engine library can't be imported. Does NOT raise on divergence —
    the caller inspects the returned list and decides.
    """
    corpus = _load_corpus(corpus_path)
    engine_version = _resolve_engine_version(engine)

    cases: list[CaseResult] = []
    divergences: list[Divergence] = []

    for rule in corpus.get("rules", []):
        # VendorError (and subclasses) propagate — they indicate misconfig, not
        # a library behaviour finding. Any other Exception gets recorded as a
        # per-rule error so one bad rule doesn't abort the full vendor run.
        try:
            case = vendor_rule(engine, rule, gpu_mode=gpu_mode)
        except VendorError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            case = CaseResult(
                id=rule.get("id", "<unknown>"),
                outcome="error",
                emission_channel="none",
                observed_exception={"type": type(exc).__name__, "message": str(exc)},
            )
        cases.append(case)

        if case.skipped_reason is not None:
            continue

        rule_divergences = compare_expected_vs_observed(
            rule_id=rule["id"],
            expected=rule.get("expected_outcome") or {},
            observed_outcome=case.outcome,
            observed_emission=case.emission_channel,
            silent_normalisations=case.observed_silent_normalisations,
        )
        if not case.positive_confirmed:
            rule_divergences.append(
                Divergence(
                    rule_id=rule["id"],
                    field="positive_confirmed",
                    expected=True,
                    observed=False,
                )
            )
        if not case.negative_confirmed:
            rule_divergences.append(
                Divergence(
                    rule_id=rule["id"],
                    field="negative_confirmed",
                    expected=True,
                    observed=False,
                )
            )
        divergences.extend(rule_divergences)

    envelope = assemble_envelope(
        engine=engine,
        engine_version=engine_version,
        image_ref=image_ref or f"llenergymeasure:{engine}",
        base_image_ref=base_image_ref or f"llenergymeasure:{engine}",
        vendor_commit=vendor_commit,
        cases=cases,
        divergences=divergences,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(envelope, indent=2, sort_keys=False) + "\n")

    return envelope, divergences


def _resolve_engine_version(engine: str) -> str:
    """Best-effort: return the installed library's version or ``"unknown"``."""
    if engine == "transformers":
        try:
            import transformers  # type: ignore

            return transformers.__version__
        except ImportError as exc:
            raise VendorEngineNotImportable(
                "transformers is not importable in this environment"
            ) from exc
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        required=True,
        choices=sorted(_ENGINE_RUNNERS),
        help="Engine whose corpus to vendor.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to the YAML corpus file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the vendored JSON envelope.",
    )
    parser.add_argument(
        "--gpu-cases",
        choices=("all", "skip", "only"),
        default="all",
        help=(
            "Which rules to run. 'skip' drops rules with requires_gpu=true "
            "(for GH-hosted CPU jobs); 'only' runs only those (for self-hosted "
            "GPU jobs); 'all' runs everything (default, useful locally)."
        ),
    )
    parser.add_argument(
        "--image-ref",
        default=None,
        help="Image reference to record in envelope.image_ref.",
    )
    parser.add_argument(
        "--base-image-ref",
        default=None,
        help="Base image reference to record in envelope.base_image_ref.",
    )
    parser.add_argument(
        "--vendor-commit",
        default=os.environ.get("GITHUB_SHA", "unknown"),
        help=(
            "Git commit SHA under which this vendor run occurred. Defaults to "
            "$GITHUB_SHA when CI runs set it; otherwise 'unknown'."
        ),
    )
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help=(
            "Exit 1 if any rule diverged from its expected_outcome. CI always "
            "passes this flag; locally it's off by default so developers can "
            "inspect the JSON without CI-style exit."
        ),
    )

    args = parser.parse_args(argv)

    try:
        _envelope, divergences = vendor_engine(
            engine=args.engine,
            corpus_path=args.corpus,
            out_path=args.out,
            gpu_mode=args.gpu_cases,
            image_ref=args.image_ref,
            base_image_ref=args.base_image_ref,
            vendor_commit=args.vendor_commit,
        )
    except VendorCorpusError as exc:
        print(f"[{args.engine}] corpus error: {exc}", file=sys.stderr)
        return 2
    except VendorEngineNotImportable as exc:
        print(f"[{args.engine}] engine not importable: {exc}", file=sys.stderr)
        return 2
    except VendorError as exc:
        print(f"[{args.engine}] vendor error: {exc}", file=sys.stderr)
        return 2

    print(f"[{args.engine}] wrote {args.out}", file=sys.stderr)
    if divergences:
        print(
            f"[{args.engine}] {len(divergences)} divergence(s) — see JSON 'divergences' array.",
            file=sys.stderr,
        )
        for d in divergences[:10]:
            print(
                f"  - {d.rule_id}: {d.field} expected={d.expected!r} observed={d.observed!r}",
                file=sys.stderr,
            )
        if len(divergences) > 10:
            print(f"  ... and {len(divergences) - 10} more.", file=sys.stderr)
        if args.fail_on_divergence:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
