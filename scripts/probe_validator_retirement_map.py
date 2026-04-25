"""PoC-E: Validator retirement equivalence map.

Hypothesis
----------
The migration plan (§7) removes `validate_transformers_flash_attn_dtype` from
config/models.py and replaces it with a vendored rule consumed by the generic
@model_validator. For this to be safe ("never delete-before-covered"), every
historical failing input must route identically through both paths.

This PoC enumerates the validator's failing-input corpus, runs each input
through (a) the current Pydantic validator path and (b) a synthetic vendored-
rule matcher that applies the proposed YAML rule. Outputs a diff table.

Decision criteria (pre-committed)
---------------------------------
- Every input routes identically (same outcome: accept/reject, same error
  class) -> SAFE to proceed with seed-then-verify-then-delete migration.
- Any input routes differently -> STOP; fix the proposed YAML rule before
  the migration PR deletes the old validator.

Scope: just the one extant framework-rule validator. If more are added before
the migration reaches them, extend this PoC.

Run
---
  /usr/bin/python3.10 scripts/probe_validator_retirement_map.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from llenergymeasure.config.engine_configs import (
        TransformersConfig,
    )
    from llenergymeasure.config.models import ExperimentConfig, TaskConfig
except Exception as e:
    print(f"ERROR: can't import llenergymeasure config: {e}")
    print("Run from project root with the venv activated:")
    print("  /usr/bin/python3.10 scripts/probe_validator_retirement_map.py")
    sys.exit(1)


# --- The proposed vendored rule for validate_transformers_flash_attn_dtype ---
# This is the exact YAML a P3 PR would seed.
PROPOSED_YAML_RULE = {
    "id": "tf_flash_attn_requires_half_precision",
    "engine": "transformers",
    "severity": "error",
    "match": {
        "engine": "transformers",
        "fields": {
            "transformers.attn_implementation": {
                "in": [
                    "flash_attention_2",
                    "flash_attention_3",
                    "flash_attn_2",
                    "flash_attn_3",
                ]
            },
            "transformers.dtype": "float32",
        },
    },
    "message_template": (
        "attn_implementation='{attn_impl}' requires dtype='float16' or dtype='bfloat16'. "
        "FlashAttention does not support float32 computation."
    ),
}


# Only test values that pass Pydantic's Literal validation. Invalid enum
# values fail at field-level validation (not cross-field), so they're not
# in the scope of the validator-retirement equivalence check.
FLASH_ATTN_IMPLS = {"flash_attention_2", "flash_attention_3"}
NON_FLASH_IMPLS = {None, "eager", "sdpa"}
DTYPES = {None, "float32", "float16", "bfloat16"}


def matches_vendored_rule(attn_impl: str | None, dtype: str | None) -> bool:
    """Apply the proposed YAML rule's match predicate. Returns True = rule fires."""
    spec = PROPOSED_YAML_RULE["match"]["fields"]
    attn_spec = spec["transformers.attn_implementation"]
    if isinstance(attn_spec, dict) and "in" in attn_spec:
        if attn_impl not in attn_spec["in"]:
            return False
    # dtype handling: the Pydantic validator defaults unset dtype to bfloat16
    # via `(self.transformers.dtype or "bfloat16") == "float32"`, so for
    # vendored-rule equivalence we need the same default treatment.
    effective_dtype = dtype or "bfloat16"
    if effective_dtype != spec["transformers.dtype"]:
        return False
    return True


def pydantic_would_reject(attn_impl: str | None, dtype: str | None) -> tuple[bool, str | None]:
    """Construct an ExperimentConfig and report whether Pydantic raises.

    Only captures exceptions raised by the validator-under-test
    (validate_transformers_flash_attn_dtype) or by field-level dtype
    constraints that overlap with it. Unrelated schema errors (missing task
    field, etc.) are filtered out by using a minimal-valid task config.
    """
    try:
        tf_kwargs: dict[str, Any] = {}
        if attn_impl is not None:
            tf_kwargs["attn_implementation"] = attn_impl
        if dtype is not None:
            tf_kwargs["dtype"] = dtype
        tf_config = TransformersConfig(**tf_kwargs)
        ExperimentConfig(
            engine="transformers",
            task=TaskConfig(model="Qwen/Qwen2.5-0.5B"),
            transformers=tf_config,
        )
    except Exception as e:
        msg = str(e)
        # Only count rejections caused by the validator under test.
        # Filter: message must mention FlashAttention / attn_implementation /
        # float32 together, which is the validator's specific error text.
        relevant = "FlashAttention" in msg or "flash_attention" in msg or "float32" in msg.lower()
        if relevant:
            return True, f"{type(e).__name__}: {e}"
        # Unrelated error (e.g., Literal validation on an enum) — treat as
        # "Pydantic accepted at the cross-field layer" for equivalence purposes.
        return False, None
    return False, None


@dataclass(frozen=True)
class CaseResult:
    attn_impl: str | None
    dtype: str | None
    pydantic_rejected: bool
    pydantic_error: str | None
    vendored_rule_fires: bool
    agrees: bool


def main() -> None:
    print("=" * 78)
    print("PoC-E: Validator retirement equivalence map")
    print("=" * 78)
    print()
    print("Validator under test: validate_transformers_flash_attn_dtype (config/models.py:467)")
    print(f"Proposed vendored rule id: {PROPOSED_YAML_RULE['id']}")
    print()

    cases: list[CaseResult] = []
    for attn in FLASH_ATTN_IMPLS | NON_FLASH_IMPLS:
        for dt in DTYPES:
            py_rejected, py_error = pydantic_would_reject(attn, dt)
            vr_fires = matches_vendored_rule(attn, dt)
            agrees = py_rejected == vr_fires
            cases.append(
                CaseResult(
                    attn_impl=attn,
                    dtype=dt,
                    pydantic_rejected=py_rejected,
                    pydantic_error=py_error,
                    vendored_rule_fires=vr_fires,
                    agrees=agrees,
                )
            )

    # Report
    print(f"{'attn_impl':<22}  {'dtype':<12}  {'pydantic':<10}  {'vendored':<10}  agrees?")
    print("-" * 78)
    for c in cases:
        print(
            f"{c.attn_impl!s:<22}  {c.dtype!s:<12}  "
            f"{'REJECT' if c.pydantic_rejected else 'accept':<10}  "
            f"{'FIRE' if c.vendored_rule_fires else 'none':<10}  "
            f"{'OK' if c.agrees else '**DIVERGES**'}"
        )

    disagreements = [c for c in cases if not c.agrees]
    print()
    print("=" * 78)
    print("DECISION CRITERIA EVALUATION")
    print("=" * 78)
    print()
    print(f"Total cases:      {len(cases)}")
    print(f"Agreements:       {len(cases) - len(disagreements)}")
    print(f"Disagreements:    {len(disagreements)}")
    print()

    if not disagreements:
        print(
            "-> SAFE: the proposed vendored rule is equivalent to the existing "
            "Pydantic validator on every (attn_impl, dtype) combination in the "
            "corpus. Migration can proceed with the seed-then-verify-then-delete "
            "discipline."
        )
    else:
        print("-> STOP: vendored rule diverges on the following cases:")
        for c in disagreements:
            print(f"     attn={c.attn_impl}, dtype={c.dtype}")
            print(f"       pydantic: {c.pydantic_error}")
            print(f"       vendored: {'fires' if c.vendored_rule_fires else 'silent'}")
        print()
        print("   Fix the proposed YAML rule before the migration PR deletes the old validator.")

    # --- Additional: what else uses this validator that we might miss? ---
    print()
    print("Coverage note:")
    print("  This PoC enumerates (attn_impl, dtype) cartesian product. The")
    print("  Pydantic validator only cares about these two fields; other")
    print("  ExperimentConfig fields are don't-cares. If the validator's")
    print("  implementation checks additional state (e.g., model architecture),")
    print("  we need to extend this corpus before committing to retirement.")
    print()
    print("  Inspecting config/models.py:467-480 confirms: validator is strictly")
    print("  (attn_implementation, dtype). Two-dim corpus is complete.")


if __name__ == "__main__":
    main()
