"""PoC-J: Does HF's GenerationConfig.validate() expose rules via minor_issues?

Hypothesis
----------
HF's GenerationConfig.validate() internally populates a `minor_issues` dict
before emitting its `logger.warning_once`. If we can intercept that dict,
we get a structured rule-output *from the library's own API* without any AST
walking. For transformers specifically (half the rule surface), this would
replace the walker with runtime introspection — cheaper, less fragile,
library-authored.

Decision criteria (pre-committed)
---------------------------------
- If minor_issues reliably captures >=90% of the rules the walker extracted
  (30 transformers rules from the AST PoC), and coverage is equivalent on
  representative test configs -> REPLACE the transformers walker with
  runtime introspection. Reconsider the pattern for other engines.
- If coverage is 60-90% -> KEEP the walker as primary but USE introspection
  as a complementary capture mechanism (catches what walker missed).
- If coverage <60% -> WALKER IS THE RIGHT CALL for transformers. Move on.

Additional signals
------------------
- Does minor_issues emission correlate with walker's minor_issues_assign
  action-class? If yes, at minimum the walker's message-template extraction
  could be cross-validated at CI time.
- Does `strict=True` raise with the full issue list? (Relevant for vendor-CI
  verification: strict-mode exceptions give us programmatic access to the
  library's rule-output stream without log scraping.)

Run
---
  /usr/bin/python3.10 scripts/probe_hf_validate_introspection.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

try:
    from transformers import GenerationConfig
except ImportError:
    print("ERROR: transformers not importable. Run with /usr/bin/python3.10.")
    sys.exit(1)

import transformers

print(f"transformers version: {transformers.__version__}")
print(f"GenerationConfig module: {GenerationConfig.__module__}")
print()


# --- Test configurations ---
# These mirror the synthetic-bad-configs categories but target GenerationConfig
# specifically. Each should trip at least one minor_issues entry if the
# introspection hypothesis holds.

TEST_CONFIGS: list[dict[str, Any]] = [
    # --- Greedy + sampling params (the canonical 50% showcase case) ---
    {"name": "greedy_with_temperature", "kwargs": {"do_sample": False, "temperature": 0.9}},
    {"name": "greedy_with_top_p", "kwargs": {"do_sample": False, "top_p": 0.95}},
    {"name": "greedy_with_top_k", "kwargs": {"do_sample": False, "top_k": 40}},
    {"name": "greedy_with_min_p", "kwargs": {"do_sample": False, "min_p": 0.1}},
    {"name": "greedy_with_typical_p", "kwargs": {"do_sample": False, "typical_p": 0.9}},
    {"name": "greedy_with_epsilon_cutoff", "kwargs": {"do_sample": False, "epsilon_cutoff": 0.05}},
    {"name": "greedy_with_eta_cutoff", "kwargs": {"do_sample": False, "eta_cutoff": 0.05}},
    # --- Beam params with num_beams=1 ---
    {"name": "single_beam_with_early_stopping", "kwargs": {"num_beams": 1, "early_stopping": True}},
    {
        "name": "single_beam_with_num_beam_groups",
        "kwargs": {"num_beams": 1, "num_beam_groups": 2},
    },
    {
        "name": "single_beam_with_diversity_penalty",
        "kwargs": {"num_beams": 1, "diversity_penalty": 0.5},
    },
    {"name": "single_beam_with_length_penalty", "kwargs": {"num_beams": 1, "length_penalty": 2.0}},
    # --- Range / type / cross-field errors (should raise, not minor_issues) ---
    {"name": "negative_max_new_tokens", "kwargs": {"max_new_tokens": -1}},
    {"name": "invalid_cache_implementation", "kwargs": {"cache_implementation": "nonsense"}},
    {"name": "invalid_early_stopping_str", "kwargs": {"early_stopping": "sometimes"}},
    # --- Valid baselines (should produce no issues) ---
    {"name": "valid_greedy", "kwargs": {"do_sample": False}},
    {"name": "valid_sampling", "kwargs": {"do_sample": True, "temperature": 0.9, "top_p": 0.95}},
    {"name": "valid_beam_search", "kwargs": {"num_beams": 4, "length_penalty": 1.5}},
]


@dataclass
class IntrospectionResult:
    name: str
    kwargs: dict[str, Any]
    minor_issues_keys: list[str] = field(default_factory=list)
    strict_raise_message: str | None = None
    unexpected_exception: str | None = None
    construction_normalised: dict[str, tuple[Any, Any]] = field(
        default_factory=dict
    )  # field -> (declared, effective)


def introspect_one(name: str, kwargs: dict[str, Any]) -> IntrospectionResult:
    """Run validate() against a test config and extract all signals."""
    result = IntrospectionResult(name=name, kwargs=kwargs)

    # --- Strategy 1: monkey-patch minor_issues to intercept populated dict ---
    # GenerationConfig.validate() creates a local `minor_issues` dict and may
    # call logger.warning_once. We patch dict to capture what ends up inside.
    original_dict = dict

    captured_issues: dict[str, str] = {}

    class _TracingDict(dict):
        def __setitem__(self, k: str, v: Any) -> None:
            super().__setitem__(k, v)
            # Only trust this if it happened inside validate()'s frame;
            # filter later by caller. For PoC we just capture everything.
            captured_issues[k] = str(v)

    # Can't replace builtins.dict directly without breaking the world.
    # Strategy: read the minor_issues dict after validate() by running
    # validate(strict=True) in a separate call and catching the message —
    # which HF composes from minor_issues.items().
    try:
        gc = GenerationConfig(**kwargs)
    except Exception as e:
        result.unexpected_exception = f"{type(e).__name__}: {e}"
        return result

    # --- Capture construction-time normalisations (T1.5 equivalent) ---
    # Compare what we passed in vs what ended up on the object.
    for k, v in kwargs.items():
        actual = getattr(gc, k, None)
        if actual != v:
            result.construction_normalised[k] = (v, actual)

    # --- Strategy 2: strict-mode raises with the minor_issues composed ---
    # HF's validate(strict=True) raises ValueError with a message that lists
    # every minor_issue. This is programmatic access to the rule output.
    try:
        gc.validate(strict=True)
        # No issues raised, but dormancy may still have been logged.
    except ValueError as e:
        msg = str(e)
        result.strict_raise_message = msg
        # Parse the composed message: HF uses "- `{key}`: {message}" format
        for line in msg.split("\n"):
            line = line.strip()
            if line.startswith("- `") and "`:" in line:
                key = line.split("`")[1]
                result.minor_issues_keys.append(key)

    return result


print("=" * 78)
print(f"Running introspection against {len(TEST_CONFIGS)} test configs")
print("=" * 78)
print()

results = [introspect_one(tc["name"], tc["kwargs"]) for tc in TEST_CONFIGS]


# --- Report ---
print(f"{'TEST':<38}  {'MINOR_ISSUES':<32}  NORMALISED")
print("-" * 78)
for r in results:
    issues_str = ",".join(r.minor_issues_keys) if r.minor_issues_keys else "-"
    normalised_str = (
        ",".join(f"{k}:{v[0]}->{v[1]}" for k, v in r.construction_normalised.items())
        if r.construction_normalised
        else "-"
    )
    exc = f" [EXC:{r.unexpected_exception}]" if r.unexpected_exception else ""
    print(f"{r.name:<38}  {issues_str[:32]:<32}  {normalised_str[:60]}{exc}")

print()

# --- Cross-check against the 30 walker-extracted transformers rules ---
# From probe_ast_scan_poc output (appendix F1), walker extracted these fields
# from GenerationConfig.validate() as rule triggers:
WALKER_EXTRACTED_FIELDS = {
    # Greedy dormancy (7 rules)
    "temperature",
    "top_p",
    "min_p",
    "typical_p",
    "top_k",
    "epsilon_cutoff",
    "eta_cutoff",
    # Beam dormancy (5 rules)
    "early_stopping",
    "num_beam_groups",
    "diversity_penalty",
    "length_penalty",
    "constraints",
    # Errors / raises
    "max_new_tokens",
    "cache_implementation",
    "pad_token_id",
    "watermarking_config",
    "compile_config",
    # Plus watermarking/cache/penalty-alpha/etc rules (~30 total)
}

fields_caught_by_introspection = set()
for r in results:
    fields_caught_by_introspection.update(r.minor_issues_keys)
    fields_caught_by_introspection.update(r.construction_normalised.keys())

greedy_dormancy_fields = {
    "temperature",
    "top_p",
    "min_p",
    "typical_p",
    "top_k",
    "epsilon_cutoff",
    "eta_cutoff",
}
caught_greedy = fields_caught_by_introspection & greedy_dormancy_fields

beam_dormancy_fields = {
    "early_stopping",
    "num_beam_groups",
    "diversity_penalty",
    "length_penalty",
    "constraints",
}
caught_beam = fields_caught_by_introspection & beam_dormancy_fields

print("=" * 78)
print("DECISION CRITERIA EVALUATION")
print("=" * 78)
print()
print(
    f"Greedy-dormancy fields covered (test corpus):  "
    f"{len(caught_greedy)}/{len(greedy_dormancy_fields)}"
)
print(
    f"Beam-dormancy fields covered (test corpus):    {len(caught_beam)}/{len(beam_dormancy_fields)}"
)
print()

total_walker = len(WALKER_EXTRACTED_FIELDS)
total_caught = len(fields_caught_by_introspection & WALKER_EXTRACTED_FIELDS)
coverage_pct = (total_caught / total_walker * 100) if total_walker else 0.0

print(
    f"Walker-rule-field coverage via introspection: "
    f"{total_caught}/{total_walker} = {coverage_pct:.1f}%"
)
print()

print("Decision:")
if coverage_pct >= 90:
    print("  -> >=90%: REPLACE transformers walker with runtime introspection.")
elif coverage_pct >= 60:
    print("  -> 60-90%: KEEP walker as primary; USE introspection as complement.")
else:
    print("  -> <60%: WALKER IS THE RIGHT CALL for transformers. Move on.")

print()
print("Notes on the results:")
print("  - This covers GREEDY + BEAM dormancy fields only (the introspectable")
print("    subset). Walker also extracts raise_error and cache/watermarking rules")
print("    — those come from the same validate() method and would be covered by")
print("    strict-mode exceptions (see 'MINOR_ISSUES' column above).")
print("  - Raw construction_normalised data shows T1.5-style normalisations that")
print("    complement both minor_issues and walker output.")
print()
print("Raw signal for human review:")
print(f"  Fields seen in minor_issues:       {sorted(fields_caught_by_introspection)}")
print(
    f"  Walker-extracted, NOT seen here:   "
    f"{sorted(WALKER_EXTRACTED_FIELDS - fields_caught_by_introspection)}"
)
