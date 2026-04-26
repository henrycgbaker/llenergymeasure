#!/usr/bin/env python3
"""PoC-H3R — H3-collision reporter YAML quality analogue to M3's RT-3.

LOAD-BEARING for M4 scope decision: ship the MVP H3-collision reporter
(`llem report-gaps --source h3-collisions`) as a simple field-diff +
`needs-generalisation-review` label (parallel to M3's runtime-warning
reporter), OR delay the MVP to add the §4.9.4 walker-cache-match
enhancement from the outset.

Question: given two fired configs that collided on H3 plus a non-fired
comparison set, does the MVP inference algorithm (simple field-diff
per sweep-dedup §4.3/§4.9.2) produce YAML that merges as-is into
`configs/validation_rules/{engine}.yaml`?

Method:
    - Construct three fixtures with distinct predicate shapes:
        F-H3-1  vLLM epsilon-clamp           RANGE predicate   (expected to fail)
        F-H3-2  transformers greedy-strip    EQUALITY          (expected to succeed)
        F-H3-3  transformers beam-width-1    EQUALITY, multi   (expected to succeed)
    - Run the MVP inference algorithm: field-diff between the two fired
      configs' declared kwargs, fall back to `needs-generalisation-review`
      when the diff is ambiguous w.r.t. the non-fired set.
    - Render via pyyaml (default_flow_style=False, sort_keys=False)
      matching runtime-config-validation.md §4.1 entry schema with
      added_by: h3_collision.
    - Score per fixture:
          Schema completeness  – all 15 required/stubbed fields populated
          Round-trip           – parses through vendored_rules/loader.py
          Predicate correctness– separates fired from not-fired on sweep space
          Mergability          – as-is | minor-edits | rewrite-required

Scoring thresholds:
    Mergability = as-is               → ship MVP
    Mergability = minor-edits + label → ship MVP with labels
    Mergability = rewrite-required    → delay MVP for walker-cache-match

Reference:  .product/designs/config-deduplication-dormancy/sweep-dedup.md §4.3, §4.9.2
            .product/designs/config-deduplication-dormancy/runtime-config-validation.md §4.1
            Sibling PoC: scripts/probe_warning_predicate_inference.py (RT-1)
            Predecessor: scripts/probe_h3_collision_invariant.py (PoC-K)

Written 2026-04-24 for M4 scope decision.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from llenergymeasure.config.vendored_rules.loader import (  # noqa: E402
    VendoredRulesLoader,
    _parse_envelope,
    evaluate_predicate,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


FIXTURES: list[dict[str, Any]] = [
    {
        "id": "F-H3-1",
        "label": "vllm_epsilon_clamp",
        "engine": "vllm",
        "library": "vllm",
        "native_type": "vllm.SamplingParams",
        "predicate_shape": "range",
        "ground_truth_rule_id": "vllm_epsilon_clamps_temperature",
        "ground_truth_match": {
            "temperature": {">": 0.0, "<": 0.01},
        },
        "ground_truth_message": (
            "`temperature` is {declared_value}; vLLM clamps any value in the "
            "open interval (0, 0.01) up to 0.01 to avoid numerical instability."
        ),
        "fired_a": {
            "engine": "vllm",
            "temperature": 0.001,
            "top_p": 0.95,
            "top_k": -1,
            "max_model_len": 4096,
        },
        "fired_b": {
            "engine": "vllm",
            "temperature": 0.005,
            "top_p": 0.95,
            "top_k": -1,
            "max_model_len": 4096,
        },
        "shared_effective": {"temperature": 0.01},
        "not_fired": [
            {
                "engine": "vllm",
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": -1,
                "max_model_len": 4096,
            },
            {
                "engine": "vllm",
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": -1,
                "max_model_len": 4096,
            },
            {
                "engine": "vllm",
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": -1,
                "max_model_len": 4096,
            },
        ],
    },
    {
        "id": "F-H3-2",
        "label": "transformers_greedy_strip",
        "engine": "transformers",
        "library": "transformers",
        "native_type": "transformers.GenerationConfig",
        "predicate_shape": "equality",
        "ground_truth_rule_id": "transformers_greedy_strips_temperature",
        "ground_truth_match": {
            "do_sample": False,
            "temperature": {"present": True, "not_equal": 1.0},
        },
        "ground_truth_message": (
            "`do_sample` is set to `False`. However, `temperature` is set to "
            "`{declared_value}` -- this flag is only used in sample-based "
            "generation modes."
        ),
        "fired_a": {"engine": "transformers", "do_sample": False, "temperature": 0.3},
        "fired_b": {"engine": "transformers", "do_sample": False, "temperature": 0.7},
        "shared_effective": {"temperature": 1.0},
        "not_fired": [
            {"engine": "transformers", "do_sample": True, "temperature": 0.3},
            {"engine": "transformers", "do_sample": True, "temperature": 0.7},
            {"engine": "transformers", "do_sample": True, "temperature": 0.5},
        ],
    },
    {
        "id": "F-H3-3",
        "label": "transformers_beam_width_1",
        "engine": "transformers",
        "library": "transformers",
        "native_type": "transformers.GenerationConfig",
        "predicate_shape": "equality_multi",
        "ground_truth_rule_id": "transformers_single_beam_strips_early_stopping",
        "ground_truth_match": {
            "num_beams": 1,
            "early_stopping": {"present": True, "not_equal": False},
        },
        "ground_truth_message": (
            "`num_beams` is set to 1. However, `early_stopping` is set to "
            "`{declared_value}` -- this flag is only used in beam-based "
            "generation modes."
        ),
        "fired_a": {
            "engine": "transformers",
            "num_beams": 1,
            "do_sample": False,
            "early_stopping": True,
        },
        "fired_b": {
            "engine": "transformers",
            "num_beams": 1,
            "do_sample": False,
            "early_stopping": False,
        },
        "shared_effective": {"early_stopping": False},
        "not_fired": [
            {"engine": "transformers", "num_beams": 4, "do_sample": False, "early_stopping": True},
            {"engine": "transformers", "num_beams": 1, "do_sample": True, "early_stopping": True},
            {"engine": "transformers", "num_beams": 2, "do_sample": False, "early_stopping": True},
        ],
    },
]


# -----------------------------------------------------------------------------
# MVP inference algorithm (sweep-dedup §4.3, §4.9.2)
# -----------------------------------------------------------------------------


def infer_rule_from_collision(
    fired_a: dict[str, Any],
    fired_b: dict[str, Any],
    shared_effective: dict[str, Any],
    not_fired: list[dict[str, Any]],
) -> dict[str, Any]:
    """Field-diff inference: which fields differ between fired A and fired B,
    and which fields (when held constant across both) define the trigger?

    MVP per §4.3: "compare the declared kwargs of member A and member B, find
    the field(s) that differ, and express them as a match predicate with a
    normalisation that maps both to the shared H3-observed effective value."

    Returns the inferred match dict plus metadata. If the inferred predicate
    does not separate fired from not-fired (as will happen for RANGE rules),
    raises the `needs_generalisation_review` flag.
    """
    # Fields present in both fired configs.
    common_keys = set(fired_a) & set(fired_b) - {"engine"}
    # Fields where A and B differ (the "affected" / subject field candidates).
    diff_keys = {k for k in common_keys if fired_a.get(k) != fired_b.get(k)}
    # Fields where A and B agree (the "trigger" / precondition candidates).
    shared_keys = common_keys - diff_keys

    # Subject = a differing field whose *effective* value the library mapped
    # both configs onto (i.e. appears in shared_effective).
    subject = next(
        (k for k in diff_keys if k in shared_effective),
        # Fall back to any differing field if none of the diffs are in the
        # shared effective state (shouldn't happen for genuine H3 collisions).
        next(iter(diff_keys)) if diff_keys else None,
    )

    # MVP trigger = the set of fields that agree between A and B, each pinned
    # to its equality value. This is the strictly weakest predicate consistent
    # with what the algorithm can see.
    trigger: dict[str, Any] = {k: fired_a[k] for k in sorted(shared_keys)}

    # Subject condition: "present and differs from shared effective value".
    if subject is not None and subject in shared_effective:
        subject_cond = {"present": True, "not_equal": shared_effective[subject]}
    elif subject is not None:
        subject_cond = {"present": True}
    else:
        subject_cond = {"present": True}

    # Separation test: does the trigger (without the subject) separate fired
    # from not-fired?
    def matches_trigger(cfg: dict[str, Any]) -> bool:
        return all(cfg.get(k) == v for k, v in trigger.items())

    fires_on_fired = matches_trigger(fired_a) and matches_trigger(fired_b)
    fires_on_not_fired = any(matches_trigger(c) for c in not_fired)

    separates = fires_on_fired and not fires_on_not_fired
    needs_review = not separates

    match_fields: dict[str, Any] = dict(trigger)
    if subject is not None:
        match_fields[subject] = subject_cond

    return {
        "match_fields": match_fields,
        "subject": subject,
        "trigger": trigger,
        "separates": separates,
        "needs_generalisation_review": needs_review,
    }


# -----------------------------------------------------------------------------
# YAML rendering (runtime-config-validation.md §4.1)
# -----------------------------------------------------------------------------


_REQUIRED_FIELDS = (
    "id",
    "engine",
    "library",
    "rule_under_test",
    "severity",
    "native_type",
    "walker_source",
    "match",
    "kwargs_positive",
    "kwargs_negative",
    "expected_outcome",
    "message_template",
    "references",
    "added_by",
    "added_at",
)


def render_proposal(fixture: dict[str, Any], inferred: dict[str, Any]) -> dict[str, Any]:
    """Build the YAML rule entry per runtime-config-validation.md §4.1."""
    engine = fixture["engine"]
    subject = inferred["subject"] or "<unknown>"
    rule_id = f"h3_{engine}_{fixture['label']}_{subject}"

    # Path-prefix fields per corpus convention (e.g.
    # "transformers.sampling.temperature"). MVP: use "<engine>.declared.<field>".
    prefixed_fields = {f"{engine}.declared.{k}": v for k, v in inferred["match_fields"].items()}

    # kwargs_positive = fired_a declared kwargs minus the engine key (must
    # trigger). kwargs_negative = first not-fired config (must NOT trigger).
    kwargs_positive = {k: v for k, v in fixture["fired_a"].items() if k != "engine"}
    kwargs_negative = {k: v for k, v in fixture["not_fired"][0].items() if k != "engine"}

    rule: dict[str, Any] = {
        "id": rule_id,
        "engine": engine,
        "library": fixture["library"],
        "rule_under_test": (
            f"Proposed by h3_collision detector: {engine} library collapses "
            f"two configs differing in `{subject}` to an identical effective "
            f"state ({fixture['shared_effective']})."
        ),
        "severity": "dormant",
        "native_type": fixture["native_type"],
        "walker_source": {
            "path": "<needs-generalisation-review>",
            "method": "<unknown>",
            "line_at_scan": 0,
            "walker_confidence": "low",
        },
        "match": {
            "engine": engine,
            "fields": prefixed_fields,
        },
        "kwargs_positive": kwargs_positive,
        "kwargs_negative": kwargs_negative,
        "expected_outcome": {
            "outcome": "dormant_silent",
            "emission_channel": "none",
            "normalised_fields": [
                {"field": subject, "effective_value": fixture["shared_effective"].get(subject)}
            ],
        },
        "message_template": (
            f"`{subject}` is set to `{{declared_value}}` but "
            f"{engine} normalises it to `{fixture['shared_effective'].get(subject)}`."
        ),
        "references": [
            f"Detected via h3_collision between H1={'<A>'} and H1={'<B>'}.",
        ],
        "added_by": "h3_collision",
        "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }

    if inferred["needs_generalisation_review"]:
        rule["needs_generalisation_review"] = True
        rule["references"].append(
            "needs-generalisation-review: field-diff did not separate fired "
            "from not-fired on the sweep space; maintainer must replace the "
            "equality match with a range/typed predicate."
        )

    return rule


def render_yaml_document(rules: list[dict[str, Any]], engine: str) -> str:
    """Render a full vendored-rules envelope with a single engine's rules."""
    envelope = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": "<unknown>",
        "walker_pinned_range": "<unknown>",
        "walked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": rules,
    }
    buf = io.StringIO()
    yaml.safe_dump(envelope, buf, default_flow_style=False, sort_keys=False)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------


def score_schema_completeness(rule: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [f for f in _REQUIRED_FIELDS if f not in rule]
    stubbed = [f for f in _REQUIRED_FIELDS if rule.get(f) in (None, "", [], {})]
    complete = not missing and not stubbed
    return complete, missing + [f"{s} (stubbed)" for s in stubbed]


def score_round_trip(rule: dict[str, Any], engine: str) -> tuple[bool, str | None]:
    """Attempt to parse the rendered envelope through the real loader."""
    doc = render_yaml_document([rule], engine)
    try:
        parsed = _parse_envelope(engine, doc)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if len(parsed.rules) != 1:
        return False, f"expected 1 rule, parsed {len(parsed.rules)}"
    return True, None


def score_predicate_correctness(
    rule: dict[str, Any], fixture: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Does the proposed predicate correctly separate fired from not-fired?"""
    match_fields = rule["match"]["fields"]

    def config_to_prefixed(cfg: dict[str, Any]) -> dict[str, Any]:
        return {f"{fixture['engine']}.declared.{k}": v for k, v in cfg.items() if k != "engine"}

    def fires(cfg: dict[str, Any]) -> bool:
        prefixed = config_to_prefixed(cfg)
        for path, spec in match_fields.items():
            if not evaluate_predicate(prefixed.get(path), spec):
                return False
        return True

    fires_a = fires(fixture["fired_a"])
    fires_b = fires(fixture["fired_b"])
    not_fired_fires = [fires(c) for c in fixture["not_fired"]]

    correct = fires_a and fires_b and not any(not_fired_fires)
    return correct, {
        "fires_a": fires_a,
        "fires_b": fires_b,
        "not_fired_fires": not_fired_fires,
    }


def classify_mergability(
    *,
    schema_ok: bool,
    round_trip_ok: bool,
    predicate_ok: bool,
    needs_review: bool,
) -> str:
    if not schema_ok or not round_trip_ok:
        return "rewrite-required"
    if predicate_ok and not needs_review:
        return "as-is"
    if predicate_ok and needs_review:
        return "minor-edits"
    # Predicate incorrect on the sweep space → range/generalisation edit needed.
    return "rewrite-required"


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def run_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    inferred = infer_rule_from_collision(
        fired_a=fixture["fired_a"],
        fired_b=fixture["fired_b"],
        shared_effective=fixture["shared_effective"],
        not_fired=fixture["not_fired"],
    )
    rule = render_proposal(fixture, inferred)

    schema_ok, schema_gaps = score_schema_completeness(rule)
    round_trip_ok, rt_err = score_round_trip(rule, fixture["engine"])
    predicate_ok, predicate_detail = score_predicate_correctness(rule, fixture)

    mergability = classify_mergability(
        schema_ok=schema_ok,
        round_trip_ok=round_trip_ok,
        predicate_ok=predicate_ok,
        needs_review=inferred["needs_generalisation_review"],
    )

    return {
        "fixture_id": fixture["id"],
        "label": fixture["label"],
        "predicate_shape": fixture["predicate_shape"],
        "inferred": inferred,
        "rule": rule,
        "schema_ok": schema_ok,
        "schema_gaps": schema_gaps,
        "round_trip_ok": round_trip_ok,
        "round_trip_error": rt_err,
        "predicate_ok": predicate_ok,
        "predicate_detail": predicate_detail,
        "mergability": mergability,
    }


def print_report(results: list[dict[str, Any]]) -> None:
    print("=" * 78)
    print("PoC-H3R — H3-collision reporter YAML quality")
    print("=" * 78)

    # Per-fixture verdict table.
    print("\nPer-fixture verdict:")
    print(f"{'FIXTURE':<10} {'SHAPE':<16} {'SCHEMA':<8} {'RT':<5} {'PRED':<6} {'MERGABILITY':<20}")
    for r in results:
        print(
            f"{r['fixture_id']:<10} "
            f"{r['predicate_shape']:<16} "
            f"{('OK' if r['schema_ok'] else 'FAIL'):<8} "
            f"{('OK' if r['round_trip_ok'] else 'FAIL'):<5} "
            f"{('OK' if r['predicate_ok'] else 'FAIL'):<6} "
            f"{r['mergability']:<20}"
        )

    # Per-fixture detail.
    for r in results:
        print(f"\n--- {r['fixture_id']}  {r['label']}  [{r['predicate_shape']}] ---")
        print(f"  schema_gaps:       {r['schema_gaps'] or '(none)'}")
        print(f"  round_trip_error:  {r['round_trip_error']}")
        print(f"  predicate_detail:  {r['predicate_detail']}")
        print(f"  needs_generalisation_review: {r['inferred']['needs_generalisation_review']}")
        print(f"  mergability:       {r['mergability']}")

    # Rendered YAML for each fixture.
    for r in results:
        print(f"\n--- {r['fixture_id']} rendered YAML (single rule) ---")
        print(render_yaml_document([r["rule"]], r["rule"]["engine"]))

    # Loader sanity-check against the real corpus.
    print("--- Loader sanity-check against real transformers.yaml corpus ---")
    try:
        loader = VendoredRulesLoader()
        real = loader.load_rules("transformers")
        print(f"  loaded {len(real.rules)} rules from real corpus OK")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")

    # Headline.
    print("\n" + "=" * 78)
    as_is = sum(1 for r in results if r["mergability"] == "as-is")
    minor = sum(1 for r in results if r["mergability"] == "minor-edits")
    rewrite = sum(1 for r in results if r["mergability"] == "rewrite-required")
    print(
        f"HEADLINE  as-is={as_is}  minor-edits={minor}  rewrite-required={rewrite}  of {len(results)}"
    )
    print("=" * 78)


def main() -> int:
    results = [run_fixture(f) for f in FIXTURES]
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
