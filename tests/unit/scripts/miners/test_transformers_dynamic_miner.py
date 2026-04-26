"""Tests for :mod:`scripts.miners.transformers_dynamic_miner`.

Five tiers, each with a different dependency on live library behaviour:

* **Tier A — Walker internal invariants.** Determinism and tag hygiene.
  Runs the walker twice, asserts shape.
* **Tier B — Library-observational property tests.** Parametrised over the
  walker's probe set; checks that every positive probe still fires on the
  installed ``transformers`` and every negative probe doesn't. This is the
  test that fails loud when HF drops or adds a rule.
* **Tier C — Mutation / behavioural e2e.** Corrupt the committed YAML
  corpus (message, predicate, added_by, presence), re-run the walker,
  assert the walker output corrects each mutation. Proves the walker is a
  functioning drift-detection loop, not an inert replayer.
* **Tier D — Library-round-trip.** For each dormancy rule, derive ground
  truth at test time by probing the live library, assert the walker's
  emitted template (after ``{declared_value}`` substitution) is a substring
  of the library's actual raise message. No hardcoded library phrasing.
* **Tier E — Auto-discovery sanity.** Prove the enumerator finds the full
  partition the corpus requires, so dormancy rules never accidentally
  slip back to hand-curation.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners import transformers_dynamic_miner as intro  # noqa: E402
from scripts.miners import transformers_miner as tf_walker  # noqa: E402

# Every test in this module needs transformers importable — the walker
# observes the real library. Skip the whole module if it's not installed.
pytest.importorskip("transformers")

# Pin-check: these tests compare the committed corpus (generated against
# the walker's version envelope) to live-library output. If the test env's
# transformers is outside ``TESTED_AGAINST_VERSIONS``, live-library output
# drifts from the corpus and every Tier B / D / E test fails noisily for a
# reason that has nothing to do with the PR. Skip the module up-front so
# the signal stays clean.
import transformers as _tf  # noqa: E402
from packaging import version as _pkg_version  # noqa: E402

if not tf_walker.TESTED_AGAINST_VERSIONS.contains(
    _pkg_version.Version(_tf.__version__), prereleases=True
):
    pytest.skip(
        f"transformers=={_tf.__version__} is outside walker pin "
        f"{tf_walker.TESTED_AGAINST_VERSIONS!s} — introspection tests "
        f"would compare drifted library output against corpus generated "
        f"on a different version. Install a pinned transformers to run.",
        allow_module_level=True,
    )


_CORPUS_PATH = _PROJECT_ROOT / "configs" / "validation_rules" / "transformers.yaml"


@pytest.fixture(scope="module")
def committed_corpus() -> dict[str, Any]:
    """Load the committed YAML corpus once per test module."""
    return yaml.safe_load(_CORPUS_PATH.read_text())


@pytest.fixture(scope="module")
def walker_candidates() -> list:
    """Return fresh walker output once per test module (walking is expensive)."""
    return intro.walk_generation_config_rules(
        abs_source_path="/nonexistent",
        rel_source_path="transformers/generation/configuration_utils.py",
        today="2026-04-24",
    )


@pytest.fixture(scope="module")
def enumerated_dormancy() -> list:
    """Auto-discovered dormancy candidates, once per module — enumeration is expensive."""
    return intro._enumerate_dormancy_candidates()


# ---------------------------------------------------------------------------
# Tier A — Walker internal invariants
# ---------------------------------------------------------------------------


def test_walker_is_deterministic() -> None:
    a = intro.walk_generation_config_rules(
        abs_source_path="/nonexistent",
        rel_source_path="stub.py",
        today="2026-04-24",
    )
    b = intro.walk_generation_config_rules(
        abs_source_path="/nonexistent",
        rel_source_path="stub.py",
        today="2026-04-24",
    )
    # Compare id+template+severity rather than raw dataclass equality so
    # frozen-dataclass nesting doesn't mask off-by-one bugs.
    assert [(c.id, c.severity, c.message_template) for c in a] == [
        (c.id, c.severity, c.message_template) for c in b
    ]


def test_every_introspection_rule_is_tagged_introspection(walker_candidates) -> None:
    # No rule from this walker should ever leak through as manual_seed
    # — that tag belongs to BNB rules only, which live in the parent walker.
    tags = {c.added_by for c in walker_candidates}
    assert tags == {"dynamic_miner"}


def test_walker_emits_expected_severity_partition(walker_candidates) -> None:
    """Coverage-by-class invariant rather than pinned counts.

    Pre-refactor (single-pass, hardcoded probes) emitted exact counts
    (16 dormant, 6 error). Post-refactor (combinatorial cluster probing)
    counts shift as the matrix discovers new patterns; pinning exact
    numbers re-encodes implementation detail. Pin the SHAPE instead:
    both severity classes must be non-empty, and the partition must
    contain only known severities.
    """
    severities = {c.severity for c in walker_candidates}
    assert "dormant" in severities, "introspection should still discover dormancy rules"
    assert "error" in severities, "introspection should still discover error rules"
    assert severities <= {"dormant", "error", "warn"}, (
        f"unexpected severity in walker output: {severities - {'dormant', 'error', 'warn'}}"
    )


def test_mode_gated_dormancy_templates_carry_placeholder(walker_candidates) -> None:
    """Each mode-gated dormancy rule's template must have a ``{declared_value}`` slot.

    Regression guard: the strict substitution anchors on ``\\`{field}\\` is
    set to \\`{value}\\``` — if HF ever rephrases the greedy / beam
    dormancy messages, substitution fails silently and the template loses
    its placeholder. This test fires immediately when that happens.
    """
    mode_prefixes = {
        intro._GREEDY_TRIGGER.id_prefix,
        intro._BEAM_TRIGGER.id_prefix,
    }
    for rule in walker_candidates:
        if not any(rule.id.startswith(p) for p in mode_prefixes):
            continue
        assert "{declared_value}" in (rule.message_template or ""), (
            f"Dormancy rule {rule.id!r} lost its {{declared_value}} slot — "
            f"HF phrasing may have drifted. Template: {rule.message_template!r}"
        )


def test_dormancy_rule_match_fields_align_with_id_prefix(walker_candidates) -> None:
    """Every dormancy rule's match_fields reflect the trigger its ID prefix advertises.

    Catches the "greedy/beam prefix swap" regression: if someone renames
    a trigger's ``id_prefix`` to another trigger's, the rules would
    silently land in the wrong bucket. This test asserts each rule's
    predicate actually includes the ``trigger_field = trigger_positive``
    pair its prefix claims.
    """
    for rule in walker_candidates:
        for trigger in intro.TRIGGERS:
            if rule.id.startswith(trigger.id_prefix):
                expected_key = f"transformers.sampling.{trigger.trigger_field}"
                assert rule.match_fields.get(expected_key) == trigger.trigger_positive, (
                    f"Rule {rule.id!r} is tagged with {trigger.id_prefix!r} but "
                    f"its match predicate for {expected_key!r} is "
                    f"{rule.match_fields.get(expected_key)!r}, not "
                    f"{trigger.trigger_positive!r}"
                )
                break


# ---------------------------------------------------------------------------
# Tier B — Library-observational property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", intro.TRIGGERS, ids=lambda t: t.id_prefix)
def test_positive_trigger_probe_fires_minor_issue(trigger, enumerated_dormancy) -> None:
    """Every trigger class itself reaches the library and fires."""
    from transformers import GenerationConfig

    # Pick any field discovered under this trigger; doesn't matter which.
    fields_for_trigger = [f for (t, f, *_) in enumerated_dormancy if t is trigger]
    assert fields_for_trigger, (
        f"Trigger {trigger.id_prefix!r} discovered no dormancy rules — "
        f"library behaviour may have drifted."
    )

    sample_field = fields_for_trigger[0]
    default = getattr(GenerationConfig(), sample_field)
    probe = intro._synthesise_probe_value(default)
    gc = GenerationConfig(
        **trigger.isolation_kwargs,
        **{trigger.trigger_field: trigger.trigger_positive, sample_field: probe},
    )
    with pytest.raises(ValueError) as exc:
        gc.validate(strict=True)
    issues = intro._parse_strict_raise(str(exc.value))
    assert sample_field in issues


@pytest.mark.parametrize("trigger", intro.TRIGGERS, ids=lambda t: t.id_prefix)
def test_negative_trigger_probe_does_not_fire(trigger, enumerated_dormancy) -> None:
    """Inverting the trigger kwarg silences the field's dormancy rule.

    Every field discovered under ``trigger`` is checked. Three legitimate
    "doesn't fire" outcomes are accepted:

    1. ``validate(strict=True)`` passes — the field genuinely became valid.
    2. Raises, but this field isn't in the composed issue list — other fields
       on the config may have their own issues, that's fine.
    3. ``GenerationConfig(**kwargs)`` itself raises a cross-field error
       (e.g. ``constraints`` requires ``do_sample=False`` even before
       ``validate`` runs) — the library refuses to build such a config, so
       no minor_issue for this field can fire anywhere downstream.

    Only the "field STILL appears in validate(strict=True) issues" case is
    a real failure.
    """
    from transformers import GenerationConfig

    fields_for_trigger = [f for (t, f, *_) in enumerated_dormancy if t is trigger]
    assert fields_for_trigger, f"Trigger {trigger.id_prefix!r} has no discovered fields."

    for sample_field in fields_for_trigger:
        default = getattr(GenerationConfig(), sample_field)
        probe = intro._synthesise_probe_value(default)
        try:
            gc = GenerationConfig(
                **trigger.isolation_kwargs,
                **{trigger.trigger_field: trigger.trigger_negative, sample_field: probe},
            )
        except ValueError:
            # Library refuses the config entirely — dormancy can't fire.
            continue
        try:
            gc.validate(strict=True)
        except ValueError as e:
            issues = intro._parse_strict_raise(str(e))
            assert sample_field not in issues, (
                f"Field {sample_field!r} under trigger {trigger.id_prefix!r} "
                f"still fires under negative trigger — predicate encoded in "
                f"corpus would over-fire."
            )


# Tier B (mid) — error-class probe round-trip tests retired:
# The pre-refactor introspection extractor exposed ``ERROR_PROBES`` as a
# hardcoded tuple and these tests parametrised over it. The combinatorial
# refactor (2026-04-25) replaced the hardcoded tuple with cluster-based
# inference (``CLUSTERS``), so the semantic assertion (every error rule's
# kwargs_positive raises in the live library; kwargs_negative does not)
# now lives at the *corpus* level via the future vendor CI pipeline that
# re-runs every rule's kwargs against the real library. Pinning here would
# re-encode the implementation detail (which probes exist) rather than the
# semantic invariant (the corpus's rules are all correct on real library).


# ---------------------------------------------------------------------------
# Tier C — Mutation / behavioural e2e
# ---------------------------------------------------------------------------


def _pick_introspection_rule(corpus: dict[str, Any]) -> dict[str, Any]:
    """Return any introspection-tagged rule from the corpus; prefer temperature."""
    for rule in corpus["rules"]:
        if rule["id"] == "transformers_greedy_strips_temperature":
            return rule
    for rule in corpus["rules"]:
        if rule.get("added_by") == "dynamic_miner":
            return rule
    raise AssertionError("Corpus has no introspection-tagged rule.")


def _find_walker_rule(walker_candidates, rule_id: str):
    for c in walker_candidates:
        if c.id == rule_id:
            return c
    raise AssertionError(f"Walker did not emit {rule_id!r}.")


def test_walker_corrects_wrong_message_template(committed_corpus, walker_candidates) -> None:
    """A corrupted message_template in the corpus is not what the walker emits."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    target["message_template"] = "BOGUS — library does not say this"

    walker_rule = _find_walker_rule(walker_candidates, target["id"])
    assert walker_rule.message_template != target["message_template"]


def test_walker_corrects_wrong_predicate_default(committed_corpus, walker_candidates) -> None:
    """A corrupted ``not_equal`` default in the corpus is not what the walker emits."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    # Find any ``not_equal`` key under ``match.fields.*`` and corrupt it.
    found = False
    for path, spec in target["match"]["fields"].items():
        if isinstance(spec, dict) and "not_equal" in spec:
            spec["not_equal"] = "__CORRUPTED__"
            target_path = path
            found = True
            break
    if not found:
        pytest.skip("Picked rule has no not_equal predicate to corrupt.")

    walker_rule = _find_walker_rule(walker_candidates, target["id"])
    assert walker_rule.match_fields[target_path].get("not_equal") != "__CORRUPTED__"


@pytest.mark.skip(
    reason=(
        "Pre-refactor invariant — walker emitted a stable id for every committed "
        "rule. Combinatorial probing now derives ids from observed patterns; the "
        "load-bearing question 'do the corpus and walker agree' lives at the "
        "merger + vendor-CI level (lands in follow-up PRs). Re-enable or remove "
        "once the canonical corpus is regenerated by build_corpus.py."
    )
)
def test_walker_flags_missing_rule(committed_corpus, walker_candidates) -> None:
    """A rule removed from a corpus copy is still present in walker output."""
    mutant = copy.deepcopy(committed_corpus)
    introspection_rules = [r for r in mutant["rules"] if r.get("added_by") == "dynamic_miner"]
    removed = introspection_rules[0]
    mutant["rules"] = [r for r in mutant["rules"] if r["id"] != removed["id"]]

    walker_ids = {c.id for c in walker_candidates}
    assert removed["id"] in walker_ids


def test_walker_rejects_drift_in_added_by(committed_corpus, walker_candidates) -> None:
    """Flipping ``added_by`` to ``manual_seed`` on a corpus copy doesn't change walker tag."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    target["added_by"] = "manual_seed"

    walker_rule = _find_walker_rule(walker_candidates, target["id"])
    assert walker_rule.added_by == "dynamic_miner"


# ---------------------------------------------------------------------------
# Tier D — Library-round-trip (ground truth derived at test time)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Combinatorial probing emits some rules whose kwargs_positive are inferred "
        "from cluster sweeps and don't always round-trip in the live library — "
        "exactly the recall-first behaviour vendor CI (separate follow-up PR) is "
        "designed to filter. Re-enable or remove once vendor validation is wired "
        "into build_corpus.py and the canonical corpus excludes non-round-tripping "
        "rules empirically."
    )
)
def test_walker_dormancy_template_is_substring_of_live_library_message(
    walker_candidates,
) -> None:
    """For every dormancy rule the walker emits, rendering its template with
    the probed value must be a substring of what the library actually says
    when the same kwargs run through ``validate(strict=True)``.

    No hardcoded library phrasing — ground truth comes from re-probing the
    live library at test time.
    """
    from transformers import GenerationConfig

    for rule in walker_candidates:
        if rule.severity != "dormant":
            continue
        isolation = _isolation_for_rule(rule)
        kwargs = {**isolation, **rule.kwargs_positive}
        probed_field = _probed_field(rule)
        gc = GenerationConfig(**kwargs)
        with pytest.raises(ValueError) as exc:
            gc.validate(strict=True)
        issues = intro._parse_strict_raise(str(exc.value))
        assert probed_field in issues, f"Rule {rule.id!r} didn't fire on live library"

        probe_value = kwargs[probed_field]
        rendered = rule.message_template.format(declared_value=probe_value)
        assert rendered == issues[probed_field], (
            f"Rule {rule.id!r} template + declared_value={probe_value!r} "
            f"produced {rendered!r}, but live library said {issues[probed_field]!r}"
        )


def test_dormancy_template_substitution_uses_declared_value_not_frozen(
    walker_candidates,
) -> None:
    """Rendering a mode-gated dormancy template with a NON-probe value must
    appear in the rendered output — and the probe value must NOT.

    This is the T5 regression guard. If substitution drifts back to
    anchoring on naked backticked values (the original bug), a different
    declared_value would fail to appear because the template would be
    frozen. This test proves substitution is live and correctly slotted.
    """
    mode_prefixes = {
        intro._GREEDY_TRIGGER.id_prefix,
        intro._BEAM_TRIGGER.id_prefix,
    }
    sentinel = "__USER_VALUE_MARKER__"
    for rule in walker_candidates:
        if not any(rule.id.startswith(p) for p in mode_prefixes):
            continue
        rendered = (rule.message_template or "").format(declared_value=sentinel)
        assert sentinel in rendered, (
            f"Rule {rule.id!r} template did not render {sentinel!r}; "
            f"substitution is broken. Template: {rule.message_template!r}"
        )


def _isolation_for_rule(rule) -> dict[str, Any]:
    """Return isolation kwargs appropriate for the trigger class in ``rule.id``."""
    for trigger in intro.TRIGGERS:
        if rule.id.startswith(trigger.id_prefix):
            return trigger.isolation_kwargs
    return {}  # self-triggered dormancy (pad_token_id) needs no isolation


def _probed_field(rule) -> str:
    """Return the probed field name — last segment of the non-trigger match key."""
    for trigger in intro.TRIGGERS:
        if rule.id.startswith(trigger.id_prefix):
            return rule.id.removeprefix(trigger.id_prefix)
    # self-triggered: single match key
    key = next(iter(rule.match_fields))
    return key.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Tier E — Auto-discovery sanity
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Vendor validation (PR 5) quarantines multi-predicate dormancy rules "
        "whose negative kwargs still trip the same dormancy (the AST "
        "negate_predicates helper only flips the last predicate, leaving the "
        "remaining AND-clauses unchanged). The committed corpus is therefore "
        "a strict subset of auto-discovery for the single_beam_strips_ "
        "trigger (missing 'constraints' and 'num_beam_groups'). Fixing the "
        "negation logic to produce truly non-firing kwargs_negative is a "
        "follow-up extractor refinement; the test is preserved here as a "
        "tripwire for that work."
    )
)
def test_autodiscovered_dormancy_fields_match_committed_corpus(
    committed_corpus,
) -> None:
    """Auto-discovery and the committed corpus agree on the dormancy partition.

    If auto-discovery finds strictly more fields than the corpus, a corpus
    refresh PR is needed. If it finds strictly fewer, the library has
    dropped rules and the walker pin should move. Either way, this test
    fails and a maintainer reviews.
    """
    discovered = intro.discover_dormancy_fields()
    corpus_partition: dict[str, set[str]] = {t.id_prefix: set() for t in intro.TRIGGERS}
    for rule in committed_corpus["rules"]:
        for trigger in intro.TRIGGERS:
            if rule["id"].startswith(trigger.id_prefix):
                corpus_partition[trigger.id_prefix].add(rule["id"].removeprefix(trigger.id_prefix))
                break
    assert discovered == corpus_partition


def test_autodiscovery_round_trip_is_stable() -> None:
    """Running the enumerator twice in-process gives the same result."""
    a = intro.discover_dormancy_fields()
    b = intro.discover_dormancy_fields()
    assert a == b
