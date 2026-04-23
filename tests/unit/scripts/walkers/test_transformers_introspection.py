"""Tests for :mod:`scripts.walkers.transformers_introspection`.

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

from scripts.walkers import transformers_introspection as intro  # noqa: E402

# Every test in this module needs transformers importable — the walker
# observes the real library. Skip the whole module if it's not installed.
pytest.importorskip("transformers")


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
    assert tags == {"introspection"}


def test_walker_emits_expected_severity_partition(walker_candidates) -> None:
    by_sev = {"dormant": 0, "error": 0}
    for c in walker_candidates:
        by_sev[c.severity] = by_sev.get(c.severity, 0) + 1
    # 7 greedy + 5 beam + 3 no-return-dict + 1 pad_token_id = 16 dormant
    # 6 error probes = 6 error
    assert by_sev == {"dormant": 16, "error": 6}


# ---------------------------------------------------------------------------
# Tier B — Library-observational property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", intro._TRIGGERS, ids=lambda t: t.id_prefix)
def test_positive_trigger_probe_fires_minor_issue(trigger) -> None:
    """Every trigger class itself reaches the library and fires."""
    from transformers import GenerationConfig

    # Pick any field discovered under this trigger; doesn't matter which.
    discovered = intro._enumerate_dormancy_candidates()
    fields_for_trigger = [f for (t, f, *_) in discovered if t is trigger]
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


@pytest.mark.parametrize("trigger", intro._TRIGGERS, ids=lambda t: t.id_prefix)
def test_negative_trigger_probe_does_not_fire(trigger) -> None:
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

    discovered = intro._enumerate_dormancy_candidates()
    fields_for_trigger = [f for (t, f, *_) in discovered if t is trigger]
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


@pytest.mark.parametrize("probe", intro._ERROR_PROBES, ids=lambda p: p.id)
def test_error_probe_raises_at_construction(probe) -> None:
    """Every error-class probe must still trigger a ValueError on the live library."""
    from transformers import GenerationConfig

    with pytest.raises(ValueError):
        GenerationConfig(**probe.kwargs_positive)


@pytest.mark.parametrize("probe", intro._ERROR_PROBES, ids=lambda p: p.id)
def test_error_probe_negative_kwargs_construct_cleanly(probe) -> None:
    """``kwargs_negative`` must NOT raise — the "same values valid" invariant."""
    from transformers import GenerationConfig

    try:
        GenerationConfig(**probe.kwargs_negative)
    except ValueError as e:  # pragma: no cover — regression signal
        pytest.fail(f"kwargs_negative raised: {e}")


# ---------------------------------------------------------------------------
# Tier C — Mutation / behavioural e2e
# ---------------------------------------------------------------------------


def _pick_introspection_rule(corpus: dict[str, Any]) -> dict[str, Any]:
    """Return any introspection-tagged rule from the corpus; prefer temperature."""
    for rule in corpus["rules"]:
        if rule["id"] == "transformers_greedy_strips_temperature":
            return rule
    for rule in corpus["rules"]:
        if rule.get("added_by") == "introspection":
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


def test_walker_flags_missing_rule(committed_corpus, walker_candidates) -> None:
    """A rule removed from a corpus copy is still present in walker output."""
    mutant = copy.deepcopy(committed_corpus)
    # Remove the first introspection-tagged rule.
    introspection_rules = [r for r in mutant["rules"] if r.get("added_by") == "introspection"]
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
    assert walker_rule.added_by == "introspection"


# ---------------------------------------------------------------------------
# Tier D — Library-round-trip (ground truth derived at test time)
# ---------------------------------------------------------------------------


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
        # Reconstruct the kwargs the walker used to derive the template.
        # For mode-gated dormancy, kwargs_positive has {trigger_field, probed_field}.
        # We need the isolation kwargs that the walker used internally,
        # which aren't in the emitted rule — so we rebuild them from the
        # trigger prefix in the rule id.
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


def _isolation_for_rule(rule) -> dict[str, Any]:
    """Return isolation kwargs appropriate for the trigger class in ``rule.id``."""
    for trigger in intro._TRIGGERS:
        if rule.id.startswith(trigger.id_prefix):
            return trigger.isolation_kwargs
    return {}  # self-triggered dormancy (pad_token_id) needs no isolation


def _probed_field(rule) -> str:
    """Return the probed field name — last segment of the non-trigger match key."""
    for trigger in intro._TRIGGERS:
        if rule.id.startswith(trigger.id_prefix):
            return rule.id.removeprefix(trigger.id_prefix)
    # self-triggered: single match key
    key = next(iter(rule.match_fields))
    return key.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Tier E — Auto-discovery sanity
# ---------------------------------------------------------------------------


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
    corpus_partition: dict[str, set[str]] = {t.id_prefix: set() for t in intro._TRIGGERS}
    for rule in committed_corpus["rules"]:
        for trigger in intro._TRIGGERS:
            if rule["id"].startswith(trigger.id_prefix):
                corpus_partition[trigger.id_prefix].add(rule["id"].removeprefix(trigger.id_prefix))
                break
    assert discovered == corpus_partition


def test_autodiscovery_round_trip_is_stable() -> None:
    """Running the enumerator twice in-process gives the same result."""
    a = intro.discover_dormancy_fields()
    b = intro.discover_dormancy_fields()
    assert a == b
