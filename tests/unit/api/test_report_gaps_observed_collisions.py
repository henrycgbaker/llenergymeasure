"""Unit tests for the observed-collisions flavour of ``llem report-gaps``.

Covers :func:`llenergymeasure.api.find_observed_collision_gaps` and the
shared :func:`render_yaml_fragment` path when ``added_by="observed_collision"``.

The two load-bearing fixtures pin Test 1 from the adversarial review at
sweep-dedup.md §10 (2026-04-25 entry):

- An *equality-trigger* fixture (``do_sample=False`` is invariant
  across the collision group and varies in contrast) MUST produce a
  concrete ``match.fields`` predicate.
- A *range-predicate* fixture (``temperature`` is the only varying
  field — ``0.001`` and ``0.005`` collapse to the same observed state)
  MUST route to evidence-only (``match_fields is None``,
  ``needs_generalisation_review=True``) because the trigger field
  varies inside the collision group.

If either of those routes incorrectly, the conservative-field-diff
contract from the design doc is broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.api import (
    GapProposal,
    ReportGapsError,
    find_observed_collision_gaps,
    render_yaml_fragment,
)
from tests.helpers.runtime_obs import (
    fake_hash as _fake_hash,
)
from tests.helpers.runtime_obs import (
    write_equivalence_groups as _write_equivalence_groups,
)
from tests.helpers.runtime_obs import (
    write_manifest as _write_manifest,
)
from tests.helpers.runtime_obs import (
    write_resolution as _write_resolution,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _seed_study(
    study_dir: Path,
    *,
    members: list[tuple[str, dict[str, object]]],
    contrasts: list[tuple[str, dict[str, object]]],
    engine: str = "transformers",
    library_version: str = "4.56.0",
    observed_hash: str = "obs_shared_xxx",
) -> None:
    """Write _resolution sidecars + manifest + equivalence_groups for one gap.

    ``members`` and ``contrasts`` are ``(experiment_id, kwargs)`` lists.
    Each entry creates a per-experiment subdir + ``_resolution.json`` and
    a manifest entry. The members all share ``observed_hash`` and have
    distinct ``resolved_config_hash`` values; the contrasts have
    distinct ``observed_config_hash`` values (so they're outside the
    collision group).
    """
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, object]] = []

    for index, (exp_id, kwargs) in enumerate(members + contrasts, start=1):
        config_hash = _fake_hash(exp_id)
        subdir = _write_resolution(study_dir, index, 1, engine, config_hash, kwargs)
        manifest_entries.append(
            {
                "experiment_id": exp_id,
                "config_hash": config_hash,
                "result_file": f"{subdir.name}/result.json",
            }
        )
    _write_manifest(study_dir, manifest_entries)

    member_ids = [exp_id for exp_id, _ in members]
    member_resolved = [_fake_hash(f"resolved_{exp_id}") for exp_id, _ in members]
    _write_equivalence_groups(
        study_dir,
        observed_collision_groups=[
            {
                "observed_config_hash": observed_hash,
                "engine": engine,
                "library_version": library_version,
                "member_resolved_config_hashes": member_resolved,
                "member_experiment_ids": member_ids,
                "gap_detected": True,
                "proposed_rule_id": None,
            }
        ],
    )


# ---------------------------------------------------------------------------
# Equality-trigger predicate (positive case for conservative field-diff)
# ---------------------------------------------------------------------------


def test_equality_trigger_emits_match_fields(tmp_path: Path) -> None:
    """Conservative field-diff: do_sample is invariant across collision and
    varies across contrast → emit ``match.fields: {do_sample: False}``."""
    study = tmp_path / "study-equality"
    _seed_study(
        study,
        members=[
            ("exp_a", {"do_sample": False, "temperature": 0.5}),
            ("exp_b", {"do_sample": False, "temperature": 1.0}),
        ],
        contrasts=[
            ("exp_c", {"do_sample": True, "temperature": 0.5}),
            ("exp_d", {"do_sample": True, "temperature": 1.0}),
        ],
    )

    proposals = find_observed_collision_gaps(study_dirs=[study])
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.match_fields == {"do_sample": False}
    assert proposal.severity == "dormant"
    assert proposal.added_by == "observed_collision"
    assert proposal.source_channel == "none"
    assert proposal.collision_count == 2
    assert proposal.contrast_count == 2
    assert proposal.needs_generalisation_review is False


# ---------------------------------------------------------------------------
# Range-predicate (PoC-H3R failure case — must route to evidence-only)
# ---------------------------------------------------------------------------


def test_range_predicate_routes_to_evidence_only(tmp_path: Path) -> None:
    """PoC-K range-predicate: temperature varies inside the collision group,
    so no field is invariant-in-collisions-AND-varying-in-contrasts.
    Conservative field-diff must refuse to propose ``match.fields``."""
    study = tmp_path / "study-range"
    _seed_study(
        study,
        members=[
            # Both clamp to temperature=0.01 in the live library; only
            # field that distinguishes them from contrast is the
            # collision-group-internal varying temperature itself, which
            # the conservative inference correctly cannot resolve to a
            # trigger predicate.
            ("exp_a", {"temperature": 0.001, "top_p": 0.95}),
            ("exp_b", {"temperature": 0.005, "top_p": 0.95}),
        ],
        contrasts=[
            ("exp_c", {"temperature": 0.5, "top_p": 0.95}),
            ("exp_d", {"temperature": 1.0, "top_p": 0.95}),
        ],
    )

    proposals = find_observed_collision_gaps(study_dirs=[study])
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.match_fields is None
    assert proposal.needs_generalisation_review is True
    # Evidence is still emitted so reviewers can see the partition.
    assert "temperature" in proposal.evidence_field_value_distribution
    distribution = proposal.evidence_field_value_distribution["temperature"]
    assert distribution["collision_configs"] == ["0.001", "0.005"]
    assert distribution["contrast_configs"] == ["0.5", "1.0"]


# ---------------------------------------------------------------------------
# Filtering and skip behaviour
# ---------------------------------------------------------------------------


def test_no_groups_file_returns_empty(tmp_path: Path) -> None:
    study = tmp_path / "study-empty"
    study.mkdir()
    assert find_observed_collision_gaps(study_dirs=[study]) == []


def test_gap_detected_false_groups_skipped(tmp_path: Path) -> None:
    study = tmp_path / "study-no-gap"
    study.mkdir()
    _write_equivalence_groups(
        study,
        observed_collision_groups=[
            {
                "observed_config_hash": "obs_x",
                "engine": "transformers",
                "library_version": "4.56.0",
                "member_resolved_config_hashes": ["res_a", "res_a"],
                "member_experiment_ids": ["exp_a", "exp_b"],
                "gap_detected": False,
            }
        ],
    )
    assert find_observed_collision_gaps(study_dirs=[study]) == []


def test_engine_filter(tmp_path: Path) -> None:
    """Multi-engine: --engine vllm filters out a transformers gap."""
    study = tmp_path / "study-multi-engine"
    _seed_study(
        study,
        members=[
            ("exp_a", {"do_sample": False}),
            ("exp_b", {"do_sample": False}),
        ],
        contrasts=[("exp_c", {"do_sample": True})],
    )

    assert find_observed_collision_gaps(study_dirs=[study], engine="vllm") == []
    # Sanity: filtering by the matching engine still finds the gap.
    assert len(find_observed_collision_gaps(study_dirs=[study], engine="transformers")) == 1


def test_no_study_dirs_raises() -> None:
    with pytest.raises(ReportGapsError, match="No study directories"):
        find_observed_collision_gaps(study_dirs=[])


def test_unsupported_engine_filter_raises(tmp_path: Path) -> None:
    with pytest.raises(ReportGapsError, match="Unsupported engine"):
        find_observed_collision_gaps(study_dirs=[tmp_path], engine="bogus_backend")


# ---------------------------------------------------------------------------
# YAML rendering — the two flavours produce schema-symmetric output
# ---------------------------------------------------------------------------


def test_yaml_fragment_carries_observed_collision_provenance(tmp_path: Path) -> None:
    study = tmp_path / "study-yaml"
    _seed_study(
        study,
        members=[
            ("exp_a", {"do_sample": False}),
            ("exp_b", {"do_sample": False}),
        ],
        contrasts=[("exp_c", {"do_sample": True})],
    )
    proposals = find_observed_collision_gaps(study_dirs=[study])
    assert proposals
    body = render_yaml_fragment(proposals[0])

    # Provenance is the new enum value.
    assert "added_by: observed_collision" in body
    # Severity for silent normalisations is dormant; outcome dormant_silent.
    assert "severity: dormant" in body
    assert "dormant_silent" in body
    # Silent channel — no message-text matching surface.
    assert "emission_channel: none" in body
    assert "observed_messages_regex" not in body
    assert "observed_messages:" not in body
    # Conservative field-diff predicate did fire on this fixture.
    assert "do_sample: false" in body


def test_yaml_fragment_evidence_only_when_predicate_unresolvable(tmp_path: Path) -> None:
    study = tmp_path / "study-evidence-only"
    _seed_study(
        study,
        members=[
            ("exp_a", {"temperature": 0.001, "top_p": 0.95}),
            ("exp_b", {"temperature": 0.005, "top_p": 0.95}),
        ],
        contrasts=[("exp_c", {"temperature": 0.5, "top_p": 0.95})],
    )
    proposals = find_observed_collision_gaps(study_dirs=[study])
    assert proposals
    body = render_yaml_fragment(proposals[0])

    assert "needs_generalisation_review: true" in body
    # Evidence block is present even when the predicate can't be inferred.
    assert "evidence_field_value_distribution:" in body
    # Empty match.fields appears as `{}`.
    assert "fields: {}" in body


# ---------------------------------------------------------------------------
# Multi-study aggregation
# ---------------------------------------------------------------------------


def test_multi_study_aggregation_sorted(tmp_path: Path) -> None:
    """Two studies, two gaps — proposals returned sorted by (engine, version, template)."""
    study_a = tmp_path / "a"
    study_b = tmp_path / "b"
    _seed_study(
        study_a,
        members=[
            ("exp_aa", {"do_sample": False}),
            ("exp_ab", {"do_sample": False}),
        ],
        contrasts=[("exp_ac", {"do_sample": True})],
        observed_hash="obs_aaaa",
    )
    _seed_study(
        study_b,
        members=[
            ("exp_ba", {"do_sample": False}),
            ("exp_bb", {"do_sample": False}),
        ],
        contrasts=[("exp_bc", {"do_sample": True})],
        observed_hash="obs_bbbb",
    )

    proposals = find_observed_collision_gaps(study_dirs=[study_b, study_a])
    assert len(proposals) == 2
    templates = [p.normalised_template for p in proposals]
    assert templates == sorted(templates)


def test_proposal_is_typed_dataclass(tmp_path: Path) -> None:
    """Sanity check that the public dataclass is what we think it is."""
    study = tmp_path / "study-types"
    _seed_study(
        study,
        members=[
            ("exp_a", {"do_sample": False}),
            ("exp_b", {"do_sample": False}),
        ],
        contrasts=[("exp_c", {"do_sample": True})],
    )
    proposals = find_observed_collision_gaps(study_dirs=[study])
    assert proposals
    assert isinstance(proposals[0], GapProposal)
