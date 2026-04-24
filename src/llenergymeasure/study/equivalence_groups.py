"""Equivalence-groups sidecar — pre-run H1 groups + post-run H3 detection.

Design: ``.product/designs/config-deduplication-dormancy/sweep-dedup.md`` §6.

Written alongside the study's results bundle. ``pre_run_groups`` is populated
at sweep-expansion time by :func:`resolve_library_effective` and serialised immediately;
``post_run_h3_groups`` is populated after the study completes by scanning
sidecars for shared H3 hashes.

The observed-config-hash collision invariant (§4.1) guarantees that in a post-resolved-config-hash dedup run set,
any group with ``len(member_resolved_config_hashes) >= 2`` is a **proven library-resolution mechanism gap**.
Phase 50.3b's ``llem report-gaps`` command consumes this file to propose
corpus additions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llenergymeasure.results.persistence import _atomic_write
from llenergymeasure.study.library_resolution import DedupResult

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreRunGroup:
    """Pre-run equivalence group recorded at sweep-expansion time."""

    resolved_config_hash: str
    canonical_config_excerpt: dict[str, Any]
    member_experiment_ids: tuple[str, ...]
    member_count: int
    representative_experiment_id: str
    would_dedup: bool
    deduplicated: bool


@dataclass(frozen=True)
class PostRunH3Group:
    """Post-run observed-config-hash collision group — a library-resolution mechanism gap if member count >= 2."""

    observed_config_hash: str
    engine: str
    library_version: str
    member_resolved_config_hashes: tuple[str, ...]
    member_experiment_ids: tuple[str, ...]
    gap_detected: bool
    proposed_rule_id: str | None = None


@dataclass
class EquivalenceGroups:
    """Top-level equivalence-groups record written as ``equivalence_groups.json``."""

    study_id: str
    dedup_mode: str
    vendored_rules_version: str = ""
    groups: list[PreRunGroup] = field(default_factory=list)
    post_run_h3_groups: list[PostRunH3Group] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-run population — from resolve_library_effective output
# ---------------------------------------------------------------------------


def build_pre_run_groups(
    dedup: DedupResult,
    *,
    experiment_ids: list[str],
) -> list[PreRunGroup]:
    """Bind :class:`DedupResult` group indices back to caller-supplied IDs.

    ``experiment_ids`` must be parallel to the declared-configs list passed
    to :func:`resolve_library_effective`. The runner is the natural source — it already
    assigns per-experiment IDs before dispatch.
    """
    if len(experiment_ids) != len(dedup.declared_resolved_hashes):
        raise ValueError(
            f"experiment_ids length {len(experiment_ids)} does not match the "
            f"declared config count {len(dedup.declared_resolved_hashes)}"
        )
    pre: list[PreRunGroup] = []
    for group in dedup.groups:
        member_ids = tuple(experiment_ids[i] for i in group.member_indices)
        pre.append(
            PreRunGroup(
                resolved_config_hash=group.resolved_config_hash,
                canonical_config_excerpt=group.canonical_excerpt,
                member_experiment_ids=member_ids,
                member_count=group.member_count,
                representative_experiment_id=experiment_ids[group.representative_index],
                would_dedup=group.member_count > 1,
                deduplicated=dedup.deduplicated and group.member_count > 1,
            )
        )
    return pre


# ---------------------------------------------------------------------------
# Post-run H3 grouping — scan sidecars after study completes
# ---------------------------------------------------------------------------


def find_h3_groups(sidecars: list[dict[str, Any]]) -> list[PostRunH3Group]:
    """Group sidecars by ``(engine, library_version, observed_config_hash)``.

    Any group with size >= 2 AND distinct ``resolved_config_hash`` across its members is
    flagged as a proven library-resolution mechanism gap — per sweep-dedup.md §4.1.

    Each sidecar dict must carry at minimum ``engine``, ``library_version``,
    ``resolved_config_hash``, ``observed_config_hash``, and ``experiment_id`` keys. Sidecars missing any
    of these are silently skipped (pre-50.3a data, or runs with dedup_mode=off
    for which H3 may be partial).
    """
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sc in sidecars:
        h3 = sc.get("observed_config_hash")
        if not h3:
            continue
        engine = str(sc.get("engine", ""))
        version = str(sc.get("library_version", ""))
        buckets[(engine, version, h3)].append(sc)

    groups: list[PostRunH3Group] = []
    for (engine, version, h3), members in buckets.items():
        if len(members) < 2:
            continue
        resolved_config_hashes = tuple(str(m.get("resolved_config_hash", "")) for m in members)
        exp_ids = tuple(str(m.get("experiment_id", "")) for m in members)
        # Gap only if the resolved_config_hashes differ — matching H1 means the
        # library-resolution mechanism already collapsed them.
        gap_detected = len(set(resolved_config_hashes)) > 1
        groups.append(
            PostRunH3Group(
                observed_config_hash=h3,
                engine=engine,
                library_version=version,
                member_resolved_config_hashes=resolved_config_hashes,
                member_experiment_ids=exp_ids,
                gap_detected=gap_detected,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# Writer / reader
# ---------------------------------------------------------------------------


def write_equivalence_groups(groups: EquivalenceGroups, path: Path) -> None:
    """Write :class:`EquivalenceGroups` to ``path`` atomically as JSON.

    ``json.dumps`` serialises frozen-dataclass tuples as arrays naturally, so
    no pre-conversion is needed.
    """
    payload = {
        "study_id": groups.study_id,
        "dedup_mode": groups.dedup_mode,
        "vendored_rules_version": groups.vendored_rules_version,
        "groups": [asdict(g) for g in groups.groups],
        "post_run_h3_groups": [asdict(g) for g in groups.post_run_h3_groups],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json.dumps(payload, indent=2, default=str), path)


def load_equivalence_groups(path: Path) -> EquivalenceGroups:
    """Load :class:`EquivalenceGroups` from a JSON file on disk."""
    data = json.loads(path.read_text())
    return EquivalenceGroups(
        study_id=str(data.get("study_id", "")),
        dedup_mode=str(data.get("dedup_mode", "")),
        vendored_rules_version=str(data.get("vendored_rules_version", "")),
        groups=[_pre_from_dict(g) for g in data.get("groups", [])],
        post_run_h3_groups=[_post_from_dict(g) for g in data.get("post_run_h3_groups", [])],
    )


def _pre_from_dict(data: dict[str, Any]) -> PreRunGroup:
    return PreRunGroup(
        resolved_config_hash=str(data["resolved_config_hash"]),
        canonical_config_excerpt=dict(data.get("canonical_config_excerpt", {})),
        member_experiment_ids=tuple(data.get("member_experiment_ids", [])),
        member_count=int(data.get("member_count", 0)),
        representative_experiment_id=str(data.get("representative_experiment_id", "")),
        would_dedup=bool(data.get("would_dedup", False)),
        deduplicated=bool(data.get("deduplicated", False)),
    )


def _post_from_dict(data: dict[str, Any]) -> PostRunH3Group:
    return PostRunH3Group(
        observed_config_hash=str(data["observed_config_hash"]),
        engine=str(data.get("engine", "")),
        library_version=str(data.get("library_version", "")),
        member_resolved_config_hashes=tuple(data.get("member_resolved_config_hashes", [])),
        member_experiment_ids=tuple(data.get("member_experiment_ids", [])),
        gap_detected=bool(data.get("gap_detected", False)),
        proposed_rule_id=data.get("proposed_rule_id"),
    )
