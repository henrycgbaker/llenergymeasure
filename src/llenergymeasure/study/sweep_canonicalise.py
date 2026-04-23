"""Sweep canonicaliser — apply vendored dormant rules to fixpoint, dedup by H1.

Design: ``.product/designs/config-deduplication-dormancy/sweep-dedup.md`` §2.

The canonicaliser is the host-side, pre-dispatch layer that normalises every
field the vendored corpus marks as ``dormant``. Each rule's fired-state
projection is taken from its match predicate's "not_equal" / "present"
operand (the sentinel value the predicate is *deviating from*) — that same
projection is what :mod:`scripts.walkers._fixpoint_test` enforces in CI, so
runtime canonicalisation and CI correctness tests apply an identical
normalisation.

Rules chain (vLLM epsilon-clamp → greedy-normalise); iteration is capped at
:data:`_MAX_ITER` to surface cycles via :class:`CanonicaliserCycleError`.

Out-of-scope per PLAN §Scope OUT: vLLM/TRT-LLM corpora don't exist yet, so
this PR mostly exercises the transformers rules. The canonicaliser itself is
engine-generic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.engines.vendored_rules.loader import (
    Rule,
    VendoredRulesLoader,
    resolve_field_path,
)
from llenergymeasure.study.hashing import build_h1_view, hash_config

logger = logging.getLogger(__name__)

_MAX_ITER = 10
"""Maximum fixpoint passes before declaring non-convergence.

PoC-F (sweep-dedup.md §10) converged every seeded-corpus case within 2
passes; 10 is generous headroom that still surfaces a rule cycle quickly.
"""


class CanonicaliserCycleError(RuntimeError):
    """The canonicaliser did not reach a fixpoint within :data:`_MAX_ITER` passes.

    Indicates a cycle in the vendored rules corpus (rule A produces state
    matching rule B which produces state matching rule A). The corpus vendor
    step's shuffle-application test is supposed to catch this at CI time,
    but this guard prevents runtime hangs if a bad corpus ships anyway.
    """

    def __init__(self, final_config: ExperimentConfig, iterations: int) -> None:
        super().__init__(
            f"Canonicaliser did not reach fixpoint within {iterations} iterations. "
            f"Likely a cycle in the vendored rules corpus. "
            f"Final engine={final_config.engine}."
        )
        self.final_config = final_config
        self.iterations = iterations


# ---------------------------------------------------------------------------
# Core canonicalise() — one config
# ---------------------------------------------------------------------------


def canonicalise(
    config: ExperimentConfig, rules: list[Rule] | tuple[Rule, ...]
) -> ExperimentConfig:
    """Apply every ``dormant``-severity rule to ``config`` repeatedly until stable.

    Returns a deep-copy of ``config`` with each dormant rule's normalisations
    projected onto the fired fields. The input is not mutated.

    Args:
        config: A validated ``ExperimentConfig``.
        rules: The rule list for the config's engine (typically from
            ``VendoredRulesLoader.load_rules(engine).rules``).

    Raises:
        CanonicaliserCycleError: If the fixpoint loop exceeds
            :data:`_MAX_ITER` passes — the vendored corpus has a rule cycle.
    """
    dormant_rules = [r for r in rules if r.severity in ("dormant", "dormant_silent")]
    if not dormant_rules:
        return config.model_copy(deep=True)

    current = config.model_copy(deep=True)
    for _iteration in range(_MAX_ITER):
        fired = False
        for rule in dormant_rules:
            match = rule.try_match(current)
            if match is None:
                continue
            updates = _rule_normalisations(rule)
            if not updates:
                continue
            for field_path, target_value in updates.items():
                if resolve_field_path(current, field_path) != target_value:
                    _assign_field_path(current, field_path, target_value)
                    fired = True
        if not fired:
            return current
    raise CanonicaliserCycleError(current, _MAX_ITER)


def _rule_normalisations(rule: Rule) -> dict[str, Any]:
    """Return ``{field_path: canonical_value}`` the rule normalises to.

    Strategy (per sweep-dedup.md §2.1 and the fixpoint test's projection):

    1. If ``expected_outcome["normalised_fields"]`` lists explicit paths, they
       collapse to ``None`` (the universal "strip this field" sentinel).
    2. Otherwise, fall back to the rule's *match* predicate: any field
       matched with a ``not_equal`` / ``present`` operator is normalised by
       stripping (setting to ``None`` or the ``not_equal`` sentinel if
       scalar). This is the fixpoint-test projection — structurally identical
       to what CI enforces for shuffle-stability.

    Rules that match only on equality (e.g. ``do_sample: false``) do not
    normalise those fields — equality predicates are *triggers*, not
    *subjects*. Subject fields are the ones marked ``present``/``not_equal``.
    """
    out: dict[str, Any] = {}

    explicit = rule.expected_outcome.get("normalised_fields") or []
    for raw_path in explicit:
        path = str(raw_path)
        out[path] = None

    if out:
        return out

    for path, spec in rule.match_fields.items():
        if not isinstance(spec, dict):
            continue
        if "not_equal" in spec:
            # The "canonical" state is the not_equal sentinel — applying the
            # rule drives the field back to the library-observed default.
            out[path] = spec["not_equal"]
        elif spec.get("present") and "not_equal" not in spec and "in" not in spec:
            # Subject field marked only as "present" — strip to None (the
            # library either ignores it, or the effective value is captured
            # later via H3).
            out[path] = None
    return out


def _assign_field_path(config: ExperimentConfig, path: str, value: Any) -> None:
    """Set ``value`` at dotted ``path`` on ``config`` in place.

    Walks nested Pydantic models / dicts, tolerant of ``None`` intermediate
    attributes (silently returns if the path doesn't resolve to an assignable
    location — mirrors :func:`resolve_field_path`'s permissive traversal).
    """
    parts = path.split(".")
    parent: Any = config
    for part in parts[:-1]:
        if parent is None:
            return
        parent = parent.get(part) if isinstance(parent, dict) else getattr(parent, part, None)
    if parent is None:
        return
    leaf = parts[-1]
    try:
        if isinstance(parent, dict):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)
    except (ValueError, TypeError) as exc:
        # Pydantic model_config={"frozen": True} or field constraints can
        # reject the assignment. Log at debug — the canonicaliser is best-
        # effort; a rejected field just stays at its declared value.
        logger.debug("Canonicaliser could not assign %s=%r: %s", path, value, exc)


# ---------------------------------------------------------------------------
# Sweep dedup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceGroup:
    """Pre-run group of declared configs that collapse to the same H1 canonical form."""

    h1_hash: str
    canonical_excerpt: dict[str, Any]
    member_indices: tuple[int, ...]
    representative_index: int

    @property
    def member_count(self) -> int:
        return len(self.member_indices)


@dataclass
class DedupResult:
    """Return bundle from :func:`dedup_sweep`.

    Attributes:
        canonical_configs: The canonicalised configs after dedup (or all
            canonicalised configs when ``deduplicate=False``, with duplicates
            kept). This is what the runner iterates over.
        groups: One :class:`EquivalenceGroup` per unique H1 hash, recording
            which indices of the input sweep collapsed together.
        declared_h1: ``declared_index → h1_hash`` — lets the runner tag the
            manifest entry for each run with its equivalence group.
        would_dedup: ``True`` iff any group has > 1 member (dedup would save
            runs even when ``deduplicate=False``).
        deduplicated: ``True`` iff dedup was actually applied to the
            ``canonical_configs`` return slot.
    """

    canonical_configs: list[ExperimentConfig]
    groups: list[EquivalenceGroup] = field(default_factory=list)
    declared_h1: list[str] = field(default_factory=list)
    would_dedup: bool = False
    deduplicated: bool = False


def dedup_sweep(
    configs: list[ExperimentConfig],
    *,
    rules: list[Rule] | tuple[Rule, ...] | None = None,
    loader: VendoredRulesLoader | None = None,
    deduplicate: bool = True,
) -> DedupResult:
    """Canonicalise then (optionally) H1-dedup ``configs``.

    Rules are resolved lazily: if ``rules`` is None the loader is consulted
    per-engine for each config (cached by the loader instance). Callers
    running homogeneous sweeps may pass ``rules`` directly to skip the
    loader hop.

    Args:
        configs: Sweep-expanded declared configs.
        rules: Optional explicit rule list. Overrides the loader when the
            sweep is single-engine and the caller has a rules handle.
        loader: Optional ``VendoredRulesLoader``. Defaults to a fresh one
            (per-process cache is internal to each instance).
        deduplicate: When ``False``, every declared config still runs —
            groups are computed for the equivalence-groups sidecar but the
            returned ``canonical_configs`` list has one entry per input.

    Returns:
        :class:`DedupResult` — see fields above.
    """
    if not configs:
        return DedupResult(canonical_configs=[])

    resolved_loader = loader or VendoredRulesLoader()
    rule_cache: dict[str, tuple[Rule, ...]] = {}

    def _rules_for(cfg: ExperimentConfig) -> tuple[Rule, ...]:
        if rules is not None:
            return tuple(rules)
        engine = cfg.engine.value if hasattr(cfg.engine, "value") else str(cfg.engine)
        cached = rule_cache.get(engine)
        if cached is not None:
            return cached
        try:
            loaded = resolved_loader.load_rules(engine).rules
        except FileNotFoundError:
            loaded = ()
        rule_cache[engine] = loaded
        return loaded

    canonicalised: list[ExperimentConfig] = []
    hashes: list[str] = []
    for cfg in configs:
        canon = canonicalise(cfg, _rules_for(cfg))
        canonicalised.append(canon)
        hashes.append(hash_config(build_h1_view(canon)))

    groups_by_hash: dict[str, list[int]] = {}
    for idx, h in enumerate(hashes):
        groups_by_hash.setdefault(h, []).append(idx)

    groups: list[EquivalenceGroup] = []
    for h1, indices in groups_by_hash.items():
        representative = indices[0]
        rep = canonicalised[representative]
        excerpt = _canonical_excerpt(rep)
        groups.append(
            EquivalenceGroup(
                h1_hash=h1,
                canonical_excerpt=excerpt,
                member_indices=tuple(indices),
                representative_index=representative,
            )
        )

    would_dedup = any(g.member_count > 1 for g in groups)

    if deduplicate:
        selected = [canonicalised[g.representative_index] for g in groups]
    else:
        selected = list(canonicalised)

    return DedupResult(
        canonical_configs=selected,
        groups=groups,
        declared_h1=hashes,
        would_dedup=would_dedup,
        deduplicated=deduplicate and would_dedup,
    )


def _canonical_excerpt(config: ExperimentConfig) -> dict[str, Any]:
    """Small human-readable excerpt of the canonical form for display/logs."""
    engine = config.engine.value if hasattr(config.engine, "value") else str(config.engine)
    excerpt: dict[str, Any] = {
        "engine": engine,
        "task.model": config.task.model,
    }
    section = getattr(config, engine, None)
    sampling = getattr(section, "sampling", None) if section is not None else None
    if sampling is not None:
        dumped = sampling.model_dump(mode="python", exclude_none=True)
        for key, value in dumped.items():
            excerpt[f"{engine}.sampling.{key}"] = value
    return excerpt
