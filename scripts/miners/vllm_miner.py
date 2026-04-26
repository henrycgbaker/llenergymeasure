"""vLLM validation-rules miner — landmark-verified orchestrator.

Composes :mod:`scripts.miners.vllm_static_miner` (AST walks) with
:mod:`scripts.miners.vllm_dynamic_miner` (sub-library lifts + cluster
probing) into a single corpus-shape YAML output. Mirrors
:mod:`scripts.miners.transformers_miner` structurally so the pipeline is
uniform per engine.

Landmark verification + version envelope guard happen inside the two
sub-miners; this orchestrator delegates and aggregates. Both sub-miners
share :data:`scripts.miners.vllm_static_miner.TESTED_AGAINST_VERSIONS` so
the version pin is single-sourced — drift the pin in one place and both
halves see it.

Usage::

    python -m scripts.miners.vllm_miner --out configs/validation_rules/vllm.yaml

This module is normally invoked via :mod:`scripts.miners.build_corpus`
(which writes per-miner staging YAMLs and merges + vendor-validates them
into the canonical corpus). The standalone ``--out`` path exists for ad-hoc
local development.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

# Project root on sys.path for both ``-m`` and direct invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners._base import RuleCandidate  # noqa: E402
from scripts.miners.vllm_dynamic_miner import walk_vllm_dynamic  # noqa: E402
from scripts.miners.vllm_static_miner import (  # noqa: E402
    TESTED_AGAINST_VERSIONS,
    walk_vllm_static,
)

# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: RuleCandidate) -> dict[str, Any]:
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "rule_under_test": c.rule_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "miner_source": {
            "path": c.miner_source.path,
            "method": c.miner_source.method,
            "line_at_scan": c.miner_source.line_at_scan,
        },
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
    """Return ``(candidates, envelope_metadata)`` for both halves combined.

    Both sub-miners run the version-envelope guard; if either fails, the
    miner exits non-zero and CI breaks.
    """
    static_candidates, version_static = walk_vllm_static()
    dynamic_candidates, version_dynamic = walk_vllm_dynamic()

    # Both halves see the same installed library; mismatches would be a
    # serious bug — fail-loud rather than paper over.
    if version_static != version_dynamic:
        raise RuntimeError(
            f"vLLM static / dynamic miners disagree on installed version: "
            f"{version_static!r} vs {version_dynamic!r}. Mid-flight library "
            f"swap?"
        )

    candidates = [*static_candidates, *dynamic_candidates]

    frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
    mined_at = (
        frozen
        if frozen
        else dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    envelope = {
        "schema_version": "1.0.0",
        "engine": "vllm",
        "engine_version": version_static,
        "walker_pinned_range": str(TESTED_AGAINST_VERSIONS),
        "mined_at": mined_at,
    }
    return candidates, envelope


def emit_yaml(candidates: list[RuleCandidate], envelope: dict[str, Any]) -> str:
    """Serialise candidates + envelope as deterministic YAML.

    Sort key matches the per-miner staging files: ``(method, id)``.
    """
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    doc = {
        "schema_version": envelope["schema_version"],
        "engine": envelope["engine"],
        "engine_version": envelope["engine_version"],
        "walker_pinned_range": envelope["walker_pinned_range"],
        "mined_at": envelope["mined_at"],
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
        f"Wrote {len(candidates)} vLLM rules to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
