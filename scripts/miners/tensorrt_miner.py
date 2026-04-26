"""TensorRT-LLM miner orchestrator — single-stage (static-only).

Per the locked invariant-miner design (decision #8 of the adversarial
review, 2026-04-26), TRT-LLM ships **without** a dynamic miner: probing
``TrtLlmArgs(...)`` with combinatorial inputs yields near-zero raises
because the constructor is permissive — all real cross-field validity
rules fire later at engine compile time, inside C++ ``Builder.build_engine``.
The static miner alone covers the surface that matters at config-validation
time.

Pipeline this orchestrator drives:

1. Verify the 0.21.0 source tree is present (canonical location:
   ``/tmp/trt-llm-0.21.0/tensorrt_llm/``). The TRT-LLM library is pinned at
   0.21.0 (CUDA 12.6.x); v1.x requires CUDA 13 and is a separate
   infrastructure milestone.
2. Read ``tensorrt_llm/version.py`` from the source tree (no import) and
   pin against :data:`TESTED_AGAINST_VERSIONS`.
3. Run :mod:`scripts.miners.tensorrt_static_miner` and emit the staging
   YAML.

This orchestrator never imports ``tensorrt_llm``. The host has 1.1.0
installed and importing it would mine the wrong source — exactly the
silent-degradation failure mode that the Haiku-era extractor PRs (#415,
#416, #417, all reverted in #423) tripped on. AST-walk over a known
extracted source tree is the only safe path.

Usage::

    PYTHONPATH=. python3 scripts/miners/tensorrt_miner.py --out path/to/tensorrt.yaml

Or via the canonical corpus builder::

    PYTHONPATH=. python3 scripts/miners/build_corpus.py --engine tensorrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE: this script's parent dir contains a sibling ``transformers_*.py``
# that would shadow the real ``transformers`` package on import. Strip the
# script directory before any third-party imports — same defensive measure
# as the static miner.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_SCRIPT_DIR).resolve()]
sys.path[:] = [p for p in sys.path if p != ""]

from scripts.miners._base import (  # noqa: E402  (late import after sys.path)
    check_installed_version,
)
from scripts.miners.tensorrt_static_miner import (  # noqa: E402
    _DEFAULT_SOURCE_ROOT,
    TESTED_AGAINST_VERSIONS,
    emit_yaml,
    walk_tensorrt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Write extracted YAML to this path.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_DEFAULT_SOURCE_ROOT,
        help=(
            "Path to the extracted tensorrt_llm 0.21.0 source tree (default: "
            f"{_DEFAULT_SOURCE_ROOT})"
        ),
    )
    args = parser.parse_args(argv)

    candidates, source_version, rel_path = walk_tensorrt(args.source_root)
    # Pin the source-tree version against TESTED_AGAINST_VERSIONS — any drift
    # (e.g. someone pointed --source-root at a 1.x checkout) becomes a fatal
    # MinerVersionMismatchError instead of silently emitting drifted rules.
    check_installed_version("tensorrt_llm", source_version, TESTED_AGAINST_VERSIONS)

    text = emit_yaml(candidates, engine_version=source_version, rel_path=rel_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} tensorrt_llm rules to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
