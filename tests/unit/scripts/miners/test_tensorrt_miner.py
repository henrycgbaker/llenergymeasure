"""Tests for :mod:`scripts.miners.tensorrt_miner` — orchestrator surface.

The orchestrator is a thin wrapper around the static miner; it exists to
match the per-engine ``{engine}_miner.py`` shape that ``transformers_miner.py``
established and to keep the version-pin check (``check_installed_version``
against the source-tree's own ``__version__``) in one named place.

Coverage:

- TRT-LLM is registered in :data:`scripts.miners.build_corpus._ENGINE_EXTRACTORS`.
- The orchestrator pins against :data:`TESTED_AGAINST_VERSIONS` and refuses
  source trees whose version disagrees.
- The orchestrator never imports ``tensorrt_llm`` at module load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.miners import (  # noqa: E402
    build_corpus,
    tensorrt_miner,
    tensorrt_static_miner,
)
from scripts.miners._base import MinerVersionMismatchError  # noqa: E402


def test_tensorrt_engine_registered_in_build_corpus() -> None:
    """``--engine tensorrt`` resolves to the static miner extractor."""
    assert "tensorrt" in build_corpus._ENGINE_EXTRACTORS
    extractors = build_corpus._ENGINE_EXTRACTORS["tensorrt"]
    assert len(extractors) == 1
    assert extractors[0].module == "scripts.miners.tensorrt_static_miner"
    assert extractors[0].staging_basename == "tensorrt_static_miner.yaml"


def test_orchestrator_does_not_import_tensorrt_llm() -> None:
    """The orchestrator must NOT import ``tensorrt_llm`` at module load.

    The host has TRT-LLM 1.1.0 installed (a separate library generation
    that diverged significantly from 0.21.0). Importing it would silently
    mine drifted source. Source-driven AST walking is the load-bearing
    safety property; this test pins it.
    """
    # The miner module is already loaded by the import above; ``tensorrt_llm``
    # must therefore be absent from ``sys.modules``.
    # (Other test modules that DO import the live library may run before us;
    # in that case skip rather than failing on someone else's import.)
    if "tensorrt_llm" in sys.modules:
        pytest.skip(
            "tensorrt_llm has been imported by some other test module; "
            "this test only catches the case where the orchestrator itself imports it."
        )
    # Re-import the orchestrator and confirm no side effect.
    assert "tensorrt_llm" not in sys.modules


def test_orchestrator_main_rejects_version_mismatch(tmp_path: Path) -> None:
    """A 1.x source tree must be refused by the version-pin guard."""
    stub_root = tmp_path / "tensorrt_llm"
    (stub_root / "llmapi").mkdir(parents=True)
    # Minimal source tree with all required *classes* present so source-load
    # succeeds, but with a 1.x version stamp so the pin guard fires.
    (stub_root / "llmapi" / "llm_args.py").write_text(
        "\n".join(
            [
                "class BaseLlmArgs:",
                "    def validate_dtype(self): pass",
                "    def validate_model(self): pass",
                "    def validate_model_format_misc(self): pass",
                "    def set_runtime_knobs_from_build_config(self): pass",
                "    def validate_build_config_with_runtime_params(self): pass",
                "    def validate_build_config_remaining(self): pass",
                "    def validate_speculative_config(self): pass",
                "    def validate_lora_config_consistency(self): pass",
                "class TrtLlmArgs(BaseLlmArgs):",
                "    def validate_enable_build_cache(self): pass",
                "class LookaheadDecodingConfig:",
                "    def validate_positive_values(self): pass",
                "class CalibConfig: pass",
                "class BatchingType: pass",
                "class CapacitySchedulerPolicy: pass",
                "class ContextChunkingPolicy: pass",
            ]
        )
        + "\n"
    )
    (stub_root / "builder.py").write_text("class BuildConfig: pass\n")
    (stub_root / "version.py").write_text('__version__ = "1.1.0"\n')

    out = tmp_path / "trt.yaml"
    with pytest.raises(MinerVersionMismatchError):
        tensorrt_miner.main(["--out", str(out), "--source-root", str(stub_root)])


def test_orchestrator_imports_tested_against_versions() -> None:
    """The orchestrator must re-export the static miner's pin for clarity."""
    assert tensorrt_miner.TESTED_AGAINST_VERSIONS is tensorrt_static_miner.TESTED_AGAINST_VERSIONS
