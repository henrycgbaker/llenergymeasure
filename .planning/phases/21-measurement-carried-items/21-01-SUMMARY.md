---
phase: 21-measurement-carried-items
plan: 01
subsystem: datasets
tags: [datasets, prompts, aienergyscore, carried-items, MEAS-03]
dependency_graph:
  requires:
    - src/llenergymeasure/config/models.py (ExperimentConfig)
    - src/llenergymeasure/core/backends/pytorch.py
    - src/llenergymeasure/core/backends/vllm.py
    - src/llenergymeasure/core/dataset_loader.py (broken import fix)
  provides:
    - src/llenergymeasure/datasets/__init__.py (public API: aienergyscore, load_prompts, BUILTIN_DATASETS)
    - src/llenergymeasure/datasets/loader.py (JSONL loading, registry, synthetic generation)
    - src/llenergymeasure/datasets/builtin/aienergyscore.jsonl (1,000 prompts)
  affects:
    - Both backends now load real prompts instead of M1 Hello placeholder
    - ExperimentConfig gains dataset_order field
tech_stack:
  added:
    - datasets (HuggingFace) used in generation script
  patterns:
    - JSONL with provenance header line pattern
    - Type-dispatcher pattern in load_prompts() (SyntheticDatasetConfig vs str)
    - Lazy import inside load_prompts() to avoid circular dependency
key_files:
  created:
    - src/llenergymeasure/datasets/__init__.py
    - src/llenergymeasure/datasets/loader.py
    - src/llenergymeasure/datasets/builtin/__init__.py
    - src/llenergymeasure/datasets/builtin/aienergyscore.jsonl
    - scripts/generate_aienergyscore_dataset.py
    - tests/unit/test_datasets.py
  modified:
    - src/llenergymeasure/config/models.py (dataset_order field added)
    - src/llenergymeasure/core/dataset_loader.py (broken import fixed, source types defined)
    - src/llenergymeasure/core/backends/pytorch.py (_prepare_prompts uses load_prompts)
    - src/llenergymeasure/core/backends/vllm.py (_prepare_prompts uses load_prompts)
    - tests/unit/test_vllm_backend.py (updated stale M1 placeholder test)
decisions:
  - AUTO_DETECT_COLUMNS and BUILTIN_DATASETS live in datasets/loader.py (not config/models)
  - FilePromptSource, HuggingFacePromptSource defined as dataclasses in dataset_loader.py (not config.models — they were never there)
  - AIEnergyScore/text_generation has single text column, no source column; grouped ordering falls back to file order
  - Dataset commit SHA captured via huggingface_hub.dataset_info() at generation time
metrics:
  duration: 462s
  completed: "2026-03-03"
  tasks_completed: 2
  files_created: 6
  files_modified: 5
---

# Phase 21 Plan 01: AIEnergyScore Dataset Module Summary

**One-liner:** Bundled 1,000-prompt aienergyscore.jsonl dataset with interleaved/grouped/shuffled ordering, wired to both inference backends, replacing M1 Hello-placeholder prompts.

## What Was Built

Created `llenergymeasure.datasets` — a new subpackage that closes the MEAS-03 carry-forward from M1. The module provides:

1. **Bundled JSONL file** (`datasets/builtin/aienergyscore.jsonl`): 1,001 lines — one provenance header followed by 1,000 text prompts downloaded from `AIEnergyScore/text_generation` (commit `2dc92b2ee2cd9776a51ccf08c6c2ab04138370c3`, Apache 2.0).

2. **Loader module** (`datasets/loader.py`): `load_prompts(config)` dispatches on `config.dataset` type — `SyntheticDatasetConfig` generates deterministic synthetic prompts, a built-in alias loads from the JSONL registry, and a `.jsonl` file path loads directly. Three ordering modes: `interleaved` (file order), `grouped` (stable sort by source field), `shuffled` (seeded random permutation).

3. **Public API** (`datasets/__init__.py`): `from llenergymeasure.datasets import aienergyscore` returns a `Path`; `load_prompts` and `BUILTIN_DATASETS` also exported.

4. **Config field** (`ExperimentConfig.dataset_order`): `Literal["interleaved", "grouped", "shuffled"]`, default `"interleaved"`. Validated by Pydantic.

5. **Backend wiring**: Both `PyTorchBackend._prepare_prompts` and `VLLMBackend._prepare_prompts` now call `load_prompts(config)` — real prompts replace the "Hello, " * N placeholder.

6. **Broken import fix** (`core/dataset_loader.py`): The file was importing `BUILTIN_DATASETS`, `AUTO_DETECT_COLUMNS`, `FilePromptSource`, `HuggingFacePromptSource`, and `PromptSourceConfig` from `config.models` where they never existed. Fixed by importing `BUILTIN_DATASETS`/`AUTO_DETECT_COLUMNS` from `datasets.loader` and defining the source types as dataclasses in `dataset_loader.py`.

## Verification Results

- `from llenergymeasure.datasets import aienergyscore; aienergyscore.exists()` — True
- 1 provenance header + 1,000 prompt lines in JSONL
- `load_prompts(ExperimentConfig(model='x', n=10))` returns 10 real WikiText prompts
- `ExperimentConfig(model='x').dataset_order == 'interleaved'` — confirmed
- `from llenergymeasure.core.dataset_loader import load_prompts_from_source` — no ImportError
- `pytest tests/unit/test_datasets.py` — 10/10 passed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `FilePromptSource`, `HuggingFacePromptSource`, `PromptSourceConfig` do not exist in config.models**
- **Found during:** Task 2, fixing the broken import in `core/dataset_loader.py`
- **Issue:** The plan said to remove `BUILTIN_DATASETS` and `AUTO_DETECT_COLUMNS` from the import and import them from `datasets.loader` instead — but `FilePromptSource`, `HuggingFacePromptSource`, and `PromptSourceConfig` were also in the broken import and don't exist anywhere in the codebase.
- **Fix:** Defined `FilePromptSource` and `HuggingFacePromptSource` as `@dataclass` classes inside `dataset_loader.py`. Set `PromptSourceConfig = FilePromptSource | HuggingFacePromptSource` as a type alias. This makes the module self-contained and correct.
- **Files modified:** `src/llenergymeasure/core/dataset_loader.py`
- **Commit:** `037e5c3`

**2. [Rule 1 - Bug] `test_vllm_backend.py::TestPreparePrompts::test_prompts_reflect_max_input_tokens` failed after backend wiring**
- **Found during:** Task 2 full test run
- **Issue:** The test asserted that longer `max_input_tokens` produced longer prompt strings — this was testing the M1 placeholder behaviour. After replacing the placeholder with `load_prompts()`, prompts are real text of fixed length regardless of `max_input_tokens`.
- **Fix:** Renamed and rewrote the test as `test_prompts_are_real_text` — asserts prompts are non-empty strings that are not the old placeholder pattern.
- **Files modified:** `tests/unit/test_vllm_backend.py`
- **Commit:** `037e5c3`

### Pre-existing Failures (Not Fixed — Out of Scope)

Five tests in `test_measurement_integration.py` were already failing before this plan (`TypeError: PyTorchBackend._build_result() missing 1 required positional argument: 'model_memory_mb'`). Confirmed pre-existing via `git stash` test run. Not caused by or related to this plan's changes.

## Self-Check: PASSED

All created files exist on disk. Both commit hashes verified in git log.

| Check | Result |
|---|---|
| `src/llenergymeasure/datasets/__init__.py` | FOUND |
| `src/llenergymeasure/datasets/loader.py` | FOUND |
| `src/llenergymeasure/datasets/builtin/__init__.py` | FOUND |
| `src/llenergymeasure/datasets/builtin/aienergyscore.jsonl` | FOUND |
| `scripts/generate_aienergyscore_dataset.py` | FOUND |
| `tests/unit/test_datasets.py` | FOUND |
| Commit `950e782` | FOUND |
| Commit `037e5c3` | FOUND |
