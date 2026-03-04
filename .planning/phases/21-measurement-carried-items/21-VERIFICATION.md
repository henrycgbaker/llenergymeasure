---
phase: 21-measurement-carried-items
verified: 2026-03-03T21:30:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 21: Measurement Carried Items Verification Report

**Phase Goal:** Close M1 measurement carry-forwards: create aienergyscore.jsonl built-in dataset (MEAS-03) and confirm peak_memory_mb semantics (MEAS-04)
**Verified:** 2026-03-03T21:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `from llenergymeasure.datasets import aienergyscore` loads a Path to the built-in JSONL file without error | VERIFIED | `datasets/__init__.py` imports from `datasets/loader.py` and assigns `BUILTIN_DATASETS["aienergyscore"]`; file exists at `datasets/builtin/aienergyscore.jsonl` |
| 2 | The JSONL file contains 1,000 prompts from AIEnergyScore/text_generation with provenance metadata | VERIFIED | `wc -l` = 1001 lines; first line is `{"_provenance": "AIEnergyScore/text_generation", "_commit": "2dc92b2ee2...", "_license": "apache-2.0", ...}`; 1,000 prompt lines follow |
| 3 | Both PyTorchBackend and VLLMBackend load real prompts from the dataset instead of M1 placeholder prompts | VERIFIED | `pytorch.py:354-356` calls `load_prompts(config)`; `vllm.py:412-414` calls `load_prompts(config)`; no "Hello, " * N pattern in either `_prepare_prompts` method |
| 4 | `core/dataset_loader.py` imports successfully (broken BUILTIN_DATASETS/AUTO_DETECT_COLUMNS import fixed) | VERIFIED | `dataset_loader.py:19` now imports `AUTO_DETECT_COLUMNS, BUILTIN_DATASETS` from `llenergymeasure.datasets.loader`; `test_dataset_loader_importable` passes |
| 5 | `ExperimentConfig.dataset_order` field exists with `Literal['interleaved', 'grouped', 'shuffled']` type defaulting to `'interleaved'` | VERIFIED | `config/models.py:304-309`: field defined with correct Literal type and default |
| 6 | `peak_memory_mb` captures inference-window-only memory; `reset_peak_memory_stats()` called after warmup and before the measurement loop in PyTorchBackend | VERIFIED | `pytorch.py:448-453`: reset called at start of `_run_measurement()`, before `PowerThermalSampler` context manager; comment states intent |
| 7 | `peak_memory_mb` field descriptions document the precise measurement semantics | VERIFIED | `metrics.py:383-387`: "inference measurement window (MB)... NOT model weights. 0.0 = not measured." |
| 8 | `inference_memory_mb` derived field exists (peak minus model baseline) for both backends | VERIFIED | `metrics.py:449-458`: `MemoryEfficiencyMetrics.inference_memory_mb` field with `default=0.0`; computed by both backends in `_build_result()` |
| 9 | `_build_result()` computes `inference_memory_mb = max(0.0, peak_memory_mb - model_memory_mb)` and sets it on MemoryEfficiencyMetrics | VERIFIED | `pytorch.py:771`: `inference_memory_mb = max(0.0, data.peak_memory_mb - model_memory_mb)` then passed to MemoryEfficiencyMetrics; `vllm.py:706`: identical pattern |
| 10 | vLLM backend captures `peak_memory_mb` with torch-first, NVML-fallback strategy | VERIFIED | `vllm.py:456,490,512-518,544-556`: `reset_peak_memory_stats()` then `max_memory_allocated()`, with heuristic fallback to `_nvml_peak_memory_mb()` |
| 11 | No regressions in pre-existing passing tests | VERIFIED | 750 unit tests pass; 7 failures (`test_measurement_integration` x5, `test_backend_protocol` x2) are pre-existing — neither test file was modified in Phase 21 and failures reproduce without Phase 21 changes |

**Score:** 11/11 truths verified

---

## Required Artifacts

### Plan 01 (MEAS-03)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/datasets/__init__.py` | Public API exporting `aienergyscore`, `load_prompts`, `BUILTIN_DATASETS` | VERIFIED | All three exported via `__all__`; imports from `datasets.loader` |
| `src/llenergymeasure/datasets/loader.py` | JSONL loading, built-in dataset registry, synthetic generation | VERIFIED | 214 lines; `BUILTIN_DATASETS`, `AUTO_DETECT_COLUMNS`, `load_prompts`, `_load_jsonl`, `_load_synthetic` all present |
| `src/llenergymeasure/datasets/builtin/aienergyscore.jsonl` | 1,000 prompts from AIEnergyScore/text_generation | VERIFIED | 1,001 lines (1 provenance header + 1,000 prompts); Apache 2.0; commit SHA recorded |
| `src/llenergymeasure/config/models.py` | `ExperimentConfig` with `dataset_order` field | VERIFIED | Field at line 304 with `Literal["interleaved", "grouped", "shuffled"]`, default `"interleaved"` |
| `tests/unit/test_datasets.py` | Unit tests for dataset module | VERIFIED | 225 lines; 10 tests covering path, count, provenance, load_prompts dispatch, ordering, validation |

### Plan 02 (MEAS-04)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/core/backends/pytorch.py` | Corrected peak memory measurement with reset before measurement loop | VERIFIED | `reset_peak_memory_stats()` at line 453, before `PowerThermalSampler` at line 465 |
| `src/llenergymeasure/core/backends/vllm.py` | vLLM peak memory capture with torch-first, NVML fallback | VERIFIED | `max_memory_allocated()` at line 490; NVML fallback at lines 512-518; `_nvml_peak_memory_mb()` helper at lines 544-556 |
| `src/llenergymeasure/domain/metrics.py` | Updated field docstrings with precise measurement semantics | VERIFIED | "inference measurement window" at line 383; "NOT model weights" at line 386; `inference_memory_mb` field at line 449 |
| `tests/unit/test_peak_memory.py` | Tests verifying memory measurement semantics | VERIFIED | 198 lines; 8 tests using source-file inspection for GPU-free verification |

---

## Key Link Verification

### Plan 01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `datasets/__init__.py` | `datasets/loader.py` | `from llenergymeasure.datasets.loader import` | WIRED | Line 16: `from llenergymeasure.datasets.loader import BUILTIN_DATASETS, load_prompts` |
| `core/backends/pytorch.py` | `datasets/loader.py` | `_prepare_prompts` calls `load_prompts` | WIRED | Lines 354-356: lazy import + call inside `_prepare_prompts` |
| `core/backends/vllm.py` | `datasets/loader.py` | `_prepare_prompts` calls `load_prompts` | WIRED | Lines 412-414: lazy import + call inside `_prepare_prompts` |
| `datasets/loader.py` | `config/models.py` | `load_prompts` reads `config.dataset_order` | WIRED | Lines 73, 84: `order=config.dataset_order` passed to `_load_jsonl` |

### Plan 02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pytorch.py` | `torch.cuda.reset_peak_memory_stats` | Reset call before measurement loop | WIRED | Line 453: before `PowerThermalSampler` at line 465 |
| `pytorch.py` | `torch.cuda.max_memory_allocated` | Peak read after measurement loop | WIRED | Line 501: after batch loop |
| `pytorch.py` | `MemoryEfficiencyMetrics` | `_build_result` computes `inference_memory_mb = max(0.0, peak - model)` | WIRED | Line 771: `inference_memory_mb = max(0.0, data.peak_memory_mb - model_memory_mb)`; assigned to MemoryEfficiencyMetrics at line 782 |
| `vllm.py` | `MemoryEfficiencyMetrics` | `_build_result` computes `inference_memory_mb = max(0.0, peak - model)` | WIRED | Line 706: same derivation; assigned to MemoryEfficiencyMetrics at line 717 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| MEAS-03 | 21-01-PLAN.md | `aienergyscore.jsonl` built-in dataset file created | SATISFIED | JSONL file at `datasets/builtin/aienergyscore.jsonl` (1001 lines); public API at `llenergymeasure.datasets`; backends wired; 10 tests pass |
| MEAS-04 | 21-02-PLAN.md | `peak_memory_mb` measurement semantics confirmed and documented | SATISFIED | `reset_peak_memory_stats()` before measurement loop in both backends; `inference_memory_mb` derived field added; field descriptions document semantics; `result-schema.md` todo removed; 8 tests pass |

No orphaned requirements: REQUIREMENTS.md maps only MEAS-03 and MEAS-04 to Phase 21 (lines 107-108), matching the plans.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/llenergymeasure/core/backends/vllm.py` | 116 | Stale comment `# M1 placeholder — same pattern as PyTorchBackend` before `self._prepare_prompts(config)` call | Info | The comment refers to the old placeholder; implementation is now correct (calls `load_prompts`). No functional impact — comment cleanup only |

No blockers or substantive warnings found.

---

## Human Verification Required

### 1. Live GPU: actual inference memory delta

**Test:** Run an experiment with `dataset="aienergyscore"`, `n=10` on a CUDA GPU. Inspect the result's `extended_metrics.memory.peak_memory_mb`, `model_memory_mb`, and `inference_memory_mb`.
**Expected:** `inference_memory_mb = max(0.0, peak_memory_mb - model_memory_mb)` and `peak_memory_mb` reflects inference-window KV cache + activations, not total GPU memory.
**Why human:** CUDA not available on host; source-inspection tests confirm the reset is in place but cannot confirm the actual MB values are sensible.

### 2. vLLM NVML fallback

**Test:** Run vLLM backend on GPU where `gpu_memory_utilization` setting triggers the NVML fallback heuristic (peak within 5% of pre-allocation ceiling).
**Expected:** `peak_memory_mb` reflects NVML current usage, not the pre-allocation ceiling; value is meaningfully lower than `total_vram * gpu_memory_utilization`.
**Why human:** Heuristic logic can only be exercised on live GPU with vLLM pre-allocating memory.

---

## Gaps Summary

No gaps. All 11 observable truths verified, all artifacts substantive and wired, all key links confirmed, MEAS-03 and MEAS-04 satisfied.

The 7 pre-existing test failures in `test_measurement_integration.py` and `test_backend_protocol.py` are not caused by Phase 21 changes — neither file was modified in this phase, and the failures reproduce on the base commit.

The stale comment at `vllm.py:116` is cosmetic only — the implementation is correct.

---

_Verified: 2026-03-03T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
