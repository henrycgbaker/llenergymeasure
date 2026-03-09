---
phase: 27-deduplication-and-circular-import
plan: 02
subsystem: core
tags: [refactor, backends, harness, deduplication, protocol]
dependency_graph:
  requires: ["27-01"]
  provides: ["MeasurementHarness", "BackendPlugin", "InferenceOutput", "thermal_floor_wait"]
  affects: ["_api.py", "study/runner.py", "infra/container_entrypoint.py"]
tech_stack:
  added: []
  patterns:
    - "BackendPlugin 4-method protocol: load_model, warmup, run_inference, cleanup"
    - "MeasurementHarness orchestrates full lifecycle - backends are thin plugins"
    - "Module-level re-exports in harness.py to enable clean test patching"
    - "InferenceOutput.extras dict for backend-specific data (e.g. hf_model for FLOPs)"
key_files:
  created:
    - src/llenergymeasure/core/harness.py
    - tests/unit/core/test_harness.py
  modified:
    - src/llenergymeasure/core/backends/protocol.py
    - src/llenergymeasure/core/backends/pytorch.py
    - src/llenergymeasure/core/backends/vllm.py
    - src/llenergymeasure/core/backends/__init__.py
    - src/llenergymeasure/core/warmup.py
    - src/llenergymeasure/_api.py
    - src/llenergymeasure/study/runner.py
    - src/llenergymeasure/infra/container_entrypoint.py
    - tests/unit/test_api.py
    - tests/unit/test_peak_memory.py
    - tests/unit/backends/test_backend_protocol.py
    - tests/unit/backends/test_vllm_backend.py
    - tests/unit/core/test_measurement_integration.py
    - tests/unit/docker/test_container_entrypoint.py
    - tests/unit/study/test_study_runner.py
decisions:
  - "hf_model for FLOPs estimation passed via InferenceOutput.extras['hf_model'] - no coupling between harness and pytorch-specific types"
  - "Module-level re-export wrapper functions in harness.py enable test patching of collect_environment_snapshot, PowerThermalSampler etc. without importing them at module level"
  - "InferenceBackend protocol kept in protocol.py for backward compat with tests/fakes.py FakeInferenceBackend - to be removed in Plan 03"
  - "model_memory_mb captured by harness._capture_model_memory_mb() after load_model(), not by backend - backends always return 0.0 for model_memory_mb in InferenceOutput"
  - "thermal_floor_wait_s set by harness on WarmupResult after backend.warmup() returns - backends set it to 0.0"
metrics:
  duration_minutes: 90
  tasks_completed: 3
  files_changed: 15
  completed_date: "2026-03-07"
---

# Phase 27 Plan 02: MeasurementHarness Extraction Summary

Extracted ~600 lines of duplicated measurement infrastructure from PyTorchBackend and VLLMBackend into a single MeasurementHarness. Both backends are now thin BackendPlugin implementations. Adding a fourth backend (TensorRT in M4) is now a ~200-line task.

## Commits

| Task | Commit | Description |
| ---- | ------ | ----------- |
| 1 | `9db50f8` | refactor(backends): add BackendPlugin protocol and InferenceOutput dataclass |
| 2 | `0f7ab9e` | feat(harness): introduce MeasurementHarness with thermal_floor_wait |
| 3 | `ffaaa86` | refactor(backends): shrink PyTorch and vLLM backends to thin plugins |

## What Was Done

### Task 1: Protocol and test scaffold
- Added `InferenceOutput` dataclass to `protocol.py` (7 fields + `total_tokens` property)
- Added `BackendPlugin` Protocol (4 methods: `load_model`, `warmup`, `run_inference`, `cleanup`)
- Kept `InferenceBackend` for backward compat with `tests/fakes.py`
- Created `tests/unit/core/test_harness.py` with 5 lifecycle tests + `FakeBackend` dataclass

### Task 2: MeasurementHarness and thermal_floor_wait
- Added `thermal_floor_wait()` to `core/warmup.py` - single canonical implementation
- Created `core/harness.py` (460 lines) with:
  - `MeasurementHarness.run(backend, config)` - full 17-step lifecycle
  - `_build_result()` - assembles `ExperimentResult` from `InferenceOutput`
  - `_capture_model_memory_mb()` - GPU memory after model load, before warmup
  - `_estimate_flops()` - reads `hf_model` from `output.extras`
  - `_collect_warnings()` - delegates to `collect_measurement_warnings()`
  - Module-level wrapper functions for patchable test hooks

### Task 3: Thin plugins and wiring
- Rewrote `PyTorchBackend`: 863 lines → 444 lines (thin plugin only)
- Rewrote `VLLMBackend`: 871 lines → 437 lines (thin plugin only)
- Removed from both backends: `run()`, `_cuda_sync()`, `_check_persistence_mode()`, `_collect_warnings()`, `_build_result()`, `_MeasurementData`/`_VLLMMeasurementData`
- Wired `_api.py`, `study/runner.py`, `container_entrypoint.py` to `harness.run(backend, config)`
- Updated 7 test files to patch `MeasurementHarness.run` instead of `backend.run`

## Verification

```
grep -rn "backend\.run(" src/ --include="*.py"  → ZERO results
pytest tests/unit/ -x -q                         → 877 passed
```

### Line counts before / after

| File | Before | After | Change |
| ---- | ------ | ----- | ------ |
| `backends/pytorch.py` | ~863 | 444 | -419 |
| `backends/vllm.py` | ~871 | 437 | -434 |
| `core/harness.py` | 0 | 460 | +460 |
| `core/warmup.py` | ~160 | 199 | +39 |

Net: -353 lines (duplication removed, new harness added)

### Deferred items for Phase 28 planner

- `InferenceBackend` protocol is now dead code in `protocol.py` - kept for `tests/fakes.py::FakeInferenceBackend` backward compat. Remove in Plan 03 or later.
- `test_peak_memory.py` source-inspection tests were updated to point at `harness.py._build_result` and `run_inference` (not `_run_measurement` which no longer exists). Both test the correct invariants post-refactor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SIM105 linter fix in vllm.py `run_inference`**
- **Found during:** Task 3 commit (pre-commit hook)
- **Issue:** `try: ... except Exception: pass` should use `contextlib.suppress(Exception)` per SIM105
- **Fix:** Pre-commit hook auto-reformatted the bare `try/except/pass` for `hf_model` extraction to `contextlib.suppress`
- **Files modified:** `src/llenergymeasure/core/backends/vllm.py`
- **Commit:** `ffaaa86` (included in Task 3 commit)

**2. [Rule 2 - Missing] test_container_entrypoint.py needed harness patch**
- **Found during:** Task 3 verification (1 failing test)
- **Issue:** `test_writes_result_json_with_config_hash_name` used `mock_backend.run.return_value` pattern but container_entrypoint now calls `harness.run(backend, config)`
- **Fix:** Added `_PATCH_HARNESS_RUN = "llenergymeasure.core.harness.MeasurementHarness.run"` and patched all 8 tests in `TestRunContainerExperiment` + `TestMainErrorHandling`
- **Files modified:** `tests/unit/docker/test_container_entrypoint.py`

**3. [Rule 2 - Missing] test_study_runner.py worker test needed harness patch**
- **Found during:** Task 3 full suite run (1 additional failing test)
- **Issue:** `test_worker_calls_get_backend` used `mock_backend.run.return_value` pattern
- **Fix:** Added `monkeypatch.setattr("llenergymeasure.core.harness.MeasurementHarness.run", ...)` to patch harness; switched to `make_result()` to avoid MagicMock type issues
- **Files modified:** `tests/unit/study/test_study_runner.py`

**4. [Rule 1 - Bug] test_peak_memory.py source-inspection tests used removed method names**
- **Found during:** Task 3 full suite run (4 failing tests)
- **Issue:** Tests inspected `_run_measurement` and `_build_result` in `pytorch.py`/`vllm.py` - both methods removed during refactor. Tests also had ordering issue with comment containing `max_memory_allocated()` text before actual call
- **Fix:** Updated tests to inspect `run_inference` in backends and `_build_result` in `harness.py`; used `= torch.cuda.max_memory_allocated()` assignment pattern to distinguish from comment text
- **Files modified:** `tests/unit/test_peak_memory.py`

## Self-Check: PASSED

All key files exist and all commits are verified:
- `src/llenergymeasure/core/harness.py` - FOUND
- `src/llenergymeasure/core/backends/protocol.py` - FOUND
- `src/llenergymeasure/core/warmup.py` - FOUND
- `tests/unit/core/test_harness.py` - FOUND
- `.planning/phases/27-deduplication-and-circular-import/27-02-SUMMARY.md` - FOUND
- commit `9db50f8` (Task 1) - FOUND
- commit `0f7ab9e` (Task 2) - FOUND
- commit `ffaaa86` (Task 3) - FOUND
- `grep -rn "backend.run(" src/` - ZERO results confirmed
- `pytest tests/unit/` - 877 passed, 0 failed confirmed
