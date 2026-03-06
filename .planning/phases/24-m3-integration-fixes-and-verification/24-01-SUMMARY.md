---
phase: 24-m3-integration-fixes-and-verification
plan: "01"
subsystem: preflight, api
tags: [fix, preflight, runner-resolution, gpu-memory, integration]
dependency_graph:
  requires: []
  provides:
    - "run_study_preflight forwards yaml_runners and user_config to resolve_study_runners"
    - "check_gpu_memory_residual called in _run_in_process single-experiment path"
  affects:
    - src/llenergymeasure/orchestration/preflight.py
    - src/llenergymeasure/_api.py
tech_stack:
  added: []
  patterns:
    - "TYPE_CHECKING guard for UserRunnersConfig type hint in preflight"
    - "lazy local import of check_gpu_memory_residual in _run_in_process"
key_files:
  created: []
  modified:
    - src/llenergymeasure/orchestration/preflight.py
    - src/llenergymeasure/_api.py
    - tests/unit/study/test_study_preflight.py
    - tests/unit/test_api.py
decisions:
  - "user_config loaded before run_study_preflight in _run() — ensures preflight and dispatch use identical runner resolution"
  - "check_gpu_memory_residual placed before Docker/local branch split in _run_in_process — covers both paths with one call"
metrics:
  duration: "~300s"
  completed: "2026-03-05"
  tasks_completed: 2
  files_modified: 4
requirements_closed: [DOCK-07, DOCK-08, DOCK-09, MEAS-01, MEAS-02]
---

# Phase 24 Plan 01: M3 Integration Fixes Summary

Two audit-flagged integration gaps closed: preflight runner resolution now mirrors actual dispatch, and the single-experiment path gains the same GPU memory residual check as multi-experiment StudyRunner.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix preflight runner resolution and update caller | 8e85db8 | preflight.py, _api.py, test_study_preflight.py |
| 2 | Add GPU memory check to _run_in_process | 864dc94 | _api.py, test_api.py |

## Changes Made

### Task 1: Preflight runner resolution (DOCK-07, DOCK-08, DOCK-09)

**Problem:** `run_study_preflight()` called `resolve_study_runners(list(backends))` with no `yaml_runners` or `user_config`, so runner resolution during preflight could diverge from actual dispatch resolution in `_api._run()`.

**Fix in `preflight.py`:**
- Added `yaml_runners: dict[str, str] | None = None` and `user_config: "UserRunnersConfig | None" = None` to `run_study_preflight()` signature
- Added `TYPE_CHECKING` guard for `UserRunnersConfig` type hint (avoids runtime circular import)
- Forwarded both params to `resolve_study_runners()`

**Fix in `_api.py`:**
- Moved `user_config = load_user_config()` to before the `run_study_preflight()` call
- Passed `yaml_runners=study.runners, user_config=user_config.runners` to `run_study_preflight()`
- Subsequent `resolve_study_runners()` call is unchanged — reuses the already-loaded `user_config`

**Tests added (`test_study_preflight.py`):**
- `test_preflight_forwards_runner_context` — verifies `resolve_study_runners` receives `yaml_runners` and `user_config` when passed
- `test_preflight_defaults_to_auto_detect_without_context` — verifies `None` defaults when called without runner context

### Task 2: GPU memory check in single-experiment path (MEAS-01, MEAS-02)

**Problem:** `_run_in_process()` did not call `check_gpu_memory_residual()` before running an experiment, unlike the multi-experiment `StudyRunner._run_one()` and `_run_one_docker()` paths.

**Fix in `_api.py`:**
- Inserted `check_gpu_memory_residual()` call after `cycle = 1` and before the Docker/local branch split
- Placed before `spec = runner_specs.get(...)` so it covers both Docker and local dispatch paths
- Uses a lazy local import to keep module-level deps minimal (pynvml not always installed)

**Test added (`test_api.py`):**
- `test_run_in_process_calls_gpu_memory_check` — mocks `check_gpu_memory_residual` and asserts it is called exactly once during `_run_in_process()`

## Verification

```
grep -n "yaml_runners" src/llenergymeasure/orchestration/preflight.py
# 160:    yaml_runners: dict[str, str] | None = None,
# 209:        list(backends), yaml_runners=yaml_runners, user_config=user_config

grep -n "check_gpu_memory_residual" src/llenergymeasure/_api.py
# 287:    from llenergymeasure.study.gpu_memory import check_gpu_memory_residual
# 289:    check_gpu_memory_residual()

pytest tests/unit/study/test_study_preflight.py tests/unit/test_api.py
# 31 passed

pytest tests/unit/
# 881 passed
```

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- `src/llenergymeasure/orchestration/preflight.py` — FOUND (yaml_runners parameter added)
- `src/llenergymeasure/_api.py` — FOUND (check_gpu_memory_residual call added)
- `tests/unit/study/test_study_preflight.py` — FOUND (2 new tests)
- `tests/unit/test_api.py` — FOUND (1 new test)
- Commit 8e85db8 — FOUND
- Commit 864dc94 — FOUND
