---
phase: 27-deduplication-and-circular-import
verified: 2026-03-07T14:00:00Z
status: passed
score: 12/12 must-haves verified
gaps: []
---

# Phase 27: Deduplication and Circular Import Verification Report

**Phase Goal:** Deduplicate code and break circular imports identified in the codebase audit. Break config/loader -> study/grid circular import, extract MeasurementHarness from duplicated backend code, consolidate nvml_context, DockerError base init, NVIDIA toolkit binary list, and save_and_record helper.
**Verified:** 2026-03-07
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

Truths are drawn from the must_haves across all three plan frontmatters.

**Plan 01 — Circular import and dict utils consolidation**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | config/loader.py imports from config.grid (not study.grid) - no circular dependency | VERIFIED | loader.py line 23: `from llenergymeasure.config.grid import (...)` |
| 2 | A single _unflatten() and _deep_merge() exist only in config/_dict_utils.py | VERIFIED | `grep -rn "def _unflatten\|def _deep_merge" src/` returns only `config/_dict_utils.py:13` and `:33` |
| 3 | study/grid.py is gone - all consumers import from config.grid | VERIFIED | `ls src/llenergymeasure/study/grid.py` → No such file. Zero study.grid refs in src/ or tests/ |
| 4 | All existing tests pass with zero changes to test logic | VERIFIED | 877 passed in 12.17s |

**Plan 02 — MeasurementHarness extraction**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 5 | MeasurementHarness owns the full measurement lifecycle - backends are thin plugins | VERIFIED | harness.py 461 lines with full 17-step lifecycle; pytorch.py 444 lines, vllm.py 438 lines (down from ~863/871) |
| 6 | PyTorchBackend and VLLMBackend each implement 4 methods: load_model, warmup, run_inference, cleanup | VERIFIED | Both backends define all 4 methods at expected line numbers |
| 7 | _api.py, study/runner.py, and container_entrypoint.py call harness.run(backend, config) - not backend.run(config) | VERIFIED | All three files confirmed: `harness = MeasurementHarness(); result = harness.run(backend, config)`. Zero `backend.run(` calls in src/ |
| 8 | thermal_floor_wait() lives once in core/warmup.py - not duplicated in both backend files | VERIFIED | warmup.py line 179: `def thermal_floor_wait`. Backends contain only docstring comments about harness setting it; no implementation |
| 9 | ExperimentResult output is identical to before the refactor (behaviour-preserving) | VERIFIED | 877 tests pass including all existing API/integration tests |

**Plan 03 — nvml_context, DockerError, NVIDIA bins, _save_and_record**

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 10 | nvml_context() context manager exists in core/gpu_info.py and is used at all nvmlInit/nvmlShutdown sites | VERIFIED | gpu_info.py line 18: `def nvml_context`. grep for raw pynvml.nvmlInit/nvmlShutdown finds only the two lines inside nvml_context itself |
| 11 | DockerError base class has __init__ in exceptions.py - all 6 subclasses in docker_errors.py have no __init__ | VERIFIED | exceptions.py line 31: `def __init__(self, message, fix_suggestion, stderr_snippet)`. docker_errors.py: zero `def __init__` lines |
| 12 | _NVIDIA_TOOLKIT_BINS is defined once in docker_preflight.py and imported by runner_resolution.py | VERIFIED | docker_preflight.py line 45: `_NVIDIA_TOOLKIT_BINS = (...)`. runner_resolution.py line 31: `from llenergymeasure.infra.docker_preflight import _NVIDIA_TOOLKIT_BINS` |
| 13 | _save_and_record() helper is defined once and replaces 3 duplicated save+mark_completed blocks | VERIFIED | runner.py line 48: `def _save_and_record`. Called at runner.py:451, runner.py:523, and _api.py:348 |

**Score: 13/13 truths verified** (12 from must_haves + 1 implicit from Plan 02)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/config/_dict_utils.py` | Canonical _unflatten() and _deep_merge() | VERIFIED | Exists, exports both functions at lines 13 and 33 |
| `src/llenergymeasure/config/grid.py` | Grid expansion moved from study/grid.py, imports from _dict_utils | VERIFIED | Exists with all 6 expected symbols; line 18: `from llenergymeasure.config._dict_utils import _deep_merge, _unflatten` |
| `src/llenergymeasure/study/grid.py` | Deleted - no tombstone | VERIFIED | File does not exist |
| `src/llenergymeasure/core/harness.py` | MeasurementHarness class, 150+ lines | VERIFIED | 461 lines; class at line 143, `run()` at line 152, 5 private methods |
| `src/llenergymeasure/core/backends/protocol.py` | BackendPlugin, InferenceOutput, InferenceBackend | VERIFIED | All three classes present at lines 14, 35, 65 |
| `src/llenergymeasure/core/warmup.py` | thermal_floor_wait() function | VERIFIED | Exists at line 179 |
| `tests/unit/core/test_harness.py` | MeasurementHarness unit tests with mock BackendPlugin | VERIFIED | 5 test functions with FakeBackend dataclass; all pass |
| `src/llenergymeasure/core/gpu_info.py` | nvml_context() context manager | VERIFIED | Exists at line 18 with correct double-try pattern |
| `src/llenergymeasure/exceptions.py` | DockerError base class with __init__(message, fix_suggestion, stderr_snippet) | VERIFIED | __init__ at line 31 with all three parameters |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config/loader.py` | `config/grid.py` | `from llenergymeasure.config.grid import` | WIRED | loader.py line 23: confirmed |
| `config/grid.py` | `config/_dict_utils.py` | `from llenergymeasure.config._dict_utils import` | WIRED | grid.py line 18: confirmed |
| `_api.py` | `core/harness.py` | `harness.run(backend, config)` | WIRED | _api.py lines 331-337: lazy import + call |
| `study/runner.py` | `core/harness.py` | `harness.run(backend, config)` | WIRED | runner.py lines 104-113: lazy import + call |
| `container_entrypoint.py` | `core/harness.py` | `harness.run(backend, config)` | WIRED | container_entrypoint.py lines 58-73: lazy import + call |
| `core/harness.py` | `core/backends/protocol.py` | `BackendPlugin` | WIRED | harness.py line 16: `from llenergymeasure.core.backends.protocol import BackendPlugin, InferenceOutput` |
| `core/harness.py` | `core/warmup.py` | `thermal_floor_wait` | WIRED | harness.py line 17: `from llenergymeasure.core.warmup import thermal_floor_wait`; called at line 188 |
| `runner_resolution.py` | `docker_preflight.py` | `import _NVIDIA_TOOLKIT_BINS` | WIRED | runner_resolution.py line 31: confirmed |
| `baseline.py` | `gpu_info.py` | `nvml_context()` | WIRED | baseline.py line 18: `from llenergymeasure.core.gpu_info import nvml_context`; used at line 81 |
| `environment.py` | `gpu_info.py` | `nvml_context()` | WIRED | environment.py line 17: `from llenergymeasure.core.gpu_info import nvml_context`; used at line 48 |

**Note on Plan 02 key_link deviation:** Plan 02 listed `pytorch.py -> warmup.py via thermal_floor_wait import` as a key link. The actual implementation correctly placed this call in harness.py (not the backends). Backends contain only docstring comments confirming harness ownership. This is a better design than planned - backends are thinner, harness is the single owner of thermal floor. Truth 8 is fully satisfied.

---

### Requirements Coverage

Phase 27 is an internal quality/refactoring phase with no REQUIREMENTS.md entries (all plans declare `requirements: []`). No requirement IDs to cross-reference.

---

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| (none) | - | - | - |

No TODO/FIXME/placeholder comments found in any of the key new or modified files. No stub implementations detected.

The only notable item is that `InferenceBackend` remains in `protocol.py` as dead code kept for backward compatibility with `tests/fakes.py::FakeInferenceBackend`. This was a documented decision in the 27-02-SUMMARY.md and is a minor cleanup item, not a blocker.

---

### Human Verification Required

None. All Phase 27 outcomes are mechanically verifiable via grep, file existence checks, and the test suite. The 877-test pass confirms behaviour-preservation.

---

## Gaps Summary

No gaps. All must-haves across all three plans are verified against the actual codebase.

**Verification evidence summary:**

- `src/llenergymeasure/study/grid.py` - DELETED (confirmed no such file)
- `src/llenergymeasure/config/_dict_utils.py` - EXISTS with `_unflatten` and `_deep_merge` as sole canonical definitions
- `src/llenergymeasure/config/grid.py` - EXISTS with all 6 grid symbols, importing from _dict_utils
- `src/llenergymeasure/config/loader.py` - imports from `config.grid` (not `study.grid`); no local `_unflatten` definition
- Zero `study.grid` references in src/ or tests/
- `src/llenergymeasure/core/harness.py` - 461 lines, full MeasurementHarness implementation
- `src/llenergymeasure/core/backends/pytorch.py` - 444 lines (down from 863); implements BackendPlugin 4-method protocol
- `src/llenergymeasure/core/backends/vllm.py` - 438 lines (down from 871); implements BackendPlugin 4-method protocol
- Zero `backend.run(` calls in src/
- `harness.run(backend, config)` wired in all three call sites (_api.py, study/runner.py, container_entrypoint.py)
- `nvml_context()` in gpu_info.py; zero raw pynvml.nvmlInit/nvmlShutdown outside it
- `DockerError.__init__` in exceptions.py; zero `def __init__` in docker_errors.py
- `_NVIDIA_TOOLKIT_BINS` defined in docker_preflight.py; imported (not redefined) in runner_resolution.py
- `_save_and_record()` defined once in runner.py; called at 3 sites (runner.py:451, runner.py:523, _api.py:348)
- **877 unit tests pass**

---

_Verified: 2026-03-07_
_Verifier: Claude (gsd-verifier)_
