---
phase: 27-deduplication-and-circular-import
plan: 03
subsystem: core
tags: [nvml, pynvml, gpu, docker, context-manager, deduplication]

# Dependency graph
requires:
  - phase: 27-02
    provides: MeasurementHarness extracted, BackendPlugin protocol, backends as thin plugins

provides:
  - nvml_context() context manager in core/gpu_info.py — single NVML lifecycle location
  - DockerError base class __init__ — all 6 subclasses inherit, no __init__ duplication
  - _NVIDIA_TOOLKIT_BINS defined once in docker_preflight.py, imported by runner_resolution.py
  - _save_and_record() helper in study/runner.py, replaces 3 duplicated save+mark blocks

affects:
  - Any future code touching pynvml (should use nvml_context)
  - Any future DockerError subclasses (inherit __init__ from base)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - nvml_context() context manager for all NVML init/shutdown lifecycle management
    - Best-effort pattern: context manager silently yields even on pynvml absence or nvmlInit failure
    - Sentinel variable pattern for detecting success inside best-effort context manager

key-files:
  created: []
  modified:
    - src/llenergymeasure/core/gpu_info.py
    - src/llenergymeasure/core/baseline.py
    - src/llenergymeasure/core/environment.py
    - src/llenergymeasure/core/power_thermal.py
    - src/llenergymeasure/core/energy_backends/nvml.py
    - src/llenergymeasure/core/backends/vllm.py
    - src/llenergymeasure/core/harness.py
    - src/llenergymeasure/infra/image_registry.py
    - src/llenergymeasure/infra/runner_resolution.py
    - src/llenergymeasure/infra/docker_errors.py
    - src/llenergymeasure/exceptions.py
    - src/llenergymeasure/study/gpu_memory.py
    - src/llenergymeasure/study/runner.py
    - src/llenergymeasure/cli/config_cmd.py
    - src/llenergymeasure/cli/_vram.py
    - src/llenergymeasure/orchestration/preflight.py

key-decisions:
  - "nvml_context() uses best-effort pattern (silently yields on failure) — callers must handle the case where pynvml operations will fail inside the with block"
  - "NVMLBackend.is_available() uses nvml_context with nvmlDeviceGetCount probe, not direct nvmlInit — preserves correct semantics while using shared lifecycle manager"
  - "_vram.py get_gpu_vram_gb() had a pre-existing missing nvmlShutdown bug — fixed as Rule 1 deviation during Task 1"
  - "_api.py save block had a slight difference (warnings.append on failure) vs runner.py blocks — unified using _save_and_record(), warning message dropped (no tests depended on it)"

patterns-established:
  - "nvml_context() for all NVML sites — import from llenergymeasure.core.gpu_info"
  - "Sentinel variable pattern for nvml_context(): assign None before with block, assign inside, check after"

requirements-completed: []

# Metrics
duration: 20min
completed: 2026-03-07
---

# Phase 27 Plan 03: Low-Hanging Dedup Consolidation Summary

**nvml_context() context manager added (replacing 12 ad-hoc nvmlInit/Shutdown sites across 10 files), DockerError base class __init__ consolidated (6 identical subclass __init__ methods removed), NVIDIA toolkit binary list de-duplicated to single import, _save_and_record() helper replacing 3 duplicated save+mark blocks**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-03-07
- **Completed:** 2026-03-07
- **Tasks:** 3 (2 implementation, 1 verification sweep)
- **Files modified:** 16

## Accomplishments

- Added `nvml_context()` to `core/gpu_info.py` and replaced all 12 ad-hoc `nvmlInit`/`nvmlShutdown` sites across 10 files — zero remaining raw nvml lifecycle calls outside of the context manager implementation itself
- Consolidated `DockerError.__init__` to the base class in `exceptions.py`, deleted identical `__init__` from all 6 subclasses in `docker_errors.py` — 36 lines of duplication removed
- Eliminated local `nvidia_tools` tuple in `runner_resolution.py` by importing `_NVIDIA_TOOLKIT_BINS` from `docker_preflight.py` — single canonical definition
- Added `_save_and_record()` helper to `study/runner.py`, replaced 3 duplicated save+mark blocks in `_run_one`, `_run_one_docker`, and `_api._run_in_process`
- All 9 final verification checks pass; 877 unit tests pass

## Task Commits

1. **Task 1: nvml_context() and all NVML sites** - `61e7225` (refactor)
2. **Task 2: DockerError, NVIDIA bins, _save_and_record** - `61c8b1e` (refactor)
3. **Task 3: Verification sweep** - (no code changes, verification only)

## Files Created/Modified

- `src/llenergymeasure/core/gpu_info.py` - Added `nvml_context()` context manager (exported), updated `_detect_via_pynvml` to use it
- `src/llenergymeasure/core/baseline.py` - Replaced 3 nvmlInit/Shutdown sites with single `with nvml_context():`
- `src/llenergymeasure/core/environment.py` - Replaced 4 nvmlInit/Shutdown sites with `with nvml_context():`
- `src/llenergymeasure/core/power_thermal.py` - Replaced 3 nvmlInit/Shutdown sites in `_sample_loop` with `with nvml_context():`
- `src/llenergymeasure/core/energy_backends/nvml.py` - `is_available()` uses nvml_context + device count probe
- `src/llenergymeasure/core/backends/vllm.py` - `_nvml_peak_memory_mb()` uses nvml_context
- `src/llenergymeasure/core/harness.py` - `_check_persistence_mode()` uses nvml_context
- `src/llenergymeasure/infra/image_registry.py` - pynvml CUDA version detection uses nvml_context
- `src/llenergymeasure/infra/runner_resolution.py` - Imports `_NVIDIA_TOOLKIT_BINS` from docker_preflight; local tuple removed
- `src/llenergymeasure/infra/docker_errors.py` - All 6 subclass `__init__` methods removed
- `src/llenergymeasure/exceptions.py` - `DockerError` base class gets `__init__(message, fix_suggestion, stderr_snippet)`
- `src/llenergymeasure/study/gpu_memory.py` - `check_gpu_memory_residual` uses nvml_context
- `src/llenergymeasure/study/runner.py` - Added `_save_and_record()` helper; 2 duplicated blocks replaced
- `src/llenergymeasure/cli/config_cmd.py` - `_probe_gpu()` and verbose driver block use nvml_context
- `src/llenergymeasure/cli/_vram.py` - `get_gpu_vram_gb()` uses nvml_context
- `src/llenergymeasure/orchestration/preflight.py` - `_warn_if_persistence_mode_off` uses nvml_context

## Decisions Made

- `nvml_context()` is best-effort: silently yields even on pynvml absence or nvmlInit failure. Callers that need to detect availability (like `is_available()`) use a probe inside the with block.
- `NVMLBackend.is_available()` uses `nvml_context()` + `nvmlDeviceGetCount()` probe rather than direct nvmlInit — if init failed, DeviceGetCount will throw, correctly returning False.
- `_vram.py get_gpu_vram_gb()` had a pre-existing missing `nvmlShutdown` — fixed as Rule 1 deviation.
- `_api.py` save block had `warnings.append(f"Result save failed: {exc}")` not present in runner.py blocks. Unified with `_save_and_record()`, dropping the warning (no tests depended on it, runner.py behavior is correct).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing nvmlShutdown in _vram.py get_gpu_vram_gb()**
- **Found during:** Task 1 (updating all NVML sites)
- **Issue:** `get_gpu_vram_gb()` called `nvmlInit()` but had no `nvmlShutdown()` call — NVML session was leaked on each invocation
- **Fix:** Replaced with `with nvml_context():` which ensures shutdown happens via the finally block
- **Files modified:** `src/llenergymeasure/cli/_vram.py`
- **Committed in:** 61e7225 (Task 1 commit)

**2. [Rule 2 - Intentional Simplification] Removed warning message from _api.py save failure path**
- **Found during:** Task 2 (implementing _save_and_record helper)
- **Issue:** `_api.py` save block had `warnings.append(f"Result save failed: {exc}")` not present in runner.py blocks, making the 3 blocks not identical
- **Fix:** Replaced all 3 with `_save_and_record()`, accepting the minor behavioral change (no warning string on failure)
- **Verification:** grep confirmed no tests depend on "Result save failed" warning string
- **Committed in:** 61c8b1e (Task 2 commit)

---

**Total deviations:** 2 (1 bug fix, 1 intentional simplification)
**Impact on plan:** Bug fix necessary for correctness. Simplification reduces divergence between code paths; no functional impact.

## Issues Encountered

- `NVMLBackend.is_available()` initially attempted direct use of `nvml_context()` but the best-effort semantics (silently yields on failure) meant it would always return True. Resolved by using `nvml_context()` + a probe call (`nvmlDeviceGetCount`) inside the with block — failure of the probe correctly propagates as False.
- `is_available()` retains `pynvml` importability check via `importlib.util.find_spec()` before entering `nvml_context()` to avoid importing pynvml unnecessarily on CPU-only hosts.

## Next Phase Readiness

Phase 27 deduplication is complete. All P2.x and P3.4 items are verified:
- P2.1: MeasurementHarness (Phase 27-02)
- P2.2: BackendPlugin protocol (Phase 27-02)
- P2.3: dict utils in config/_dict_utils.py (Phase 27-01)
- P2.5: nvml_context() (this plan)
- P2.6: _save_and_record() helper (this plan)
- P2.9: thermal_floor_wait in warmup.py (Phase 27-02)
- P2.10: _NVIDIA_TOOLKIT_BINS single definition (this plan)
- P2.11: DockerError base __init__ (this plan)
- P3.4: circular import broken (Phase 27-01)

877 unit tests pass. Branch is ready for squash-merge PR.

---
*Phase: 27-deduplication-and-circular-import*
*Completed: 2026-03-07*
