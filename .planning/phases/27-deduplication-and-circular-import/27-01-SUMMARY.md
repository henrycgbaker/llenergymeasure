---
phase: 27-deduplication-and-circular-import
plan: "01"
subsystem: config
tags: [refactor, circular-import, config, grid, dict-utils]

# Dependency graph
requires:
  - phase: 19.1-vllm-config-schema
    provides: "local _unflatten/_deep_merge in study/grid.py (the workaround being replaced)"
provides:
  - "config/_dict_utils.py: canonical _unflatten and _deep_merge utilities"
  - "config/grid.py: grid expansion functions moved from study/grid.py"
  - "config/ is now self-contained, no circular import with study/"
affects: [27-02, any future plan importing from config or study]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "config/_dict_utils.py as canonical home for shared dict utilities"
    - "config/grid.py as canonical home for grid expansion (moved from study/)"

key-files:
  created:
    - src/llenergymeasure/config/_dict_utils.py
    - src/llenergymeasure/config/grid.py
  modified:
    - src/llenergymeasure/config/loader.py
    - src/llenergymeasure/cli/run.py
    - tests/unit/study/test_study_grid.py
    - tests/unit/study/test_study_runner.py
    - tests/unit/study/test_study_manifest.py
  deleted:
    - src/llenergymeasure/study/grid.py

key-decisions:
  - "config/_dict_utils.py is the single canonical location for _unflatten and _deep_merge — no re-exports from study/"
  - "study/grid.py deleted outright — no tombstone/compat shim per CONTEXT.md decision"
  - "loader.py imports _unflatten from _dict_utils (not redefining locally)"

patterns-established:
  - "config/ package is self-contained: loader.py -> config.grid -> config._dict_utils, no study/ dependency"

requirements-completed: []

# Metrics
duration: 4min
completed: 2026-03-07
---

# Phase 27 Plan 01: Circular Import Fix and Dict Utils Consolidation Summary

**Broke config/loader.py -> study/grid.py circular import by moving grid into config package and consolidating three copies of _unflatten/_deep_merge into a single config/_dict_utils.py**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-07T12:59:27Z
- **Completed:** 2026-03-07T13:03:47Z
- **Tasks:** 3
- **Files modified:** 7 (2 created, 1 deleted, 5 updated)

## Accomplishments

- Created `config/_dict_utils.py` as canonical home for `_unflatten` and `_deep_merge`, replacing three separate copies (study/grid.py x2, config/loader.py x1)
- Moved all grid expansion logic (`expand_grid`, `apply_cycles`, `CycleOrder`, etc.) from `study/grid.py` to `config/grid.py`, importing from `_dict_utils` instead of defining locally
- Deleted `study/grid.py` and updated all 6 import sites (loader.py, cli/run.py, 3 test files with 5 import statements) to point to `config.grid`
- 872 unit tests pass with zero changes to test logic

## Task Commits

1. **Task 1: Create config/_dict_utils.py and config/grid.py** - `6f62d81` (refactor)
2. **Task 2: Update all importers and delete study/grid.py** - `1a9c43e` (refactor)
3. **Task 3: Verify test suite passes** - (verification only, no new files)

## Files Created/Modified

- `src/llenergymeasure/config/_dict_utils.py` - New canonical `_unflatten` and `_deep_merge` utilities
- `src/llenergymeasure/config/grid.py` - Grid expansion moved from study/, imports from `_dict_utils`
- `src/llenergymeasure/study/grid.py` - **Deleted**
- `src/llenergymeasure/config/loader.py` - Updated: import from `config.grid` (not `study.grid`), `_unflatten` imported from `_dict_utils`
- `src/llenergymeasure/cli/run.py` - Updated: lazy import updated from `study.grid` to `config.grid`
- `tests/unit/study/test_study_grid.py` - Updated: top-level import from `config.grid`
- `tests/unit/study/test_study_runner.py` - Updated: 3 lazy imports from `config.grid`
- `tests/unit/study/test_study_manifest.py` - Updated: 1 lazy import from `config.grid`

## Decisions Made

- `study/grid.py` deleted with no compat shim - direct imports only per CONTEXT.md decision
- `_unflatten` in `loader.py` was a private function (not public API), safe to replace with import
- `loader.py` already had `deep_merge` as a public function with a different signature (no leading underscore); the private `_unflatten` was the only one removed and replaced with an import

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- `config/` is now self-contained with no circular dependency on `study/`
- Plan 02 (harness refactor) can cleanly import from `config/` without triggering circular imports
- Zero `study.grid` references remain in `src/` or `tests/`

---
*Phase: 27-deduplication-and-circular-import*
*Completed: 2026-03-07*
