---
phase: 22-testing-and-ci
plan: 03
subsystem: testing
tags: [pytest, unit-tests, coverage, cli, config, backend-detection, docker-detection, provenance]

# Dependency graph
requires:
  - phase: 22-01
    provides: Test directory restructure, pytest markers, xdist, conftest factories

provides:
  - Unit tests for cli/experiment.py (4 test classes, 19 tests)
  - Unit tests for config/backend_detection.py (3 test classes, 12 tests)
  - Unit tests for config/docker_detection.py (2 test classes, 13 tests)
  - Unit tests for config/provenance.py (6 test classes, 39 tests)
affects:
  - 22-04
  - coverage reporting

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Injecting undefined module names (F821 dead code) via monkeypatch.setattr(mod, name, value, raising=False)"
    - "Patching builtins.__import__ with side_effect to simulate package unavailability"
    - "Patching llenergymeasure.config.env_setup.ensure_env_file (lazy import inside function) at its definition site"

key-files:
  created:
    - tests/unit/cli/test_cli_experiment.py
    - tests/unit/config/test_backend_detection.py
    - tests/unit/config/test_docker_detection.py
    - tests/unit/config/test_provenance.py
  modified: []

key-decisions:
  - "cli/experiment.py is v1.x dead code (F821 suppressed) — tests inject missing names via monkeypatch.setattr(raising=False) rather than patching module-level attributes that don't exist"
  - "backend_detection.py uses direct import inside try/except — patching builtins.__import__ is the correct interception point"
  - "ensure_env_file imported inside function body in _run_experiment_in_docker — patch at definition site (llenergymeasure.config.env_setup.ensure_env_file)"

patterns-established:
  - "Pattern: For modules with deleted imports (dead code), inject fakes via monkeypatch.setattr(module, name, value, raising=False)"
  - "Pattern: For functions using bare try/import, patch builtins.__import__ with side_effect that raises for the target package name"

requirements-completed: [TEST-01]

# Metrics
duration: 8min
completed: 2026-03-04
---

# Phase 22 Plan 03: Coverage Gap Tests Summary

**Unit test coverage added for 4 zero-coverage active modules: cli/experiment.py, config/backend_detection.py, config/docker_detection.py, and config/provenance.py — 83 new tests across 15 test classes.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-04T15:01:18Z
- **Completed:** 2026-03-04T15:09:19Z
- **Tasks:** 2
- **Files modified:** 4 created

## Accomplishments

- 19 tests for `cli/experiment.py` covering `_is_json_output_mode()`, `_display_measurement_summary()` JSON path, `resolve_prompts()` priority logic (CLI > file > config.dataset > config.prompts), and `_run_experiment_in_docker()` command construction
- 12 tests for `config/backend_detection.py` covering `is_backend_available()` with mocked ImportError/OSError/RuntimeError, `get_available_backends()` filtering, `get_backend_install_hint()` strings, and pytorch-before-vllm ordering verification
- 13 tests for `config/docker_detection.py` covering `/.dockerenv` detection, `/proc/1/cgroup` detection (docker/containerd), FileNotFoundError/PermissionError handling, and `should_use_docker_for_campaign()` dispatch logic
- 39 tests for `config/provenance.py` covering `ParameterSource` enum, `ParameterProvenance` model (construction, str, serialisation), all `ResolvedConfig` query methods, and `flatten_dict`/`unflatten_dict` roundtrip/`compare_dicts`
- Full unit suite: 859 tests passing (up from 776 in plan 22-01)

## Task Commits

Each task was committed atomically:

1. **Task 1: cli/experiment.py and config/backend_detection.py** - `5dc8dc2` (test)
2. **Task 2: config/docker_detection.py and config/provenance.py** - `5d447ce` (test)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified

- `tests/unit/cli/test_cli_experiment.py` — 19 tests for legacy cli/experiment.py utility functions
- `tests/unit/config/test_backend_detection.py` — 12 tests for backend import detection logic
- `tests/unit/config/test_docker_detection.py` — 13 tests for Docker container detection
- `tests/unit/config/test_provenance.py` — 39 tests for provenance models and dict utilities

## Decisions Made

- `cli/experiment.py` is dead code (v1.x, not registered in v2.0 CLI) with all internal imports deleted and F821 suppressed. Tests inject undefined names via `monkeypatch.setattr(module, name, value, raising=False)` — this is the correct pattern for testing dead code with deleted imports.
- `ensure_env_file` is imported inside `_run_experiment_in_docker()`'s body via `from llenergymeasure.config.env_setup import ensure_env_file`. Patched at its definition site (`llenergymeasure.config.env_setup.ensure_env_file`).
- `console` in `experiment.py` is also an undefined name (deleted v1.x import). Injected via `monkeypatch.setattr(_exp_mod, "console", MagicMock(), raising=False)` in Docker tests.

## Deviations from Plan

None — plan executed exactly as written. The approach for testing dead code with deleted imports required a pattern decision (inject undefined names) but this was within the plan's scope of "mock any I/O or hardware dependencies".

## Issues Encountered

`cli/experiment.py` references multiple undefined names (`HuggingFacePromptSource`, `load_prompts_from_source`, `load_prompts_from_file`, `console`) because all v1.x internal imports were deleted. These names don't exist as module attributes, so `monkeypatch.setattr(mod, name, ...)` requires `raising=False`. Discovered during test iteration and resolved cleanly.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- 4 zero-coverage active modules now have test files
- Full unit suite at 859 tests, all passing
- Ready for plan 22-04 (any remaining coverage or CI work)

---
*Phase: 22-testing-and-ci*
*Completed: 2026-03-04*
