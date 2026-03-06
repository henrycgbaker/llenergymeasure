---
phase: 24-m3-integration-fixes-and-verification
verified: 2026-03-05T23:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 24: M3 Integration Fixes and Retroactive Verification Report

**Phase Goal:** Close audit tech debt - fix 2 integration issues found by the integration checker and create retroactive VERIFICATION.md files for 3 unverified phases
**Verified:** 2026-03-05T23:00:00Z
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `run_study_preflight()` receives `yaml_runners` and `user_config` from the caller and forwards them to `resolve_study_runners()` | VERIFIED | `preflight.py` line 160-161: params in signature; line 208-209: forwarded to `resolve_study_runners()` |
| 2 | `_run_in_process()` calls `check_gpu_memory_residual()` before running a single experiment (both local and Docker paths) | VERIFIED | `_api.py` lines 285-289: lazy import + call placed before Docker/local branch split at line 292 |
| 3 | Phase 18 (Docker pre-flight) has a VERIFICATION.md with DOCK-07, DOCK-08, DOCK-09 verified against current codebase | VERIFIED | `.planning/phases/18-docker-pre-flight/18-VERIFICATION.md` exists; all three IDs present with file path + line number evidence from `docker_preflight.py` |
| 4 | Phase 20 (Docker image and CI) has a VERIFICATION.md with DOCK-10 verified against current codebase | VERIFIED | `.planning/phases/20-docker-image-and-ci/20-VERIFICATION.md` exists; DOCK-10 verified against `docker-publish.yml`, `release.yml`, and three Dockerfiles |
| 5 | Phase 21 (Measurement carried items) has a VERIFICATION.md with MEAS-03, MEAS-04 verified against current codebase | VERIFIED | `.planning/phases/21-measurement-carried-items/21-VERIFICATION.md` exists; MEAS-03 verified against `aienergyscore.jsonl` + `loader.py`; MEAS-04 verified against both backend files |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/orchestration/preflight.py` | Preflight with `yaml_runners` and `user_config` forwarding | VERIFIED | Lines 157-214: signature updated, `TYPE_CHECKING` guard for `UserRunnersConfig`, forwarding on line 209 |
| `src/llenergymeasure/_api.py` | GPU memory check in `_run_in_process` for both local and Docker branches | VERIFIED | Lines 285-289: `check_gpu_memory_residual()` called before branch split at line 292 |
| `tests/unit/study/test_study_preflight.py` | Tests confirming preflight forwards runner context | VERIFIED | `test_preflight_forwards_runner_context` (line 70) and `test_preflight_defaults_to_auto_detect_without_context` (line 99) - both substantive, use monkeypatch to capture call args |
| `tests/unit/test_api.py` | Test confirming GPU memory check in single-experiment path | VERIFIED | `test_run_in_process_calls_gpu_memory_check` (line 691) - substantive, asserts call count = 1 |
| `.planning/phases/18-docker-pre-flight/18-VERIFICATION.md` | Retroactive verification of Docker pre-flight requirements; contains DOCK-07 | VERIFIED | File exists; DOCK-07, DOCK-08, DOCK-09 all present with line-number evidence from `docker_preflight.py` |
| `.planning/phases/20-docker-image-and-ci/20-VERIFICATION.md` | Retroactive verification of Docker image and CI requirement; contains DOCK-10 | VERIFIED | File exists; DOCK-10 verified with evidence from `docker-publish.yml`, `release.yml`, and `docker/` Dockerfiles |
| `.planning/phases/21-measurement-carried-items/21-VERIFICATION.md` | Retroactive verification of measurement carried items; contains MEAS-03 | VERIFIED | File exists; MEAS-03 verified against `datasets/builtin/aienergyscore.jsonl` + `loader.py`; MEAS-04 verified against both backend files + docs |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_api.py (_run)` | `preflight.py (run_study_preflight)` | `yaml_runners=study.runners, user_config=user_config.runners` | WIRED | Lines 188-193: call passes both params; `user_config` loaded before call (line 183) |
| `_api.py (_run_in_process)` | `study/gpu_memory.py (check_gpu_memory_residual)` | Direct call before backend.run() and docker_runner.run() | WIRED | Lines 287-289: lazy import + call; positioned before Docker/local split at line 292, so covers both branches |
| `18-VERIFICATION.md` | `src/llenergymeasure/infra/docker_preflight.py` | grep verification of `_check_nvidia_toolkit`, `_probe_container_gpu`, CUDA compat | WIRED | `docker_preflight.py` confirmed: `_check_nvidia_toolkit` at line 64, `_probe_container_gpu` at line 117, called from `run_docker_preflight` at lines 249 and 264 |
| `20-VERIFICATION.md` | `.github/workflows/docker-publish.yml` | grep verification of GHCR publish chain | WIRED | `docker-publish.yml` confirmed: `REGISTRY: ghcr.io` (line 21), matrix over [pytorch, vllm, tensorrt], `release.yml` triggers via `needs: release` |
| `21-VERIFICATION.md` | `src/llenergymeasure/datasets/` | grep verification of aienergyscore dataset and `inference_memory_mb` | WIRED | `datasets/builtin/aienergyscore.jsonl` exists; `BUILTIN_DATASETS` registry in `loader.py`; `inference_memory_mb` in both `pytorch.py` and `vllm.py` |

---

### Requirements Coverage

Requirements declared across Phase 24 plans: DOCK-07, DOCK-08, DOCK-09, DOCK-10, MEAS-01, MEAS-02, MEAS-03, MEAS-04

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOCK-07 | 24-01 (integration fix), 24-02 (retroactive verification) | NVIDIA Container Toolkit detection in pre-flight | SATISFIED | `docker_preflight.py` line 64: `_check_nvidia_toolkit()`; line 249: called in `run_docker_preflight()`; 24-01 fixes the runner resolution divergence that could have caused DOCK-07 checks to fire incorrectly |
| DOCK-08 | 24-01, 24-02 | GPU visibility check inside container | SATISFIED | `docker_preflight.py` line 117: `_probe_container_gpu()`; line 264: called in `run_docker_preflight()`; runner resolution fix ensures preflight fires correctly |
| DOCK-09 | 24-01, 24-02 | CUDA/driver version compatibility check | SATISFIED | `docker_preflight.py` lines 159-157: four CUDA compat detection patterns; descriptive error message with host driver info |
| DOCK-10 | 24-02 | Docker image for each backend published to GHCR | SATISFIED | Three Dockerfiles in `docker/`; `docker-publish.yml` matrix over all three backends pushing to `ghcr.io`; triggered from `release.yml` |
| MEAS-01 | 24-01 | GPU memory residual check before each experiment | SATISFIED | `check_gpu_memory_residual()` now called in `_run_in_process()` (single-experiment path) at line 289, before both Docker and local dispatch branches |
| MEAS-02 | 24-01 | Warning on excessive residual GPU memory | SATISFIED | `check_gpu_memory_residual()` in `gpu_memory.py` handles the warning; the call is now present in both single- and multi-experiment paths |
| MEAS-03 | 24-02 | `aienergyscore.jsonl` built-in dataset bundled and loadable | SATISFIED | `datasets/builtin/aienergyscore.jsonl` exists; `BUILTIN_DATASETS` registry; `load_prompts()` dispatches to it; public API in `datasets/__init__.py` |
| MEAS-04 | 24-02 | `inference_memory_mb` semantics confirmed and documented | SATISFIED | `pytorch.py` and `vllm.py` both compute `inference_memory_mb = max(0.0, peak_memory_mb - model_memory_mb)`; documented in `docs/energy-measurement.md` |

No orphaned requirements found. All 8 requirement IDs from ROADMAP Phase 24 are accounted for across the two plans.

---

### Anti-Patterns Found

No blockers or warnings detected in the modified files.

Files checked:
- `src/llenergymeasure/orchestration/preflight.py` - no TODOs, no stub returns, fully wired
- `src/llenergymeasure/_api.py` - no TODOs, no stub returns, `check_gpu_memory_residual()` is a real call not a placeholder
- `tests/unit/study/test_study_preflight.py` - substantive assertions on captured call args
- `tests/unit/test_api.py` - substantive assertion (`len(gpu_check_calls) == 1`)

---

### Human Verification Required

None. All five success criteria are verifiable programmatically by inspecting source files and test implementations. No visual, real-time, or external service behaviour is involved.

---

### Summary

Phase 24 achieved its goal. Both integration issues identified by the milestone audit are closed:

1. **Preflight runner resolution (DOCK-07, DOCK-08, DOCK-09):** `run_study_preflight()` now accepts `yaml_runners` and `user_config` and forwards both to `resolve_study_runners()`. The caller in `_run()` loads `user_config` before the preflight call and passes `study.runners` and `user_config.runners`. Runner resolution during preflight is now identical to actual dispatch resolution.

2. **GPU memory check in single-experiment path (MEAS-01, MEAS-02):** `_run_in_process()` calls `check_gpu_memory_residual()` via a lazy local import before the Docker/local branch split. This matches the pattern used in `StudyRunner._run_one()` and `_run_one_docker()`, ensuring the residual check fires for all experiment dispatch paths.

Retroactive documentation gap closed:
- Phase 18 VERIFICATION.md: DOCK-07, DOCK-08, DOCK-09 verified with line-number evidence from `docker_preflight.py`
- Phase 20 VERIFICATION.md: DOCK-10 verified against Dockerfiles and CI workflows
- Phase 21 VERIFICATION.md: MEAS-03 and MEAS-04 verified against the datasets module and backend files

All 8 requirements (DOCK-07 through DOCK-10, MEAS-01 through MEAS-04) are fully satisfied with codebase evidence.

---

_Verified: 2026-03-05T23:00:00Z_
_Verifier: Claude (gsd-verifier)_
