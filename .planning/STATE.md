---
gsd_state_version: 1.0
milestone: v1.17
milestone_name: milestone
status: unknown
last_updated: "2026-02-28T11:20:34.162Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 8
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 19 — vLLM Backend Activation

## Current Position

Phase: 19 of 23 in progress (vLLM Backend Activation — Plan 02 complete)
Next: Phase 20 (or remaining Phase 19 plans if any)
Status: Phase 19 Plan 02 shipped (VLLMBackend unit tests, 42 tests, VLLM-01/02/03 verified)
Last activity: 2026-02-28 — Phase 19 Plan 02 complete on gsd/phase-19-vllm-backend-activation

Progress: [████░░░░░░] 29%

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 8
- Average duration: 235s
- Total execution time: 3686s (173s + 492s + 300s + 300s + 1020s + 793s + 429s + 179s)

*Updated after each plan completion*

## Accumulated Context

### Decisions

- StudyRunner dispatches via `multiprocessing.spawn` + Pipe IPC (local path, M2)
- Docker path is a parallel dispatch method in StudyRunner — not a separate runner class
- Multi-backend without Docker → hard error at pre-flight (M2, DOCK-05 extends this to auto-elevation)
- One backend per milestone: M3=vLLM, M4=TRT-LLM, M5=SGLang
- Phase 13 (docs) folded into Phase 22 of M3 — write docs once against final backend story
- Local import in _run_one() keeps pynvml lazy; avoids module-level ImportError when pynvml not installed (Phase 16)
- GPU memory threshold hardcoded at 100 MB for M3; configurability deferred until researcher demand (Phase 16)
- [Phase 17]: parse_runner_value raises ValueError on empty 'docker:' and unrecognised values — strict contract prevents silent fallbacks
- [Phase 17]: Container entrypoint uses core.backends.get_backend path (same as StudyRunner worker) — not orchestration factory — for identical measurement behaviour
- [Phase 17]: Error JSON format {type, message, traceback} mirrors StudyRunner worker payloads — consistent upstream consumer handling
- [Phase 17-02]: user_config=None enables auto-detection; user_config provided (even with default 'local') blocks auto-detection — explicit presence beats inference
- [Phase 17-02]: UserRunnersConfig accepts bare "docker" (not just "docker:<image>") to match YAML runner config syntax
- [Phase 17-03]: DockerRunner returns error payload dicts as-is (no effective_config injection) — error dicts have no result fields to annotate
- [Phase 17-03]: exchange_dir=None sentinel in finally-block prevents double-cleanup on unexpected exceptions
- [Phase 17-04]: Auto-elevation is info-log-only (no user prompt) — multi-backend with Docker proceeds automatically
- [Phase 17-04]: DockerErrors caught in _run_one_docker() and _run_in_process() — converted to non-fatal failure dicts, study continues
- [Phase 17-04]: test_study_preflight.py tests must mock is_docker_available() — host machine has Docker + NVIDIA CT installed
- [Phase 18-01]: DockerPreFlightError inherits PreFlightError (not DockerError) so existing CLI error handler catches it without changes
- [Phase 18-01]: CUDA compat detection uses specific patterns (cuda+version, driver/library mismatch, nvml+driver) to avoid false positives from generic GPU access errors containing "device driver"
- [Phase 18-01]: run_docker_preflight imported inside function body in run_study_preflight — lazy import prevents circular dependency
- [Phase 18-01]: resolve_study_runners called without yaml_runners/user_config in pre-flight — uses auto-detection to check if Docker runners are active
- [Phase 19-01]: VLLMBackend uses offline batch mode only — CM-07 (streaming bug) resolved structurally by design, not patched
- [Phase 19-01]: FLOPs estimation wrapped in try/except — vLLM exposes HF model via internal API path, defaults to 0.0 on failure
- [Phase 19-01]: top_k=0 (our disabled sentinel) maps to top_k=-1 (vLLM's disabled sentinel) in _build_sampling_params()
- [Phase 19-01]: PyTorch takes priority over vLLM in detect_default_backend() — pytorch is the simpler, always-available default
- [Phase 19-02]: Streaming source check targets API calls (stream=True, AsyncEngine) not docstring text — docstrings legitimately reference 'streaming' as context for CM-07
- [Phase 19-02]: _FakeSamplingParams dataclass used over MagicMock — captures **kwargs cleanly for SamplingParams construction tests

### Carried Items

1. `aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)
2. `peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)
3. Manual Ctrl+C SIGINT test on GPU hardware — Phase 23 (TEST-01)

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 19-vllm-backend-activation-19-02-PLAN.md (VLLMBackend unit tests, 42 tests, 1 file).
Resume file: None
