---
gsd_state_version: 1.0
milestone: v1.17
milestone_name: milestone
status: unknown
last_updated: "2026-03-03T23:37:31.445Z"
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 12
  completed_plans: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 19.2 complete — next phase TBD (M3 continues)

## Current Position

Phase: 19.2 Plan 02 of 2 complete (vLLM Runtime Wiring and Tests)
Next: Next M3 phase (Phase 20 documentation or next backend phase)
Status: Phase 19.2 complete — all schema changes wired into runtime backends, SSOT updated, 24 new unit tests added
Last activity: 2026-03-03 — Phase 19.2 Plan 02 complete on gsd/phase-19.2-vllm-extended-parameters-and-passthrough

Progress: [████░░░░░░] 32%

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 13
- Average duration: 209s
- Total execution time: 5910s (173s + 492s + 300s + 300s + 1020s + 793s + 429s + 179s + 420s + 420s + 308s + 176s + 900s)

*Updated after each plan completion*

## Accumulated Context

### Decisions

- StudyRunner dispatches via `multiprocessing.spawn` + Pipe IPC (local path, M2)
- Docker path is a parallel dispatch method in StudyRunner — not a separate runner class
- Multi-backend without Docker → hard error at pre-flight (M2, DOCK-05 extends this to auto-elevation)
- One backend per milestone: M3=vLLM, M4=TRT-LLM, M5=SGLang
- Phase 13 (docs) folded into Phase 23 of M3 — write docs once against final backend story
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
- [Phase 19.1-01]: Nested VLLMConfig (engine + sampling) replaces flat 5-field schema — mirrors vLLM's own LLM()/SamplingParams API separation
- [Phase 19.1-01]: Local _unflatten/_deep_merge helpers in grid.py avoid circular import with config.loader (loader imports grid, grid cannot import loader)
- [Phase 19.1-01]: block_size validated via Literal[8, 16, 32] (not model_validator) — idiomatic Pydantic for discrete value sets
- [Phase 19.1-01]: gpu_memory_utilization bounds: ge=0.0, lt=1.0 (strict less-than) — value of 1.0 would leave no headroom for model weights
- [Phase 19.1-02]: dict-then-instantiate in _build_sampling_params() — single return point enables clean VLLMSamplingConfig override injection for both greedy and sampling paths
- [Phase 19.1-02]: get_backend_capabilities() reads VLLMEngineConfig.model_fields (not VLLMConfig) — capability matrix reflects actual engine fields, not container fields
- [Phase 19.1-02]: speculative_model + num_speculative_tokens → speculative_config dict — vLLM v0.6+ removed direct speculative_model kwarg
- [Phase 19.2-01]: extra="allow" on all backend configs — unknown YAML fields stored in model_extra, forwarded to native API at runtime wiring (Plan 02)
- [Phase 19.2-01]: VLLMAttentionConfig.backend is str not Literal — vLLM owns attention backend name validation, not our schema
- [Phase 19.2-01]: compilation_config typed as dict[str, Any] — vLLM CompilationConfig has ~30 internal fields, full passthrough dict avoids fragile upstream coupling
- [Phase 19.2-01]: offload_params typed as list[str] in YAML — YAML-friendly; Plan 02 converts to set[str] when forwarding to vLLM if needed
- [Phase 19.2-02]: Beam search branching at _run_measurement() via isinstance(sampling_params, BeamSearchParams) — type carries routing, no flag needed
- [Phase 19.2-02]: engine.model_extra merged LAST in _build_llm_kwargs() — user passthrough extras can override explicit fields (escape hatch for vLLM API changes)
- [Phase 19.2-02]: attention.backend -> attention_backend name mapping applied at wiring time — schema uses vLLM-agnostic name, wiring handles API-specific rename
- [Phase 19.2-02]: offload_params converts list[str] -> set[str] at wiring time — YAML list is user-friendly, vLLM API expects set[str]

### Roadmap Evolution

- Phase 19.2 inserted after Phase 19.1: vLLM Extended Parameters and Passthrough — adds beam search, attention config, compilation passthrough, CPU offload params, kv_cache_memory_bytes, and extra="allow" passthrough architecture for all backend configs

### Carried Items

1. `aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)
2. `peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)
3. Manual Ctrl+C SIGINT test on GPU hardware — Phase 22 (TEST-01)

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 19.2-vllm-extended-parameters-and-passthrough-19.2-02-PLAN.md (runtime wiring: 7 engine fields, attention config, model_extra passthrough, beam search, SSOT updates, 24 new tests).
Resume file: None
