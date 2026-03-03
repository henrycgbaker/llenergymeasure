---
gsd_state_version: 1.0
milestone: v1.17
milestone_name: milestone
status: unknown
last_updated: "2026-03-03T18:23:14.441Z"
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 10
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 21 — Measurement Carried Items

## Current Position

Phase: 21 of 23 in progress (Measurement Carried Items — plan 01 done)
Next: Phase 21 plan 02 (peak_memory_mb semantics)
Status: Phase 21 plan 01 complete — aienergyscore.jsonl dataset, datasets module, backend wiring, broken import fix
Last activity: 2026-03-03 — Phase 21 plan 01 complete on gsd/phase-21-measurement-carried-items

Progress: [████░░░░░░] 34%

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 14
- Average duration: 209s
- Total execution time: 6044s (173s + 492s + 300s + 300s + 1020s + 793s + 429s + 179s + 420s + 420s + 308s + 60s + 62s + 308s + 126s + 462s)

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
- [Phase 19.1-01]: Nested VLLMConfig (engine + sampling) replaces flat 5-field schema — mirrors vLLM's own LLM()/SamplingParams API separation
- [Phase 19.1-01]: Local _unflatten/_deep_merge helpers in grid.py avoid circular import with config.loader (loader imports grid, grid cannot import loader)
- [Phase 19.1-01]: block_size validated via Literal[8, 16, 32] (not model_validator) — idiomatic Pydantic for discrete value sets
- [Phase 19.1-01]: gpu_memory_utilization bounds: ge=0.0, lt=1.0 (strict less-than) — value of 1.0 would leave no headroom for model weights
- [Phase 19.1-02]: dict-then-instantiate in _build_sampling_params() — single return point enables clean VLLMSamplingConfig override injection for both greedy and sampling paths
- [Phase 19.1-02]: get_backend_capabilities() reads VLLMEngineConfig.model_fields (not VLLMConfig) — capability matrix reflects actual engine fields, not container fields
- [Phase 19.1-02]: speculative_model + num_speculative_tokens → speculative_config dict — vLLM v0.6+ removed direct speculative_model kwarg
- [Phase 20-02]: Separate docker-publish.yml workflow (not merged into release.yml) — Docker builds are slow and should fail independently
- [Phase 20-02]: fail-fast:false in matrix ensures vLLM build failure does not cancel PyTorch build
- [Phase 20-02]: Layer cache scoped per backend (scope=matrix.backend) prevents cross-backend GHA cache collisions
- [Phase 20-02]: latest=auto in metadata-action attaches latest only on default branch tags, not pre-releases
- [Phase 20-01]: vllm/vllm-openai and pytorch/pytorch as upstream bases — CUDA/framework compatibility guaranteed by upstream, no shared Dockerfile.base dependency
- [Phase 20-01]: ENTRYPOINT [] reset in Dockerfile.vllm — DockerRunner appends command after image name (CMD override), not after ENTRYPOINT; vllm/vllm-openai sets its own ENTRYPOINT to API server
- [Phase 20-01]: --no-deps on .[vllm] pip install — avoids reinstalling vLLM/torch already present in base image
- [Phase 20-01]: ghcr.io/henrycgbaker/llenergymeasure/ as GHCR namespace — corrects stale placeholder lacking the GitHub user prefix
- [Phase 21-01]: AUTO_DETECT_COLUMNS and BUILTIN_DATASETS live in datasets/loader.py (not config/models) — config.models stays pure config, no dataset logic
- [Phase 21-01]: FilePromptSource/HuggingFacePromptSource defined as dataclasses in dataset_loader.py — they were never in config.models, self-contained in the loader that uses them
- [Phase 21-01]: AIEnergyScore/text_generation has single text column, no source field — grouped ordering falls back to file order (stable)

### Carried Items

1. ~~`aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)~~ **DONE** (Phase 21 plan 01)
2. `peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)
3. Manual Ctrl+C SIGINT test on GPU hardware — Phase 23 (TEST-01)

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed phase 21 plan 01 (aienergyscore.jsonl dataset module + backend wiring).
Resume file: None
