---
gsd_state_version: 1.0
milestone: v0.1
milestone_name: milestone
status: unknown
last_updated: "2026-03-05T16:59:00.000Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 15
  completed_plans: 13
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 23 (Documentation) or milestone completion

## Current Position

Phase: 23 complete. All 5 plans complete.
Next: Milestone completion (v0.9.0 M3)
Status: Full 13-file documentation set complete — 9 researcher docs + 4 policy maker guides + rewritten README.
Last activity: 2026-03-05 - Phase 23 plan 05 complete (branch: gsd/phase-23-documentation)

Progress: [██████████] 100% (Phase 23 complete)

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 21
- Average duration: 205s
- Total execution time: 7022s (173s + 492s + 300s + 300s + 1020s + 793s + 429s + 179s + 420s + 420s + 308s + 242s + 58s + 430s + 83s + 481s + 114s + 106s + 161s + 290s + 223s)

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
- [Phase 21.1-01]: dev-gpu split into dev-vllm/dev-tensorrt — vllm and tensorrt pin incompatible torch versions, cannot co-exist in one extra
- [Phase 21.1-01]: [tool.uv] conflicts = vllm/tensorrt extras declared mutually exclusive — allows uv to fork lockfile resolution independently
- [Phase 21.1-01]: index-strategy = unsafe-best-match required for tensorrt-llm transitive deps (nvidia-cudnn-cu12) present on both nvidia and pypi indexes
- [Phase 21.1-02]: Duplicate CI jobs in release.yml (not workflow_run) — needs: only works within same file; workflow_run has race condition caveats
- [Phase 21.1-02]: docker job permissions set at job level — reusable workflow calls require explicit grant for callee's GITHUB_TOKEN to have packages:write
- [Phase 21.1-02]: type=raw Docker tags over type=semver — input version is full ref_name (v0.9.0), preserves exact string including v prefix
- [Phase 22-01]: Patch find_spec at module reference (llenergymeasure.core.backends.importlib.util.find_spec) not global importlib.util — module captures reference at import time, global patch is shadowed
- [Phase 22-01]: tests/CLAUDE.md added with git add -f — .git/info/exclude globally ignores CLAUDE.md but this is project test documentation that belongs in the repo
- [Phase 22-02]: Tier 1 test job skips on docs-only changes via dorny/paths-filter; package-validation runs always (packaging bugs from any change)
- [Phase 22-02]: Tier 2 GPU CI triggers on push to main only (post-merge); no -n auto — GPU tests must run serially; --extra vllm excluded (requires Docker runtime)
- [Phase 22-02]: HF cache keyed on tests/fixtures/model_list.txt hash with restore-keys fallback
- [Phase 22-03]: cli/experiment.py is dead v1.x code with deleted imports (F821 suppressed) — inject missing names via monkeypatch.setattr(module, name, value, raising=False) for testability
- [Phase 22-03]: backend_detection.py uses direct try/import — patch builtins.__import__ with side_effect that raises for target package name
- [Phase 22-03]: ensure_env_file imported inside function body — patch at definition site (llenergymeasure.config.env_setup), not at experiment module level
- [Phase 22-04]: SIGINT script polls ./results/ (not --output dir) — _api.py hardcodes Path("results") as study manifest parent; --output only affects per-experiment result files
- [Phase 22-04]: Study fixture uses experiments: list with warmup.enabled=false and baseline.enabled=false — avoids 30s+ thermal floor wait in CI
- [Phase 23-01]: typer 0.24.1 has no get_docs_for_typer_app — use typer.main.get_command(app) + click introspection for CLI reference generation
- [Phase 23-01]: Base install (uv sync --dev, no extras) sufficient for docs scripts — model_json_schema() does not import pytorch at runtime
- [Phase 23-02]: Annotated output in getting-started.md constructed from _display.py source — host cannot run pytorch outside containers; constructed output matches exact print_result_summary() format
- [Phase 23-02]: CLI reference supplements auto-generated flag table with manually written context (effective defaults, exit codes, examples) — generator output alone is incomplete as a reference
- [Phase 23-documentation]: Parameter support matrix in backends.md manually constructed from Pydantic config models; GPU test results required for full regeneration via generate_param_matrix.py
- [Phase 23-04]: WarmupConfig thermal_floor_seconds default is 60.0s (not 30s as stated in plan) — corrected from Pydantic model source; aligned with MLPerf Power minimum
- [Phase 23-05]: README rewritten as concise 75-line overview with links to all 13 docs — no inline content per CONTEXT.md decision
- [Phase 23-05]: Policy maker guides are self-contained — guide-comparison-context.md explains CodeCarbon and Zeus as measurement backends (not separate benchmarks) to avoid reader confusion

### Carried Items

1. ~~`aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)~~ RESOLVED - PR #54, datasets module on main
2. ~~`peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)~~ RESOLVED - PR #53, inference_memory_mb on main
3. ~~Manual Ctrl+C SIGINT test on GPU hardware — Phase 23 (TEST-01)~~ RESOLVED by Phase 22-04 automation

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-03-05
Stopped at: Phase 23 complete. All 13 docs written (9 researcher + 4 policy maker guides). README rewritten as overview with links.
Resume file: None
