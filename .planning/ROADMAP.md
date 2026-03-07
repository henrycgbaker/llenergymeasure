# Roadmap: LLenergyMeasure

## Milestones

- [x] **v0.1.0–v0.6.0 Foundation & Planning** — Phases 1–4.5 (shipped 2026-02-26)
- [x] **v0.7.0 M1 — Core Single-Experiment** — Phases 1–8.2 (shipped 2026-02-27)
- [x] **v0.8.0 M2 — Study / Sweep** — Phases 9–15 (shipped 2026-02-27)
- [ ] **v0.9.0 M3 — Docker + vLLM** — Phases 16–31 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...): Planned milestone work
- Decimal phases (9.1, 10.1...): Urgent insertions (marked with INSERTED)

**Phase Discussion Protocol (Phases 25+):**
Every phase follows a structured workflow that moves from ambiguity to informed decisions before any code is written:

1. **Identify Grey Areas** — Review the phase scope and surface open questions, ambiguous requirements, and decisions where multiple valid approaches exist. These are the "Discussion" topics listed under each phase below.
2. **Peer Research** — For each grey area, research how peer codebases and tools handle the same problem (e.g., for dead code: how mlflow/wandb/pytorch-lightning manage deprecation; for dedup: how they structure shared utilities; for testing: their test strategies). Research produces concrete recommendations with rationale, not just summaries.
3. **Informed Decisions** — Discuss research findings, weigh trade-offs, and make explicit decisions for each grey area. Every decision has a rationale grounded in peer evidence.
4. **CONTEXT.md** — All decisions, rationale, and relevant research findings are captured in a phase-level `CONTEXT.md` file (`.planning/phases/{phase}/CONTEXT.md`). This is the decision record that all subsequent work executes against.
5. **Phase-Level Research** — During planning, the planner agent conducts additional targeted research (implementation details, API specifics, library docs) as needed to flesh out the plan.
6. **Planning** — Plans are written against the CONTEXT.md decisions. Each plan references the decisions it implements.

```
Grey areas → Research → Decisions → CONTEXT.md → Plan research → Plans
```

This ensures every refactoring decision is informed by real-world patterns, not just internal opinion.

**Cross-Cutting Architectural Principle — Backend Plugin Model:**
The refactoring phases (25-28) should collectively move towards a clear plugin architecture where:
- **The tool IS the measurement harness**: energy sampling, NVML lifecycle, result construction, CUDA sync, thermal management, warmup, persistence, and progress reporting are shared library infrastructure.
- **Backends ARE thin inference plugins**: each backend (PyTorch, vLLM, future TensorRT, SGLang) implements only the inference-specific logic (model loading, prompt formatting, generation call) and plugs into the shared harness.
- **New backends should not require re-implementing** measurement infrastructure. Adding a fourth backend should be a small, focused piece of work - not a copy-paste of 800+ lines from an existing backend.
- Phase 27 is where this architecture is researched and the shared extraction happens. Phases 25-26 clear the ground; Phase 28 cleans up remaining inconsistencies.

<details>
<summary>✅ v0.7.0 M1 — Core Single-Experiment (Phases 1–8.2) — SHIPPED 2026-02-27</summary>

- [x] **Phase 1: Package Foundation** - Dead code removal, src/ layout, pyproject.toml, protocols, state machine, resilience carry-forwards (completed 2026-02-26)
- [x] **Phase 2: Config System** - ExperimentConfig composition model, YAML loader, user config, SSOT introspection (completed 2026-02-26)
- [x] **Phase 3: Library API** - `__init__.py` public API, `run_experiment()`, internal `_run(StudyConfig)`, API stability contract (completed 2026-02-26)
- [x] **Phase 4: PyTorch Backend and Pre-flight** - PyTorch inference backend (P0 bug fix), InferenceBackend protocol, pre-flight checks, environment snapshot (completed 2026-02-26)
- [x] **Phase 4.1: PyTorch Parameter Audit** - INSERTED — Audit PyTorchConfig fields against upstream `transformers`/`torch` APIs (completed 2026-02-26)
- [x] **Phase 5: Energy Measurement** - NVML poller, Zeus optional, CodeCarbon optional, baseline power, warmup, FLOPs estimation, timeseries (completed 2026-02-26)
- [x] **Phase 6: Results Schema and Persistence** - ExperimentResult schema, EnergyBreakdown, persistence API, late aggregation, output layout (completed 2026-02-26)
- [x] **Phase 7: CLI** - `llem run`, `llem config`, `llem --version`, plain text display, exit codes, error hierarchy (completed 2026-02-27)
- [x] **Phase 8: Testing and Integration** - Unit + integration test tiers, protocol mocks, GPU CI workflow, UAT against M1 exit criteria (completed 2026-02-27)
- [x] **Phase 8.1: PyTorch Result Wiring Fixes** - INSERTED — Fix `_build_result()` field wiring: timeseries, effective_config, baseline fields. Add `extra="forbid"`. Gap closure. (completed 2026-02-27)
- [x] **Phase 8.2: M1 Tech Debt Cleanup** - INSERTED — Phase 2 VERIFICATION.md, REQUIREMENTS.md status drift, v1.x import breakages, orphaned exports cleanup. (completed 2026-02-27)

Full details: `milestones/v1.17.0-ROADMAP.md`

</details>

<details>
<summary>✅ v0.8.0 M2 — Study / Sweep (Phases 9–15) — SHIPPED 2026-02-27</summary>

- [x] **Phase 9: Grid Expansion and StudyConfig** - Sweep YAML grammar, `StudyConfig` + `ExecutionConfig` models, Cartesian grid expander, cycle ordering, pre-flight count display (completed 2026-02-27)
- [x] **Phase 10: Manifest Writer** - `StudyManifest` checkpoint model, `ManifestWriter` with atomic writes, study output directory layout (completed 2026-02-27)
- [x] **Phase 11: Subprocess Isolation and StudyRunner** - Subprocess dispatch via `spawn`, `Pipe`/`Queue` IPC, timeout handling, SIGINT, skip-and-continue, thermal gaps (completed 2026-02-27)
- [x] **Phase 12: Integration** - `StudyRunner.run()`, `run_study()` public API, `_run()` body, CLI study flags, study progress display, `StudyResult` assembly, multi-backend hard error (completed 2026-02-27)
- [x] **Phase 14: Multi-Cycle Execution Fixes** - Fix double `apply_cycles()`, cycle tracking, manifest completion status. Gap closure. (completed 2026-02-27)
- [x] **Phase 15: M2 Tech Debt and Progress Wiring** - Wire progress display, fix phantom field, ROADMAP/SUMMARY drift. (completed 2026-02-27)

Full details: `milestones/v1.18.0-ROADMAP.md`

</details>

### M3 — Docker + vLLM (v0.9.0)

**Milestone Goal:** Docker container infrastructure with ephemeral per-experiment lifecycle, vLLM backend activation, Docker pre-flight validation, GPU memory cleanup, full user documentation, comprehensive codebase refactoring, and test hardening.

- [x] **Phase 16: GPU Memory Verification** - NVML residual memory check before each experiment dispatch in both local and Docker paths (completed 2026-02-27)
- [x] **Phase 17: Docker Runner Infrastructure** - StudyRunner Docker dispatch path, config/result transfer via volume, per-backend runner configuration (completed 2026-02-28)
- [x] **Phase 18: Docker Pre-flight** - NVIDIA Container Toolkit detection, GPU visibility validation, CUDA/driver compatibility check (completed 2026-02-28)
- [x] **Phase 19: vLLM Backend Activation** - Fix streaming and shm-size P0 bugs, activate vLLM backend end-to-end via Docker, container entrypoint (completed 2026-02-28)
- [x] **Phase 19.1: vLLM Parameter Audit** - INSERTED — Research upstream vLLM API, expand VLLMConfig fields, wire energy-relevant params (ref: Phase 4.1 PyTorch audit) (completed 2026-03-03)
- [x] **Phase 19.2: vLLM Extended Parameters and Passthrough** - INSERTED — VLLMAttentionConfig, VLLMBeamSearchConfig, passthrough kwargs, 3-segment sweep paths (completed 2026-03-04)
- [x] **Phase 20: Docker Image and CI** - Docker publish workflow for all backends to GHCR, release pipeline with GPU CI gate (completed 2026-03-05)
- [x] **Phase 21: Measurement Carried Items** - `aienergyscore.jsonl` built-in dataset, `peak_memory_mb` inference-only semantics, dataset loader module (completed 2026-03-05)
- [x] **Phase 21.1: CI & Versioning Scaffold** - INSERTED — Migrate CI to uv, reset version scheme to 0.x, revert docker-publish removal, improve release workflow (completed 2026-03-04)
- [x] **Phase 22: Testing and CI** - Test strategy review, CI pipeline improvements, Docker path SIGINT verification, coverage analysis (completed 2026-03-05)
- [x] **Phase 23: Documentation** - Full user docs: installation, getting started, Docker setup guide, backend config guide, study YAML reference (completed 2026-03-05)
- [x] **Phase 24: M3 Integration Fixes and Retroactive Verification** - Fix preflight runner resolution, add GPU memory check to single-experiment path, retroactive verification of phases 18/20/21 (completed 2026-03-05)
- [x] **Phase 25: Dead Code Deletion** - Salvage audit of dead modules, then delete ~4,500 lines of v1.x dead code, ghost packages, and unreachable modules (completed 2026-03-06)
- [x] **Phase 26: Bug Fixes and Security** - Fix 3 active bugs (thermal throttle, FLOPs fields, study count) + HF_TOKEN security issue (completed 2026-03-07)
- [ ] **Phase 27: Deduplication and Circular Import** - Extract shared backend methods, config utilities, NVML context manager; break config-study circular import
- [ ] **Phase 28: Logging and Performance** - Unify logging to loguru, fix 8 performance issues (dead subprocesses, measurement boundary overhead, caching)
- [ ] **Phase 29: Test Cleanup and Quality** - Delete dead tests, replace 8 source-inspection tests, fix 6 tautological assertions, fix robustness issues
- [ ] **Phase 30: Test Coverage** - Close coverage gaps in core measurement + CLI paths, add E2E integration tests, target ≥75% adjusted coverage
- [ ] **Phase 31: CI Pipeline Improvements** - Remove duplicate GPU CI, add version checks, extend Docker smoke, add dependabot, coverage reporting

## Phase Details

### Phase 16: GPU Memory Verification
**Goal**: Users can be confident that residual GPU state from a prior experiment does not contaminate the next measurement
**Depends on**: Phase 15 (M2 complete)
**Requirements**: MEAS-01, MEAS-02
**Success Criteria** (what must be TRUE):
  1. Before each experiment dispatch (local and Docker paths), NVML is queried for current GPU memory usage
  2. If residual memory exceeds the configured threshold, a warning is logged before the experiment starts
  3. A clean-state experiment (no prior GPU use) produces no warning
**Plans**: 1 plan

Plans:
- [x] 16-01-PLAN.md — NVML residual memory check module + StudyRunner pre-dispatch wiring + unit tests

### Phase 17: Docker Runner Infrastructure
**Goal**: StudyRunner can dispatch experiments to ephemeral Docker containers, with config passed in and results passed out via shared volume
**Depends on**: Phase 16
**Requirements**: DOCK-01, DOCK-02, DOCK-03, DOCK-04, DOCK-05, DOCK-06, DOCK-11
**Success Criteria** (what must be TRUE):
  1. Setting `runner: docker` in config causes the experiment to execute inside an ephemeral container (`docker run --rm`)
  2. Container receives ExperimentConfig via a mounted JSON file (`LLEM_CONFIG_PATH` env var)
  3. Container writes its ExperimentResult to a shared volume; StudyRunner reads it after the process exits
  4. A multi-backend study with incompatible backends auto-elevates to Docker and surfaces guidance to the user
  5. Runner can be configured per-backend via `runners:` config section or `LLEM_RUNNER_VLLM=docker:image` env var
  6. Container entrypoint calls `ExperimentOrchestrator` directly (library API, not CLI re-entry)
**Plans**: 4 plans

Plans:
- [ ] 17-01-PLAN.md — Docker error hierarchy, container entrypoint (library API), built-in image registry (DOCK-02, DOCK-03, DOCK-11)
- [ ] 17-02-PLAN.md — Runner resolution module with precedence chain, runners config model (DOCK-06)
- [ ] 17-03-PLAN.md — DockerRunner dispatch class: subprocess.run, volume management, result collection (DOCK-01, DOCK-04)
- [ ] 17-04-PLAN.md — StudyRunner integration, auto-elevation for multi-backend studies (DOCK-05)

### Phase 18: Docker Pre-flight
**Goal**: Docker pre-flight checks catch misconfigured host environments before any container is launched, giving users actionable error messages
**Depends on**: Phase 17
**Requirements**: DOCK-07, DOCK-08, DOCK-09
**Success Criteria** (what must be TRUE):
  1. Running `llem run` with a Docker runner on a host without NVIDIA Container Toolkit produces a clear, actionable error before any container starts
  2. A container-level GPU visibility check confirms the GPU is accessible inside the container before the experiment starts
  3. A CUDA/driver version mismatch between host and container image produces a descriptive error naming the versions
**Plans**: 1 plan

Plans:
- [ ] 18-01-PLAN.md — Tiered Docker pre-flight checks (NVIDIA CT, GPU visibility, CUDA/driver), --skip-preflight CLI flag, wiring into study execution path (DOCK-07, DOCK-08, DOCK-09)

### Phase 19: vLLM Backend Activation
**Goal**: Users can run experiments using the vLLM backend via Docker and receive a valid ExperimentResult
**Depends on**: Phase 18
**Requirements**: VLLM-01, VLLM-02, VLLM-03
**Success Criteria** (what must be TRUE):
  1. `llem run config.yaml` with `backend: vllm` and `runner: docker` completes without error and produces a valid ExperimentResult
  2. vLLM streaming output is collected correctly (P0 fix: CM-07)
  3. The container is launched with `--shm-size 8g` (P0 fix: CM-09)
**Plans**: 2 plans

Plans:
- [ ] 19-01-PLAN.md — VLLMBackend class (ground-up rewrite, offline batch inference) + get_backend() registration (VLLM-01, VLLM-02)
- [ ] 19-02-PLAN.md — Unit tests for VLLMBackend + shm-size verification (VLLM-01, VLLM-02, VLLM-03)

### Phase 19.1: vLLM Parameter Audit
**Goal**: VLLMConfig exposes all energy-relevant vLLM parameters, validated against the upstream vLLM API — researchers can sweep vLLM-specific params like they do PyTorch params
**Depends on**: Phase 19
**Requirements**: VLLM-04
**Success Criteria** (what must be TRUE):
  1. VLLMConfig fields are audited against upstream vLLM `LLM()` and `SamplingParams` APIs
  2. All energy-relevant params are exposed (enforce_eager, block_size, kv_cache_dtype, swap_space, dtype, speculative_model, etc.)
  3. Sweep grammar supports vLLM-scoped dotted notation (e.g. `vllm.engine.block_size: [8, 16, 32]`)
  4. SSOT introspection updated for vLLM fields (test values, constraint metadata)
**Plans**: 2 plans

Plans:
- [ ] 19.1-01-PLAN.md — VLLMEngineConfig + VLLMSamplingConfig schema + sweep grammar fix for three-segment paths
- [ ] 19.1-02-PLAN.md — VLLMBackend wiring for nested config + SSOT introspection update + test migration

### Phase 20: Docker Image and CI
**Goal**: An official vLLM Docker image is published to GHCR and automatically updated on each release
**Depends on**: Phase 19
**Requirements**: DOCK-10
**Success Criteria** (what must be TRUE):
  1. `docker pull ghcr.io/llenergymeasure/vllm:{version}-cuda{major}` succeeds and produces a working image
  2. Pushing a release tag triggers CI to build and publish the image to GHCR automatically
**Plans**: TBD

Plans:
- [ ] 20-01: Dockerfile — vLLM base image, llenergymeasure install, entrypoint wiring
- [ ] 20-02: CI publish workflow — build and push on release tag to GHCR with version + CUDA tags

### Phase 21: Measurement Carried Items
**Goal**: The built-in dataset file exists and peak_memory_mb semantics are confirmed, closing two long-standing M1 carry-forwards
**Depends on**: Phase 16
**Requirements**: MEAS-03, MEAS-04
**Success Criteria** (what must be TRUE):
  1. `from llenergymeasure.datasets import aienergyscore` loads the built-in `.jsonl` dataset without error
  2. `peak_memory_mb` in ExperimentResult is documented with a precise definition of what it measures and when it is captured
**Plans**: TBD

Plans:
- [ ] 21-01: aienergyscore dataset — create `aienergyscore.jsonl` and expose via datasets module
- [ ] 21-02: peak_memory_mb semantics — confirm measurement point and document in code + result schema

### Phase 21.1: CI & Versioning Scaffold
**Goal**: CI pipeline uses uv with dependency caching, version scheme is reset to 0.x (pre-1.0), and release workflow is fit for the new versioning
**Depends on**: Phase 21
**Requirements**: TEST-01
**Success Criteria** (what must be TRUE):
  1. CI workflows (lint, type-check, test) use uv with `uv.lock` caching instead of pip
  2. Version in pyproject.toml and `__init__.py` is 0.8.0; `llem --version` shows 0.8.0
  3. GitHub releases page shows clean 0.x version history (v0.1.0 through v0.8.0)
  4. Release workflow builds with uv and attaches artifacts on version tags
  5. Docker publish workflow is properly integrated (not removed ad-hoc)
**Plans**: 2 plans

Plans:
- [ ] 21.1-01-PLAN.md — uv lockfile + CI migration + version reset + Dockerfile COPY fix
- [ ] 21.1-02-PLAN.md — Release workflow with CI gate + Docker publish reusable workflow

### Phase 22: Testing and CI
**Goal**: Test suite and CI pipeline are reviewed, gaps identified, and the testing strategy is fit-for-purpose for a multi-backend Docker-based measurement tool
**Depends on**: Phase 21
**Requirements**: TEST-01
**Success Criteria** (what must be TRUE):
  1. Test strategy is researched and documented: unit/integration/E2E boundaries, GPU test approach, Docker path testing
  2. CI pipeline is reviewed and improved: test matrix, caching, Docker build validation
  3. Docker path SIGINT handling is verified on real GPU hardware (Ctrl+C preserves manifest, exits 130)
  4. Test coverage gaps are identified and critical gaps addressed
**Plans**: TBD

Plans:
- [ ] 22-01: Research — test strategy for multi-backend Docker measurement tools, CI best practices, peer approaches
- [ ] 22-02: CI improvements — test matrix, caching, Docker build smoke test, coverage reporting
- [ ] 22-03: Docker SIGINT test — manual Ctrl+C on GPU hardware, verify manifest preservation and exit 130
- [ ] 22-04: Coverage gap analysis — identify untested paths, add critical missing tests

### Phase 23: Documentation
**Goal**: A researcher new to the tool can install it, run their first experiment, and configure Docker for vLLM without reading source code
**Depends on**: Phase 22
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. A user following the installation and getting started guide can run `llem run` successfully on their first attempt
  2. A user following the Docker setup guide can configure NVIDIA Container Toolkit and run a vLLM experiment via Docker
  3. The backend configuration guide explains how to switch between PyTorch (local) and vLLM (Docker) and what each requires
  4. The study YAML reference covers all sweep grammar options with working examples that a user can copy and modify
**Plans**: 5 plans

Plans:
- [ ] 23-01-PLAN.md — Delete stale docs, rewrite auto-gen scripts (config docs + CLI reference), add CI freshness check
- [ ] 23-02-PLAN.md — Core onboarding: installation.md, getting-started.md, cli-reference.md
- [ ] 23-03-PLAN.md — Docker setup guide + backend configuration guide (with param matrix)
- [ ] 23-04-PLAN.md — Study YAML reference, energy measurement, methodology, troubleshooting
- [ ] 23-05-PLAN.md — Policy maker guides (4 files) + README.md rewrite

### Phase 24: M3 Integration Fixes and Retroactive Verification
**Goal**: Close audit tech debt — fix 2 integration issues found by the integration checker and create retroactive VERIFICATION.md files for 3 unverified phases
**Depends on**: Phase 23
**Requirements**: DOCK-07, DOCK-08, DOCK-09, DOCK-10, MEAS-01, MEAS-02, MEAS-03, MEAS-04
**Gap Closure**: Closes gaps from v0.9.0 milestone audit
**Success Criteria** (what must be TRUE):
  1. `run_study_preflight()` receives `yaml_runners` and `user_config` from the caller and forwards them to `resolve_study_runners()` — runner resolution in preflight matches `_api._run()`
  2. `_run_in_process()` calls `check_gpu_memory_residual()` before running a single experiment (both local and Docker paths)
  3. Phase 18 (Docker pre-flight) has a VERIFICATION.md with DOCK-07, DOCK-08, DOCK-09 verified against current codebase
  4. Phase 20 (Docker image and CI) has a VERIFICATION.md with DOCK-10 verified against current codebase
  5. Phase 21 (Measurement carried items) has a VERIFICATION.md with MEAS-03, MEAS-04 verified against current codebase
**Plans**: 2 plans

Plans:
- [ ] 24-01-PLAN.md — Fix preflight runner resolution (forward yaml_runners/user_config) + add GPU memory check to _run_in_process (DOCK-07, DOCK-08, DOCK-09, MEAS-01, MEAS-02)
- [ ] 24-02-PLAN.md — Retroactive VERIFICATION.md for phases 18, 20, 21 (DOCK-07, DOCK-08, DOCK-09, DOCK-10, MEAS-03, MEAS-04)

### Phase 25: Dead Code Deletion
**Goal**: Audit dead code for salvageable patterns, then remove all v1.x dead code, ghost packages, and unreachable modules - reducing the codebase by ~4,500 lines and eliminating the dead dual backend system
**Depends on**: Phase 24
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. A salvage audit has been performed: all dead modules reviewed for reusable logic, patterns worth preserving documented before deletion
  2. All dead v1.x CLI files deleted (`cli/experiment.py`, `cli/config.py`, `cli/results.py`, `cli/utils.py`)
  3. Entire `cli/display/` Rich package deleted (5 files, ~550 lines)
  4. Dead `core/inference_backends/` system deleted (4 files, ~3,500 lines) - resolves DA1/DA2
  5. Dead orchestration functions deleted (`launcher.py` dead functions, `orchestration/runner.py`, `context.py`, `factory.py`)
  6. Dead core modules deleted (`compute_metrics.py`, `model_loader.py`, `distributed.py`, `parallelism.py`, `gpu_utilisation.py`, `prompts.py`, `implementations.py`, `inference.py`)
  7. Dead results modules deleted (`exporters.py`, `repository.py`, `timeseries.py`)
  8. Dead misc modules deleted (`logging.py`, `resilience.py`, `state/`, `notifications/`, `infra/subprocess.py`, `config/quantization.py`)
  9. Dead functions in `config/validation.py` deleted (`validate_parallelism_constraints`)
  10. Dead `core/dataset_loader.py` deleted (active system is `datasets/loader.py`)
  11. All `__init__.py` re-exports and `__all__` lists updated to reflect deletions
  12. Tests pass, lint clean, no broken imports
**Plans**: 3 plans
**Source**: `.planning/simplify-audit/AUDIT-REPORT.md` (P1.1-P1.9), `test-coverage-gaps.md` (dead code inventory)

Discussion (research → CONTEXT.md → plans):
- Salvage inventory: which dead modules contain logic worth preserving for M4+ (TensorRT skeleton, distributed launch, CSV export, FlopsEstimator strategies)?
- What is the safe deletion order to avoid broken imports mid-refactor?

Plans:
- [ ] 25-01-PLAN.md — Salvage audit: document TensorRT skeleton + InferenceBackend protocol in .product/designs/, preserve CSV column ordering comment in aggregation.py
- [ ] 25-02-PLAN.md — Bulk file deletion: 34 dead files/directories (cli/display/, core/inference_backends/, orchestration dead cluster + launcher.py, dead results, dead core, dead misc, ghost packages), co-delete test_cli_experiment.py + test_csv_exporter_no_loguru
- [ ] 25-03-PLAN.md — Targeted cleanup: remove dead functions from live files (validate_parallelism_constraints, _parse_flops_string, flops_reduction_factor), final verification pass

### Phase 26: Bug Fixes and Security
**Goal**: Fix all confirmed bugs that produce incorrect measurement output and close the HF_TOKEN security issue
**Depends on**: Phase 25
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. B1 fixed: `thermal_bit` uses `HwThermalSlowdown` and `sw_thermal_bit` uses `SwThermalSlowdown` in `power_thermal.py` - thermal throttle metadata is no longer silently corrupted
  2. B2 fixed: `flops_per_token` and `flops_per_second` are computed from available data (`flops_total`, `output_tokens`, `inference_time_sec`) instead of hardcoded to 0.0
  3. B3 fixed: `StudySummary.total_experiments` counts correctly without double-multiplying by `n_cycles`
  4. S1 fixed: `HF_TOKEN` passed via `--env-file` with a temp file instead of plaintext CLI arg in `docker_runner.py`
  5. All existing tests pass; new tests added for each fix
**Plans**: 2 plans
**Source**: `.planning/simplify-audit/AUDIT-REPORT.md` (B1-B3, S1)

Note: B4-B7 are in dead code deleted by Phase 25. Active bugs are B1, B2, B3, and S1.

Discussion (research → CONTEXT.md → plans):
- How do peer tools pass secrets to Docker containers? (env-file, Docker secrets, tmpfs mount patterns)
- NVML thermal throttle constant naming: verify against upstream `pynvml`/`nvidia-ml-py` source to confirm correct Hw vs Sw constants
- How do peer measurement tools compute derived FLOPs metrics (per-token, per-second)?
- How do peer study/sweep tools count experiments when cycles are involved?

Plans:
- [ ] 26-01-PLAN.md — Measurement bug fixes: thermal throttle constants (B1), FLOPs derived fields (B2), study experiment count (B3) + unit tests for each
- [ ] 26-02-PLAN.md — Security fix: HF_TOKEN env-file pattern (S1), temp file lifecycle, process table verification

### Phase 27: Deduplication and Circular Import
**Goal**: Consolidate duplicated code across backends, config, and infrastructure - extract shared utilities and break the config-study circular import
**Depends on**: Phase 26
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. P2.1: Shared backend methods extracted to `core/backends/_shared.py` (`_build_result`, `_collect_warnings`, `_cuda_sync`, `_check_persistence_mode`, `_MeasurementData`) - ~280 lines deduplicated
  2. P2.2: Single `compute_config_hash()` utility replaces 5 independent SHA-256[:16] implementations
  3. P2.3: `_unflatten` and `_deep_merge` extracted to `config/_dict_utils.py` - 3 copies consolidated to 1
  4. P2.4: Single `is_backend_available` implementation (if both survive Phase 25)
  5. P2.5: `nvml_context()` context manager in `core/gpu_info.py` replaces 12+ ad-hoc `try/finally: nvmlShutdown()` sites
  6. P2.6: `_save_and_record()` helper replaces 3 duplicated save-result + manifest-update blocks
  7. P2.7: Docker dispatch logic consolidated between `_api.py` and `study/runner.py`
  8. P2.9: `thermal_floor_wait()` extracted to `core/warmup.py` from both backends
  9. P2.10: NVIDIA toolkit binary list defined once, imported in both `runner_resolution.py` and `docker_preflight.py`
  10. P2.11: Docker error `__init__` moved to `DockerError` base class, removed from 6 subclasses
  11. P3.4: `config/loader.py` no longer imports from `study/grid.py` - circular import broken
  12. All existing tests pass; deduplication is behaviour-preserving
**Plans**: 3 plans
**Source**: `.planning/simplify-audit/AUDIT-REPORT.md` (P2.1-P2.11, P3.4)

Discussion (research → CONTEXT.md → plans):
- **Backend plugin architecture**: How do peer measurement/benchmarking tools (lm-eval-harness, mlperf, zeus, codecarbon) structure the boundary between "measurement harness" and "inference backend"? What is the minimal interface a backend plugin must implement? How do they ensure new backends don't re-implement measurement infra?
- How do peer tools (pytorch-lightning, huggingface transformers, vllm) structure shared backend utilities and avoid code duplication across backends?
- NVML lifecycle patterns: how do tools that wrap pynvml (zeus, codecarbon, nvitop) manage init/shutdown? Context manager vs singleton vs lazy init?
- How do peer Python projects break circular imports? (private util modules, lazy imports, dependency inversion)
- Config hash consolidation: what hashing patterns do peer tools use for experiment deduplication?
- What should the `InferenceBackend` protocol look like post-refactor? Current `run(config) -> ExperimentResult` bundles too much - should backends return raw inference data while the harness handles measurement?

Plans:
- [ ] 27-01-PLAN.md — Circular import + dict utils: move `study/grid.py` → `config/grid.py`, create `config/_dict_utils.py`, update all importers (P3.4, P2.3)
- [ ] 27-02-PLAN.md — MeasurementHarness extraction: `BackendPlugin` protocol, `InferenceOutput` dataclass, `core/harness.py`, shrink backends to thin plugins, wire callers, `thermal_floor_wait()` to `core/warmup.py` (P2.1, P2.9)
- [ ] 27-03-PLAN.md — Infrastructure dedup: `nvml_context()` replaces 12+ NVML sites, `DockerError.__init__` base class, NVIDIA toolkit list canonical, `_save_and_record()` helper (P2.5, P2.6, P2.10, P2.11)

### Phase 28: Logging Standardisation and Performance
**Goal**: Unify all logging to loguru with f-strings, and fix performance issues that add unnecessary latency to measurements
**Depends on**: Phase 27
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. P2.8: All 9+ files using `logging.getLogger(__name__)` migrated to `from loguru import logger` with f-string formatting
  2. P4.1: Dead `nvidia-smi` subprocess removed from `collect_compute_metrics()` - saves 150-200ms per measurement
  3. P4.2: `_cuda_sync()` no longer calls `importlib.util.find_spec("torch")` at the measurement boundary
  4. P4.3: `ThreadPoolExecutor` hoisted outside the per-sample loop in FLOPs fallback
  5. P4.4: `collect_environment_snapshot()` runs `pip freeze`/`conda list` in a background thread (or made opt-in)
  6. P4.5: `is_docker_available()` cached with `@functools.lru_cache`
  7. P4.7: `_load_jsonl` short-circuits after `n` records for the interleaved path
  8. P4.8: `compute_measurement_config_hash` results cached to avoid O(N*C) recomputation in study loop
  9. P4.9: Dead `nvidia-smi` subprocess removed from `get_cuda_major_version()`
  10. No behavioural changes beyond performance improvements; all tests pass
**Plans**: 2 plans
**Source**: `.planning/simplify-audit/AUDIT-REPORT.md` (P2.8, P4.1-P4.9)

Note: P4.6 (StateManager O(N) filename scan) deferred - requires filename schema change with low frequency impact.

Discussion (research → CONTEXT.md → plans):
- Logging in measurement tools: how do peer tools (zeus, codecarbon, mlflow) handle logging? loguru vs structlog vs stdlib - what do mature Python tools choose and why?
- loguru configuration patterns: sinks, formatting, level filtering, test integration (caplog compatibility)
- Environment snapshot performance: how do peer tools collect environment metadata without blocking experiment start? (lazy, background thread, opt-in)
- Measurement boundary overhead: what precision do energy measurement tools need at the CUDA sync point? What overhead is acceptable?

Plans:
- [ ] 28-01-PLAN.md — Logging migration: replace `logging.getLogger` with `loguru.logger` across `core/baseline.py`, `core/flops.py`, `core/power_thermal.py`, `core/warmup.py`, `core/state.py`, `domain/environment.py`, `core/backends/pytorch.py`, `core/backends/vllm.py`, plus %-style to f-string conversion
- [ ] 28-02-PLAN.md — Performance fixes: dead nvidia-smi removal (P4.1, P4.9), find_spec guard removal (P4.2), ThreadPoolExecutor hoist (P4.3), background env snapshot (P4.4), lru_cache on is_docker_available (P4.5), JSONL short-circuit (P4.7), config hash caching (P4.8)

### Phase 29: Test Cleanup and Quality
**Goal**: Fix broken, fragile, and meaningless tests so the suite is trustworthy before adding new coverage
**Depends on**: Phase 28
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. Dead test file `test_cli_experiment.py` deleted
  2. All 8 source-inspection tests replaced with behavioural equivalents (test_cli_run.py ×2, test_cli_config.py ×1, test_vllm_backend.py ×2, test_aggregation_v2.py ×2, test_peak_memory.py ×all)
  3. All 6 tautological assertions fixed to test real conditions (test_backend_protocol.py ×2, test_backend_detection.py ×2, test_config_introspection.py, test_flops_v2.py)
  4. Robustness issues fixed: timing assertion in test_study_gaps.py, probabilistic assertion in test_study_grid.py, `builtins.open` patch narrowed in test_environment_snapshot.py, `lru_cache` cleanup via autouse fixture in test_image_registry.py, `Path.exists` patch narrowed in test_docker_detection.py
  5. `FakeResultsRepository.load()` uses path-based lookup instead of returning last-saved
  6. All relative paths in tests replaced with `__file__`-relative or `importlib.resources` paths
  7. All tests pass; no new test failures introduced
**Plans**: 2 plans
**Source**: `.planning/simplify-audit/test-quality-audit.md`

Discussion (research → CONTEXT.md → plans):
- How do peer Python tools test hardware-dependent code? (mock strategies for pynvml, CUDA, Docker daemon)
- Source-inspection tests: are there legitimate uses, or should they always be behavioural? What do testing best-practice guides say?
- Fake/stub design patterns: how do peer tools (pytest-mock patterns, httpx test client, respx) design test doubles that faithfully model real behaviour?
- Flaky test mitigation: what strategies do large Python projects use? (deterministic seeds, time-independent assertions, fixture-based cleanup)

Plans:
- [ ] 29-01-PLAN.md — Delete dead tests and replace fragile tests: delete test_cli_experiment.py, replace 8 source-inspection tests with behavioural equivalents, fix all relative paths
- [ ] 29-02-PLAN.md — Fix assertions and robustness: fix 6 tautological assertions, fix timing/probabilistic flakiness, narrow global patches, fix FakeResultsRepository, lru_cache cleanup fixtures

### Phase 30: Test Coverage
**Goal**: Close critical coverage gaps in core measurement, CLI, and integration paths - target adjusted coverage ≥75%
**Depends on**: Phase 29
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. `core/baseline.py` coverage: 37% → ≥80% (cache hit/miss paths, NVML failure paths, invalidation)
  2. `core/power_thermal.py` coverage: 30% → ≥75% (sample loop with mocked pynvml, mean power, throttle detection)
  3. `core/flops.py` coverage: 27% → ≥70% (FlopsEstimator dispatch chain, precision detection, architecture/parameter fallbacks)
  4. `core/environment.py` coverage: 12% → ≥70% (GPU/CUDA/thermal/CPU/container collection with mocked NVML)
  5. `cli/_vram.py` coverage: 10% → ≥70% (VRAM estimation, HF Hub mock, precision bytes mapping)
  6. `cli/run.py` coverage: 52% → ≥75% (error exit codes, dry-run, study detection, CLI defaults)
  7. `cli/config_cmd.py` coverage: 47% → ≥70% (GPU probe, backend version probe, verbose paths)
  8. `cli/_display.py` coverage: 54% → ≥70% (result summary branches, dry-run output, study progress)
  9. E2E pipeline integration test: YAML → StudyConfig → FakeBackends → StudyResult (unit-level, no GPU)
  10. CLI E2E test: `llem run config.yaml` via CliRunner with tmp_path YAML and fake backends
  11. Adjusted test coverage ≥75% (excluding GPU-only code paths)
**Plans**: 3 plans
**Source**: `.planning/simplify-audit/test-coverage-gaps.md`

Discussion (research → CONTEXT.md → plans):
- How do peer measurement/benchmarking tools (mlperf, lm-eval-harness, zeus) structure their test suites? What coverage targets do they aim for?
- E2E integration test patterns: how do tools with complex pipelines (config → runner → result) test the full chain without real hardware?
- Coverage reporting: Codecov vs Coveralls vs HTML artifacts - what works best for single-developer open-source projects?
- What is a realistic and useful coverage target for a GPU-dependent measurement tool where significant code paths require real hardware?

Plans:
- [ ] 30-01-PLAN.md — Core measurement coverage: baseline.py (8 tests), power_thermal.py (10 tests), flops.py (14 tests), environment.py (12 tests)
- [ ] 30-02-PLAN.md — CLI coverage: _vram.py (8 tests), run.py (9 tests), config_cmd.py (8 tests), _display.py (10+ tests)
- [ ] 30-03-PLAN.md — Integration tests: E2E pipeline test (YAML → StudyConfig → FakeBackends → StudyResult), CLI E2E test (llem run config.yaml), coverage reporting setup

### Phase 31: CI Pipeline Improvements
**Goal**: Fix CI issues identified by the pipeline audit - remove redundancy, add safety checks, improve reliability
**Depends on**: Phase 25 (can run in parallel with Phases 29-30)
**Requirements**: (internal quality - no REQUIREMENTS.md entries)
**Success Criteria** (what must be TRUE):
  1. GPU CI no longer runs twice in the release path (removed from `release.yml`, kept in `auto-release.yml`)
  2. Version tag ↔ pyproject.toml consistency check in `release.yml` - mismatched tags fail early
  3. Test markers aligned between `ci.yml` and `release.yml` (`-m "not gpu and not docker"` in both)
  4. `mkdir -p results/` added to `gpu-ci.yml` before first Docker bind mount
  5. `-n auto` added to `release.yml` test job for parallel test execution
  6. Docker smoke test extended to include `Dockerfile.pytorch` (not just vllm)
  7. Version extraction in `auto-release.yml` uses `tomllib` instead of fragile grep/sed
  8. `dependabot.yml` added for GitHub Actions version pinning
  9. Version sync check added (pyproject.toml ↔ `__init__.py`) in CI
  10. Coverage reporting configured (`--cov` flag, XML output, optional Codecov upload)
**Plans**: 2 plans
**Source**: `.planning/simplify-audit/ci-pipeline-audit.md`

Discussion (research → CONTEXT.md → plans):
- How do peer Python/ML projects structure their GitHub Actions CI? (pytorch, huggingface, vllm, mlflow)
- Release pipeline patterns: how do mature projects gate releases on GPU CI without redundant runs?
- Dependabot configuration for GitHub Actions: SHA pinning, update frequency, security-only vs all updates
- Version consistency enforcement: how do projects with dual version sources (pyproject.toml + __init__.py) keep them in sync?

Plans:
- [ ] 31-01-PLAN.md — CI fixes: remove duplicate GPU CI (MEDIUM), version tag check (MEDIUM), align test markers (MEDIUM), mkdir results (MEDIUM), parallel release tests (LOW)
- [ ] 31-02-PLAN.md — CI additions: Docker smoke matrix, tomllib version extraction, dependabot.yml, version sync check, coverage reporting

## Progress

**Execution Order:**
Phases execute in numeric order: 16 → ... → 24 → 25 → 26 → 27 → 28 → 29 → 30 → 31
Phase 21 can run in parallel with Phase 20 (no dependency between them).
Phase 21.1 (CI scaffold) before Phase 22 (Testing) — CI must be stable before testing improvements.
Phase 22 (Testing/CI) before Phase 23 (Docs) — testing informs documentation.
Phase 24 (Gap closure) after milestone audit.
Phase 25 (Dead code) → 26 (Bugs) → 27 (Dedup) → 28 (Logging/Perf): strict sequence - each builds on prior cleanup.
Phase 29 (Test cleanup) → 30 (Test coverage): tests must target clean code only.
Phase 31 (CI) can run in parallel with Phases 29-30 (touches workflow files, not source or tests).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 16. GPU Memory Verification | 1/1 | Merged (PR #24) | 2026-02-28 |
| 17. Docker Runner Infrastructure | 4/4 | Complete    | 2026-02-28 |
| 18. Docker Pre-flight | 1/1 | Complete   | 2026-02-28 |
| 19. vLLM Backend Activation | 2/2 | Complete    | 2026-02-28 |
| 19.1. vLLM Parameter Audit | 2/2 | Complete    | 2026-03-03 |
| 19.2. vLLM Extended Params | 2/2 | Complete    | 2026-03-04 |
| 20. Docker Image and CI | 2/2 | Complete    | 2026-03-05 |
| 21. Measurement Carried Items | 2/2 | Complete    | 2026-03-05 |
| 21.1. CI & Versioning Scaffold | 2/2 | Complete    | 2026-03-04 |
| 22. Testing and CI | 4/4 | Complete    | 2026-03-05 |
| 23. Documentation | 5/5 | Complete    | 2026-03-05 |
| 24. Integration Fixes + Verification | 2/2 | Complete    | 2026-03-05 |
| 25. Dead Code Deletion | 3/3 | Complete    | 2026-03-06 |
| 26. Bug Fixes and Security | 2/2 | Complete   | 2026-03-07 |
| 27. Deduplication + Circular Import | 2/3 | In Progress|  |
| 28. Logging + Performance | 0/2 | Not started | — |
| 29. Test Cleanup + Quality | 0/2 | Not started | — |
| 30. Test Coverage | 0/3 | Not started | — |
| 31. CI Pipeline | 0/2 | Not started | — |

---

*M1 roadmap created: 2026-02-26*
*M2 roadmap appended: 2026-02-27*
*M3 roadmap appended: 2026-02-27*
*M1 shipped: 2026-02-27 (v0.7.0)*
*M2 shipped: 2026-02-27 (v0.8.0)*
*Version scheme reset: 2026-03-04 (1.x → 0.x)*
*Phase 13 (docs) folded into M3 Phase 22: 2026-02-27*
*Phases 25-31 rewritten: 2026-03-06 (refactor + test hardening, replacing 2-phase stubs)*
