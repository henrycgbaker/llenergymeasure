# Roadmap: LLenergyMeasure

## Milestones

- [x] **v1.x Foundation & Planning** — Phases 1–4.5 (shipped 2026-02-26)
- [x] **v1.17.0 M1 — Core Single-Experiment** — Phases 1–8.2 (shipped 2026-02-27)
- [x] **v1.18.0 M2 — Study / Sweep** — Phases 9–15 (shipped 2026-02-27)
- [ ] **v1.19.0 M3 — Docker + vLLM** — Phases 16–23 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3...): Planned milestone work
- Decimal phases (9.1, 10.1...): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v1.17.0 M1 — Core Single-Experiment (Phases 1–8.2) — SHIPPED 2026-02-27</summary>

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
<summary>✅ v1.18.0 M2 — Study / Sweep (Phases 9–15) — SHIPPED 2026-02-27</summary>

- [x] **Phase 9: Grid Expansion and StudyConfig** - Sweep YAML grammar, `StudyConfig` + `ExecutionConfig` models, Cartesian grid expander, cycle ordering, pre-flight count display (completed 2026-02-27)
- [x] **Phase 10: Manifest Writer** - `StudyManifest` checkpoint model, `ManifestWriter` with atomic writes, study output directory layout (completed 2026-02-27)
- [x] **Phase 11: Subprocess Isolation and StudyRunner** - Subprocess dispatch via `spawn`, `Pipe`/`Queue` IPC, timeout handling, SIGINT, skip-and-continue, thermal gaps (completed 2026-02-27)
- [x] **Phase 12: Integration** - `StudyRunner.run()`, `run_study()` public API, `_run()` body, CLI study flags, study progress display, `StudyResult` assembly, multi-backend hard error (completed 2026-02-27)
- [x] **Phase 14: Multi-Cycle Execution Fixes** - Fix double `apply_cycles()`, cycle tracking, manifest completion status. Gap closure. (completed 2026-02-27)
- [x] **Phase 15: M2 Tech Debt and Progress Wiring** - Wire progress display, fix phantom field, ROADMAP/SUMMARY drift. (completed 2026-02-27)

Full details: `milestones/v1.18.0-ROADMAP.md`

</details>

### M3 — Docker + vLLM (v1.19.0)

**Milestone Goal:** Docker container infrastructure with ephemeral per-experiment lifecycle, vLLM backend activation, Docker pre-flight validation, GPU memory cleanup, and full user documentation.

- [x] **Phase 16: GPU Memory Verification** - NVML residual memory check before each experiment dispatch in both local and Docker paths (completed 2026-02-27)
- [x] **Phase 17: Docker Runner Infrastructure** - StudyRunner Docker dispatch path, config/result transfer via volume, per-backend runner configuration (completed 2026-02-28)
- [x] **Phase 18: Docker Pre-flight** - NVIDIA Container Toolkit detection, GPU visibility validation, CUDA/driver compatibility check (completed 2026-02-28)
- [x] **Phase 19: vLLM Backend Activation** - Fix streaming and shm-size P0 bugs, activate vLLM backend end-to-end via Docker, container entrypoint (completed 2026-02-28)
- [x] **Phase 19.1: vLLM Parameter Audit** - INSERTED — Research upstream vLLM API, expand VLLMConfig fields, wire energy-relevant params (ref: Phase 4.1 PyTorch audit) (completed 2026-03-03)
- [x] **Phase 19.2: vLLM Extended Parameters and Passthrough** - INSERTED — Beam search, attention config, compilation passthrough, CPU offload, passthrough architecture for all backend configs (completed 2026-03-03)
- [ ] **Phase 20: Docker Image and CI** - Official vLLM Docker image published to GHCR, CI publish on release tag
- [ ] **Phase 21: Measurement Carried Items** - `aienergyscore.jsonl` built-in dataset, `peak_memory_mb` semantics confirmed and documented
- [ ] **Phase 22: Testing** - Manual Ctrl+C SIGINT test on GPU hardware for Docker path
- [ ] **Phase 23: Documentation** - Full user docs: installation, getting started, Docker setup guide, backend config guide, study YAML reference

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

### Phase 19.2: vLLM Extended Parameters and Passthrough (INSERTED)
**Goal**: All backend configs accept arbitrary passthrough kwargs alongside explicitly-typed energy-relevant fields — researchers can use any upstream parameter without waiting for us to type it
**Depends on**: Phase 19.1
**Requirements**: VLLM-05
**Success Criteria** (what must be TRUE):
  1. VLLMEngineConfig, VLLMSamplingConfig use `extra="allow"` — unknown fields pass through to vLLM
  2. PyTorchConfig uses `extra="allow"` — same passthrough architecture for consistency
  3. VLLMBeamSearchConfig exists as third nested section with beam_width, length_penalty, early_stopping, max_tokens
  4. VLLMAttentionConfig exists with full native mirror of vLLM's AttentionConfig (~11 fields)
  5. compilation_config passthrough dict passes directly to vLLM's CompilationConfig
  6. kv_cache_memory_bytes field with mutual exclusion against gpu_memory_utilization
  7. CPU offload fields (offload_group_size, offload_num_in_group, offload_prefetch_step, offload_params) wired
  8. Sampling `n` parameter (num sequences) wired
  9. SSOT introspection covers all new explicitly-typed fields
  10. All backends' extra="allow" passthrough kwargs forwarded correctly at runtime
**Plans**: TBD

Plans:
- [ ] 19.2-01-PLAN.md — Schema changes: VLLMBeamSearchConfig, VLLMAttentionConfig, new fields, extra="allow", cross-validators (VLLM-05)
- [ ] 19.2-02-PLAN.md — Runtime wiring (new field forwarding, model_extra passthrough, beam search path), PyTorch model_extra, SSOT introspection updates, unit tests (VLLM-05)

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

### Phase 22: Testing
**Goal**: Docker path SIGINT handling is verified on real GPU hardware
**Depends on**: Phase 19
**Requirements**: TEST-01
**Success Criteria** (what must be TRUE):
  1. Pressing Ctrl+C during a Docker-dispatched study gracefully stops the container, preserves the manifest, and exits with code 130
**Plans**: TBD

Plans:
- [ ] 22-01: Manual SIGINT test — Docker path Ctrl+C on GPU hardware, verify manifest preservation and exit 130

### Phase 23: Documentation
**Goal**: A researcher new to the tool can install it, run their first experiment, and configure Docker for vLLM without reading source code
**Depends on**: Phase 19, Phase 21, Phase 22
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. A user following the installation and getting started guide can run `llem run` successfully on their first attempt
  2. A user following the Docker setup guide can configure NVIDIA Container Toolkit and run a vLLM experiment via Docker
  3. The backend configuration guide explains how to switch between PyTorch (local) and vLLM (Docker) and what each requires
  4. The study YAML reference covers all sweep grammar options with working examples that a user can copy and modify
**Plans**: TBD

Plans:
- [ ] 23-01: Core user docs — installation, getting started, configuration reference
- [ ] 23-02: Docker setup guide — NVIDIA Container Toolkit, host requirements, image selection
- [ ] 23-03: Backend config guide — PyTorch local, vLLM Docker, runner selection
- [ ] 23-04: Study YAML reference — sweep grammar, cycle ordering, examples

## Progress

**Execution Order:**
Phases execute in numeric order: 16 → 17 → 18 → 19 → 19.1 → 19.2 → 20 → 21 → 22 → 23
Phase 21 can run in parallel with Phase 20 (no dependency between them).
Phase 22 (SIGINT testing) runs before Phase 23 (docs) so documentation reflects verified behaviour.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 16. GPU Memory Verification | 1/1 | Merged (PR #24) | 2026-02-28 |
| 17. Docker Runner Infrastructure | 4/4 | Complete    | 2026-02-28 |
| 18. Docker Pre-flight | 1/1 | Complete   | 2026-02-28 |
| 19. vLLM Backend Activation | 2/2 | Complete    | 2026-02-28 |
| 19.1. vLLM Parameter Audit | 2/2 | Complete    | 2026-03-03 |
| 19.2. vLLM Extended Params & Passthrough | 2/2 | Complete    | 2026-03-03 |
| 20. Docker Image and CI | 0/TBD | Not started | - |
| 21. Measurement Carried Items | 0/TBD | Not started | - |
| 22. Testing | 0/TBD | Not started | - |
| 23. Documentation | 0/TBD | Not started | - |

---

*M1 roadmap created: 2026-02-26*
*M2 roadmap appended: 2026-02-27*
*M3 roadmap appended: 2026-02-27*
*M1 shipped: 2026-02-27 (v1.17.0)*
*M2 shipped: 2026-02-27 (v1.18.0)*
*Phase 13 (docs) folded into M3 Phase 23: 2026-02-27*
