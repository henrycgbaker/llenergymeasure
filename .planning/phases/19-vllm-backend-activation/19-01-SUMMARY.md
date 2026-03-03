---
phase: 19-vllm-backend-activation
plan: 01
subsystem: infra
tags: [vllm, inference, backend, energy-measurement, offline-batch]

# Dependency graph
requires:
  - phase: 17-docker-runner
    provides: container entrypoint that calls get_backend(config.backend)
  - phase: 18-docker-pre-flight
    provides: Docker pre-flight checks that gate vLLM container execution
provides:
  - VLLMBackend class in core/backends/vllm.py with full measurement lifecycle
  - get_backend('vllm') registration in core/backends/__init__.py
  - detect_default_backend() vllm detection (after pytorch priority)
affects:
  - phase: 19-02 (vLLM parameter audit — builds on this backend)
  - container_entrypoint (can now dispatch backend=vllm experiments)
  - future TRT-LLM and SGLang backends (same lifecycle pattern)

# Tech tracking
tech-stack:
  added: [vllm.LLM, vllm.SamplingParams (lazy imports)]
  patterns:
    - Offline batch inference via single llm.generate(prompts, sampling_params) call
    - Lazy vLLM/torch imports — module importable on hosts without vLLM installed
    - Best-effort FLOPs estimation with graceful None fallback for vLLM
    - top_k sentinel mapping: 0 (our disabled) → -1 (vLLM disabled)
    - Greedy decode detection: temperature==0.0 or do_sample==False

key-files:
  created:
    - src/llenergymeasure/core/backends/vllm.py
  modified:
    - src/llenergymeasure/core/backends/__init__.py

key-decisions:
  - "VLLMBackend uses offline batch mode only — no streaming, CM-07 resolved structurally"
  - "FLOPs estimation wrapped in try/except — vLLM exposes HF model via internal path, defaults to 0.0 on failure"
  - "top_k=0 (our disabled sentinel) maps to top_k=-1 (vLLM's disabled sentinel)"
  - "PyTorch takes priority over vLLM in detect_default_backend() — pytorch is simpler default"

patterns-established:
  - "Lazy vLLM/torch imports in all method bodies — module-level imports would break hosts without vLLM"
  - "Measurement lifecycle for vLLM: snapshot → baseline → load → warmup(1 prompt, 1 token) → thermal → energy → sync → generate(all) → sync → stop"
  - "Token counting from RequestOutput: prompt_token_ids for input, outputs[0].token_ids for output"

requirements-completed: [VLLM-01, VLLM-02]

# Metrics
duration: 7min
completed: 2026-02-28
---

# Phase 19 Plan 01: vLLM Backend Activation Summary

**VLLMBackend with offline batch inference via vllm.LLM().generate(), registered in get_backend(), CM-07 streaming bug resolved structurally by design**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-28T11:02:25Z
- **Completed:** 2026-02-28T11:09:34Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- VLLMBackend class written from scratch (642 lines) with the full 17-step measurement lifecycle matching PyTorchBackend's structure but using vLLM's offline batch API
- CM-07 (streaming P0 bug) resolved structurally — no streaming code exists, offline batch mode only
- Backend registered in get_backend() and detect_default_backend(); error messages updated to list both pytorch and vllm
- All vLLM/torch imports are lazy — module importable on hosts without vLLM installed
- mypy strict + ruff pass on both modified files

## Task Commits

Each task was committed atomically:

1. **Task 1: Write VLLMBackend class with offline batch inference lifecycle** - `59b5c2c` (feat)
2. **Task 2: Register vllm in get_backend() and detect_default_backend()** - `3eb45cb` (feat)

**Plan metadata:** (docs commit — see final commit hash)

## Files Created/Modified

- `src/llenergymeasure/core/backends/vllm.py` — VLLMBackend class, 642 lines, full lifecycle implementation
- `src/llenergymeasure/core/backends/__init__.py` — Added vllm branch to get_backend(), vllm detection in detect_default_backend(), updated error messages

## Decisions Made

- **Offline batch only**: Single llm.generate(prompts, sampling_params) call with all prompts. No streaming code. Resolves CM-07 structurally — not a patch but an architectural decision.
- **FLOPs estimation best-effort**: vLLM doesn't expose a standard HuggingFace model object. Attempted via internal API path (llm.llm_engine.model_executor.driver_worker.model_runner.model) wrapped in try/except, defaults to 0.0 on failure.
- **top_k sentinel mapping**: Our config uses 0 as "disabled"; vLLM uses -1. Mapped explicitly in _build_sampling_params().
- **PyTorch priority in detect_default_backend()**: If both transformers and vllm are installed, pytorch is returned first — it's the simpler, always-available default.
- **Lazy imports throughout**: All `from vllm import ...` and `import torch` calls are inside method bodies. Module can be imported on any host regardless of backend availability.

## Deviations from Plan

None - plan executed exactly as written.

Note: mypy type annotations were added with `Any` for lazy-loaded vLLM types (llm, sampling_params) to satisfy strict mypy. This is the correct pattern for untyped lazy imports — matches the project's existing approach.

## Issues Encountered

None — all 6 verification checks passed immediately after implementation.

## Next Phase Readiness

- VLLMBackend is ready — container entrypoint can now dispatch `backend: vllm` experiments
- Phase 19-02 (vLLM parameter audit and VLLMConfig expansion) can build directly on this backend
- FLOPs estimation will return 0.0 until a better internal API path is confirmed with a running vLLM instance

## Self-Check: PASSED

- FOUND: src/llenergymeasure/core/backends/vllm.py
- FOUND: .planning/phases/19-vllm-backend-activation/19-01-SUMMARY.md
- FOUND: commit 59b5c2c (Task 1 — VLLMBackend class)
- FOUND: commit 3eb45cb (Task 2 — backend registration)
- FOUND: commit bd0abbe (docs — plan metadata)

---
*Phase: 19-vllm-backend-activation*
*Completed: 2026-02-28*
