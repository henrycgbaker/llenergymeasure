---
phase: 23-documentation
plan: 03
subsystem: docs
tags: [docker, nvidia-container-toolkit, vllm, pytorch, backends, documentation]

requires:
  - phase: 23-01
    provides: auto-generation pipeline, stale docs cleared

provides:
  - docs/docker-setup.md — full NVIDIA Container Toolkit walkthrough with troubleshooting
  - docs/backends.md — backend comparison, runner config system, parameter support matrix

affects:
  - 23-02 (getting-started.md references docker-setup.md)
  - 23-04 (study-config.md may reference backends.md for backend-specific sweep syntax)

tech-stack:
  added: []
  patterns:
    - "Docker setup guide: step-by-step with verification at each stage"
    - "Parameter matrix: manually constructed from Pydantic config models, regeneratable from GPU test results"
    - "vLLM nested config structure: vllm.engine / vllm.sampling mirrors vLLM's own API separation"

key-files:
  created:
    - docs/docker-setup.md
    - docs/backends.md
  modified: []

key-decisions:
  - "Parameter matrix manually constructed from PyTorchConfig, VLLMEngineConfig, VLLMSamplingConfig, TensorRTConfig model fields — GPU test results not available on host"
  - "docker-setup.md uses ubuntu22.04-based probe image consistent with docker_preflight.py _PROBE_IMAGE constant"
  - "backends.md covers vLLM beam_search section as mutually exclusive with sampling section per VLLMConfig validator"

patterns-established:
  - "Cross-link pattern: docker-setup.md links back to getting-started.md; backends.md links to docker-setup.md as prerequisite"
  - "Passthrough documentation: unknown fields under vllm.engine/sampling forwarded to vLLM native APIs"

requirements-completed: [DOCS-02, DOCS-03]

duration: 3min
completed: 2026-03-05
---

# Phase 23 Plan 03: Docker Setup and Backend Configuration Summary

**Docker setup guide (full NVIDIA CT walkthrough from zero) and backend configuration guide (PyTorch vs vLLM switching, runner precedence, manually constructed parameter support matrix) written against actual config models.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-05T16:48:15Z
- **Completed:** 2026-03-05T16:51:13Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `docs/docker-setup.md` — 5-step walkthrough from Docker install through NVIDIA CT to vLLM experiment, with troubleshooting table covering all pre-flight check failures
- `docs/backends.md` — PyTorch and vLLM sections with full parameter references sourced from `backend_configs.py`, runner configuration precedence chain, parameter support matrix
- All vLLM YAML examples use the nested `vllm.engine:` / `vllm.sampling:` structure (Phase 19.1 schema)
- Runner precedence documented: env var > study YAML > user config > default (local)
- Cross-links: docker-setup → getting-started, backends → docker-setup

## Task Commits

Each task was committed atomically:

1. **Task 1: Write docker-setup.md** - `08a76c9` (docs)
2. **Task 2: Write backends.md with parameter support matrix** - `88ff68c` (docs)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `docs/docker-setup.md` — Full NVIDIA Container Toolkit setup guide with per-step verification
- `docs/backends.md` — Backend comparison, runner config, parameter matrix

## Decisions Made

- Parameter matrix constructed manually from Pydantic config model fields (`PyTorchConfig`, `VLLMEngineConfig`, `VLLMSamplingConfig`, `TensorRTConfig`) because `generate_param_matrix.py` requires GPU test result JSON files not available on host. The HTML comment `<!-- Parameter matrix — regenerate from GPU test results: uv run python scripts/generate_param_matrix.py -->` marks the section for future regeneration.
- Probe image in docker-setup.md verification step aligned with `_PROBE_IMAGE = "nvidia/cuda:12.0.0-base-ubuntu22.04"` in `docker_preflight.py` — the same image used by `llem`'s pre-flight.
- `vllm.beam_search:` section documented as mutually exclusive with `vllm.sampling:` per `VLLMConfig.validate_beam_search_exclusive` validator.

## Deviations from Plan

None - plan executed exactly as written. Parameter matrix was specified as "manually construct from config models" if the script requires GPU test results, which it does.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `docs/docker-setup.md` and `docs/backends.md` are complete and cross-linked.
- Plan 02 (`installation.md`, `getting-started.md`, `cli-reference.md`) was not yet executed — those files are the next dependency for the full researcher onboarding path.
- Plan 04 (`study-config.md`, `troubleshooting.md`) can reference `backends.md` for backend-specific sweep syntax.

---
*Phase: 23-documentation*
*Completed: 2026-03-05*
