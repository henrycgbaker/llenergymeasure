---
phase: 20-docker-image-and-ci
plan: 01
subsystem: infra
tags: [docker, dockerfile, ghcr, vllm, pytorch, image-registry]

# Dependency graph
requires:
  - phase: 17-docker-runner-infrastructure
    provides: DockerRunner image invocation pattern requiring ENTRYPOINT [] reset
  - phase: 19-vllm-backend-activation
    provides: VLLMBackend as the container payload for Dockerfile.vllm
provides:
  - Greenfield Dockerfile.vllm using vllm/vllm-openai upstream base with ENTRYPOINT [] reset
  - Greenfield Dockerfile.pytorch using pytorch/pytorch upstream base
  - Corrected GHCR registry path ghcr.io/henrycgbaker/llenergymeasure/ in image_registry.py
affects: [20-02-PLAN.md, CI publish workflows, DockerRunner consumers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Upstream-image-first: use official vllm/vllm-openai and pytorch/pytorch bases, not a shared internal base"
    - "ENTRYPOINT [] reset in vLLM Dockerfile: required because vllm/vllm-openai sets its own ENTRYPOINT, conflicting with DockerRunner command override"
    - "--no-deps on .[vllm] install: avoids re-installing vLLM/torch already in base image"

key-files:
  created: []
  modified:
    - docker/Dockerfile.vllm
    - docker/Dockerfile.pytorch
    - src/llenergymeasure/infra/image_registry.py
    - src/llenergymeasure/infra/docker_runner.py
    - tests/unit/test_image_registry.py

key-decisions:
  - "vllm/vllm-openai and pytorch/pytorch as upstream bases — CUDA/framework compatibility guaranteed by upstream"
  - "ENTRYPOINT [] reset in Dockerfile.vllm — DockerRunner appends command after image name (CMD override), not after ENTRYPOINT"
  - "--no-deps on .[vllm] install — avoids pip reinstalling vLLM/torch that are already present in base image"
  - "ghcr.io/henrycgbaker/llenergymeasure/ as GHCR namespace — corrects stale placeholder that lacked the GitHub user prefix"

patterns-established:
  - "Greenfield Dockerfiles: single-stage, upstream-image-first, no shared Dockerfile.base dependency"
  - "HF_HOME=/app/.cache/huggingface with chmod 777 for non-root container users"

requirements-completed:
  - DOCK-10

# Metrics
duration: 3min
completed: 2026-03-03
---

# Phase 20 Plan 01: Docker Image Redesign and Registry Fix Summary

**Greenfield Dockerfile.vllm (FROM vllm/vllm-openai, ENTRYPOINT [] reset) and Dockerfile.pytorch (FROM pytorch/pytorch), with corrected GHCR namespace ghcr.io/henrycgbaker/llenergymeasure/ in image_registry.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-03T18:14:42Z
- **Completed:** 2026-03-03T18:17:55Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Replaced multi-stage shared-base Dockerfiles with single-stage upstream-image-first designs
- Reset `ENTRYPOINT []` in Dockerfile.vllm — critical for DockerRunner compatibility (vllm/vllm-openai sets its own ENTRYPOINT to the API server)
- Fixed `DEFAULT_IMAGE_TEMPLATE` in image_registry.py from `ghcr.io/llenergymeasure/` to `ghcr.io/henrycgbaker/llenergymeasure/` — all tests, docstrings, and examples updated

## Task Commits

Each task was committed atomically:

1. **Task 1: Greenfield Dockerfiles for vLLM and PyTorch backends** - `897f739` (feat)
2. **Task 2: Fix registry path in image_registry.py and update tests** - `10ac308` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `docker/Dockerfile.vllm` - Greenfield FROM vllm/vllm-openai, ENTRYPOINT [] reset, single-stage
- `docker/Dockerfile.pytorch` - Greenfield FROM pytorch/pytorch, single-stage, sentencepiece included
- `src/llenergymeasure/infra/image_registry.py` - DEFAULT_IMAGE_TEMPLATE and docstrings updated to henrycgbaker namespace
- `src/llenergymeasure/infra/docker_runner.py` - Docstring example updated to correct GHCR path
- `tests/unit/test_image_registry.py` - Test assertion updated to match corrected registry path

## Decisions Made

- Used `--no-deps` on `.[vllm]` pip install in Dockerfile.vllm — vLLM and torch are already in the vllm/vllm-openai base, reinstalling them would risk version conflicts and increase build time
- ENTRYPOINT [] reset is the only way to allow DockerRunner's command override to work — `docker run image cmd` overrides CMD, not ENTRYPOINT; the vLLM upstream image sets ENTRYPOINT to its API server
- pytorch/pytorch uses `-runtime` variant (not `-devel`) — no nvcc needed for inference workloads, smaller image

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pytest was running against the editable install from the main workspace (not the worktree's src). All test runs used `PYTHONPATH` pointing to the worktree src to ensure correct source was tested. This is expected in worktree-based development.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Dockerfile.vllm and Dockerfile.pytorch ready for CI build and push (Plan 02)
- image_registry.py namespace correct — `get_default_image('vllm')` returns `ghcr.io/henrycgbaker/llenergymeasure/vllm:...`
- No blockers for Plan 02 (GitHub Actions CI workflow for Docker publish)

## Self-Check: PASSED

All files found and both task commits verified.

---
*Phase: 20-docker-image-and-ci*
*Completed: 2026-03-03*
