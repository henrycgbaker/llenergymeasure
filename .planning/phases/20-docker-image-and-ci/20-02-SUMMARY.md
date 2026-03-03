---
phase: 20-docker-image-and-ci
plan: 02
subsystem: infra
tags: [github-actions, docker, ghcr, matrix-strategy, buildx, metadata-action]

# Dependency graph
requires:
  - phase: 20-docker-image-and-ci-01
    provides: docker/Dockerfile.vllm and docker/Dockerfile.pytorch built in plan 01

provides:
  - GitHub Actions workflow (.github/workflows/docker-publish.yml) that builds and publishes Docker images to GHCR on release tag push
  - Build-only PR validation of Dockerfiles without pushing
  - Matrix strategy supporting vllm and pytorch backends, extensible for future backends

affects:
  - Phase 22 (docs) — pull commands in docs reference ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda12
  - M4 (TensorRT), M5 (SGLang) — add one entry to matrix.include to enable

# Tech tracking
tech-stack:
  added:
    - docker/build-push-action@v6
    - docker/login-action@v3
    - docker/setup-buildx-action@v3
    - docker/metadata-action@v5
  patterns:
    - Matrix strategy with fail-fast:false for independent backend builds
    - Per-backend GHA layer cache scope to prevent cross-backend cache collisions
    - Conditional push (push only on tag events, build-only on PRs)
    - latest=auto flavor in metadata-action to prevent pre-release tag overwrite

key-files:
  created:
    - .github/workflows/docker-publish.yml
  modified: []

key-decisions:
  - "Separate docker-publish.yml workflow (not merged into release.yml) — Docker builds are slow and should fail independently"
  - "fail-fast: false ensures vLLM build failure does not cancel PyTorch build"
  - "Layer cache scoped per backend (scope=${{ matrix.backend }}) prevents cross-backend cache collisions"
  - "latest=auto in metadata-action attaches latest only on default branch tags, not pre-releases"
  - "GHCR login only on non-PR events, using GITHUB_TOKEN (no PAT required)"

patterns-established:
  - "Adding a new Docker backend = one entry added to matrix.include in docker-publish.yml"
  - "All image tags carry -cuda12 suffix (CUDA 12 only, matches vLLM 0.6+ requirement)"

requirements-completed: [DOCK-10]

# Metrics
duration: 2min
completed: 2026-03-03
---

# Phase 20 Plan 02: Docker Publish CI Workflow Summary

**GitHub Actions matrix workflow publishing vllm and pytorch images to ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda12 on release tag push**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T18:14:28Z
- **Completed:** 2026-03-03T18:15:30Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `.github/workflows/docker-publish.yml` with matrix strategy covering vllm and pytorch backends
- Workflow triggers on `v*` tag push (build + push to GHCR) and PR to main (build-only validation)
- Images tagged as `{version}-cuda12` and `{major}.{minor}-cuda12` with `latest=auto` via metadata-action
- Per-backend GHA layer cache prevents cross-backend cache collisions
- Commented-out TensorRT and SGLang entries show the one-entry extension pattern for M4/M5

## Task Commits

Each task was committed atomically:

1. **Task 1: Create docker-publish.yml workflow with matrix strategy** - `7227f59` (feat)

**Plan metadata:** (to be added with final commit)

## Files Created/Modified

- `.github/workflows/docker-publish.yml` — GitHub Actions CI workflow: builds and publishes backend Docker images to GHCR on release tag push; build-only on PRs

## Decisions Made

None — followed plan specification exactly. All design decisions were pre-resolved in CONTEXT.md and the plan action block.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

Minor: PyYAML parses `on:` as boolean `True` (YAML 1.1 spec). The verification script in the plan used `wf['on']` which fails with PyYAML. Used `wf[True]` instead. The workflow YAML itself is correct — this is a verification script quirk, not a workflow bug.

## User Setup Required

None — GHCR publishing uses `GITHUB_TOKEN` (auto-available in GitHub Actions). No PAT or secrets to configure manually.

## Next Phase Readiness

- CI workflow is complete and ready for the first release tag push
- Researchers can run `docker pull ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12` after M3 release tag
- Adding TensorRT (M4) or SGLang (M5) requires only adding one entry to `matrix.include` in docker-publish.yml

---
*Phase: 20-docker-image-and-ci*
*Completed: 2026-03-03*
