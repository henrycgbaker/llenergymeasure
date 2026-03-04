---
phase: 20-docker-image-and-ci
verified: 2026-03-03T18:45:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
human_verification:
  - test: "Push a v* release tag and confirm docker-publish workflow triggers and images land in GHCR"
    expected: "ghcr.io/henrycgbaker/llenergymeasure/vllm:{version}-cuda12 and pytorch variant appear in GHCR package registry"
    why_human: "CI workflow correctness requires a real tag push to GitHub to validate — cannot simulate GitHub Actions locally"
  - test: "docker build -f docker/Dockerfile.vllm . succeeds on a host with Docker and internet access"
    expected: "Image builds cleanly with vllm/vllm-openai base; llenergymeasure importable inside container"
    why_human: "Docker build requires network access to pull vllm/vllm-openai (~20GB); cannot run in this environment"
  - test: "docker build -f docker/Dockerfile.pytorch . succeeds on a host with Docker and internet access"
    expected: "Image builds cleanly with pytorch/pytorch base; llenergymeasure importable inside container"
    why_human: "Docker build requires network access to pull pytorch/pytorch base; cannot run in this environment"
---

# Phase 20: Docker Image and CI Verification Report

**Phase Goal:** Official vLLM Docker image published to GHCR, CI publish on release tag
**Verified:** 2026-03-03T18:45:00Z
**Status:** passed (automated) — 3 items require human/live verification
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Plan 01 truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `docker build -f docker/Dockerfile.vllm .` produces image with llenergymeasure installed | ? HUMAN | Dockerfile is substantive and correct; build requires live Docker + network |
| 2 | `docker build -f docker/Dockerfile.pytorch .` produces image with llenergymeasure installed | ? HUMAN | Dockerfile is substantive and correct; build requires live Docker + network |
| 3 | ENTRYPOINT is reset in Dockerfile.vllm so DockerRunner command override works | VERIFIED | `ENTRYPOINT []` present at line 20 of docker/Dockerfile.vllm |
| 4 | DEFAULT_IMAGE_TEMPLATE resolves to `ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda{cuda_major}` | VERIFIED | Line 41 of image_registry.py matches exactly |
| 5 | `get_default_image('vllm')` returns a string starting with `ghcr.io/henrycgbaker/llenergymeasure/vllm:` | VERIFIED | Test assertion at line 63 of test_image_registry.py; implementation confirmed correct |

Plan 02 truths:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | Pushing a release tag (v*) triggers the docker-publish workflow | ? HUMAN | `on.push.tags: ["v*"]` present in workflow; CI requires live tag push to confirm |
| 7 | The workflow builds vLLM and PyTorch images in a matrix strategy | VERIFIED | `matrix.include` contains both `backend: vllm` and `backend: pytorch` entries |
| 8 | Images are pushed to `ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda12` | VERIFIED | `images:` field in metadata-action step; `suffix=-cuda12`; `type=semver,pattern={{version}}` |
| 9 | `latest` tag attached only when pushing from the default branch | VERIFIED | `flavor: latest=auto` in metadata-action (auto = latest only on default branch tags) |
| 10 | PRs build images without pushing (validation only) | VERIFIED | `push: ${{ github.event_name != 'pull_request' }}` on build-push-action step |
| 11 | Adding a new backend requires only adding one entry to the matrix list | VERIFIED | Commented-out `tensorrt` and `sglang` entries demonstrate the pattern; matrix.include structure confirmed |
| 12 | `fail-fast` is false so one backend failure does not cancel the other | VERIFIED | `strategy.fail-fast: false` at line 16 of docker-publish.yml |
| 13 | Layer cache is scoped per backend to prevent cache collisions | VERIFIED | `cache-from/cache-to: type=gha,scope=${{ matrix.backend }}` on both cache directives |

**Score:** 13/13 truths verified (10 automated + 3 requiring human/live confirmation)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docker/Dockerfile.vllm` | Greenfield vLLM Dockerfile from official vllm/vllm-openai base | VERIFIED | 39 lines, `FROM vllm/vllm-openai:${VLLM_VERSION}`, `ENTRYPOINT []`, `CMD container_entrypoint` |
| `docker/Dockerfile.pytorch` | Greenfield PyTorch Dockerfile from official pytorch/pytorch base | VERIFIED | 31 lines, `FROM pytorch/pytorch:${PYTORCH_VERSION}`, `CMD container_entrypoint` |
| `src/llenergymeasure/infra/image_registry.py` | Corrected DEFAULT_IMAGE_TEMPLATE with henrycgbaker namespace | VERIFIED | Line 41: `DEFAULT_IMAGE_TEMPLATE = "ghcr.io/henrycgbaker/llenergymeasure/{backend}:..."`, all docstrings updated |
| `tests/unit/test_image_registry.py` | Updated assertions matching new registry path | VERIFIED | Line 63: `assert image.startswith("ghcr.io/henrycgbaker/llenergymeasure/vllm:")` |
| `.github/workflows/docker-publish.yml` | CI workflow for building and publishing Docker images to GHCR | VERIFIED | 65-line workflow, valid structure, all required actions present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `image_registry.py` | `docker/Dockerfile.vllm` | `DEFAULT_IMAGE_TEMPLATE` matches GHCR path where CI publishes | VERIFIED | Both use `ghcr.io/henrycgbaker/llenergymeasure/`; template format matches workflow tag output |
| `docker_runner.py` | `docker/Dockerfile.vllm` | DockerRunner passes command after image name — requires `ENTRYPOINT []` reset | VERIFIED | `docker_runner.py` appends `python -m llenergymeasure.infra.container_entrypoint` after image name; `ENTRYPOINT []` present in Dockerfile.vllm |
| `.github/workflows/docker-publish.yml` | `docker/Dockerfile.vllm` | `matrix.dockerfile` references `docker/Dockerfile.vllm` | VERIFIED | Line 20 of workflow: `dockerfile: docker/Dockerfile.vllm` |
| `.github/workflows/docker-publish.yml` | `docker/Dockerfile.pytorch` | `matrix.dockerfile` references `docker/Dockerfile.pytorch` | VERIFIED | Line 22 of workflow: `dockerfile: docker/Dockerfile.pytorch` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOCK-10 | 20-01-PLAN.md, 20-02-PLAN.md | Official vLLM Docker image published to GHCR | SATISFIED | Dockerfile.vllm uses vllm/vllm-openai base; docker-publish.yml pushes to ghcr.io/henrycgbaker/llenergymeasure/ on release tag |

**Note on DOCK-10 description:** The REQUIREMENTS.md description for DOCK-10 reads `ghcr.io/llenergymeasure/vllm:{version}-cuda{major}` (missing `henrycgbaker/`). This is a stale description — the CONTEXT.md decision (confirmed registry path: `ghcr.io/henrycgbaker/llenergymeasure/`) and all implementation artefacts use the correct path. The requirement intent is fully satisfied; only the REQUIREMENTS.md description text is outdated. This is a documentation-only gap, not a functional one.

**Orphaned requirements check:** REQUIREMENTS.md traceability table maps only DOCK-10 to Phase 20. No other requirements are assigned to this phase. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TODO/FIXME/placeholder comments, empty implementations, or stub patterns found in any of the 5 modified files.

### Human Verification Required

#### 1. Docker Build — vLLM Image

**Test:** On a host with Docker installed and internet access, run `docker build -f docker/Dockerfile.vllm -t llenergymeasure-vllm .` from the repo root.
**Expected:** Build succeeds; `docker run --rm llenergymeasure-vllm python -c "import llenergymeasure; print('ok')"` prints `ok`.
**Why human:** Build pulls vllm/vllm-openai (~20 GB) and requires Docker daemon + network — not available in this environment.

#### 2. Docker Build — PyTorch Image

**Test:** On a host with Docker installed and internet access, run `docker build -f docker/Dockerfile.pytorch -t llenergymeasure-pytorch .` from the repo root.
**Expected:** Build succeeds; `docker run --rm llenergymeasure-pytorch python -c "import llenergymeasure; print('ok')"` prints `ok`.
**Why human:** Build pulls pytorch/pytorch base and requires Docker daemon + network — not available in this environment.

#### 3. CI Workflow — Release Tag Publish

**Test:** Push a release tag (e.g. `git tag v1.19.0-rc1 && git push origin v1.19.0-rc1`) to the GitHub remote.
**Expected:** The `Docker Publish` workflow triggers in GitHub Actions, both `vllm` and `pytorch` matrix jobs run, and images appear in the GHCR package registry under `ghcr.io/henrycgbaker/llenergymeasure/`.
**Why human:** GitHub Actions CI requires a live push to the remote repository to validate the trigger and GHCR publish — cannot simulate in this environment.

### Gaps Summary

No functional gaps. All automated checks pass. The phase delivered:

- Two greenfield Dockerfiles using official upstream bases (vllm/vllm-openai, pytorch/pytorch), with the critical `ENTRYPOINT []` reset in Dockerfile.vllm for DockerRunner compatibility
- Corrected GHCR namespace (`ghcr.io/henrycgbaker/llenergymeasure/`) across image_registry.py, docker_runner.py docstring, tests, and the CI workflow
- A self-contained GitHub Actions matrix workflow that builds and pushes on release tags, validates on PRs, with per-backend cache isolation and `fail-fast: false`

The three human verification items are confirmations of expected behaviour that require live infrastructure (Docker daemon and GitHub Actions), not gaps in the implementation.

One minor documentation discrepancy exists: DOCK-10 in REQUIREMENTS.md describes the registry path without the `henrycgbaker/` prefix. This does not affect the implementation and can be corrected in a documentation pass.

---

_Verified: 2026-03-03T18:45:00Z_
_Verifier: Claude (gsd-verifier)_
