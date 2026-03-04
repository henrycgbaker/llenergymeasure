# Phase 20: Docker Image and CI - Research

**Researched:** 2026-03-03
**Domain:** Docker image publishing, GitHub Actions CI/CD, GHCR
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Image architecture:**
- Each backend image is self-contained — FROM its upstream official image (e.g., vLLM official for vLLM, PyTorch official for PyTorch)
- No shared base image; each Dockerfile is independent
- llenergymeasure installed via COPY source + `pip install .` (no PyPI dependency)
- Existing Dockerfiles (Dockerfile.base, Dockerfile.vllm, Dockerfile.pytorch, Dockerfile.tensorrt) are greenfield — redesign from scratch for optimal result

**Tag and registry strategy:**
- Registry path: `ghcr.io/henrycgbaker/llenergymeasure/{backend}:{tag}`
- Tag format: `{version}-cuda{major}` (e.g., `1.19.0-cuda12`)
- `latest` tag always points to most recent release
- CUDA 12 only (vLLM 0.6+ requirement, single variant)

**CI workflow design:**
- Separate `docker-publish.yml` workflow (not merged into release.yml)
- Multi-backend matrix from day one — backends defined as a list, future milestones add entries
- Phase 20 enables: vLLM + PyTorch
- Future milestones enable: TensorRT (M4), SGLang (M5)
- Triggered on release tag push

### Claude's Discretion

- Exact CUDA minor version (follows upstream official image)
- Entrypoint design (minimal vs shell wrapper, informed by DockerRunner invocation pattern)
- Image staging strategy (single-stage vs multi-stage for published image)
- Build caching strategy
- Whether PRs should build images without pushing (validation)

### Deferred Ideas (OUT OF SCOPE)

- PyPI distribution — captured as todo. When implemented, reconsider Docker install strategy (switch from COPY+install to `pip install` from PyPI)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DOCK-10 | Official vLLM Docker image published to GHCR (`ghcr.io/henrycgbaker/llenergymeasure/vllm:{version}-cuda{major}`) | GitHub Actions `docker/build-push-action@v6` + GHCR login with `GITHUB_TOKEN`. Matrix strategy covers vLLM + PyTorch. |
</phase_requirements>

## Summary

Phase 20 delivers two concrete artefacts: a redesigned `docker/Dockerfile.vllm` (and `docker/Dockerfile.pytorch`) that build self-contained images from upstream official bases, and a `docker-publish.yml` GitHub Actions workflow that builds and pushes both images to GHCR automatically on every release tag.

The key architectural shift from the existing Dockerfiles is upstream-image-first: instead of building from `nvidia/cuda:...` and manually installing vLLM, the new `Dockerfile.vllm` uses `FROM vllm/vllm-openai:{version}` as base and layers llenergymeasure on top. This keeps the image composition predictable, ensures CUDA/vLLM compatibility is guaranteed by vLLM project maintainers, and minimises maintenance burden. The PyTorch image follows the same pattern from `pytorch/pytorch:...`.

There is one code-level fix required alongside the Dockerfile work: `image_registry.py` currently uses `ghcr.io/llenergymeasure/{backend}` but the locked CONTEXT.md decision sets the registry path to `ghcr.io/henrycgbaker/llenergymeasure/{backend}`. This must be corrected so the runtime default image resolution matches the published images. Both `DEFAULT_IMAGE_TEMPLATE` and its docstring/tests must be updated.

**Primary recommendation:** Use `docker/build-push-action@v6` with a matrix strategy over `[{backend: vllm, dockerfile: docker/Dockerfile.vllm}, {backend: pytorch, dockerfile: docker/Dockerfile.pytorch}]`. Build-only on PRs (push: false), build+push on release tags. Use `docker/metadata-action@v5` with a `suffix=-cuda12` flavour to auto-generate `{version}-cuda12` and `latest` tags.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `docker/build-push-action` | v6 | Build + push Docker images from GitHub Actions | Official Docker action, standard across ecosystem |
| `docker/login-action` | v3 | Authenticate with GHCR | Official Docker action, supports GITHUB_TOKEN natively |
| `docker/metadata-action` | v5 | Generate tags + labels from Git refs | Handles semver patterns, latest tag, suffix injection |
| `docker/setup-buildx-action` | v3 | Set up Docker Buildx | Required for BuildKit features (layer caching, multi-stage) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| GitHub Actions cache (gha) | N/A | Layer cache between workflow runs | Always — dramatically reduces rebuild time for large images |
| `actions/checkout` | v4 | Checkout repo in workflow | Required first step |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `docker/build-push-action@v6` | `docker build` + `docker push` in shell | Actions approach provides layer caching, provenance attestation, build summaries |
| GitHub Actions cache (gha) | Registry cache | gha is simpler, zero registry config; registry cache useful for cross-workflow sharing but overkill here |
| metadata-action suffix | Hardcoded tags | metadata-action auto-handles latest pointer and semver parsing |

## Architecture Patterns

### Recommended File Structure
```
docker/
├── Dockerfile.vllm      # FROM vllm/vllm-openai:{version} — redesigned
├── Dockerfile.pytorch   # FROM pytorch/pytorch:{version} — redesigned
├── Dockerfile.base      # Keep or archive (no longer used in published images)
├── Dockerfile.tensorrt  # Placeholder — activated in M4
.github/workflows/
├── ci.yml               # Existing (lint, type-check, unit tests) — no change
├── release.yml          # Existing (PyPI + GitHub release) — no change
└── docker-publish.yml   # NEW: build + push Docker images on release tag
```

### Pattern 1: Upstream-Image-First Dockerfile (vLLM)

**What:** Start FROM the official vLLM image which already has CUDA, Python, and vLLM pinned and tested. Install llenergymeasure on top.

**When to use:** Any backend where the upstream project publishes an official Docker image (vLLM, PyTorch).

**Key insight from vLLM docs:** The official image is `vllm/vllm-openai:{version}` on Docker Hub. The `-openai` suffix is standard — this is the image used for serving. To add packages: `FROM vllm/vllm-openai:{version}` then `RUN pip install ...`. The vLLM project uses `uv` internally but `pip` works fine for layering.

**Example Dockerfile.vllm structure:**
```dockerfile
# syntax=docker/dockerfile:1
# ============================================================
# vLLM backend — extends official vllm/vllm-openai image
# ============================================================
ARG VLLM_VERSION=v0.7.3

FROM vllm/vllm-openai:${VLLM_VERSION}

WORKDIR /app

# Install llenergymeasure (no PyPI yet — COPY source + install)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[vllm]" --no-deps \
    && pip install --no-cache-dir \
        pydantic loguru typer codecarbon nvidia-ml-py datasets \
        python-dotenv schedule

# Container entrypoint: reads LLEM_CONFIG_PATH, runs experiment, writes result
CMD ["python", "-m", "llenergymeasure.infra.container_entrypoint"]
```

**Notes on the CMD vs ENTRYPOINT choice:**
The DockerRunner calls the container as:
```
docker run ... {image} python -m llenergymeasure.infra.container_entrypoint
```
The command is passed explicitly after the image, which overrides CMD. vLLM's official image sets `ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]` — this WILL conflict unless we override it. Use `ENTRYPOINT ["python", "-m", "llenergymeasure.infra.container_entrypoint"]` (or reset ENTRYPOINT to an empty shell) to avoid the upstream ENTRYPOINT. Alternatively, verify the DockerRunner passes the command as exec-form after the image name, which overrides CMD but NOT ENTRYPOINT. **Resolution: reset `ENTRYPOINT []` and use `CMD ["python", "-m", "llenergymeasure.infra.container_entrypoint"]`** to maintain full Docker run-time override capability.

### Pattern 2: GitHub Actions Matrix for Multi-Backend Builds

**What:** Single workflow job that iterates over a list of `{backend, dockerfile}` pairs. Adding a new backend = one line added to the matrix.

**Example docker-publish.yml structure:**
```yaml
name: Docker Publish

on:
  push:
    tags: ["v*"]
  pull_request:
    branches: [main]     # Build-only (no push) on PRs

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      matrix:
        include:
          - backend: vllm
            dockerfile: docker/Dockerfile.vllm
          - backend: pytorch
            dockerfile: docker/Dockerfile.pytorch
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/henrycgbaker/llenergymeasure/${{ matrix.backend }}
          flavor: |
            suffix=-cuda12
          tags: |
            type=semver,pattern={{version}}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ${{ matrix.dockerfile }}
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=${{ matrix.backend }}
          cache-to: type=gha,mode=max,scope=${{ matrix.backend }}
```

**Tag output for `git push v1.19.0`:**
- `ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12`
- `ghcr.io/henrycgbaker/llenergymeasure/vllm:latest`  ← only on default branch tags

**Note on `latest` tag:** `type=raw,value=latest,enable={{is_default_branch}}` only attaches `latest` when pushing from the default branch. This prevents pre-release branch tags from overwriting `latest`.

### Pattern 3: Registry Path Fix in image_registry.py

**Current (wrong):** `DEFAULT_IMAGE_TEMPLATE = "ghcr.io/llenergymeasure/{backend}:{version}-cuda{cuda_major}"`

**Correct (matches CONTEXT.md decision):** `DEFAULT_IMAGE_TEMPLATE = "ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda{cuda_major}"`

This fix also requires updating:
- The module-level docstring in `image_registry.py`
- The docstring example in `get_default_image()`
- The docstring example in `DockerRunner.__init__`
- The assertion in `tests/unit/test_image_registry.py::TestGetDefaultImage::test_returns_well_formed_image_string`

### Anti-Patterns to Avoid

- **Using `ENTRYPOINT` without overriding upstream:** vLLM official image sets its own ENTRYPOINT. The published llenergymeasure image must reset it explicitly.
- **Pinning vLLM version as a build ARG default only:** The `ARG VLLM_VERSION=v0.7.3` default must match what's installed in pyproject.toml `vllm>=0.6`. Document the pin policy clearly.
- **Merging docker-publish.yml into release.yml:** The user explicitly decided these should be separate workflows. Docker builds are slow; keeping them separate lets CI and release jobs fail independently.
- **Using `fail-fast: true` in matrix:** If vLLM build fails (huge image, slow), PyTorch build should still complete. Set `fail-fast: false`.
- **Not scoping gha cache per backend:** Without `scope=${{ matrix.backend }}`, all matrix jobs share one cache slot and overwrite each other.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tag generation (`1.19.0-cuda12`, `latest`) | Custom shell script parsing git tags | `docker/metadata-action@v5` with `type=semver` + `suffix=-cuda12` | Handles edge cases: pre-release suffixes, tag-vs-branch events, `latest` pointer logic |
| GHCR login | `echo $TOKEN \| docker login ...` in shell | `docker/login-action@v3` | Handles token masking, error messages, re-login on expiry |
| BuildKit setup | Docker CLI flags | `docker/setup-buildx-action@v3` | Ensures correct buildx version, layer cache compatibility |
| Layer caching | No cache (slow rebuilds on every tag push) | `type=gha,scope={backend}` | vLLM image is ~10GB; without cache, every release tag triggers a multi-hour rebuild |

**Key insight:** The vLLM Docker image is very large. Without layer caching, the CI build could take 60–120 minutes per backend. The `gha` cache backend reduces this dramatically on subsequent builds by reusing unchanged layers.

## Common Pitfalls

### Pitfall 1: vLLM Upstream ENTRYPOINT Conflict
**What goes wrong:** `vllm/vllm-openai` sets `ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]`. If the published llenergymeasure image inherits this, `docker run {image} python -m llenergymeasure.infra.container_entrypoint` will fail because the exec-form command is appended as arguments to the vLLM API server, not as a new command.
**Why it happens:** Docker ENTRYPOINT is inherited from base image. CMD overrides CMD, but ENTRYPOINT overrides require explicit `ENTRYPOINT []` reset.
**How to avoid:** Add `ENTRYPOINT []` in Dockerfile.vllm to reset, then set `CMD ["python", "-m", "llenergymeasure.infra.container_entrypoint"]`.
**Warning signs:** Container exits immediately with vLLM server argument parse errors.

### Pitfall 2: Registry Path Mismatch (image_registry.py vs GHCR)
**What goes wrong:** Images are pushed to `ghcr.io/henrycgbaker/llenergymeasure/vllm:...` but `image_registry.py` resolves default images to `ghcr.io/llenergymeasure/vllm:...`. DockerRunner pulls from the wrong path and gets `ImageNotFound` errors.
**Why it happens:** `DEFAULT_IMAGE_TEMPLATE` was written before the CONTEXT.md registry path decision was finalised.
**How to avoid:** Update `DEFAULT_IMAGE_TEMPLATE` in Plan 01 before or alongside the Dockerfile work.
**Warning signs:** `ImagePull` errors in DockerRunner when using bare `docker` runner config.

### Pitfall 3: `latest` Tag on Pre-Release Branches
**What goes wrong:** A branch tag (e.g., `v1.19.0-rc1` pushed from a feature branch) overwrites `latest` in GHCR.
**Why it happens:** `type=raw,value=latest` without a condition fires on any tag push.
**How to avoid:** Use `enable={{is_default_branch}}` condition on the `latest` tag type, or use `type=semver` with `latest=auto` in the `flavor` block.
**Warning signs:** `latest` tag pointing to a pre-release version.

### Pitfall 4: GitHub Actions Cache API v2 Requirement (from April 2025)
**What goes wrong:** `cache-from/cache-to: type=gha` fails after April 15, 2025 with old tooling.
**Why it happens:** GitHub deprecated Cache API v1 in favour of v2.
**How to avoid:** Use `docker/setup-buildx-action@v3` (ships with Buildx 0.21+) which supports Cache API v2 natively. Confirmed current as of March 2026.
**Warning signs:** Cache-related workflow failures with API version error messages.

### Pitfall 5: vLLM Version Pinning Drift
**What goes wrong:** `ARG VLLM_VERSION=v0.7.3` in Dockerfile drifts out of sync with `vllm>=0.6` in pyproject.toml. Incompatible versions between the vLLM base image and the installed llenergymeasure extras.
**Why it happens:** Two separate version pins that must be kept in sync.
**How to avoid:** Document the pin explicitly. Consider passing version as a build ARG in the CI workflow so it is set in one place (`docker-publish.yml` matrix entry). For Phase 20, a pinned ARG default is acceptable.

## Code Examples

Verified patterns from official sources:

### Complete docker-publish.yml
```yaml
# Source: Official GitHub Actions Docker docs + docker/build-push-action@v6 README
name: Docker Publish

on:
  push:
    tags: ["v*"]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    strategy:
      fail-fast: false
      matrix:
        include:
          - backend: vllm
            dockerfile: docker/Dockerfile.vllm
          - backend: pytorch
            dockerfile: docker/Dockerfile.pytorch
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/henrycgbaker/llenergymeasure/${{ matrix.backend }}
          flavor: |
            latest=auto
            suffix=-cuda12
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ${{ matrix.dockerfile }}
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha,scope=${{ matrix.backend }}
          cache-to: type=gha,mode=max,scope=${{ matrix.backend }}
```

**Tag output for tag `v1.19.0` on default branch:**
- `ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12`
- `ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19-cuda12`
- `ghcr.io/henrycgbaker/llenergymeasure/vllm:latest`  ← `latest=auto` in flavor

### Dockerfile.vllm (greenfield)
```dockerfile
# syntax=docker/dockerfile:1
# ============================================================
# vLLM backend — llenergymeasure on top of official vLLM image
# ============================================================
ARG VLLM_VERSION=v0.7.3

FROM vllm/vllm-openai:${VLLM_VERSION}

# Reset upstream ENTRYPOINT (vLLM sets api_server entrypoint)
ENTRYPOINT []

WORKDIR /app

# Copy source and install llenergymeasure (no PyPI yet)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps ".[vllm]" \
    && pip install --no-cache-dir \
        pydantic loguru typer codecarbon nvidia-ml-py datasets \
        python-dotenv schedule

# HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod 777 /app/.cache/huggingface

CMD ["python", "-m", "llenergymeasure.infra.container_entrypoint"]
```

### Dockerfile.pytorch (greenfield)
```dockerfile
# syntax=docker/dockerfile:1
# ============================================================
# PyTorch backend — llenergymeasure on top of official PyTorch image
# ============================================================
ARG PYTORCH_VERSION=2.5.1-cuda12.4-cudnn9-runtime

FROM pytorch/pytorch:${PYTORCH_VERSION}

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[pytorch]" sentencepiece

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod 777 /app/.cache/huggingface

CMD ["python", "-m", "llenergymeasure.infra.container_entrypoint"]
```

### image_registry.py template fix
```python
# Source: image_registry.py DEFAULT_IMAGE_TEMPLATE — fix registry path
DEFAULT_IMAGE_TEMPLATE = "ghcr.io/henrycgbaker/llenergymeasure/{backend}:{version}-cuda{cuda_major}"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom base image (FROM nvidia/cuda) | FROM upstream official (FROM vllm/vllm-openai) | 2024+ | Smaller maintenance burden, guaranteed CUDA/vLLM compat |
| docker push in shell scripts | docker/build-push-action@v6 | 2023+ | Layer caching, provenance attestation, build summaries |
| Personal Access Tokens for GHCR | GITHUB_TOKEN | 2021+ | No secret management, auto-scoped to repo, recommended by GitHub |
| GitHub Cache API v1 | Cache API v2 (required from April 2025) | April 2025 | Buildx 0.21+ / setup-buildx-action@v3 required |

**Deprecated/outdated:**
- `ghcr.io/llenergymeasure/` namespace: was used in Phase 17 code but CONTEXT.md decision sets `ghcr.io/henrycgbaker/llenergymeasure/`. Phase 20 Plan 01 corrects this.
- Existing `Dockerfile.base` / multi-stage-from-base pattern: replaced by upstream-image-first approach per locked CONTEXT.md decision.

## Open Questions

1. **vLLM version to pin in Dockerfile ARG default**
   - What we know: `pyproject.toml` requires `vllm>=0.6`; latest release as of 2026-03-03 is approximately v0.7.3 (inferred from Docker Hub tag listing)
   - What's unclear: Exact latest stable tag — needs verification at build time
   - Recommendation: Set ARG default to the version known to work with Phase 19 tests; add a comment saying "update to latest stable on each milestone release"

2. **PUID/PGID entrypoint for published images**
   - What we know: The existing `scripts/entrypoint.sh` handles PUID/PGID. DockerRunner in Phase 17 does NOT set PUID/PGID — it relies on volume permissions.
   - What's unclear: Should the published image include the PUID/PGID entrypoint shell script, or is the minimal `CMD python -m container_entrypoint` sufficient for the llenergymeasure use case?
   - Recommendation: Skip PUID/PGID in Phase 20 published images — DockerRunner does not use it. The shell entrypoint was designed for `docker compose` interactive usage. Container measurements use fixed paths under `/run/llem` which DockerRunner mounts.

3. **PyTorch official image tag format**
   - What we know: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` is a known pattern
   - What's unclear: Whether `-runtime` vs `-devel` is the right variant for inference (no nvcc needed)
   - Recommendation: Use `-runtime` — inference does not need the full CUDA development toolkit. Consistent with existing `Dockerfile.pytorch` which installs from wheel index.

## Sources

### Primary (HIGH confidence)
- `docker/build-push-action` GitHub README (fetched) — v6 current, workflow structure, push: false pattern
- GitHub Actions official docs (fetched) — GITHUB_TOKEN permissions, packages: write scope
- Docker official GHA caching docs (fetched) — gha backend, Cache API v2 requirement from April 2025
- `docker/metadata-action` GitHub README (fetched) — tag types, semver patterns, suffix/flavor config

### Secondary (MEDIUM confidence)
- vLLM Docker docs (WebFetch) — `vllm/vllm-openai` is official image name; `FROM vllm/vllm-openai` extend pattern confirmed
- WebSearch results — GHCR matrix strategies, multiple Dockerfiles per workflow, confirmed against official docs

### Tertiary (LOW confidence)
- vLLM version `v0.7.3` as latest stable — inferred from Docker Hub tag listing via search; verify at planning time

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — official Docker actions, verified versions, confirmed current API
- Architecture: HIGH — CONTEXT.md decisions are locked; Dockerfile and workflow patterns verified from official docs
- Pitfalls: HIGH for ENTRYPOINT conflict and registry mismatch (grounded in codebase inspection); MEDIUM for version drift (general Docker pattern)

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable domain; main risk is vLLM upstream version changes)
