---
phase: 20-docker-image-and-ci
status: passed
verified: 2026-03-05T22:28:00Z
retroactive: true
requirements_verified:
  - DOCK-10
evidence_source: "Retroactive verification against codebase on main (PRs #34, #41)"
---

# Phase 20 Verification: Docker Image and CI

Retroactive verification of DOCK-10 against the codebase on `main`. Evidence gathered
by direct inspection of source files on 2026-03-05.

---

## DOCK-10 — Docker Image on GHCR

**Requirement:** A Docker image for each backend must be built and published to GHCR
(GitHub Container Registry) as part of the release pipeline.

**Status: PASS**

### Evidence

**Dockerfiles exist for all backends** — `docker/` directory:

```
docker/Dockerfile.base
docker/Dockerfile.pytorch
docker/Dockerfile.vllm
docker/Dockerfile.tensorrt
```

**Dockerfile.vllm installs llenergymeasure with vLLM** — `docker/Dockerfile.vllm`, lines 31-34:

```dockerfile
RUN uv pip install --system --no-cache --no-deps ".[vllm]" \
    && uv pip install --system --no-cache \
        pydantic loguru typer pyyaml platformdirs nvidia-ml-py numpy \
        pyarrow tqdm codecarbon datasets python-dotenv
```

Extends `vllm/vllm-openai:v0.7.3` base image (line 16).

**`docker-publish.yml` workflow pushes to GHCR** — `.github/workflows/docker-publish.yml`:

Registry set at line 21: `REGISTRY: ghcr.io`

Matrix strategy builds all three backends (lines 26-28):

```yaml
strategy:
  matrix:
    backend: [pytorch, vllm, tensorrt]
```

Image pushed to GHCR (lines 44-57):

```yaml
- id: meta
  uses: docker/metadata-action@v5
  with:
    images: ${{ env.REGISTRY }}/${{ github.repository }}/${{ matrix.backend }}
    tags: |
      type=raw,value=${{ inputs.version }}
      type=raw,value=latest

- uses: docker/build-push-action@v6
  with:
    context: .
    file: docker/Dockerfile.${{ matrix.backend }}
    push: true
    tags: ${{ steps.meta.outputs.tags }}
    labels: ${{ steps.meta.outputs.labels }}
```

**`release.yml` triggers `docker-publish.yml` after tests pass** — `.github/workflows/release.yml`,
lines 81-88:

```yaml
docker:
  needs: release
  uses: ./.github/workflows/docker-publish.yml
  with:
    version: ${{ github.ref_name }}
  permissions:
    contents: read
    packages: write
```

The `docker` job declares `needs: release`, which itself `needs: [lint, type-check, test, gpu]`.
This ensures Docker images are only published after the full test suite passes.

**Image tags include version** — `docker-publish.yml` line 48:

```yaml
type=raw,value=${{ inputs.version }}
```

The `version` input receives `github.ref_name` from `release.yml` (e.g., `v0.9.0`).

**Permissions at job level for GHCR write** — `docker-publish.yml` lines 29-31:

```yaml
permissions:
  contents: read
  packages: write
```

---

## Summary

| Requirement | Description                              | Status |
| ----------- | ---------------------------------------- | ------ |
| DOCK-10     | Docker image for each backend on GHCR    | PASS   |

Three Dockerfiles (pytorch, vllm, tensorrt) exist. The `docker-publish.yml` workflow
builds and pushes all three to GHCR, triggered by `release.yml` after tests pass.
Phase 20 code was merged to main in PRs #34 and #41.
