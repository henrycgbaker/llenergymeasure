# Phase 20: Docker Image and CI - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Official Docker images for llenergymeasure backends published to GHCR, automatically built and pushed on release tags. This phase delivers the Dockerfiles (greenfield redesign) and CI workflow. It does not add new backends or change runtime behaviour.

</domain>

<decisions>
## Implementation Decisions

### Image architecture
- Each backend image is self-contained — FROM its upstream official image (e.g., vLLM official for vLLM, PyTorch official for PyTorch)
- No shared base image; each Dockerfile is independent
- llenergymeasure installed via COPY source + `pip install .` (no PyPI dependency)
- Existing Dockerfiles (Dockerfile.base, Dockerfile.vllm, Dockerfile.pytorch, Dockerfile.tensorrt) are greenfield — redesign from scratch for optimal result

### Tag and registry strategy
- Registry path: `ghcr.io/henrycgbaker/llenergymeasure/{backend}:{tag}`
- Tag format: `{version}-cuda{major}` (e.g., `1.19.0-cuda12`)
- `latest` tag always points to most recent release
- CUDA 12 only (vLLM 0.6+ requirement, single variant)

### CI workflow design
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

</decisions>

<specifics>
## Specific Ideas

- Full pull command example: `docker pull ghcr.io/henrycgbaker/llenergymeasure/vllm:1.19.0-cuda12`
- Matrix should be structured so adding a backend is a one-line change (add entry to list)
- PyTorch image published alongside vLLM even though PyTorch runs locally — gives users a containerised option

</specifics>

<deferred>
## Deferred Ideas

- PyPI distribution — captured as todo. When implemented, reconsider Docker install strategy (switch from COPY+install to `pip install` from PyPI)

</deferred>

---

*Phase: 20-docker-image-and-ci*
*Context gathered: 2026-03-03*
