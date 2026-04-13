# Installation

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Hard requirement (TensorRT-LLM compatibility) |
| OS | Linux | Required for vLLM and TensorRT-LLM backends |
| GPU | NVIDIA with CUDA 12.x | Required for all inference engines |
| CUDA (host) | 12.x | For container image compatibility |
| Docker + NVIDIA Container Toolkit | Latest | Required for vLLM and TensorRT-LLM |
| Docker Compose | v2.32+ recommended | Required for build cache (see below). v2.11+ minimum |
| Docker Buildx | v0.17+ recommended | Required for build cache. Bundled with Docker Engine 24+ |

**macOS/Windows:** Transformers engine only. Docker-based engines (vLLM, TensorRT-LLM) require Linux.

---

## Install

The base package includes no inference engine. Install with the extras for your use case.

```bash
pip install "llenergymeasure[transformers]"
```

### Available extras

| Extra | What it installs | When to use |
|-------|-----------------|-------------|
| `pytorch` | PyTorch, Transformers, Accelerate | Local inference (default path) |
| `vllm` | vLLM | vLLM inference via Docker |
| `tensorrt` | TensorRT-LLM | TensorRT-LLM inference via Docker |
| `zeus` | Zeus energy monitor | GPU energy via Zeus (alternative to NVML) |
| `codecarbon` | CodeCarbon | Carbon-aware energy tracking |

Install multiple extras together:

```bash
pip install "llenergymeasure[pytorch,zeus]"
```

The base install (`pip install llenergymeasure`) includes no inference engine and cannot run experiments.

---

## Install from Source (Development)

The project uses [uv](https://docs.astral.sh/uv/) as its package manager.

```bash
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool
uv sync --dev --extra pytorch
uv run llem --version
```

Expected output:

```
llem v0.9.0
```

---

## Docker Setup

For vLLM or TensorRT-LLM backends, Docker with NVIDIA Container Toolkit is required. See the [Docker Setup Guide](docker-setup.md) for a complete walkthrough covering driver installation, toolkit setup, and verification.

---

## BuildKit Builder Setup

Before building Docker images locally, set up a dedicated BuildKit builder with sufficient
cache space. Without this, the default builder may evict cached layers when building
multiple engines, causing expensive recompilation.

```bash
make docker-builder-setup
```

This creates a `llem-builder` with a 200 GiB GC limit. To use it, set
`BUILDX_BUILDER=llem-builder` in your `.env` file or export it in your shell. Run once per
machine. See [Docker Setup - BuildKit](docker-setup.md#buildkit-builder-setup-recommended)
for details.

## Building Docker Images from Source

The pre-built images from GHCR work for most users. If you need to rebuild images locally
(e.g. after modifying the source code), use the Make targets:

```bash
# Build all engines (pytorch, vllm, tensorrt)
make docker-build-all

# Build a specific engine
make docker-build-pytorch
make docker-build-vllm
make docker-build-tensorrt
```

These use `docker compose build` under the hood. You can also call it directly:

```bash
docker compose build pytorch vllm tensorrt
```

Or with plain `docker build` (no Compose, no build cache):

```bash
docker build -f docker/Dockerfile.transformers -t llenergymeasure:transformers .
docker build -f docker/Dockerfile.vllm -t llenergymeasure:vllm .
docker build -f docker/Dockerfile.tensorrt -t llenergymeasure:tensorrt .
```

Local builds produce images tagged `llenergymeasure:{engine}`. When present, `llem`
prefers these over registry images. See [Image Management](docker-setup.md#image-management)
for the full resolution chain.

> **When to rebuild.** Images bundle the `llenergymeasure` source at build time. If you
> modify config models, engines, or the container entrypoint, you must rebuild for changes
> to take effect inside containers. Local-runner experiments (Transformers) use the installed
> source directly and do not need a rebuild.

### Other Docker Make targets

| Target | Description |
|--------|-------------|
| `make docker-pull` | Pull all registry images for your installed version |
| `make docker-images` | Show which image each engine resolves to (local vs registry) |
| `make docker-check` | Validate `docker-compose.yml` configuration |

### Fast rebuilds and first-pull cost

> **Most users never need to build.** `make docker-pull` (or letting `llem run`
> resolve the registry image automatically) gives you a working environment with
> no compilation. Building from source is for contributors and for hosts where
> you've modified `src/llenergymeasure/`.

Every engine image declares `cache_from` entries pointing at the published GHCR tags.
CI populates the cache on each release with `cache-to=type=registry,mode=max`, which
exports intermediate layers to `ghcr.io/henrycgbaker/llenergymeasure/{engine}:latest`
(and the immutable `:v${LLEM_PKG_VERSION}` tag). For Transformers this lets fresh
machines skip the ~30-min flash-attn FA3 Hopper compile.

Measured on `ds01` (AMD EPYC 7742, 128 cores, 504 GB RAM — Docker 27.0.3 / Buildx
v0.32.1 / llenergymeasure 0.9.0):

| Engine | Image size | Cold build | First GHCR pull | Warm local rebuild |
|--------|-----------|------------|-----------------|--------------------|
| Transformers | 7.9 GB | 33m 56s | 2m 33s (10 layers reused) | seconds |
| vLLM | 15.6 GB | 4m 12s | 4m 16s (0 layers reused) | seconds |
| TensorRT-LLM | 50.6 GB | 13m 24s | 13m 32s (0 layers reused) | seconds |

**Reading the table.** Times are measured on a 128-core/504 GB host; on smaller
machines cold builds scale roughly with `MAX_JOBS` (FA3 compile is CPU-bound).

- **Cold build** — fresh builder, `--no-cache`, no GHCR. Simulates an offline
  first-ever build.
- **First GHCR pull** — fresh builder, `cache_from` populated. What a new
  contributor gets after `make docker-builder-setup`.
- **Warm local rebuild** — second and subsequent local builds. Stable layers
  (FA3, base deps) sit before `COPY src/` in the Dockerfiles, so source-only
  edits typically rebuild in seconds for all three engines.

**Why does the GHCR cache only help Transformers?** The vLLM and TensorRT-LLM
images are thin layers (`COPY src/` + one `pip install`) on top of heavy
upstream bases (`vllm/vllm-openai`, `nvcr.io/nvidia/tensorrt-llm/release`).
The dominant cost on a fresh machine is pulling that upstream base from Docker
Hub / NGC, which BuildKit's `cache_from` cannot accelerate — only layers from
*our* Dockerfile are eligible. Our own steps add ~30 s on top, so even a perfect
cache hit caps the saving at ~30 s. The cache is still wired up (and works for
`:v${LLEM_PKG_VERSION}`-pinned rebuilds within a release) but the architecture
limits its impact for these two engines. Transformers benefits because the FA3
compile is in our Dockerfile, ahead of `COPY src/`, so it gets cached.

Once the upstream base is in local Docker storage (after the first build),
subsequent rebuilds for vLLM/TRT are seconds — the slow part doesn't repeat.

**Build as normal:**

```bash
make docker-build-transformers   # or docker-build-vllm / docker-build-tensorrt
```

**How the cache pipeline is wired:**

- CI's `docker-publish.yml` runs `build-push-action` with
  `cache-to=type=registry,...,mode=max` on each release, exporting the full layer
  graph (including FA3 intermediates) to two refs:
  `ghcr.io/henrycgbaker/llenergymeasure/{engine}:v${LLEM_PKG_VERSION}` (immutable
  per release) and `:latest` (rolling).
- `docker-compose.yml` declares `cache_from: [:v${LLEM_PKG_VERSION}, :latest]`
  for every engine — version-pinned first (best layer match within a release),
  rolling-latest as fallback.
- `make docker-builder-setup` provisions a `docker-container` BuildKit driver
  with a 200 GiB GC limit; the default `docker` driver cannot import registry
  caches at all.
- The Transformers FA3 compile (the only layer where caching is load-bearing)
  exceeds GitHub-hosted runner capacity (4 cores / 16 GB), so the seed step is
  manual: `make docker-seed-transformers` from a host with ≥32 cores and
  `docker login ghcr.io` (`write:packages`). After the seed, CI rebuilds warm
  off `:latest` for every subsequent release.
- Pulling the cache is unauthenticated for public packages.

**How to tell if the cache actually warmed:** `make docker-build-{engine}` runs the build
under `BUILDKIT_PROGRESS=plain` and emits a one-line summary when it finishes:

- `✓ transformers build: 4m 18s — GHCR cache imported, 27 layers reused` — cache hit,
  FA3 layer not recompiled.
- `⚠ transformers build: 18m 03s — no GHCR cache imported (cold build)` — silent fallback.
  Cross-check [troubleshooting → Docker rebuild is slow](troubleshooting.md#docker-rebuild-is-slow--recompiling-flash-attn).

The full BuildKit log for the most recent build is at `/tmp/llem-build-{engine}.log`.

**Authentication:** GHCR packages are public. No `docker login` is required to pull them.
If you hit rate limits or are behind a corporate proxy, `docker login ghcr.io` with a
[personal access token](https://github.com/settings/tokens) (scope `read:packages`) may help.

**Push access (contributors).** You do not need push access to develop on this project —
contributors only ever pull cache. Cache publication on releases is fully automated by
`docker-publish.yml` using the repo's auto-issued `GITHUB_TOKEN`, so any merged release
PR ships a fresh cache without human intervention. Manual seeding via
`make docker-seed-transformers` is restricted to the package owner (the packages live
under the `henrycgbaker` user namespace, not an org); this is the standard OSS pattern
for solo-maintained projects and reflects the supply-chain principle that manual pushes
should bypass neither code review nor CI. If you have a legitimate need to push the
cache manually (e.g. infra recovery, base-image emergency reseed), open an issue and
the maintainer can either publish on your behalf or grant per-package collaborator
access in GHCR settings.

**Offline builds:** BuildKit degrades gracefully. When the registry is unreachable the
`cache_from` entries are skipped and the build falls back to local layer cache (cold on a
fresh builder). No errors, just slower.

**First-pull cost:** the first build on any new machine downloads the full cache graph
(sizes above). Subsequent builds are incremental.

### FlashAttention-3

The Transformers Docker image ships with both FlashAttention-2 (FA2) and FlashAttention-3 (FA3)
pre-built. FA3 is compiled from source during the image build, which is the slowest build
step (~20 min). On warm rebuilds the FA3 layer is reused from the GHCR cache (see
[Fast rebuilds and first-pull cost](#fast-rebuilds-and-first-pull-cost) above) and the
build completes in minutes.

FA3 provides Hopper-optimised attention kernels. Use it via
`transformers.attn_implementation: flash_attention_3` in your experiment configs.

**To skip FA3** (e.g. for faster CI builds):

```bash
docker build -f docker/Dockerfile.transformers \
  --build-arg INSTALL_FA3=false \
  -t llenergymeasure:transformers .
```

**Why FA3 takes so long from scratch:** FA3 has no pre-built PyPI wheel. It is compiled from
the `hopper/` subdirectory of the [flash-attention](https://github.com/Dao-AILab/flash-attention)
repository using `nvcc` for CUDA architectures SM 8.0 (A100) and SM 9.0 (H100). CUDA kernel
compilation is inherently slow - each architecture target requires a separate compilation
pass. This is why the build cache is so valuable.

**FA3 hardware requirements:**

| GPU generation | SM | FA2 | FA3 |
|----------------|-----|-----|-----|
| Ampere (A100) | 8.0 | Yes | Yes |
| Hopper (H100) | 9.0 | Yes | Yes (optimised) |
| Ada Lovelace (L40S, RTX 4090) | 8.9 | Yes | Yes |
| Turing or older | < 8.0 | No | No |

**For local (non-Docker) installs**, FA3 must be built manually:

```bash
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
pip install flash-attention/hopper --no-build-isolation
```

This produces the `flash_attn_3` and `flash_attn_interface` packages that `transformers`
checks for at runtime.

---

## Verify Installation

Run `llem config` to check your environment:

```bash
llem config
```

Example output:

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Engines
  transformers: installed
  vllm: not installed  (pip install llenergymeasure[vllm])
  tensorrt: not installed  (pip install llenergymeasure[tensorrt])
Energy
  Energy: nvml
Config
  Path: /home/user/.config/llenergymeasure/config.yaml
  Status: using defaults (no config file)
Python
  3.12.0
```

What each section means:

- **GPU** — NVIDIA GPU detected via pynvml. If this shows "No GPU detected", experiments will fail.
- **Engines** — Which inference engines are installed. You need at least one to run experiments.
- **Energy** — Active energy measurement backend. `nvml` (pynvml) is the default and ships with the base install.
- **Config** — Path to the user config file. "Using defaults" is normal for new installs.
- **Python** — Python version in use.

Run `llem config --verbose` for driver version, engine versions, and full config values.

---

## Next Steps

Follow [Getting Started](getting-started.md) to run your first experiment.
