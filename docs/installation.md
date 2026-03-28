# Installation

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Hard requirement (TensorRT-LLM compatibility) |
| OS | Linux | Required for vLLM and TensorRT-LLM backends |
| GPU | NVIDIA with CUDA 12.x | Required for all inference backends |
| CUDA (host) | 12.x | For container image compatibility |
| Docker + NVIDIA Container Toolkit | Latest | Required for vLLM and TensorRT-LLM |
| Docker Compose | v2.32+ recommended | Required for build cache (see below). v2.11+ minimum |

**macOS/Windows:** PyTorch backend only. Docker-based backends (vLLM, TensorRT-LLM) require Linux.

---

## Install

The base package includes no inference backend. Install with the extras for your use case.

```bash
pip install "llenergymeasure[pytorch]"
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

The base install (`pip install llenergymeasure`) includes no inference backend and cannot run experiments.

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
multiple backends, causing expensive recompilation.

```bash
make docker-builder-setup
```

This creates a `llem-builder` with a 200 GiB GC limit. To use it, set
`BUILDX_BUILDER=llem-builder` in your `.env` file or export it in your shell. Run once per
machine. See [Docker Setup - BuildKit](docker-setup.md#buildkit-builder-setup-recommended)
for details.

## Building Docker Images from Source

The pre-built images from GHCR work for most users. If you need to rebuild images locally
(e.g. after modifying the source code, or to include FlashAttention-3), use Docker Compose:

```bash
# Build a specific backend
docker compose build pytorch

# Build all backends
docker compose build pytorch vllm tensorrt
```

Or with plain `docker build`:

```bash
docker build -f docker/Dockerfile.pytorch -t llenergymeasure:pytorch .
docker build -f docker/Dockerfile.vllm -t llenergymeasure:vllm .
docker build -f docker/Dockerfile.tensorrt -t llenergymeasure:tensorrt .
```

> **When to rebuild.** Images bundle the `llenergymeasure` source at build time. If you
> modify config models, backends, or the container entrypoint, you must rebuild for changes
> to take effect inside containers. Local-runner experiments (PyTorch) use the installed
> source directly and do not need a rebuild.

### Build Cache (recommended)

Docker image builds can be slow - especially the PyTorch image which compiles FlashAttention
from source (~1 hour cold build). To skip this by pulling pre-compiled layers from GHCR:

**1. Log in to GitHub Container Registry:**

```bash
docker login ghcr.io
```

You need a GitHub account with a
[personal access token](https://github.com/settings/tokens) (read:packages scope).

**2. Enable COMPOSE_BAKE in your `.env`:**

```bash
COMPOSE_BAKE=true
```

This is already set in `.env.example`. It tells Docker Compose to delegate builds to
BuildKit's [bake](https://docs.docker.com/build/bake/) engine, which has full support for
registry-based build cache.

**3. Build as normal:**

```bash
docker compose build pytorch
```

Compose will pull cached layers from GHCR (written by CI on each release) and only rebuild
layers that have changed locally. A cached PyTorch build typically completes in under 2
minutes instead of ~1 hour.

**Requirements:** Docker Compose v2.32+ (`docker compose version` to check). If you have
an older version, see the
[Docker Compose install docs](https://docs.docker.com/compose/install/) to upgrade.

**Without COMPOSE_BAKE:** Builds work normally but don't use registry cache. The `cache_from`
entries in `docker-compose.yml` are silently ignored. No errors, just slower builds.

### FlashAttention-3 (optional)

The PyTorch Docker image ships with FlashAttention-2 (FA2) pre-built. FlashAttention-3 (FA3)
is **not included by default** because it must be compiled from source, which adds
approximately 1 hour to the image build.

FA3 provides Hopper-optimised attention kernels. It is required if you want to use
`pytorch.attn_implementation: flash_attention_3` in your experiment configs.

**To build the PyTorch image with FA3:**

```bash
docker build -f docker/Dockerfile.pytorch \
  --build-arg INSTALL_FA3=true \
  -t ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0 .
```

**Why FA3 takes so long:** FA3 has no pre-built PyPI wheel. It is compiled from the
`hopper/` subdirectory of the [flash-attention](https://github.com/Dao-AILab/flash-attention)
repository using `nvcc` for CUDA architectures SM 8.0 (A100) and SM 9.0 (H100). CUDA kernel
compilation is inherently slow - each architecture target requires a separate compilation
pass.

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
Backends
  pytorch: installed
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
- **Backends** — Which inference backends are installed. You need at least one to run experiments.
- **Energy** — Active energy measurement backend. `nvml` (pynvml) is the default and ships with the base install.
- **Config** — Path to the user config file. "Using defaults" is normal for new installs.
- **Python** — Python version in use.

Run `llem config --verbose` for driver version, backend versions, and full config values.

---

## Next Steps

Follow [Getting Started](getting-started.md) to run your first experiment.
