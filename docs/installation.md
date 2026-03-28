# Installation

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Hard requirement (TensorRT-LLM compatibility) |
| OS | Linux | Required for vLLM and TensorRT-LLM backends |
| GPU | NVIDIA with CUDA 12.x | Required for all inference backends |
| CUDA (host) | 12.x | For container image compatibility |
| Docker + NVIDIA Container Toolkit | Latest | Required for vLLM and TensorRT-LLM |

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

This creates a `llem-builder` with a 200 GiB GC limit (vs ~93 GiB default). Run once per
machine. See [Docker Setup - BuildKit](docker-setup.md#buildkit-builder-setup-recommended)
for details.

## Building Docker Images from Source

The pre-built images from GHCR work for most users. If you need to rebuild images locally
(e.g. after modifying the source code, or to include FlashAttention-3), build from the
repository root:

```bash
# PyTorch image (includes FA2, excludes FA3 by default)
docker build -f docker/Dockerfile.pytorch -t ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0 .

# vLLM image
docker build -f docker/Dockerfile.vllm -t ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0 .

# TensorRT-LLM image
docker build -f docker/Dockerfile.tensorrt -t ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0 .
```

Replace `v0.9.0` with the version shown by `llem --version`. The tag must match the
installed version so the tool finds the correct image.

> **When to rebuild.** Images bundle the `llenergymeasure` source at build time. If you
> modify config models, backends, or the container entrypoint, you must rebuild for changes
> to take effect inside containers. Local-runner experiments (PyTorch) use the installed
> source directly and do not need a rebuild.

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
