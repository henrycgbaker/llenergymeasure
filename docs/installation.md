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
llem v0.8.0
```

---

## Docker Setup

For vLLM or TensorRT-LLM backends, Docker with NVIDIA Container Toolkit is required. See the [Docker Setup Guide](docker-setup.md) for a complete walkthrough covering driver installation, toolkit setup, and verification.

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
