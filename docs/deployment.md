# Deployment Guide

Docker deployment, GPU configuration, and troubleshooting.

## Quick Setup

```bash
# Install locally
pip install -e .

# Check your environment
lem doctor

# Run experiment
lem experiment configs/examples/pytorch_example.yaml -n 10
```

For multi-backend campaigns requiring Docker:

```bash
# Create .env file
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env

# Build backend images
docker compose build pytorch vllm

# Run campaign
lem campaign multi-backend-campaign.yaml
```

## Running Modes

| Mode | Best For | Setup |
|------|----------|-------|
| **Local (recommended)** | Quick experiments | `pip install -e .` then `lem doctor` |
| **Docker** | Multi-backend campaigns | `docker compose build`, then `lem campaign` |
| **Docker dev** | Test in container | `docker compose --profile dev run --rm pytorch-dev` |
| **VS Code devcontainer** | Full IDE + GPU | "Reopen in Container" |

## Docker

### Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA 12.4** compatible drivers (base image uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- **nvidia-container-toolkit** installed and configured
- **Privileged mode** for energy metrics (docker-compose.yml sets `privileged: true`)

### Backend Services

The project provides separate Docker services for each inference backend:

| Service | Image Tag | Use Case |
|---------|-----------|----------|
| `pytorch` | `lem:pytorch` | Default, most compatible |
| `vllm` | `lem:vllm` | High throughput with PagedAttention |
| `tensorrt` | `lem:tensorrt` | Maximum performance (Ampere+ GPUs) |
| `pytorch-dev` | `lem:pytorch-dev` | Development with PyTorch |
| `vllm-dev` | `lem:vllm-dev` | Development with vLLM |
| `tensorrt-dev` | `lem:tensorrt-dev` | Development with TensorRT |

**Note**: vLLM and TensorRT have conflicting PyTorch dependencies. Use separate images rather than installing multiple backends in one environment.

### Environment Variables

For Docker mode, create a `.env` file:

```bash
# Create .env with required user IDs
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env

# Optional: Add HuggingFace token for gated models (Llama, Mistral, etc.)
echo "HF_TOKEN=your_huggingface_token" >> .env

# Optional: GPU selection
echo "CUDA_VISIBLE_DEVICES=0,1" >> .env

# Optional: CodeCarbon logging level
echo "CODECARBON_LOG_LEVEL=warning" >> .env
```

**PUID/PGID are REQUIRED** for Docker mode. The entrypoint will exit with an error if not set. This ensures files created in containers are owned by your host user.

**Note:** For local install (not Docker), `.env` is not needed — campaign command auto-generates it on first multi-backend run.

### Volume Architecture

The tool uses a hybrid volume strategy for optimal performance and usability:

| Mount | Type | Purpose |
|-------|------|---------|
| `results/` | Bind mount | User-accessible experiment results |
| `configs/` | Bind mount (RO) | User-editable config files |
| `lem-hf-cache` | Named volume | HuggingFace model cache (10-100+ GB) |
| `lem-trt-engine-cache` | Named volume | TensorRT compiled engines |
| `lem-experiment-state` | Named volume | Experiment state and progress |

**Why named volumes for caches?**
- No permission issues (Docker-managed)
- Persist across container rebuilds
- Fast startup (no recursive chown)

**Viewing named volume contents:**
```bash
# Inspect volume
docker volume inspect lem-hf-cache

# Copy files from volume
docker run --rm -v lem-hf-cache:/data -v $(pwd):/backup alpine cp -r /data /backup/hf-cache-backup
```

### Cleaning Up Volumes

```bash
# Clean specific volumes
make lem-clean-state    # Clear experiment state
make lem-clean-cache    # Clear HuggingFace cache (will re-download models)
make lem-clean-trt      # Clear TensorRT engines
make lem-clean-all      # Clear all named volumes

# Or use docker directly
docker volume rm lem-hf-cache lem-experiment-state lem-trt-engine-cache
```

**Important**: `docker compose down -v` deletes ALL named volumes. Use `docker compose down` (without `-v`) to preserve caches.

**Results directory precedence:**
1. `--results-dir` CLI flag (highest)
2. `io.results_dir` in config YAML
3. `LLM_ENERGY_RESULTS_DIR` environment variable
4. Default `results/` (lowest)

### Makefile (Recommended)

The Makefile handles UID/GID mapping automatically:

```bash
# Build backends
make docker-build-pytorch     # Build PyTorch backend (default)
make docker-build-vllm        # Build vLLM backend
make docker-build-tensorrt    # Build TensorRT backend
make docker-build-all         # Build all backends
make docker-build-dev         # Build PyTorch dev image

# Run commands
make datasets                 # List available datasets
make validate CONFIG=test.yaml # Validate a config
make experiment CONFIG=test.yaml DATASET=alpaca SAMPLES=100
make lem CMD="results list"   # Run any lem command

# Interactive shells
make docker-shell             # Production shell (pytorch)
make docker-dev               # Development shell (pytorch-dev)
```

### Docker Compose

#### Building Images

Build the base image first, then the backend you need:

```bash
# Build base + PyTorch (default)
docker compose build base pytorch

# Build base + vLLM
docker compose build base vllm

# Build base + TensorRT
docker compose build base tensorrt

# Build all backends
docker compose build base pytorch vllm tensorrt

# Build dev images (include --profile dev)
docker compose --profile dev build base pytorch-dev
```

#### Running Experiments

```bash
# PyTorch backend (default)
docker compose run --rm pytorch \
  lem experiment /app/configs/test.yaml --dataset alpaca -n 100

# vLLM backend
docker compose run --rm vllm \
  lem experiment /app/configs/test.yaml --dataset alpaca -n 100

# TensorRT backend
docker compose run --rm tensorrt \
  lem experiment /app/configs/test.yaml --dataset alpaca -n 100

# Interactive shell
docker compose run --rm pytorch /bin/bash
```

#### Development Mode

Development containers mount the source code for live editing:

```bash
# Build dev image
docker compose --profile dev build base pytorch-dev

# Interactive shell (source mounted, editable install)
docker compose --profile dev run --rm pytorch-dev

# Run command in dev container
docker compose --profile dev run --rm pytorch-dev \
  lem experiment /app/configs/test.yaml -d alpaca -n 10
```

### Container Strategies

When running multi-backend campaigns with Docker, you can choose between two container strategies:

#### Overview

| Strategy | Startup Overhead | Isolation | Best For |
|----------|------------------|-----------|----------|
| **ephemeral** (default) | 3-5s per experiment | Perfect (fresh GPU state) | Most campaigns, reproducibility |
| **persistent** | Once at campaign start | Shared (GPU memory accumulates) | Many short experiments, development |

#### Ephemeral Mode (Default)

Uses `docker compose run --rm` to create fresh containers for each experiment.

**Characteristics:**
- Fresh container = fresh GPU memory and CUDA context
- Automatic cleanup on errors
- Perfect isolation between experiments
- Negligible overhead (~1.3% of typical campaign time)

**Recommended for:**
- Production measurements
- Reproducibility requirements
- Long experiments (>1 minute each)

**Example:**
```bash
# Each experiment runs in a fresh container
lem campaign multi-backend.yaml
```

#### Persistent Mode

Uses `docker compose up -d` + `docker compose exec` to run experiments in long-running containers.

**Characteristics:**
- Containers stay running between experiments
- Faster (no container startup overhead)
- GPU memory may accumulate across experiments
- Requires health monitoring and manual cleanup
- Confirmation prompt (or `--yes` flag to skip)

**Recommended for:**
- Development and debugging
- Many short experiments (<30 seconds each)
- Iterating on configurations

**Example:**
```bash
# Containers persist across experiments
lem campaign multi-backend.yaml --container-strategy persistent --yes
```

#### Configuration

**Via CLI flag:**
```bash
lem campaign config.yaml --container-strategy persistent
```

**Via user config (`.lem-config.yaml`):**
```yaml
docker:
  strategy: persistent  # or ephemeral (default)
  warmup_delay: 5.0     # seconds to wait after container start
  auto_teardown: true   # cleanup containers after campaign completion
```

**Precedence:** CLI flag > user config > default (ephemeral)

#### Choosing a Strategy

**Use ephemeral (default) when:**
- Running formal benchmark measurements
- Reproducibility is critical
- Experiments are long (>1 minute each)
- Overhead of 3-5s per experiment is negligible

**Use persistent when:**
- Developing or debugging experiments
- Running many short experiments (<30s each)
- Container startup overhead is significant relative to experiment time
- You understand the isolation tradeoffs

**Performance note:** For typical campaigns (6 experiments, 5 minutes each), ephemeral overhead is ~24 seconds out of 30 minutes (1.3% of total time).

### VS Code Devcontainer

1. Install [VS Code Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Set `HF_TOKEN` in your shell environment
3. Open project and use `Ctrl+Shift+P` then "Dev Containers: Reopen in Container"

The devcontainer:
- Runs as root (avoids permission issues)
- Mounts source code (edits sync instantly)
- Mounts HuggingFace cache (models persist)
- Has GPU passthrough enabled

### Legacy Volume Mounts Reference

This table shows how volumes are mounted in the container:

| Container Path | Source | Purpose |
|----------------|--------|---------|
| `/app/configs` | `./configs` (bind) | Experiment configs (read-only) |
| `/app/results` | `./results` (bind) | Output results (user-accessible) |
| `/app/scripts` | `./scripts` (bind, ro) | Entrypoint scripts |
| `/app/.cache/huggingface` | `lem-hf-cache` (named) | Model cache |
| `/app/.cache/tensorrt-engines` | `lem-trt-engine-cache` (named) | TensorRT engines |
| `/app/.state` | `lem-experiment-state` (named) | Experiment state |

### Backend-Specific Notes

#### vLLM

- Uses `ipc: host` for shared memory (required for multiprocessing)
- Installs its own PyTorch version (2.8+) for compatibility

#### TensorRT

- Uses `ipc: host` for shared memory (required for multiprocessing)
- Requires compute capability >= 8.0 (Ampere: A100, A10, RTX 30xx/40xx, H100, L40)
- NOT supported: V100, T4, RTX 20xx, GTX series
- Mounts engine cache for compiled TensorRT engines
- Requires MPI libraries (included in image)

## MIG GPU Support

NVIDIA MIG (Multi-Instance GPU) allows partitioning A100/H100 GPUs into isolated instances.

### What Works

- Single-process experiments on individual MIG instances
- Parallel independent experiments on different MIG instances
- MIG detection and metadata recording (`gpu_is_mig`, `gpu_mig_profile`)

### What Does NOT Work

- **Multi-process distributed inference** across MIG instances (hardware limitation)
- Using parent GPU index when MIG is enabled (must use UUID)

### Usage

```bash
# 1. List MIG instances
nvidia-smi -L

# Example output:
# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-xxx)
#   MIG 3g.20gb Device 0: (UUID: MIG-abc123)
#   MIG 3g.20gb Device 1: (UUID: MIG-def456)

# 2. Run on specific MIG instance
CUDA_VISIBLE_DEVICES=MIG-abc123 lem experiment config.yaml --dataset alpaca -n 100

# 3. Parallel experiments (separate terminals)
# Terminal 1:
CUDA_VISIBLE_DEVICES=MIG-abc123 lem experiment config1.yaml --dataset alpaca -n 100
# Terminal 2:
CUDA_VISIBLE_DEVICES=MIG-def456 lem experiment config2.yaml --dataset alpaca -n 100
```

### MIG with Docker

```bash
# Use specific MIG instance
CUDA_VISIBLE_DEVICES=MIG-abc123 docker compose run --rm pytorch \
  lem experiment /app/configs/test.yaml --dataset alpaca -n 100

# Or set in .env
echo "CUDA_VISIBLE_DEVICES=MIG-abc123" >> .env
```

### Energy Measurement Limitation

Energy readings on MIG instances reflect the **parent GPU's total power**, not per-instance usage. This is a hardware/driver limitation. Results include `energy_measurement_warning` when running on MIG.

For accurate energy measurements:
- Use full GPUs (disable MIG)
- Or ensure no other workloads on sibling MIG instances

## CI/CD Pipeline

### Overview

The project uses a two-tier CI strategy:

| Tier | Workflow | Runs on | Trigger | Purpose |
|------|----------|---------|---------|---------|
| **Tier 1** | `ci.yml` | GitHub-hosted (`ubuntu-latest`) | Every PR + push to main | Lint, type-check, unit tests, package validation |
| **Tier 2** | `gpu-ci.yml` | Self-hosted (`ds01-gpu`) | Manual, release tags, `gpu-ci` label | Full test suite with GPU + Docker + SIGINT verification |
| **Release** | `release.yml` | Both | `v*` tag push | Tier 1 + Tier 2 gate, then build + publish |

### Tier 2 GPU CI Triggers

GPU CI does **not** run on every push to main. This is intentional — GPU tests are
expensive (10-30 min) and most changes don't affect GPU behaviour. Instead:

1. **Manual** (`workflow_dispatch`): Run any time via GitHub UI or CLI:
   ```bash
   gh workflow run gpu-ci.yml
   gh run watch    # follow progress
   ```

2. **Release gate**: Automatically called by `release.yml` when a `v*` tag is pushed.
   The GitHub Release is only created if GPU CI passes.

3. **Label-gated on PRs**: Add the `gpu-ci` label to a PR to trigger GPU tests on
   that PR. Useful when a change touches GPU-sensitive code (backends, measurement,
   Docker). Remove and re-add the label to re-trigger.

### Self-Hosted Runner (ds01-gpu)

The Tier 2 runner is a GitHub Actions agent running on DS01 (4x A100-PCIE-40GB).

**Location**: `~/actions-runner/`
**Service**: User-level systemd (`github-runner.service`)
**Labels**: `self-hosted`, `Linux`, `X64`, `gpu`

#### Managing the runner

```bash
# Status
systemctl --user status github-runner.service

# Restart
systemctl --user restart github-runner.service

# Logs
journalctl --user -u github-runner.service -f

# Stop (e.g. for maintenance)
systemctl --user stop github-runner.service
```

#### Runner survives logout (requires admin)

The runner requires lingering to persist across user sessions. An admin must run
(one-time):

```bash
sudo loginctl enable-linger h.baker@hertie-school.lan
```

Verify: `loginctl show-user h.baker@hertie-school.lan --property=Linger` should
show `Linger=yes`.

Without lingering, the runner stops when you log out of DS01.

#### Re-registering the runner

If the runner loses its registration (e.g. deleted from GitHub settings):

```bash
cd ~/actions-runner
TOKEN=$(gh api repos/henrycgbaker/LLenergyMeasure/actions/runners/registration-token -X POST --jq '.token')
./config.sh --url https://github.com/henrycgbaker/LLenergyMeasure --token "$TOKEN" --name ds01-gpu --labels self-hosted,linux,gpu --work _work --unattended
systemctl --user restart github-runner.service
```

### Release Pipeline

Releases are fully automated via the `release` label.

#### Automated flow

```
1. Create PR: version bump in pyproject.toml + __init__.py
2. Add 'release' label to the PR
3. Tier 1 CI runs (lint, type-check, test)
4. Review and merge

   ┌─── auto-release.yml triggers ───┐
   │                                  │
   │  Tier 2 GPU CI (Docker + GPU)   │
   │         │                        │
   │    ┌────┴────┐                   │
   │  Pass      Fail                  │
   │    │         │                   │
   │  Create    Create GitHub Issue   │
   │  git tag   (no tag, no release)  │
   └────┬─────────────────────────────┘
        │
   release.yml triggers on v* tag
        │
   ├── lint + type-check + test (re-check)
   ├── GPU CI (belt-and-suspenders)
   │
   release (build wheel, create GitHub Release)
        │
   docker-publish (build + push all backends to GHCR)
```

#### Manual release (fallback)

If the automated flow fails or you need to release manually:

```bash
# 1. Verify GPU CI passes
gh workflow run gpu-ci.yml
gh run watch

# 2. Create tag
git tag v0.9.0
git push origin v0.9.0

# 3. release.yml runs automatically from tag push
```

If a tag needs to be re-done:

```bash
git tag -d v0.9.0 && git push origin :v0.9.0   # delete tag
# fix the issue, then:
git tag v0.9.0 && git push origin v0.9.0        # re-tag
```

## Troubleshooting

### Energy Metrics are Zero

**Cause**: CodeCarbon needs NVML access, which requires privileged mode.

**Solution**: Ensure `privileged: true` in docker-compose.yml. Without compose:
```bash
docker run --privileged --gpus all ...
```

### CUDA Version Mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available`

**Solution**: Ensure NVIDIA drivers support CUDA 12.4+:
```bash
nvidia-smi  # Check driver version
```

### Permission Denied on Results/State

The Docker image requires PUID/PGID to be set for proper file ownership.

**Solution 1: Create .env file (recommended)**
```bash
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env
```

**Solution 2: Pass inline**
```bash
PUID=$(id -u) PGID=$(id -g) docker compose run --rm pytorch ...
```

If you previously ran as root and have root-owned files:
```bash
# Fix ownership
docker run --rm -v $(pwd)/results:/results alpine chown -R $(id -u):$(id -g) /results
```

### PUID/PGID Required Error

If you see:
```
ERROR: PUID and PGID environment variables are required.
```

Create the `.env` file manually:
```bash
echo "PUID=$(id -u)" >> .env
echo "PGID=$(id -g)" >> .env
```

Or run `lem doctor` to check your environment setup.

### MIG Device Errors

**Symptom**: `RuntimeError: CUDA error: invalid device ordinal` or `NCCL error`

**Cause**: MIG instances are hardware-isolated.

**Solutions**:

1. **Single-process on MIG** (recommended):
   ```bash
   CUDA_VISIBLE_DEVICES=MIG-abc123 lem experiment config.yaml --dataset alpaca -n 100
   ```

2. **Distributed inference** - use full GPUs (non-MIG)

3. **Parallel independent experiments** on separate MIG instances

### Model Downloads Every Run

**Cause**: Named volume for HF cache was deleted.

**Solution**: The HuggingFace cache is stored in a Docker named volume (`lem-hf-cache`) that persists across container runs. If models are re-downloading:

1. Check if volume exists:
   ```bash
   docker volume ls | grep lem-hf-cache
   ```

2. If missing, it will be recreated automatically on next run. Models will be cached after first download.

3. **Don't use** `docker compose down -v` as it deletes all named volumes including the cache.

**Pre-populating the cache**: If you have models cached on the host, you can copy them in:
```bash
docker run --rm -v lem-hf-cache:/cache -v ~/.cache/huggingface:/host alpine cp -r /host/* /cache/
```

### Config Validation Errors

**Solutions**:
- Ensure YAML format (not Python)
- Validate first: `lem config validate config.yaml`
- Check field names match schema

### Out of Memory (OOM)

**Solutions**:
- Reduce `batching.batch_size`
- Reduce `max_input_tokens` / `max_output_tokens`
- Enable quantization: `quantization.load_in_8bit: true`
- Lower precision: `fp_precision: float16`

### vLLM/TensorRT Shared Memory Errors

**Symptom**: Errors about shared memory or IPC

**Cause**: vLLM and TensorRT require shared memory for multiprocessing.

**Solution**: The docker-compose.yml sets `ipc: host` for these services. If running manually:
```bash
docker run --ipc=host --gpus all ...
```
