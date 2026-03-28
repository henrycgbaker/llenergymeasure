# Docker Setup Guide

`llenergymeasure` uses Docker to run vLLM and TensorRT-LLM (and future SGLang) backends in
isolated containers with GPU access. This guide walks through setting up Docker with GPU support
from scratch on a Linux host.

> **Recommended path.** Docker is the recommended way to run llenergymeasure. The PyTorch
> backend can run locally for quick tests, but Docker gives you vLLM and reproducible
> container-isolated measurements.

---

## Prerequisites

Before starting, confirm you have:

- A Linux host (GPU passthrough in Docker is Linux-only — no macOS or Windows Docker GPU support)
- An NVIDIA GPU installed
- Root or sudo access to install system packages

No prior Docker knowledge is assumed. Each step includes a verification command.

---

## Step 1: Install Docker

If Docker is already installed, skip to the verification step.

Follow the [official Docker Engine install guide](https://docs.docker.com/engine/install/) for
your distribution. For Ubuntu/Debian, Docker's own script is the easiest path:

```bash
# Install Docker Engine (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Post-install step:** Add your user to the `docker` group so you can run Docker without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

**Verify Docker is working:**

```bash
docker run hello-world
```

Expected output includes `Hello from Docker!`.

### BuildKit builder setup (recommended)

Docker image builds use BuildKit under the hood. The default builder has a conservative
garbage-collection limit (~10% of disk) that is too small when building all three backend
images (PyTorch, vLLM, TensorRT). This causes build cache eviction and expensive
recompilation (FA3 takes ~1 hour from scratch).

Create a dedicated builder with a 200 GiB cache limit:

```bash
make docker-builder-setup
```

This creates a `docker-container` driver builder called `llem-builder` with tuned GC limits
(configured in `docker/buildkitd.toml`). To use it, set the `BUILDX_BUILDER` environment
variable:

```bash
export BUILDX_BUILDER=llem-builder
docker compose build
```

Or add `BUILDX_BUILDER=llem-builder` to your `.env` file for project-scoped use.

The command is idempotent - running it again is a no-op if the builder already exists.

To recreate the builder (e.g. after changing `buildkitd.toml`):

```bash
make docker-builder-rm
make docker-builder-setup
```

---

## Step 2: Install NVIDIA Drivers

If `nvidia-smi` already works on your host, skip this step.

```bash
nvidia-smi
```

If `nvidia-smi` is not found or returns an error, install NVIDIA drivers for your distribution.
Follow the [NVIDIA driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/)
for your OS and GPU model.

Expected `nvidia-smi` output (your GPU name and driver version will differ):

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================================================================|
|   0  NVIDIA A100-SXM4-80GB          On  |   00000000:00:04.0 Off |                    0 |
| N/A   34C    P0             54W /  400W |       0MiB /  81920MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```

---

## Step 3: Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to access the host GPU. This is the
critical step that makes `docker run --gpus all` work.

**Ubuntu/Debian:**

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker for the configuration to take effect
sudo systemctl restart docker
```

**Other distributions:** Follow the
[NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for RHEL/CentOS, Fedora, SUSE, or Arch Linux.

---

## Step 4: Verify GPU Access in Docker

Run a container with GPU access and check that `nvidia-smi` works inside it:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

This command:
- Pulls a small CUDA base image (~100 MB)
- Launches a container with all host GPUs passed through (`--gpus all`)
- Runs `nvidia-smi` inside the container
- Exits and removes the container (`--rm`)

Expected output: the same `nvidia-smi` table you saw in Step 2, but printed from inside
the container.

> This is exactly the check that `llem`'s Docker pre-flight performs automatically before
> launching any backend container. If this command works, `llem` pre-flight will pass.

**If this command fails**, see [Troubleshooting](#troubleshooting) below.

---

## Step 5: Verify with llenergymeasure

With Docker and NVIDIA CT installed, verify that `llem` sees the GPU:

```bash
llem config
```

Expected output will show your GPU under the `GPU` section and Docker runner availability.
See [Installation](installation.md) for the full `llem config` output format.

**Run a vLLM experiment** by creating a YAML file:

```yaml
# experiment.yaml
model: gpt2
backend: vllm
n: 50
runners:
  vllm: docker
```

Then run it:

```bash
llem run experiment.yaml
```

`llem` will automatically pull the vLLM Docker image on first use, launch a container, run
the experiment inside it, and return the results. See [Getting Started](getting-started.md) for
an annotated walkthrough.

**Run a TensorRT-LLM experiment** by creating a YAML file:

```yaml
# experiment.yaml
model: meta-llama/Llama-2-7b-hf
backend: tensorrt
n: 50
runners:
  tensorrt: docker
```

Then run it:

```bash
llem run experiment.yaml
```

`llem` will pull the TensorRT-LLM Docker image, compile a TensorRT engine (first run only —
takes several minutes), cache the engine on disk, then run inference. See [Getting Started](getting-started.md)
for the full TensorRT-LLM walkthrough.

---

## Image Management

llenergymeasure uses pre-built Docker images from GitHub Container Registry. Images follow
this naming convention:

```
ghcr.io/henrycgbaker/llenergymeasure/{backend}:v{version}
```

For example:
```
ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0
ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0
```

**Auto-pull on first use.** When you run a backend for the first time, `llem` automatically
pulls the correct image. No manual `docker pull` is needed.

**Pre-fetch an image** (optional, useful for offline environments):

```bash
docker pull ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0
docker pull ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0
```

Replace `0.9.0` with your installed `llenergymeasure` version (run `llem --version` to check).

**Future backends.** SGLang (M5) images will follow the same naming convention and auto-pull
behaviour. No additional setup is needed when SGLang ships.

---

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"

NVIDIA Container Toolkit is not installed, or Docker was not restarted after configuration.

**Fix:**
1. Confirm toolkit is installed: `which nvidia-ctk` should return a path.
2. Re-run: `sudo nvidia-ctk runtime configure --runtime=docker`
3. Restart Docker: `sudo systemctl restart docker`
4. Retry the `docker run --gpus all ...` command.

---

### "nvidia-container-cli: initialization error"

A driver version mismatch between the host NVIDIA driver and the CUDA version in the container.

**Fix:** Check your host driver version with `nvidia-smi`. The container image requires a
minimum driver version for its CUDA release. See the
[CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) to confirm
your driver supports the container's CUDA version.

For example, CUDA 12.4 requires driver >= 525.60.13.

---

### "Permission denied" when running docker commands

Your user is not in the `docker` group.

**Fix:**
```bash
sudo usermod -aG docker $USER
newgrp docker   # apply group change in current shell
```

Or log out and log back in. Verify with: `groups | grep docker`.

---

### GPU not visible inside container

The `--gpus all` flag is missing from the `docker run` command.

`llem` adds `--gpus all` automatically when launching backend containers. If you are running
Docker commands manually, ensure you include `--gpus all` (or `--gpus device=0` for a specific
GPU).

---

### Shared memory errors with vLLM

vLLM requires more than the default 64 MB of shared memory (`/dev/shm`). `llem` automatically
sets `--shm-size 8g` when launching vLLM containers. If you are running the vLLM container
manually, add `--shm-size 8g` to your `docker run` command.

---

### Pre-flight check failures

`llem` runs pre-flight checks before launching any Docker container. The checks and their
failure modes:

| Check | Failure message | Fix |
|-------|----------------|-----|
| Docker CLI on PATH | `Docker not found on PATH` | Install Docker Engine (Step 1) |
| NVIDIA Container Toolkit binary | `NVIDIA Container Toolkit not found` | Install NVIDIA CT (Step 3) |
| Host `nvidia-smi` | Warning (non-blocking) | Expected if using remote Docker daemon |
| GPU visibility in container | `GPU not accessible inside Docker container` | Re-run Steps 3-4 |
| CUDA/driver compatibility | `CUDA/driver compatibility error inside container` | Update host driver |

To bypass pre-flight checks temporarily (not recommended for production):

```bash
llem run experiment.yaml --skip-preflight
```

---

## Next Steps

- [Getting Started](getting-started.md) — run your first vLLM or TensorRT-LLM experiment
- [Backend Configuration](backends.md) — configure vLLM, TensorRT-LLM, and switch between backends
