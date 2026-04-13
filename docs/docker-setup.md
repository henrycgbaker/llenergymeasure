# Docker Setup Guide

`llenergymeasure` uses Docker to run vLLM and TensorRT-LLM (and future SGLang) engines in
isolated containers with GPU access. This guide walks through setting up Docker with GPU support
from scratch on a Linux host.

> **Recommended path.** Docker is the recommended way to run llenergymeasure. The PyTorch
> engine can run locally for quick tests, but Docker gives you vLLM and reproducible
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

**Verify Docker Compose and Buildx versions** (v2.32+ and v0.17+ recommended for
[fast rebuilds](installation.md#fast-rebuilds-and-first-pull-cost)):

```bash
docker compose version   # need v2.32+
docker buildx version    # need v0.17+
```

If your versions are below these, you can upgrade the plugins directly:

```bash
# Upgrade Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo chmod 755 /usr/libexec/docker/cli-plugins/docker-compose

# Upgrade Docker Buildx (find latest version at https://github.com/docker/buildx/releases)
BUILDX_VERSION=v0.32.1
sudo curl -L "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-amd64" \
  -o /usr/libexec/docker/cli-plugins/docker-buildx
sudo chmod 755 /usr/libexec/docker/cli-plugins/docker-buildx
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
garbage-collection limit (~10% of disk) that is too small when building all three engine
images (Transformers, vLLM, TensorRT). This causes build cache eviction and expensive
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
> launching any engine container. If this command works, `llem` pre-flight will pass.

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
engine: vllm
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
engine: tensorrt
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

### Image sources

each engine has two possible image sources:

| Source | Tag pattern | Built by | Use case |
|--------|------------|----------|----------|
| **Local build** | `llenergymeasure:{engine}` | `make docker-build-{engine}` | Development - reflects current source tree |
| **Registry** | `ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}` | CI on release tags | Production, CI, pip-install users |

### Image resolution

When you run `llem run`, the tool resolves which image to use for each engine. Resolution
follows a precedence chain (highest wins):

1. **Environment variable** `LLEM_IMAGE_{ENGINE}` (e.g. `LLEM_IMAGE_VLLM=my/custom:tag`)
2. **Study YAML** `images:` section
3. **Runner spec** shorthand (`docker:my/custom:tag` in `runners:`)
4. **User config** `images:` section (`~/.config/llenergymeasure/config.yaml`)
5. **Smart default**: local build image if present, otherwise registry image

In practice, most users rely on the smart default (level 5). If you have built images locally
with `make docker-build-all`, those are used. Otherwise, `llem` uses the GHCR registry image
matching your installed version.

### Auto-pull on first use

When `llem` needs a registry image that is not cached locally, it pulls it automatically
before the experiment runs. The preflight panel shows which images will be used and whether
they are cached or need pulling:

```
Runners
  vllm              docker
    ↳ llenergymeasure:vllm                              ← local build
  pytorch           docker
    ↳ ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0  ← registry
```

### Study-level image preparation

For multi-experiment studies, `llem` checks and pulls all required Docker images **once**
before the first experiment runs, not per-experiment. This avoids redundant pulls when
multiple experiments share the same engine image. The CLI shows this as a
"Preparing Docker images" section with per-image status (cached vs pulled) and metadata
(image ID, size, age, layers).

### Pre-fetch images manually

For offline environments or to avoid pull latency during experiments:

```bash
# Pull all registry images for your version
make docker-pull

# Or pull individually
docker pull ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0
docker pull ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0
docker pull ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0
```

Replace `0.9.0` with your installed version (`llem --version`).

### Check current image resolution

To see which images `llem` will use for each engine:

```bash
make docker-images
```

Output shows local vs registry source for each engine:

```
=== Image resolution ===
  pytorch    -> llenergymeasure:transformers  (local_build)
  tensorrt   -> ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0  (registry)
  vllm       -> llenergymeasure:vllm  (local_build)
```

### Building images locally

Build images from source when you have modified the codebase or need images that reflect
your local changes:

```bash
# Build all engines
make docker-build-all

# Build a specific engine
make docker-build-pytorch
make docker-build-vllm
make docker-build-tensorrt
```

These targets use `docker compose build` under the hood and pull cached layers from the
GHCR registry on first build (see
[Fast rebuilds and first-pull cost](installation.md#fast-rebuilds-and-first-pull-cost)
for the full mechanism).

> **Advanced.** Setting `COMPOSE_BAKE=true` routes builds through `buildx bake` for
> parallel multi-engine builds. With the current cache architecture this is rarely
> worth enabling — vLLM/TRT cold builds are already 4–13 min and warm rebuilds are
> seconds, so the parallelism gain is small. Left out of `.env.example` to avoid
> noise; opt in only if you frequently run `make docker-build-all` from cold.

> **When to rebuild.** Images bundle the `llenergymeasure` source at build time. If you
> modify config models, engines, or the container entrypoint, rebuild for changes to take
> effect inside containers. Local-runner experiments (Transformers without Docker) use the
> installed source directly and do not need a rebuild.

### Override images in YAML

```yaml
runners:
  transformers: local                       # host execution, no Docker
  vllm: docker                         # default resolution (local → registry)
  tensorrt: "docker:my/custom:tag"     # explicit image override
```

### Future engines

SGLang (M5) images will follow the same naming convention, resolution logic, and auto-pull
behaviour. No additional setup is needed when SGLang ships.

### Layer cache sharing via GHCR registry

See [installation.md — Fast rebuilds and first-pull cost](installation.md#fast-rebuilds-and-first-pull-cost)
for the user-facing walkthrough (mechanism, sizes, authentication, offline fallback).

Operator notes:

- `cache-to` pushes only to `:latest` (never to immutable version tags), so
  storage growth is bounded by image drift between releases.
- Inspect what's cached on the active builder: `docker buildx du --builder llem-builder`.
- If the cache is corrupt, recreate it with `make docker-builder-rm && make docker-builder-setup`.

### Image labels and versioning

Every engine image is stamped at build time with OCI labels that llem reads at
study start-up to detect host/container schema skew:

| Label | Source | Purpose |
|-------|--------|---------|
| `org.opencontainers.image.version` | `LLEM_PKG_VERSION` build-arg (from `_version.py`) | Human-readable llenergymeasure release baked into the image |
| `org.opencontainers.image.source` | Dockerfile | Points at the GitHub repository |
| `llem.expconf.schema.fingerprint` | `LLEM_EXPCONF_SCHEMA_FINGERPRINT` build-arg | SHA-256 of `ExperimentConfig.model_json_schema()`; the blocking signal for schema skew |

The package version is *not* what catches skew during dev: phase PRs don't
touch `_version.py`, so both host and image report the same version through
an entire milestone of schema churn. The fingerprint is the blocking signal —
it changes whenever any `ExperimentConfig` field (or nested model) is added,
renamed, or restructured.

Inspect the labels on a local image:

```bash
docker image inspect llenergymeasure:transformers \
    --format '{{json .Config.Labels}}' | python3 -m json.tool
```

`Makefile` and `.github/workflows/docker-publish.yml` both call
`scripts/compute_expconf_fingerprint.py` so locally-built and CI-published
images carry identical fingerprints.

See [troubleshooting.md](troubleshooting.md#schema-skew-between-host-and-docker-image)
for the remediation flow when a mismatch is reported.

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

`llem` adds `--gpus all` automatically when launching engine containers. If you are running
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
- [Engine Configuration](engines.md) — configure vLLM, TensorRT-LLM, and switch between engines
- [Fast rebuilds and first-pull cost](installation.md#fast-rebuilds-and-first-pull-cost) — how the GHCR layer cache speeds up local Docker builds
