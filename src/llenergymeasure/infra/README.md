# infra/ - Docker Infrastructure

Docker runner, container entrypoint, pre-flight checks, and runner resolution. Layer 1 in the six-layer architecture.

## Purpose

Manages the full container lifecycle for Docker-isolated experiment execution. Also owns environment metadata collection and runner mode resolution (local vs Docker).

## Modules

| Module | Description |
|--------|-------------|
| `docker_runner.py` | `DockerRunner` — dispatches a single experiment to an ephemeral container |
| `container_entrypoint.py` | Container-side entry point (invoked inside Docker) |
| `runner_resolution.py` | `resolve_runner()`, `resolve_study_runners()` — local vs Docker selection |
| `docker_preflight.py` | `run_docker_preflight()` — GPU visibility, CUDA/driver compat checks |
| `docker_errors.py` | `translate_docker_error()`, typed Docker error classes |
| `image_registry.py` | `get_default_image()`, `parse_runner_value()` — Docker image registry |
| `environment.py` | `collect_environment_metadata()` — hardware/software snapshot |
| `__init__.py` | Package marker |

## DockerRunner

Dispatches one experiment to an ephemeral `docker run --rm --gpus all` container:

```python
from llenergymeasure.infra.docker_runner import DockerRunner

runner = DockerRunner(image="ghcr.io/.../llenergymeasure/pytorch:v0.9.0", timeout=600)
result = runner.run(config)  # ExperimentResult or error dict
```

Container lifecycle:
1. Create temp exchange directory (`tempfile.mkdtemp(prefix='llem-')`)
2. Serialise `ExperimentConfig` to JSON in the exchange dir
3. Start `docker run --rm --gpus all --mount ...` with exchange dir as `/run/llem`
4. Block until container exits
5. Read `result.json` or `error.json` written by the container entrypoint
6. Clean up exchange dir on success; preserve on failure for debugging

## Container entrypoint

Inside the container, `container_entrypoint.py` is the entry point:

```
LLEM_CONFIG_PATH=/run/llem/{hash}_config.json python -m llenergymeasure.infra.container_entrypoint
```

Reads config JSON, runs via library API (not CLI re-entry), writes result JSON back to the shared volume.

## Runner resolution

Determines whether each engine runs locally or in Docker:

```python
from llenergymeasure.infra.runner_resolution import resolve_study_runners, is_docker_available

# Precedence: env var > YAML runners: > user config > auto-detection > default
runner_specs = resolve_study_runners(["pytorch", "vllm"], yaml_runners=..., user_config=...)
```

Precedence chain (highest wins):
1. `LLEM_RUNNER` environment variable
2. `runners:` section in study YAML
3. User config (`~/.config/llem/config.yaml`)
4. Auto-detection (Docker + NVIDIA Container Toolkit available? → docker; else → local)

## Docker image resolution

### Image sources

Two image sources exist for each engine:

| Source | Tag pattern | Built by | Use case |
|--------|------------|----------|----------|
| **Local** | `llenergymeasure:{engine}` | `make docker-build-{engine}` | Dev iteration - always reflects current source |
| **Registry** | `ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}` | CI on release tags | Production, CI, pip-install users |

### Resolution precedence (`resolve_image()`)

The full image resolution chain (highest wins):

1. **Env var** `LLEM_IMAGE_{ENGINE}` (e.g. `LLEM_IMAGE_VLLM=my/custom:tag`)
2. **Study YAML** `images:` section
3. **Runner spec** shorthand (`docker:my/custom:tag` in `runners:`)
4. **User config** `images:` section
5. **Smart default** via `get_default_image()`: local build → registry fallback

Each level returns an `(image, image_source)` tuple. The `image_source` string
(`"env"`, `"yaml"`, `"runner_override"`, `"user_config"`, `"local_build"`,
`"registry_cached"`, `"registry"`) tracks provenance for display and diagnostics.

### Smart default behaviour

`get_default_image(engine)` checks for a local image first, then falls back to the registry tag:

```python
from llenergymeasure.infra.image_registry import get_default_image

image = get_default_image("vllm")
# → "llenergymeasure:vllm"  (if local image exists)
# → "ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0"  (otherwise)
```

The distinction between `"registry"` and `"registry_cached"` source: when a registry image
is already present in the local Docker cache (from a prior pull), the source is
`"registry_cached"`. When it will need pulling, the source is `"registry"`.

### Building local images

```bash
make docker-build-all            # all 3 engines
make docker-build-transformers   # just transformers
make docker-build-vllm           # just vllm
make docker-build-tensorrt       # just tensorrt
```

These pull cached layers from GHCR on first build (Transformers: <5 min warm vs ~30 min cold).
See `docs/installation.md#fast-rebuilds-and-first-pull-cost` for the full mechanism.

### Study-level image preparation

For multi-experiment studies, `StudyRunner._prepare_images()` checks and pulls all required
Docker images **once** before the first experiment, rather than per-experiment. This avoids
redundant `docker pull` calls when multiple experiments share the same engine. The CLI shows
this as a "Preparing Docker images" section with per-image status and metadata.

### YAML overrides

```yaml
runners:
  transformers: local                                         # host, no Docker
  vllm: docker                                           # default (local → registry)
  tensorrt: "docker:ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0"  # force registry
```

```python
runner_type, image_override = parse_runner_value("docker:my/custom-image:v1")
# runner_type = "docker", image_override = "my/custom-image:v1"
```

## Environment metadata

```python
from llenergymeasure.infra.environment import collect_environment_metadata

metadata = collect_environment_metadata(device_index=0)
# metadata.gpu — GPUEnvironment (name, memory, compute capability)
# metadata.cuda — CUDAEnvironment (version, driver)
# metadata.cpu — CPUEnvironment
# metadata.container — ContainerEnvironment (is_docker, image)
```

Gracefully degrades when NVML is unavailable.

## Layer constraints

- Layer 1 — may import from layer 0 only
- Can import from: `config/`, `domain/`, `device/`, `utils/`
- Cannot import from: `energy/`, `engines/`, `harness/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../study/runner.py` for StudyRunner which uses DockerRunner for Docker dispatch
- See `../api/_impl.py` for direct DockerRunner usage in single-experiment path
