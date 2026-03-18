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

Determines whether each backend runs locally or in Docker:

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

## Docker image registry

```python
from llenergymeasure.infra.image_registry import get_default_image, parse_runner_value

image = get_default_image("pytorch")  # "ghcr.io/henrycgbaker/llenergymeasure/pytorch:v0.9.0"

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
- Cannot import from: `energy/`, `backends/`, `harness/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../study/runner.py` for StudyRunner which uses DockerRunner for Docker dispatch
- See `../api/_impl.py` for direct DockerRunner usage in single-experiment path
