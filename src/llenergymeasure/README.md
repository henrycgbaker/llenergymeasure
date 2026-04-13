# llenergymeasure Package

Main package for the LLM efficiency measurement framework.

## Package Structure

```
llenergymeasure/
├── __init__.py         # Public API (run_experiment, run_study)
├── _version.py         # Package version (zero internal imports)
├── api/                # Public Python API (_impl.py)
├── engines/           # Engine plugins (pytorch, vllm, tensorrt) + protocol
├── cli/                # Typer CLI (run, config)
├── config/             # Configuration system (SSOT models)
├── datasets/           # Built-in prompt datasets
├── device/             # GPU info, NVML, power/thermal, gpu_indices, env metadata
├── domain/             # Domain models (ExperimentResult, etc.) — pure Pydantic
├── energy/             # Energy samplers (NVML, Zeus, CodeCarbon)
├── entrypoints/        # Docker container entry point
├── harness/            # MeasurementHarness, preflight, environment, warmup
├── infra/              # Docker runner, image registry, runner resolution
├── results/            # Results persistence, aggregation, extended metrics
├── study/              # Study (sweep) runner, grid expansion, preflight
└── utils/              # Shared exceptions, constants, security
```

## Layer Architecture

```
Layer 8: cli/ | entrypoints/  - Typer CLI (llem run, llem config) + Docker container entry point
Layer 7: api/                 - Public Python API (run_experiment, run_study)
Layer 6: study/               - Study runner, grid expansion, manifest, study preflight
Layer 5: harness/ | results/  - MeasurementHarness, experiment preflight, env snapshot; results persistence
Layer 4: engines/ | energy/ | datasets/  - Engine plugins, energy samplers, prompt loading
Layer 3: infra/               - Docker runner, image registry, runner resolution
Layer 2: device/              - GPU info, NVML, power/thermal, gpu_indices, env metadata collection
Layer 1: config/ | domain/    - Config models, domain result models (pure Pydantic)
Layer 0: utils/               - Exceptions, constants, security
```

Layer rules are enforced in CI by `import-linter` (see `[tool.importlinter]` in `pyproject.toml`).

Upper layers may import lower layers but not vice versa.

## Key Files

### api/
Public Python API entry point:
- `run_experiment(config, **kwargs)` — single experiment
- `run_study(config)` — multi-experiment sweep
- `save_result(result, output_dir)` — re-exported from results.persistence for CLI use

### cli/
Modular Typer CLI with two commands:
- `llem run [config.yaml]` — run single experiment or multi-experiment study
- `llem config [-v]` — show environment, GPU, engine, and energy status
- Uses `_version.py` directly for version string (not the package root)

### entrypoints/ (Layer 8)
- `container.py` — Docker container entry point (was `infra/container_entrypoint.py`)
- Imports from `api/`, `engines/`, `harness/`, `device/` — all valid downward from the top layer

### utils/exceptions.py
Exception hierarchy rooted at `LLEMError`:
- `ConfigError` — config loading/validation
- `EngineError` — inference engine failures
- `PreFlightError` — pre-flight check failures
- `ExperimentError` — experiment execution errors
- `StudyError` — study orchestration errors
- `DockerError` — Docker container dispatch errors

### utils/security.py
Security utilities:
- `sanitize_experiment_id()` — sanitise IDs for filesystem
- `is_safe_path()` — prevent path traversal

## Submodules

| Module          | Description                                                      |
|-----------------|------------------------------------------------------------------|
| `api/`          | Public Python API (run_experiment, run_study)                    |
| `engines/`     | Inference engine plugins (pytorch, vllm, tensorrt)              |
| `cli/`          | Typer CLI commands (run, config)                                 |
| `config/`       | Configuration loading, SSOT models, introspection                |
| `datasets/`     | Built-in prompt datasets                                         |
| `device/`       | GPU info, power/thermal querying, gpu_indices, env metadata      |
| `domain/`       | Pydantic models for experiments and results (pure data, no I/O)  |
| `energy/`       | Energy samplers (NVML, Zeus, CodeCarbon)                         |
| `entrypoints/`  | Docker container entry point                                     |
| `harness/`      | MeasurementHarness, experiment preflight, environment snapshot   |
| `infra/`        | Docker runner, image registry, runner resolution                 |
| `results/`      | Results persistence, aggregation, extended efficiency metrics    |
| `study/`        | Study runner, grid expansion, manifest, study preflight          |
| `utils/`        | Shared exceptions, constants, security utilities                 |

## Layer-by-Layer Notes

### `cli/` and `entrypoints/` (Layer 8)
- `cli/` imports only from `api/`. Must not import harness, engines, energy, infra, study, or
  results directly.
- `cli/__init__.py` imports version from `_version.py` directly (not the package root), avoiding
  the heavy `__init__.py` import chain.
- `entrypoints/container.py` is the Docker-side entry point; it may import from any lower layer.

### `api/` (Layer 7)
- `_impl.py` — implementation of `run_experiment` / `run_study`
- `__init__.py` — re-exports `run_experiment`, `run_study`, and `save_result`

### `study/` (Layer 6)
- `runner.py` — orchestrates multi-experiment sweeps
- `preflight.py` — study-level pre-flight validation (multi-engine Docker requirements)
- `_progress.py` — progress display (belongs here, not in cli/)

### `harness/` and `results/` (Layer 5)
- `harness/__init__.py` — `MeasurementHarness`, `select_energy_sampler()`
- `harness/preflight.py` — experiment-level pre-flight checks (CUDA, engine, model)
- `harness/environment.py` — environment snapshot collection
- `harness/warmup.py` — thermal floor wait and warmup utilities
- `results/persistence.py` — `save_result()` / `load_result()` repository functions
- `results/extended_metrics.py` — efficiency metrics computation (tokens/joule, joules/token, etc.)
- Harness owns the NVML measurement window; engine compilation must never occur inside it

### `engines/` (Layer 4)
- `protocol.py` — `EnginePlugin` protocol
- `transformers.py`, `vllm.py`, `tensorrt.py` — engine implementations
- `_helpers.py` — shared warmup utilities

### `energy/` (Layer 4)
- `base.py` — `EnergySampler` base class
- `nvml.py`, `zeus.py`, `codecarbon.py` — energy sampler implementations

### `infra/` (Layer 3)
- `docker_runner.py` — Docker dispatch (DockerRunner)
- `runner_resolution.py` — local vs Docker selection
- `image_registry.py` — Docker image registry and version tagging
- `docker_preflight.py` — Docker-level pre-flight checks

### `device/` (Layer 2)
- `gpu_info.py` — `GPUInfo`, `nvml_context()`, `_resolve_gpu_indices()`
- `power_thermal.py` — `PowerThermalSampler`, `ThermalThrottleInfo`
- `environment.py` — hardware metadata collection via NVML
- Placed above `config/` and `domain/` because `power_thermal.py` returns `ThermalThrottleInfo`
  from `domain/metrics.py` (valid downward import from Layer 2 to Layer 1)

### `config/` and `domain/` (Layer 1)
- Zero imports from upper layers. Pure configuration and data models.
- `domain/` models are purely Pydantic: no collection logic, no Active Record methods.
  Use `results.persistence.save_result()` / `load_result()` for persistence.

### `utils/` (Layer 0)
- `exceptions.py`, `constants.py`, `security.py` — shared utilities
- No imports from any other llenergymeasure layer

## Version Access

`_version.py` at package root contains only `__version__`. Modules needing the version string
import from `llenergymeasure._version` (not the package root) to avoid triggering the heavy
`__init__.py` import chain (which loads `api._impl` and all of its dependencies).

```python
# Correct: zero-dependency version access
from llenergymeasure._version import __version__

# Also works (public API), but triggers the full import chain
import llenergymeasure
llenergymeasure.__version__
```

## Usage

```python
from llenergymeasure import run_experiment, run_study
from llenergymeasure.config import ExperimentConfig
from llenergymeasure.domain import ExperimentResult
```

## Related

- See `cli/README.md` for CLI architecture
- See `config/README.md` for configuration system
- See `CLAUDE.md` (git-excluded) for layered architecture and import rules
