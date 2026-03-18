# llenergymeasure Package

Main package for the LLM efficiency measurement framework.

## Package Structure

```
llenergymeasure/
├── __init__.py         # Public API (run_experiment, run_study)
├── api/                # API implementation (_impl.py, _gpu.py, preflight.py)
├── backends/           # Backend plugins (pytorch, vllm, tensorrt) + protocol
├── cli/                # Typer CLI (run, config)
├── config/             # Configuration system (SSOT models)
├── datasets/           # Built-in prompt datasets
├── device/             # GPU info and power/thermal querying
├── domain/             # Domain models (ExperimentResult, etc.)
├── energy/             # Energy samplers (NVML, Zeus, CodeCarbon)
├── harness/            # MeasurementHarness, warmup, energy selection
├── infra/              # Docker runner, image registry, entrypoint
├── results/            # Results persistence, aggregation
├── study/              # Study (sweep) runner, grid expansion
└── utils/              # Shared exceptions, constants, security
```

## Key Files

### api/
Public Python API entry point:
- `run_experiment(config, **kwargs)` — single experiment
- `run_study(config)` — multi-experiment sweep

### cli/
Modular Typer CLI with two commands:
- `llem run [config.yaml]` — run single experiment or multi-experiment study
- `llem config [-v]` — show environment, GPU, backend, and energy status

### utils/exceptions.py
Exception hierarchy rooted at `LLEMError`:
- `ConfigError` — config loading/validation
- `BackendError` — inference backend failures
- `PreFlightError` — pre-flight check failures
- `ExperimentError` — experiment execution errors
- `StudyError` — study orchestration errors
- `DockerError` — Docker container dispatch errors

### utils/security.py
Security utilities:
- `sanitize_experiment_id()` — sanitise IDs for filesystem
- `is_safe_path()` — prevent path traversal

## Submodules

| Module        | Description                                              |
|---------------|----------------------------------------------------------|
| `api/`        | Public Python API (run_experiment, run_study)            |
| `backends/`   | Inference backend plugins (pytorch, vllm, tensorrt)      |
| `cli/`        | Typer CLI commands (run, config)                         |
| `config/`     | Configuration loading, SSOT models, introspection        |
| `datasets/`   | Built-in prompt datasets                                 |
| `device/`     | GPU info, power/thermal querying                         |
| `domain/`     | Pydantic models for experiments and results              |
| `energy/`     | Energy samplers (NVML, Zeus, CodeCarbon)                 |
| `harness/`    | MeasurementHarness, warmup, energy selection             |
| `infra/`      | Docker runner, image registry, container entrypoint      |
| `results/`    | FileSystemRepository, aggregation logic                  |
| `study/`      | Study runner, grid expansion, manifest, preflight        |
| `utils/`      | Shared exceptions, constants, security utilities         |

## Usage

```python
from llenergymeasure import run_experiment, run_study
from llenergymeasure.config import ExperimentConfig
from llenergymeasure.domain import ExperimentResult
```

## Related

- See `cli/CLAUDE.md` for CLI architecture
- See `config/README.md` for configuration system
- See `CLAUDE.md` for layered architecture and import rules
