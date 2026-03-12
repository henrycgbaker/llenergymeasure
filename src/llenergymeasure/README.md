# llenergymeasure Package

Main package for the LLM efficiency measurement framework.

## Package Structure

```
llenergymeasure/
├── __init__.py         # Public API (run_experiment, run_study)
├── _api.py             # API implementation
├── constants.py        # Global constants
├── exceptions.py       # Exception hierarchy
├── protocols.py        # Protocol definitions (interfaces for DI)
├── security.py         # Path sanitisation, validation
├── cli/                # Typer CLI (run, config)
├── config/             # Configuration system (SSOT models)
├── core/               # Inference engine, metrics, energy backends
├── datasets/           # Built-in prompt datasets
├── domain/             # Domain models (ExperimentResult, etc.)
├── infra/              # Docker runner, image registry, entrypoint
├── orchestration/      # Experiment preflight checks
├── results/            # Results persistence, aggregation
├── state/              # Experiment state machine
└── study/              # Study (sweep) runner, grid expansion
```

## Key Files

### cli/
Modular Typer CLI with two commands:
- `llem run [config.yaml]` — run single experiment or multi-experiment study
- `llem config [-v]` — show environment, GPU, backend, and energy status

### protocols.py
Protocol definitions for dependency injection:
- `ModelLoader` — load model and tokeniser
- `InferenceEngine` — run inference
- `EnergyBackend` — energy tracking (Zeus, NVML, CodeCarbon)
- `MetricsCollector` — collect metrics
- `ResultsRepository` — persist results

### exceptions.py
Exception hierarchy rooted at `LLEMError`:
- `ConfigError` — config loading/validation
- `BackendError` — inference backend failures
- `PreFlightError` — pre-flight check failures
- `ExperimentError` — experiment execution errors
- `StudyError` — study orchestration errors
- `DockerError` — Docker container dispatch errors

### security.py
Security utilities:
- `sanitize_experiment_id()` — sanitise IDs for filesystem
- `is_safe_path()` — prevent path traversal

## Submodules

| Module           | Description                                              |
|------------------|----------------------------------------------------------|
| `cli/`           | Typer CLI commands (run, config)                         |
| `config/`        | Configuration loading, SSOT models, introspection        |
| `core/`          | Inference backends, model loading, FLOPs, energy, GPU    |
| `datasets/`      | Built-in prompt datasets                                 |
| `domain/`        | Pydantic models for experiments and results              |
| `infra/`         | Docker runner, image registry, container entrypoint      |
| `orchestration/` | Experiment preflight checks                              |
| `results/`       | FileSystemRepository, aggregation logic                  |
| `state/`         | Experiment state machine                                 |
| `study/`         | Study runner, grid expansion, manifest, preflight        |

## Usage

```python
from llenergymeasure import run_experiment, run_study
from llenergymeasure.config import ExperimentConfig
from llenergymeasure.domain import ExperimentResult
```

## Related

- See `cli/CLAUDE.md` for CLI architecture
- See `config/README.md` for configuration system
- See `core/README.md` for inference engine details
