# Package Structure Peer Review

Research into how peer ML/inference measurement tools structure their Python packages.
Informs the planned restructure phase (post-cleanup).

## 1. Peer Project Structures

### lm-eval-harness (EleutherAI) - Most Relevant Peer

```
lm_eval/
├── _cli/                 # CLI-specific command handling
├── api/                  # Core API for programmatic use
├── config/              # Configuration management
├── models/              # Model backend integrations (HF, vLLM, SGLang, OpenVINO, etc.)
├── tasks/               # Task definitions and benchmark implementations
├── caching/             # Caching mechanisms
├── decontamination/     # Data decontamination utilities
├── filters/             # Filtering utilities
├── loggers/             # Logging infrastructure
├── prompts/             # Prompt handling
├── evaluator.py         # Core evaluation logic (harness orchestrator)
├── evaluator_utils.py   # Evaluation utilities
├── result_schema.py     # Result data structure definitions
└── defaults.py          # Default configurations
```

**Key pattern:** Clear separation with `_cli/` module containing CLI-specific code, while `api/` and core modules handle library functionality. The `evaluator.py` is the harness orchestrator.

### vLLM (vllm-project) - Inference Architecture Reference

```
vllm/
├── engine/              # Core inference engine (scheduler, KV cache, coordination)
├── model_executor/      # Model execution layer (forward passes)
├── entrypoints/         # API entry points and interfaces
├── distributed/         # Multi-GPU and distributed inference
├── device_allocator/    # GPU/hardware memory management
├── inputs/              # Input processing and validation
├── config/              # Configuration management
├── compilation/         # Model compilation and optimisation
├── tokenizers/          # Tokenisation implementations
├── transformers_utils/  # HuggingFace Transformers integration
├── ray/                 # Ray distributed computing integration
├── multimodal/          # Multi-modal input handling
└── [12+ more specialised modules]
```

**Key patterns:**
- Layered architecture: config -> entrypoints -> engine -> model_executor
- Clear separation: core logic (engine) vs. hardware abstraction (device_allocator, distributed)
- Extensibility through integrations (ray/, transformers_utils/, third_party/)

### zeus (ML-Energy) - Energy Measurement Library

```
zeus/
├── monitor/     # Energy and power measurement (programmatic + CLI)
├── optimizer/   # Time and energy optimisation algorithms
├── device/      # Hardware abstraction (CPU/GPU)
├── utils/       # Utility functions
├── _legacy/     # Legacy code for reproducibility
├── metric.py    # Prometheus metric export
├── callback.py  # Training callback base class
└── show_env.py  # Environment detection verification
```

**Key pattern:** Functional domain separation (measure, optimise, abstract hardware). Small, focused modules. Measurement and CLI integrated in same module.

### MLPerf Inference - Benchmark Harness

```
MLPerf/
├── vision/           # Computer vision benchmarks
├── language/         # Language model benchmarks
├── recommendation/   # Recommendation system benchmarks
├── loadgen/          # Load generation tool (core harness)
├── tools/            # Utility scripts
├── compliance/       # Compliance verification
└── calibration/      # Model calibration tools
```

**Key pattern:** Domain-based organisation (what's being measured), not architectural layering. Loadgen is the orchestration core.

## 2. Four-Layer Model

Consensus architectural layering from peer analysis:

```
PRESENTATION LAYER
├── CLI (click/typer/argparse)
├── HTTP API (FastAPI)
└── Web UI (future)
    ↓
APPLICATION / ORCHESTRATION LAYER
├── Runner/Coordinator
├── Study/Experiment orchestration
└── Configuration resolution
    ↓
DOMAIN / CORE LAYER
├── Config models (Pydantic)
├── Core logic (measurement, execution)
├── Result schemas
└── Business rules
    ↓
INFRASTRUCTURE LAYER
├── Backend plugins (Protocol-based)
├── Device abstraction
├── I/O operations
└── External integrations
```

Dependencies flow downward only. CLI never imports from infrastructure directly.

## 3. CLI vs Library Separation

**Pattern 1: Separate CLI Module** (lm-eval-harness style) - **Recommended**
- Core library API in main modules (`api/`, `evaluator.py`)
- CLI as thin wrapper in dedicated `_cli/` directory
- Library fully usable without CLI dependencies
- Entry point defined in `pyproject.toml` points to CLI module
- Matches our existing `cli/` layout

**Pattern 2: Integrated CLI** (zeus style)
- Measurement/monitoring capabilities include CLI alongside programmatic API
- Both share the same module
- Best when CLI and library use identical logic

**Pattern 3: API-First Architecture** (FastAPI/vLLM style)
- Design library API first
- CLI is a presentation layer consuming the API
- Multiple interfaces (HTTP API, CLI, direct imports) all call same orchestration

## 4. Protocol-Based Plugin Architecture

Python 3.8+ Protocols are the preferred approach across peers:

```python
# Define plugin contract (no ABC inheritance needed)
class BackendProtocol(Protocol):
    def execute(self, request: Request) -> Result: ...
    def validate_config(self, config: dict) -> bool: ...

# Plugin implementation - no explicit inheritance required
class MyBackend:
    def execute(self, request): ...
    def validate_config(self, config): ...
```

**Benefits over ABC:**
- Structural subtyping: implementations don't need to explicitly inherit
- Third-party code can implement protocol without knowing about it
- Cleaner for distributed/optional plugins
- Better for dynamic discovery

## 5. Entry-Point-Based Plugin Discovery

```toml
[project.entry-points."llenergymeasure.backends"]
pytorch = "llenergymeasure.backends.pytorch:PyTorchBackend"
vllm = "llenergymeasure.backends.vllm:VLLMBackend"
tensorrt = "llenergymeasure.backends.tensorrt:TensorRTBackend"
```

Runtime discovery:

```python
from importlib.metadata import entry_points

backends = entry_points(group="llenergymeasure.backends")
for ep in backends:
    backend_class = ep.load()
```

**Benefits:**
- Third-party packages can add backends without modifying source
- Core doesn't depend on specific backends
- Works with namespace packages
- Standard Python packaging mechanism

## 6. Config-Driven Patterns

From Hydra and experiment runner frameworks:

```python
# 1. Parse YAML at edge (no Pydantic yet)
raw_config = yaml.safe_load(file)

# 2. Resolve sweeps/templating (create multiple experiment variations)
expanded_configs = resolve_sweeps(raw_config)

# 3. Validate with Pydantic models
study_config = StudyConfig.model_validate(expanded_configs)

# 4. Pass to core orchestrator
result = runner.run(study_config)
```

**Key insight:** Sweep resolution happens before Pydantic validation, not after. We already follow this pattern.

## 7. Experiment/Study Execution Patterns

From Experiment Runner (ER), PRISM, idmtools:

**Core components:**
1. **ExperimentConfig** - Single measurement point, pure data structure
2. **StudyConfig** - Collection of experiments + execution parameters
3. **Orchestrator/Runner** - Manages execution lifecycle
4. **Harness** - Measurement infrastructure (handles backend invocation, data collection, cleanup)
5. **Result aggregation** - Collects outputs from all experiments

**Orchestrator responsibilities:**
- Resolve studies into experiments
- Schedule execution
- Handle failures/retries
- Collect results
- Clean up resources

## 8. Recommended Target Structure

Based on all peer analysis, the target structure for llenergymeasure:

```
src/llenergymeasure/
├── __init__.py           # Public API (run_experiment, run_study)
├── __main__.py           # CLI entry point
│
├── api/                  # Library API (public interface)
│   ├── __init__.py       # Exports: run_experiment, run_study
│   └── _impl.py          # Implementation (from current _api.py)
│
├── cli/                  # CLI layer (thin wrapper around API)
│   ├── __init__.py
│   ├── run.py            # `llem run` command
│   └── config_cmd.py     # `llem config` command
│
├── config/               # Configuration models (Pydantic)
│   ├── models.py         # ExperimentConfig, StudyConfig
│   ├── backend_configs.py
│   ├── loader.py
│   └── ...
│
├── backends/             # Backend plugin implementations (promoted from core/)
│   ├── __init__.py       # get_backend() factory
│   ├── protocol.py       # BackendPlugin protocol
│   ├── pytorch.py
│   ├── vllm.py
│   └── tensorrt.py
│
├── core/                 # Domain/measurement logic
│   ├── harness.py        # MeasurementHarness
│   ├── warmup.py
│   ├── flops.py
│   ├── timeseries.py
│   └── ...
│
├── energy/               # Energy measurement (promoted from core/energy_backends/)
│   ├── __init__.py       # select_energy_backend()
│   ├── base.py           # EnergyBackend protocol
│   ├── nvml.py
│   ├── zeus.py
│   └── codecarbon.py
│
├── domain/               # Domain models (Pydantic)
│   ├── experiment.py
│   ├── metrics.py
│   └── ...
│
├── datasets/             # Built-in prompt datasets
│   └── ...
│
├── infra/                # Docker runner, image registry
│   └── ...
│
├── study/                # Study (sweep) runner
│   └── ...
│
├── _internal/            # Private utilities
│   └── ...
│
└── results/              # Results persistence, aggregation
    └── ...
```

**Rationale table:**

| Change | Rationale | Peer precedent |
|--------|-----------|----------------|
| `core/backends/` -> `backends/` | Backends are top-level concerns, not sub-domain of core | lm-eval `models/`, vLLM `engine/` |
| `_api.py` -> `api/` | API surface deserves its own package for public exports | lm-eval `api/` |
| `core/energy_backends/` -> `energy/` | Energy measurement is its own domain, not a core sub-module | zeus `monitor/` |
| Add `_internal/` | Private utilities shouldn't pollute public packages | Common Python pattern |
| Keep `cli/` | Already correct - matches Pattern 1 | lm-eval `_cli/` |
| Keep `config/` | Already correct location | All peers |
| Keep `domain/` | Already correct location | DDD pattern |

## 9. Entry Point Configuration

```toml
[project.scripts]
llem = "llenergymeasure.cli:app"

[project.entry-points."llenergymeasure.backends"]
pytorch = "llenergymeasure.backends.pytorch:PyTorchBackend"
vllm = "llenergymeasure.backends.vllm:VLLMBackend"
tensorrt = "llenergymeasure.backends.tensorrt:TensorRTBackend"

[project.entry-points."llenergymeasure.energy"]
nvml = "llenergymeasure.energy.nvml:NVMLBackend"
zeus = "llenergymeasure.energy.zeus:ZeusBackend"
codecarbon = "llenergymeasure.energy.codecarbon:CodeCarbonBackend"
```

## 10. Public API Design

```python
# api/__init__.py
from llenergymeasure.api._impl import run_experiment, run_study

__all__ = ["run_experiment", "run_study"]
```

Re-exported from package root `__init__.py` for convenience:

```python
# __init__.py
from llenergymeasure.api import run_experiment, run_study
```

This matches lm-eval's pattern where `lm_eval.evaluate()` delegates to `lm_eval.evaluator.evaluate()`.

## Sources

- lm-eval-harness (EleutherAI/lm-evaluation-harness)
- vLLM (vllm-project/vllm)
- zeus (ML-Energy/zeus)
- MLPerf Inference (mlcommons/inference)
- Python Packaging User Guide - Creating and Discovering Plugins
- PEP 544 - Protocols: Structural Subtyping
