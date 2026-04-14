# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Breaking Changes

- **`engine: pytorch` → `engine: transformers` rename (YAML / API / engine identifier).** The `pytorch` engine identifier has been renamed to `transformers` throughout. The engine runs HuggingFace Transformers `.generate()` — PyTorch is the tensor substrate, not the engine. This aligns the identifier with `pip install transformers` and the library that owns the inference API.

  **Migrate YAML configs** (one-liner):
  ```bash
  sed -i 's/engine: pytorch/engine: transformers/g; s/^pytorch:/transformers:/g' your-study.yaml
  ```

  **Affected identifiers:**
  - YAML engine value: `engine: pytorch` → `engine: transformers`
  - YAML section key: `pytorch:` → `transformers:`
  - Python class: `PyTorchConfig` → `TransformersConfig`
  - Python constant: `ENGINE_PYTORCH = "pytorch"` → `ENGINE_TRANSFORMERS = "transformers"`
  - PyPI extra: `pip install llenergymeasure[pytorch]` → `llenergymeasure[transformers]`
  - Env vars: `LLEM_RUNNER_PYTORCH` → `LLEM_RUNNER_TRANSFORMERS`, `LLEM_IMAGE_PYTORCH` → `LLEM_IMAGE_TRANSFORMERS`
  - Docker image tags: `llenergymeasure:pytorch` → `llenergymeasure:transformers`, `Dockerfile.pytorch` → `Dockerfile.transformers`

  **Preserved** (PyTorch the library — unchanged):
  - `import torch` and all `torch.*` API references
  - `torch_dtype` field
  - `FROM pytorch/pytorch:...` Docker base image tag
  - `PYTORCH_VERSION` / `PYTORCH_DEVEL_VERSION` build args
  - `torch_compile_backend` field (PyTorch's `torch.compile(backend=...)` parameter)

- **`backend:` → `engine:` rename (YAML / CLI / API).** The experiment configuration field, CLI flag, and all public symbols now use "engine" instead of "backend" throughout. This aligns terminology with how vLLM, TRT-LLM, and HuggingFace themselves use "engine" (EngineArgs, LLM engine, etc.) and removes ambiguity with tensor/compute/attention backends.

  **Migrate YAML configs** (one-liner):
  ```bash
  sed -i 's/^\(\s*\)backend:/\1engine:/g' your-study.yaml
  ```

  **Affected identifiers:**
  - YAML field: `backend: pytorch/vllm/tensorrt` → `engine: transformers/vllm/tensorrt`
  - CLI flag: `llem run --backend` → `llem run --engine` (short: `-b` → `-e`)
  - Result JSON fields: `"backend"` / `"backend_version"` → `"engine"` / `"engine_version"`
  - Python symbols: `BackendPlugin` → `EnginePlugin`, `BackendError` → `EngineError`, `BACKEND_*` constants → `ENGINE_*`, `get_backend()` → `get_engine()`, `detect_default_backend()` → `detect_default_engine()`

  **Preserved** (these are kernel/compute backends, not inference engines — unchanged):
  - `vllm.attention.backend` — vLLM's attention kernel selector (Flash/FlashInfer/SDPA)
  - `transformers.torch_compile_backend` — PyTorch `torch.compile(backend=...)` parameter
  - `TensorRTConfig.backend` — TRT-LLM's internal `LLM(backend=...)` parameter
  - Energy measurement backends (Zeus, NVML, CodeCarbon)

### Fixed

- **`Dockerfile.transformers` stale references.** The Dockerfile was renamed in #261 but still installed the now-nonexistent `[pytorch]` extra and carried header comments referencing the old `Dockerfile.pytorch` file and `llenergymeasure-pytorch` tag. Updated to install `.[transformers]` and corrected all header comments. The `pytorch/pytorch:*` base image tags are preserved - PyTorch the library is unchanged, only the engine identifier was renamed.

- **`tensorrt.tp_size` → `tensorrt.tensor_parallel_size`.** Aligns the TensorRT-LLM Pydantic field name with `TrtLlmArgs.tensor_parallel_size`. `transformers.tp_size` is **preserved** — it follows the `accelerate` convention and HF has no single native `tensor_parallel_size` equivalent.

  **Migrate YAML configs** (scope-limited — manual edit is safer as `tp_size` also appears legitimately under `transformers:`; in your `tensorrt:` section rename `tp_size: N` → `tensor_parallel_size: N`).

  **Affected identifiers:**
  - YAML field: `tensorrt.tp_size` → `tensorrt.tensor_parallel_size`
  - Python: `TensorRTConfig(tp_size=N)` → `TensorRTConfig(tensor_parallel_size=N)`
  - Sweep key: `"tensorrt.tp_size"` → `"tensorrt.tensor_parallel_size"`

  **Preserved:**
  - `transformers.tp_size` — HF accelerate convention, unchanged
  - `vllm.engine.tensor_parallel_size` — already native name, unchanged

### Added

- **Host/container schema fingerprint verification.** Docker images are now stamped at build time with a `llem.expconf.schema.fingerprint` OCI label (SHA-256 of `ExperimentConfig.model_json_schema()`) plus `org.opencontainers.image.version`. `StudyRunner._prepare_images` compares the label to the host fingerprint before any experiment runs and aborts with an actionable rebuild hint on mismatch. The check is bypassable via `LLEM_SKIP_IMAGE_CHECK=1`.
- **`llem doctor` CLI command.** Reports per-backend image status (OK / MISMATCH / UNVERIFIED / UNREACHABLE) and exits non-zero on mismatch for CI-friendly gating.
- **Inline schema status in the image-prep progress line** (`schema: ok` / `schema: mismatch` / `schema: unverified` / `schema: bypassed`), rendered via the existing metadata display with no changes to the progress protocol.
- **Engine parameter discovery (`scripts/discover_engine_schemas.py`).** Introspects installed engine packages inside their Docker images and emits JSON schemas describing every configurable parameter (types, defaults, descriptions where available, discovery limitations). Supports `vllm`, `tensorrt`, and `transformers`; `--all` discovers every engine found in the current image.
- **Vendored engine schemas at `src/llenergymeasure/config/discovered_schemas/{vllm,tensorrt,transformers}.json`.** These are the canonical SSOT for "what CAN I configure per engine", shipped inside the wheel. Regenerate with `make discover-schema ENGINE=<engine>` (writes to the vendored path and prints `git diff`; committing is the review gate).
- **`make discover-schema` / `make discover-schemas-all` targets.** Rebuild vendored engine schemas via `./scripts/update_engine_schema.sh`.
- **`SchemaLoader` class (`llenergymeasure.config.SchemaLoader`).** Reads vendored engine schemas via `importlib.resources` with per-instance caching and major-version envelope validation. Raises `UnsupportedSchemaVersionError` on envelope breaking changes. Exports `DiscoveredSchema`, `DiscoveryLimitation`, and `UnsupportedSchemaVersionError` from `llenergymeasure.config`.

### Changed

- **Per-experiment timeout is now configurable** via `study_execution.experiment_timeout_seconds` (default 600s). Replaces the previous `max(n_prompts*2, 600)` heuristic. Both the local subprocess path and the Docker container path honour the same field, and Docker-path timeouts are normalised to `TimeoutError` so the circuit breaker counts them consistently across both paths.

### Fixed

- **`ImportError: cuKernelGetName` when importing `tensorrt_llm` in our image.** `docker/Dockerfile.tensorrt` prepended `/usr/local/tensorrt/lib` to `LD_LIBRARY_PATH` but left the NGC-inherited ordering intact, placing `/usr/local/cuda/compat/lib.real` (the image-bundled compat library, CUDA 12.2) ahead of `/usr/local/cuda/compat/lib` (where nvidia-container-toolkit bind-mounts the host driver at `--gpus` time). `libtensorrt_llm.so` therefore resolved `libcuda.so.1` against the bundled library and failed to find `cuKernelGetName`, a symbol added in CUDA 12.4. Fix: prepend `/usr/local/cuda/compat/lib` so the host-driver mount takes precedence.

### Removed

- Internal helper `llenergymeasure.study.runner._calculate_timeout` (replaced by direct config reads; also removes a layer-boundary import from `api/_impl.py`).

## [v2.0.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v2.0.0) (2026-01-14)

Refactored CLI-based tool with clean architecture, comprehensive documentation, and improved configuration UX.

### Added

- **Architectural Refactor**
  - Dependency injection via protocol-based components (`EnergyBackend`, `ModelLoader`, `InferenceEngine`)
  - `ExperimentOrchestrator` manages lifecycle with injected dependencies
  - Late aggregation pattern: raw per-process results saved separately, aggregated on-demand
  - Pydantic models throughout for validated configs and results
  - Modular package structure: `config/`, `core/`, `domain/`, `orchestration/`, `results/`, `state/`

- **Documentation Overhaul**
  - New `docs/` directory with user-facing guides: `quickstart.md`, `cli.md`, `deployment.md`
  - Comprehensive CLI reference with all commands, flags, and examples
  - Implemented testing parameters table in README for quick reference
  - Streamlined README focusing on essentials; detailed docs moved to dedicated files

- **Enhanced Configuration UX**
  - Intuitive YAML field names: `gpus`, `batching`, `decoder`, `quantization` (legacy names still supported)
  - `config show` displays resolved config with colour-coded sections
  - Grid config validation with `--validate` and `--strict` flags
  - Config param wiring tests ensuring all YAML options reach their targets

- **Decoder Sampling Presets**
  - `preset` field: `deterministic`, `standard`, `creative`, `factual`
  - Presets expand to appropriate temperature/top_p/top_k combinations
  - Additional decoder params: `min_p`, `no_repeat_ngram_size`

- **Multi-Cycle Experiments** for statistical robustness
  - `--cycles N` flag runs the same experiment N times (1–10)
  - Statistical aggregation: mean, standard deviation, 95% confidence intervals
  - Coefficient of variation (CV) for measurement stability assessment
  - Uses t-distribution for small sample confidence intervals

- **Scheduled Experiments** (daemon mode) for temporal variation studies
  - Interval-based: `--interval 6h`, time-of-day: `--at 09:00`
  - Day filtering: `--days mon,wed,fri` or `--days weekdays`
  - Graceful shutdown on SIGINT/SIGTERM with progress tracking

- **MLPerf-Style Traffic Simulation**
  - `TrafficSimulation` config with Poisson and constant arrival modes
  - `target_qps` parameter for queries-per-second arrival rate
  - Poisson mode uses exponential inter-arrival times (statistically realistic)

- **Industry-Standard Batching Strategies** (MLPerf/vLLM terminology)
  - `strategy` field: `static`, `dynamic`, `sorted_static`, `sorted_dynamic`
  - `max_tokens_per_batch` for dynamic token-aware batching
  - Length sorting reduces padding waste

- **MIG GPU Support**
  - Topology detection and UUID handling for NVIDIA MIG instances
  - Correct energy attribution per MIG partition

- **Proper Multi-GPU Parallelism** (replaces v1.x naive `device_map="auto"` approach)
  - v1.x used `accelerate launch` with `CUDA_VISIBLE_DEVICES` for multi-GPU, which auto-distributed layers but without coordinated parallel execution
  - v2.0 adds explicit parallelism strategies with proper distributed execution

- **Tensor Parallelism (TP)** for large model inference
  - Native HuggingFace tensor parallelism via `tp_plan="auto"`
  - Splits layers horizontally so GPUs compute in parallel
  - Supported models: Llama, Mistral, Mixtral, Qwen, Phi, Gemma, Falcon, MPT, BLOOM, OPT
  - Uses `torchrun` launcher for distributed initialisation

- **Pipeline Parallelism (PP)** for multi-GPU inference
  - Splits model vertically into sequential stages across GPUs
  - Each GPU holds a subset of layers (e.g., layers 0-15 on GPU 0, 16-31 on GPU 1)
  - Useful when model doesn't fit on single GPU but TP isn't supported

- **Parallelism Configuration** via `sharding` config
  - `strategy`: `none`, `tensor_parallel`, `pipeline_parallel`
  - `num_shards`: Number of GPUs for parallelism
  - `tp_plan`: HuggingFace native tensor parallel plan
  - Validation for GPU count, model compatibility, and configuration conflicts

- **E2E Experiment Workflow**
  - Auto-aggregation on experiment completion
  - Experiment resumption for interrupted runs

### Changed

- YAML config uses shorter, intuitive field names (backwards-compatible aliases preserved)
- Traffic simulation uses proper Poisson arrival process instead of simple delays
- Batching config uses explicit `strategy` field instead of boolean `dynamic_batching`

### Dropped (From v1.x Plans)

- **FSDP Support**: Confirmed as training-only; `device_map="auto"` is correct for inference
- **Scenario Metadata**: Covered by existing `extra_metadata` field

### References

- [MLPerf Inference](https://docs.mlcommons.org/inference/) — Traffic simulation patterns
- [vLLM](https://blog.vllm.ai/) — Batching strategies
- [TokenPowerBench](https://arxiv.org/html/2512.03024v1) — Statistical robustness methodology

---

## [v1.16.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v1.16.0) (2025-01-07)

Production-ready containerisation with full GPU support and streamlined developer experience.

### Added
- **Multi-stage Dockerfile** with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image
  - Builder stage for dependency compilation
  - Runtime stage for production deployment (~3GB image)
  - Dev stage for local development with editable installs
- **Docker Compose profiles** separating production and development workflows
  - `lem-app`: Production service with baked-in package
  - `lem-dev`: Development service with source mounting
- **VS Code devcontainer** configuration for seamless IDE integration
  - GPU passthrough with `--gpus all`
  - Privileged mode for NVML energy metrics
  - Pre-configured Python extensions (Pylance, Ruff)
- **Makefile targets** for common Docker operations (`make docker-build`, `make experiment`, `make datasets`)

### Improved
- CI workflow reliability with concurrency groups preventing parallel releases
- Test runner now validates both `src/` and `tests/` directories
- Dev container runs as root, eliminating permission complexity with virtual environments
- Documentation expanded with "Running the Tool" section covering all four execution modes

### Fixed
- Docker CUDA 12.4 base image now matches host driver requirements
- Volume permission errors resolved by running dev containers as root
- Deprecated `torch_dtype` parameter replaced with `dtype` in model loading
- Removed obsolete `TRANSFORMERS_CACHE` environment variable (superseded by `HF_HOME`)
- CodeCarbon pandas `FutureWarning` suppressed via targeted filter
- `nvidia-smi` GPU utilisation parsing handles `[N/A]` values gracefully

---

## [v1.15.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v1.15.0) (2025-12-21)

Comprehensive test coverage ensuring reliability across all components.

### Added
- **End-to-end CLI tests** (8 tests) validating complete benchmark workflows
  - Config validation through to results aggregation
  - Dataset listing and prompt source configuration
  - Error handling for invalid inputs
- **Integration tests** (47 tests) covering non-GPU workflows
  - Configuration loading with inheritance chains
  - Results repository file operations lifecycle
  - CLI command parsing and execution
  - Aggregation pipeline from raw results to exports
- **Methodology documentation** (`docs/methodology.md`) explaining measurement approach
  - Energy tracking via CodeCarbon with NVML backend
  - FLOPs estimation strategies and confidence levels
  - Distributed GPU result aggregation logic

### Changed
- Total test count: **416 passing tests** (unit + integration + e2e)
- All tests run without GPU access using mocked/simulated data

### Removed
- `requirements.txt` (306 frozen packages) — all dependencies now managed via Poetry lockfile

---

## [v1.13.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v1.13.0) (2025-12-21)

User-friendly command-line interface replacing legacy entry points.

### Added
- **Typer-based CLI** (`lem`) with intuitive subcommands:
  - `experiment <config> --dataset <name> -n <samples>`: Run experiments with automatic `accelerate launch` wrapping
  - `aggregate <exp_id> | --all [--force]`: Combine raw per-process JSON results into aggregated metrics
  - `config validate <file>`: Check configuration syntax and required fields
  - `config show <file>`: Display resolved configuration with inheritance applied
  - `results list [--all]`: Show available experiment runs
  - `results show <exp_id> [--raw] [--json]`: Inspect experiment results
  - `datasets`: List built-in HuggingFace datasets (alpaca, gsm8k, mmlu, sharegpt)
- **ExperimentOrchestrator** (~100 lines) with clean dependency injection
  - Accepts protocol-based components for energy backend, model loader, inference engine
  - Manages experiment lifecycle from config loading through result persistence
- **ExperimentContext** dataclass encapsulating runtime state
  - Accelerator instance, model, tokenizer, prompts
  - Automatic cleanup of distributed resources on context exit
- **Accelerate launcher** with configurable retry logic for transient failures
- **25 CLI tests** and **27 orchestration unit tests**

### Removed
- Legacy `MAIN_*.py` entry points (6 files) — all functionality now accessible via CLI

### Usage Examples
```bash
# Run experiment with built-in dataset
lem experiment configs/llama2-7b.yaml --dataset alpaca -n 1000

# Aggregate all pending results
lem aggregate --all

# Export results as JSON
lem results show exp_20240115_123456 --json
```

---

## [v1.10.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v1.10.0) (2025-12-20)

Major architectural refactor establishing clean module boundaries.

### Breaking Changes
- **Package renamed**: `llm-bench` → `lem`
- All imports now use `llenergymeasure` instead of `llm_bench`

### Added
- **Energy backend plugin registry** with automatic CodeCarbon registration
  - `register_backend()`, `get_backend()`, `list_backends()` API
  - Protocol-based interface for custom energy tracking backends
- **FlopsEstimator** with three-strategy fallback chain:
  1. `calflops` (high confidence) — direct computation graph measurement
  2. `architecture` (medium confidence) — calculation from `model.config` parameters
  3. `parameter_estimate` (low confidence) — approximation via `2 × params × seq_len`
  - Returns `FlopsResult` with value, method, confidence level, and precision
- **Results aggregation** with verification checks:
  - Temporal overlap detection (ensures concurrent GPU execution)
  - GPU attribution verification (prevents double-counting across processes)
  - Derived efficiency metrics (tokens/joule, FLOPs/watt)
- **Export functionality** for CSV and JSON formats
  - Flattened Pydantic models with logical column ordering
  - `ResultsExporter` class for unified export interface
- **Core modules** migrated from legacy `experiment_core_utils`:
  - `distributed.py`: Accelerator setup, unique ID generation, barrier synchronisation
  - `model_loader.py`: HuggingFace model/tokeniser loading with BitsAndBytes quantisation
  - `prompts.py`: Prompt filtering, sorting, tokenisation, batching strategies
  - `inference.py`: Batch inference engine with memory-efficient generation
  - `compute_metrics.py`: FLOPs calculation, peak memory stats, GPU utilisation tracking
  - `energy_backends/codecarbon.py`: CodeCarbon wrapper implementing `EnergyBackend` protocol
- **Pydantic domain models** for all configurations and results
  - `ExperimentConfig`, `BatchingOptions`, `DecoderConfig`, `QuantisationConfig`
  - `EnergyMetrics`, `InferenceMetrics`, `ComputeMetrics`, `RawProcessResult`, `AggregatedResult`
- **296 unit tests** covering all new modules

### Changed
- Replaced `print()` statements with Loguru structured logging throughout
- Comprehensive type hints and docstrings on all public interfaces
- BitsAndBytes quantisation correctly reports fp16 precision in FLOPs calculations

---

## [v1.0.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v1.0.0) (2025-12-16)

Research phase complete — stable multi-model benchmarking validated on production hardware.

### Features
- **Multi-model experiment support** with scenario-based configuration
  - Run sequential experiments across model families (Llama, Mistral, Phi, etc.)
  - Scenario configs defining model × precision × batch size combinations
- **Experiment suite CSV export** with consistent naming conventions
  - Timestamped output files with model name and config hash
  - Append mode for incremental experiment runs
- **Failed experiment detection** with cycle tracking
  - Automatic retry on transient failures
  - Quarantine of consistently failing configurations
- **Minimum output token enforcement** ensuring comparable generation lengths
- **Large model stability improvements**
  - Gradient checkpointing for memory-constrained runs
  - Proper CUDA cache clearing between experiments

### Research Capabilities
- **Data wrangling pipelines** for experiment result analysis
  - Pandas-based cleaning and normalisation
  - Outlier detection and filtering
- **Plotting functionality** for efficiency metrics visualisation
  - Tokens/second vs energy consumption scatter plots
  - Model size vs efficiency Pareto frontiers
- **FLOPs caching** preventing redundant calculations for repeated model runs

### Validation
- Tested with 1B and 3B parameter models on A100 GPUs
- Verified energy measurements against manual nvidia-smi readings
- Cross-validated FLOPs estimates with published model specifications

---

## [v0.5.0](https://github.com/henrycgbaker/LLenergyMeasure/releases/tag/v0.5.0) (2025-03-22)

Core measurement functionality establishing the foundation for all subsequent development.

### Added
- **Distributed results aggregation** across multiple GPUs
  - Per-process JSON result files with process rank metadata
  - Aggregation logic summing energy, averaging throughput
  - Support for 1-8 GPU configurations
- **FLOPs calculation** with quantisation awareness
  - Correct handling of INT8/INT4 operations
  - Integration with `calflops` library
- **Robust process cleanup** preventing zombie processes
  - Signal handlers for graceful shutdown
  - Distributed barrier synchronisation before exit
- **Optimum benchmark integration** for standardised measurements

### Improved
- **Distributed execution stability**
  - Proper NCCL initialisation and teardown
  - Timeout handling for stalled processes
- **Code organisation** with major directory restructuring
  - Separation of config, core, and result handling
  - Modular utility functions
