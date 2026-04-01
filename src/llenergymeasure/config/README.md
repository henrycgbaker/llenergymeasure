# config/ - Configuration System

Configuration loading, validation, and models for experiment setup.

## Purpose

Provides Pydantic-based configuration models and a loader that supports YAML/JSON files with inheritance via `_extends`. The system supports three configuration modes: config files, presets, and CLI overrides.

## Parameter Resolution

Configuration parameters are resolved with the following precedence (highest to lowest):

```
CLI flags  >  Config file  >  Preset  >  Defaults
```

This allows flexible workflows:
- **Formal experiments**: Full config files for reproducibility
- **Quick exploration**: Presets with minimal flags
- **Parameter sweeps**: Base config with CLI overrides

## CLI vs YAML Philosophy

**Core Principle**: CLI flags are for workflow control; YAML configs are for testable experiment parameters.

Think of it as: **CLI = "how to run"** vs **YAML = "what to measure"**

### Workflow Params (CLI Recommended)

These control execution workflow, not the experiment itself:

| Flag | Purpose |
|------|---------|
| `--model`, `-m` | Which model to benchmark |
| `--dataset`, `-d` | Which prompt source to use |
| `--preset` | Quick-start configuration template |
| `--cycles`, `-c` | Statistical repetition count |
| `--seed` | Reproducibility seed |
| `--max-tokens` | Generation limit |
| `--fresh` | Ignore incomplete experiments |
| `--resume` | Continue interrupted experiment |
| `--no-aggregate` | Skip auto-aggregation |
| `--results-dir` | Output location |

### Testable Params (YAML Required for Formal Experiments)

These define the experiment configuration that affects measurements:

| Config Field | What It Controls |
|--------------|------------------|
| `batching.batch_size` | Inference batch size |
| `batching.strategy` | static/dynamic/sorted_static/sorted_dynamic |
| `dtype` | float32/float16/bfloat16 |
| `num_processes` | Distributed workers |
| `gpus` | GPU allocation |
| `decoder.temperature` | Sampling temperature |
| `quantization.*` | 4-bit/8-bit quantisation |
| `traffic_simulation.*` | Traffic patterns (Poisson/constant) |
| `sharding.*` | Tensor/pipeline parallelism |

### Deprecated CLI Flags

These CLI flags **still work** but emit deprecation warnings:

```
--batch-size, -b    → Use batching.batch_size in YAML
--dtype             → Use dtype in YAML
--num-processes     → Use num_processes in YAML
--gpu-list          → Use gpus in YAML
--temperature       → Use decoder.temperature in YAML
--quantization      → Use quantization.quantization in YAML
```

**Why deprecated?** They encourage ad-hoc experiments harder to reproduce.
**Why keep working?** Backwards compatibility and quick iteration.

### Override Tracking

All CLI overrides are recorded in result metadata for traceability:

```json
{
  "effective_config": { "batch_size": 8, ... },
  "cli_overrides": {
    "batching.batch_size": { "original": 1, "new": 8 }
  }
}
```

### Workflow Examples

```bash
# 1. Formal experiment (fully reproducible)
lem experiment configs/llama2-7b-benchmark.yaml

# 2. Quick exploration (preset + model)
lem experiment --preset quick-test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d alpaca

# 3. Parameter sweep (override tracked in metadata)
for batch in 1 2 4 8; do
  lem experiment config.yaml --batch-size $batch
done

# 4. Statistical robustness (use campaigns)
lem campaign campaign.yaml --cycles 5
```

## Key Files

### models.py
Pydantic configuration models.

**ExperimentConfig** - Main experiment configuration:
```python
from llenergymeasure.config import ExperimentConfig

config = ExperimentConfig(
    config_name="my-experiment",
    model_name="meta-llama/Llama-2-7b-hf",
    max_input_tokens=512,
    max_output_tokens=128,
    gpus=[0, 1],
    num_processes=2,
    random_seed=42,  # For reproducibility
)
```

**Sub-configurations:**
- `DatasetConfig` - source, n_prompts, order (nested sub-object)
- `DecoderConfig` - temperature, sampling presets, repetition control
- `PyTorchConfig` - PyTorch backend options
- `VLLMConfig` - vLLM backend options
- `TensorRTConfig` - TensorRT-LLM backend options

## Decoder Sampling Configuration

Industry-standard sampling parameters aligned with vLLM, HuggingFace, and MLPerf.

### Sampling Presets

Use `preset` for common configurations:

```yaml
decoder:
  preset: deterministic  # Greedy decoding (temp=0, do_sample=false)
```

| Preset | Temperature | do_sample | Notes |
|--------|-------------|-----------|-------|
| `deterministic` | 0.0 | false | Greedy decoding for reproducible benchmarks |
| `standard` | 1.0 | true | Balanced sampling (top_p=0.95, top_k=50) |
| `creative` | 0.8 | true | Higher variance (top_p=0.9, repetition_penalty=1.1) |
| `factual` | 0.3 | true | Lower variance (top_k=10) |

**Preset + override**: Explicit parameters override preset values:
```yaml
decoder:
  preset: deterministic
  temperature: 0.5  # This overrides the preset's temperature
```

### Full Parameter Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | [0, 2] | 1.0 | 0=greedy, 1=standard, >1=more random |
| `do_sample` | bool | true | Enable sampling (ignored if temp=0) |
| `top_p` | [0, 1] | 1.0 | Nucleus sampling (1.0=disabled) |
| `top_k` | ≥0 | 50 | Top-k sampling (0=disabled) |
| `min_p` | [0, 1] | 0.0 | Min prob relative to top token (0=disabled) |
| `repetition_penalty` | [0.1, 10] | 1.0 | Repetition penalty (1.0=disabled) |
| `no_repeat_ngram_size` | ≥0 | 0 | Prevent n-gram repetition (0=disabled) |
| `preset` | enum | None | Shortcut: deterministic/standard/creative/factual |

### Example Configurations

```yaml
# Greedy decoding (most reproducible)
decoder:
  preset: deterministic

# Custom sampling
decoder:
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3

# Preset with override
decoder:
  preset: creative
  repetition_penalty: 1.5  # Override preset value
```

### Seed Handling for Reproducibility

`random_seed` is the single seed source for all per-experiment stochasticity:
- Backend inference RNG (`torch.manual_seed`, vLLM `seed=`, TRT-LLM `random_seed=`)
- Dataset prompt ordering (when `dataset.order: shuffled`)

Study-level cycle shuffling uses a separate `shuffle_seed` (defaults to `study_design_hash`).

```yaml
random_seed: 42
decoder:
  preset: standard  # Sampling enabled but reproducible with seed
```

## Sharding / Parallelism Configuration

Multi-GPU parallelism for large model inference.

**Background**: v1.x used `accelerate launch` with `device_map="auto"`, which auto-distributes layers across visible GPUs based on `CUDA_VISIBLE_DEVICES`. This worked but was naive—each GPU holds different layers with no coordinated parallel execution.

v2.0 adds proper parallelism strategies:
- **Tensor Parallelism**: Splits layers horizontally so GPUs compute in parallel
- **Pipeline Parallelism**: Explicit stage-based splitting with coordinated execution

Two strategies available:

### Tensor Parallelism (TP)

Splits individual layers across GPUs. Each GPU processes a portion of each layer in parallel.

```yaml
sharding:
  strategy: tensor_parallel
  num_shards: 2          # Number of GPUs
  tp_plan: auto          # HuggingFace native TP
gpus: [0, 1]
```

**Supported models**: Llama, Mistral, Mixtral, Qwen, Phi, Gemma, Falcon, MPT, BLOOM, OPT

**Note**: Uses `torchrun` launcher instead of `accelerate launch`.

### Pipeline Parallelism (PP)

Splits model vertically into sequential stages across GPUs. Each GPU holds a subset of layers (e.g., layers 0-15 on GPU 0, layers 16-31 on GPU 1). Forward passes run sequentially through stages.

```yaml
sharding:
  strategy: pipeline_parallel
  num_shards: 4              # Number of stages/GPUs
gpus: [0, 1, 2, 3]
```

**Use cases**:
- Model too large for single GPU but TP not supported
- Simple multi-GPU inference without specialised backends

**Note**: For production serving with optimised batching and continuous batching, consider using the vLLM backend (when available).

### Parallelism Parameter Reference

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `strategy` | `none`, `tensor_parallel`, `pipeline_parallel` | `none` | Parallelism strategy |
| `num_shards` | 1+ | 1 | Number of GPUs for parallelism |
| `tp_plan` | `auto` | `auto` | Tensor parallel plan (HF native, TP only) |

### Configuration Warnings

The validator (`validate_config()`) checks for problematic configurations and returns warnings at three severity levels:

| Severity | Behaviour |
|----------|-----------|
| **error** | Blocks experiment without `--force` flag |
| **warning** | Shows warning, prompts for confirmation (skip with `--yes`) |
| **info** | Informational, doesn't block |

#### All Configuration Warnings

| Category | Condition | Severity | Message |
|----------|-----------|----------|---------|
| Distributed | `num_processes > 1` with single GPU | info | Multiple processes with single GPU may not provide parallelism benefits |
| Tokens | `max_output_tokens > 2048` | warning | Very high max_output_tokens may cause memory issues |
| Quantization | `quantization=True` with `dtype=float32` | warning | Quantization typically uses float16 compute, not float32 |
| Quantization | `quantization=True` without 4bit/8bit specified | error | Must specify load_in_4bit or load_in_8bit |
| Batching | Dynamic strategy without `max_tokens_per_batch` | info | Will use max_input_tokens as token budget |
| Batching | Dynamic strategy with non-default `batch_size` | warning | batch_size is ignored for dynamic strategies |
| Batching | Sorted strategy with `batch_size=1` | info | Sorting provides no benefit with batch_size=1 |
| Parallelism | `data_parallel` with vLLM backend | error | data_parallel not supported for vLLM |
| Parallelism | `degree > len(gpus)` | error | degree exceeds available GPUs |
| Sharding | `num_shards > len(gpus)` | error | num_shards exceeds available GPUs |
| Sharding | Sharding strategy with single GPU | info | Sharding provides no benefit with single GPU |
| Sharding | TP with unsupported model | warning | Model may not support HF native tensor parallelism |
| Sharding | TP with quantization | warning | Quantization with tensor parallelism is experimental |
| Decoder | Non-default sampling params in deterministic mode | error | Sampling params ignored when temp=0 or do_sample=False |
| Decoder | `do_sample=True` with `temperature=0` | info | do_sample has no effect when temperature=0 |
| Decoder | Both `temperature` and `top_p` modified | warning | Not recommended - alter one or the other |
| Traffic | `target_qps > 100` | warning | Very high QPS may not be achievable |

#### Handling Warnings

```bash
# Show warnings, prompt for confirmation
lem experiment config.yaml --dataset alpaca -n 100

# Skip confirmation prompts (auto-accept warnings)
lem experiment config.yaml --dataset alpaca -n 100 --yes

# Run despite blocking errors
lem experiment config.yaml --dataset alpaca -n 100 --force
```

### loader.py
Configuration loading with inheritance.

```python
from llenergymeasure.config import load_config, validate_config

config = load_config("configs/experiment.yaml")
warnings = validate_config(config)
```

**Key functions:**
- `load_config(path)` - Load and validate config
- `validate_config(config)` - Return warnings (not errors)
- `load_config_dict(path)` - Load raw dict (YAML/JSON)
- `resolve_inheritance(dict, path)` - Apply `_extends`
- `deep_merge(base, overlay)` - Merge nested dicts

**Inheritance example:**
```yaml
# base.yaml
max_input_tokens: 512
dtype: float16

# experiment.yaml
_extends: base.yaml
config_name: my-experiment
model_name: meta-llama/Llama-2-7b-hf
```

## Built-in Presets

Presets provide sensible defaults for common scenarios. Use with `--preset <name>`:

| Preset | Purpose | Settings |
|--------|---------|----------|
| `quick-test` | Fast validation runs | batch=1, max_in=64, max_out=32, deterministic |
| `benchmark` | Formal measurements | batch=1, max_in=2048, max_out=512, fp16, deterministic |
| `throughput` | Throughput-optimised | batch=8, max_in=512, max_out=256, fp16, dynamic batching |

**Usage:**
```bash
# Preset with model (quick exploration)
lem experiment --preset quick-test --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d alpaca

# Start new config from preset
lem config new --preset benchmark
```

## Configuration Reference

### Parameter Table (CLI vs Config)

| Config Field | CLI Flag | Type | Default | Description |
|--------------|----------|------|---------|-------------|
| `config_name` | - | str | Required | Unique identifier |
| `model_name` | `--model / -m` | str | Required | HuggingFace model path |
| `adapter` | - | str\|None | None | LoRA adapter (HF Hub ID or local path) |
| `backend` | `--backend` | str | pytorch | Inference backend (pytorch/vllm/tensorrt) |
| `max_input_tokens` | - | int | 512 | Max input tokens |
| `max_output_tokens` | `--max-tokens` | int | 128 | Max generated tokens |
| `min_output_tokens` | - | int | 0 | Min generated tokens |
| `dtype` | `--dtype` | str | bfloat16 | float32/float16/bfloat16 |
| `num_processes` | `--num-processes` | int | 1 | Worker processes |
| `gpus` | `--gpu-list` | list[int] | [0] | GPU indices |
| `random_seed` | `--seed` | int\|None | None | Random seed |
| `streaming` | `--streaming` | bool | false | Enable TTFT/ITL measurement |
| `streaming_warmup_requests` | `--streaming-warmup` | int | 5 | Warmup requests (excluded) |
| `batching.batch_size` | `--batch-size / -b` | int | 1 | Batch size |
| `batching.strategy` | - | str | static | Batching strategy |
| `traffic_simulation.enabled` | - | bool | false | Enable traffic simulation |
| `traffic_simulation.mode` | - | str | poisson | Traffic mode (poisson/constant) |
| `traffic_simulation.target_qps` | - | float | 1.0 | Target queries per second |
| `decoder.temperature` | `--temperature` | float | 1.0 | Decoder temperature |
| `quantization.quantization` | `--quantization` | bool | false | Enable quantization |
| `io.results_dir` | `--results-dir` | str\|None | None | Results directory override |

### Required Fields
| Field | Type | Description |
|-------|------|-------------|
| `config_name` | str | Unique identifier |
| `model_name` | str | HuggingFace model path |

### Token Settings
| Field | Default | Description |
|-------|---------|-------------|
| `max_input_tokens` | 512 | Max input tokens |
| `max_output_tokens` | 128 | Max generated tokens |
| `min_output_tokens` | 0 | Min generated tokens |

### Distributed Settings
| Field | Default | Description |
|-------|---------|-------------|
| `gpus` | [0] | GPU indices |
| `num_processes` | 1 | Worker processes |

### Precision Settings
| Field | Default | Options |
|-------|---------|---------|
| `dtype` | bfloat16 | float32, float16, bfloat16 |
| `backend` | pytorch | pytorch, tensorrt, vllm |

### Reproducibility
| Field | Default | Description |
|-------|---------|-------------|
| `random_seed` | None | Random seed (None = non-deterministic) |

## Backend-Specific Configuration

Each inference backend exposes its own configuration section for advanced optimisation. These are defined in `backend_configs.py` and set via `vllm:` or `pytorch:` blocks in YAML.

**Important:** Backend-specific settings are IN ADDITION to shared config (model, batching, decoder, etc.). Some shared settings map to backend params, others may be overridden:

| Shared Config | vLLM Mapping | Notes |
|--------------|--------------|-------|
| `parallelism.degree` | `tensor_parallel_size` | Automatic mapping |
| `batching.batch_size` | Ignored | vLLM uses continuous batching via `max_num_seqs` |
| `decoder.*` | `SamplingParams` | Converted to vLLM sampling params |
| `quantization.*` | May be overridden | `vllm.quantization_method` takes precedence |

```yaml
# vLLM backend
backend: vllm
vllm:
  gpu_memory_utilization: 0.9   # Most important: controls KV cache size
  max_num_seqs: 256             # Max concurrent sequences (replaces batch_size)
  enable_prefix_caching: true   # 30-50% throughput gain for repeated prefixes
  kv_cache_dtype: fp8           # Memory-efficient KV cache

# PyTorch backend (default)
backend: pytorch
pytorch:
  attn_implementation: flash_attention_2
  torch_compile: reduce-overhead
```

**Most users only need:** `gpu_memory_utilization`, `max_num_seqs`, `enable_prefix_caching`

### backend_configs.py

| Config Class | Purpose |
|--------------|---------|
| `VLLMConfig` | vLLM engine kwargs and sampling params |
| `VLLMAttentionConfig` | Attention backend selection |
| `VLLMSpeculativeConfig` | Speculative decoding setup |
| `VLLMLoRAConfig` | LoRA adapter configuration |
| `PyTorchConfig` | PyTorch/Transformers options |
| `PyTorchAssistedGenerationConfig` | Assisted generation (speculative) |
| `TensorRTConfig` | TensorRT-LLM engine and build options |
| `TensorRTQuantizationConfig` | TensorRT quantization (FP8/INT8/INT4) |
| `TensorRTCalibrationConfig` | INT8 calibration data settings |

**Full documentation:** See [docs/backends.md](../../../docs/backends.md) for comprehensive parameter reference.

## I/O Configuration

Control results directory and data paths.

```yaml
io:
  results_dir: /custom/results/path  # Override default results location
```

**Precedence:**
1. `--results-dir` CLI flag (highest)
2. `io.results_dir` in config YAML
3. `LLM_ENERGY_RESULTS_DIR` environment variable
4. Default `results/` directory (lowest)

## Streaming Latency Configuration

Enable Time to First Token (TTFT) and Inter-Token Latency (ITL) measurement.

```yaml
streaming: true                    # Enable streaming mode
streaming_warmup_requests: 5       # Warmup requests (reuses first N prompts)
```

**Notes:**
- Streaming mode processes prompts sequentially for accurate per-token timing
- `batch_size` is ignored when streaming is enabled
- Warmup is additional overhead: first N prompts run for warmup, then ALL prompts run for measurement
- Recommended: 30+ prompts for reliable latency percentiles

## Prompt Source Configuration

Prompts can be loaded from files or HuggingFace datasets.

### File-based prompts
```yaml
prompts:
  type: file
  path: ./prompts.txt  # One prompt per line
```

### HuggingFace datasets
```yaml
prompts:
  type: huggingface
  dataset: alpaca          # Built-in alias or full HF path
  split: train             # Dataset split (default: train)
  column: instruction      # Column to extract (auto-detected if omitted)
  sample_size: 1000        # Limit prompts (optional)
  shuffle: true            # Shuffle before sampling (default: false)
  seed: 42                 # Random seed (default: 42)
```

**Built-in dataset aliases:**
| Alias | HuggingFace Path | Default Column | Notes |
|-------|-----------------|----------------|-------|
| `ai-energy-score` | AIEnergyScore/text_generation | text | **Default** when no dataset specified |
| `alpaca` | tatsu-lab/alpaca | instruction | |
| `sharegpt` | anon8231489123/ShareGPT_Vicuna_unfiltered | conversations | |
| `gsm8k` | gsm8k (main subset) | question | |
| `mmlu` | cais/mmlu (all subset) | question | |

**Auto-detect columns:** text, prompt, question, instruction, input, content

## Batching Strategies (MLPerf/vLLM Terminology)

Industry-standard batching strategies for benchmarking:

```yaml
batching:
  strategy: sorted_dynamic    # static | dynamic | sorted_static | sorted_dynamic
  batch_size: 4               # Only used with static/sorted_static strategies
  max_tokens_per_batch: 512   # Only used with dynamic/sorted_dynamic strategies
```

| Strategy | Description | Key Parameter |
|----------|-------------|---------------|
| `static` | Fixed batch size (default) | `batch_size` |
| `dynamic` | Token-aware batching by token budget | `max_tokens_per_batch` |
| `sorted_static` | Sort prompts by length, then fixed batches | `batch_size` |
| `sorted_dynamic` | Sort prompts by length, then token-aware batching | `max_tokens_per_batch` |

**Important:** `batch_size` is ignored for dynamic strategies. Dynamic strategies group prompts by total token count using `max_tokens_per_batch` instead of a fixed number of prompts.

**Length sorting** reduces padding waste by grouping similar-length prompts together.

## Traffic Simulation (MLPerf LoadGen Style)

Simulate realistic request arrival patterns for load testing:

```yaml
traffic_simulation:
  enabled: true
  mode: poisson             # poisson | constant
  target_qps: 2.0           # Target queries per second (arrival rate λ)
  seed: 42                  # For reproducibility
```

| Mode | Description |
|------|-------------|
| `poisson` | Exponential inter-arrival times (realistic traffic) |
| `constant` | Fixed inter-arrival = 1/target_qps |

**Poisson arrivals** model real-world traffic patterns where requests arrive randomly but at a known average rate.

## Scheduled Experiments (Daemon Mode)

Run experiments on a schedule for temporal variation studies (e.g., energy consumption at different times of day).

### CLI Usage
```bash
# Run every 6 hours for 24 hours
lem schedule config.yaml --interval 6h --duration 24h --dataset alpaca -n 100

# Run daily at 9am for a week
lem schedule config.yaml --at 09:00 --duration 7d

# Run at 9am on weekdays only
lem schedule config.yaml --at 09:00 --days weekdays --duration 14d

# Run every 12 hours on weekends
lem schedule config.yaml --interval 12h --days sat,sun --duration 48h
```

### YAML Configuration
```yaml
schedule:
  enabled: true
  interval: "6h"              # Run every 6 hours (alternative to 'at')
  at: "09:00"                 # Run at specific time (alternative to 'interval')
  days: ["mon", "wed", "fri"] # Filter to specific days (optional)
  total_duration: "7d"        # Stop daemon after 7 days
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable scheduled mode |
| `interval` | str | None | Interval between runs (e.g., '6h', '30m') |
| `at` | str | None | Time of day to run (e.g., '09:00') |
| `days` | list | None | Days to run: ['mon','tue',...] or ['weekdays'] |
| `total_duration` | str | '24h' | How long to run daemon |

**Day aliases**: `weekdays` (mon-fri), `weekends` (sat-sun)

## Validation Rules

Pydantic validators enforce:
- `num_processes <= len(gpus)`
- `min_output_tokens <= max_output_tokens`
- `load_in_4bit` and `load_in_8bit` are mutually exclusive
- `target_qps > 0` (traffic simulation)
- `sample_size >= 1` (prompt source)

## Grid Generation

Generate configs for parameter sweeps using Cartesian product:

```bash
lem config generate-grid base.yaml \
    --vary batch_size=1,2,4,8 \
    --vary dtype=float16,float32 \
    --output-dir ./grid/
```

This creates 8 configs (4 batch sizes × 2 dtypes) in `./grid/`.

### ⚠️ Grid Validation Flags

**Important**: Parameter sweeps can produce invalid combinations. For example, varying `temperature` and `top_k` together will create configs where sampling params are set in deterministic mode (temp=0), which is an error.

| Flag | Behaviour |
|------|-----------|
| (default) | Generate **all** configs, show warnings for invalid ones |
| `--validate` | Only generate **valid** configs (skip invalid) |

**Usage:**
```bash
# Default: generates all, warns about invalid (may produce unusable configs!)
lem config generate-grid base.yaml \
    --vary decoder.temperature=0.0,1.0 \
    --vary decoder.top_k=50,100

# Recommended: skip invalid configs
lem config generate-grid base.yaml \
    --vary decoder.temperature=0.0,1.0 \
    --vary decoder.top_k=50,100 \
    --validate
```

**Common invalid combinations detected:**

| Combination | Error |
|-------------|-------|
| `temperature=0` + non-default `top_k`/`top_p`/`min_p` | Sampling params ignored in deterministic mode |
| `quantization=true` without bit mode | Missing `load_in_4bit` or `load_in_8bit` |
| `num_shards > len(gpus)` | More shards than GPUs |
| `do_sample=false` + sampling params | Sampling params ignored |

**Output example (default):**
```
Generating 6 configs from 2 parameters...
✓ Generated 6 configs in /tmp/grid/

⚠ 2 config(s) with blocking errors:
  base_temperature_0_0_top_k_100.yaml:
    [ERROR] decoder: Sampling params [top_k=100] have no effect in deterministic mode
  base_temperature_0_0_top_k_50.yaml:
    [ERROR] decoder: ... (if top_k=50 differs from default)
```

**Output example (--validate):**
```
Generating 6 configs from 2 parameters...
✓ Generated 4 configs in /tmp/grid/

Skipped 2 invalid configs due to --validate
```

**Best practice**: Use `--validate` when generating grids for batch experiments to avoid runtime failures.

## Interactive Config Builder

Create configs interactively with sensible defaults:

```bash
lem config new                    # Start from scratch
lem config new --preset benchmark # Start from preset
lem config new -o my-config.yaml  # Specify output path
```

## Reproducibility

Results include `effective_config` and `cli_overrides` fields for full reproducibility:

```json
{
  "effective_config": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "batch_size": 8,
    "dtype": "float16"
  },
  "cli_overrides": {
    "batching.batch_size": {"new": 8, "original": 1}
  }
}
```

## Related

- See `../cli/` for CLI commands (`llem run`, `llem config`)
- See `../harness/__init__.py` for config usage in inference
- See `../utils/constants.py` for shared constants
- See `../domain/experiment.py` for result models with config tracking
