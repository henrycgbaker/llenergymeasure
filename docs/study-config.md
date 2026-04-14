# Study and Experiment Configuration

llenergymeasure uses YAML files to define experiments and studies. `llem run` auto-detects
whether a YAML file is an **experiment** (single run) or a **study** (sweep or multi-experiment
run) by inspecting its top-level keys. Files with a `sweep:` or `experiments:` key are loaded
as studies; all others are loaded as single experiments.

## Single Experiment

The minimal experiment YAML requires only `model:`:

```yaml
model: gpt2
engine: transformers
```

A fuller example with sub-configs:

```yaml
model: gpt2
engine: transformers
dtype: bfloat16
max_input_tokens: 256
max_output_tokens: 256

dataset:
  source: aienergyscore
  n_prompts: 100
  order: interleaved

decoder:
  preset: deterministic   # greedy decoding

transformers:
  batch_size: 4
  attn_implementation: sdpa

warmup:
  enabled: true
  n_warmup: 5

baseline:
  enabled: true
  duration_seconds: 30

energy_sampler: auto
gpu_telemetry: true
```

A vLLM experiment (requires Docker runner):

```yaml
model: gpt2
engine: vllm

dataset:
  source: aienergyscore
  n_prompts: 100

runners:
  vllm: docker

vllm:
  engine:
    enforce_eager: false
    gpu_memory_utilization: 0.9
    kv_cache_dtype: auto
  sampling:
    max_tokens: 128
```

Run either with:

```bash
llem run experiment.yaml
```

## Study / Sweep

A study runs multiple experiments with different configurations. There are two ways to define
the experiment set:

- **`sweep:`** — defines a grid of parameter values. Supports two entry types:
  - **Independent axes** (list of scalars) — Cartesian product across all axes.
  - **Dependent groups** (list of dicts) — alternatives within a group, crossed with other
    groups and axes. Use for parameters with mutual exclusivity or co-dependencies.
- **`experiments:`** — explicit list of experiment configs. Each entry is merged with any
  top-level shared fields.

Both can be combined: sweep configs are produced first, then explicit entries are appended.

The study YAML also accepts a `base:` key pointing to a base experiment config file, and a
`study_execution:` block controlling how many cycles to run and in what order.

## Sweep Grammar

### Single dimension sweep

```yaml
# 2 configs: dtype=float16 and dtype=bfloat16
name: dtype-sweep
model: gpt2
engine: transformers

dataset:
  n_prompts: 100

sweep:
  dtype: [float16, bfloat16]

study_execution:
  n_cycles: 3
  experiment_order: shuffle
```

Run with `llem run dtype-sweep.yaml`. Produces 2 configs × 3 cycles = 6 runs.

---

### Multi-dimension sweep (Cartesian product)

```yaml
# 4 configs: fp16+50, fp16+100, bf16+50, bf16+100
name: dtype-n-sweep
model: gpt2
engine: transformers

sweep:
  dtype: [float16, bfloat16]
  dataset.n_prompts: [50, 100]

study_execution:
  n_cycles: 3
  experiment_order: shuffle
```

Produces 4 configs × 3 cycles = 12 runs.

---

### Engine-scoped sweep (2-segment path)

Use dotted paths (`engine.param`) to sweep a Engine-specific parameter:

```yaml
# 4 configs: batch_size 1, 2, 4, 8 — all with engine=transformers
name: batch-size-sweep
model: gpt2
engine: transformers

sweep:
  pytorch.batch_size: [1, 2, 4, 8]

study_execution:
  n_cycles: 3
  experiment_order: shuffle
```

Produces 4 configs × 3 cycles = 12 runs. The `transformers.batch_size` path expands to a
`pytorch: { batch_size: N }` section in each generated experiment config.

---

### Engine-scoped sweep (3-segment path)

For nested engine configs (e.g. vLLM's `engine` sub-section):

```yaml
# 6 configs: 3 block_sizes × 2 kv_cache_dtypes
name: kv-cache-sweep
model: gpt2
engine: vllm

sweep:
  vllm.engine.block_size: [8, 16, 32]
  vllm.engine.kv_cache_dtype: [auto, fp8]

runners:
  vllm: docker

study_execution:
  n_cycles: 3
```

Produces 6 configs × 3 cycles = 18 runs. The path `vllm.engine.block_size` expands to
`vllm: { engine: { block_size: N } }`.

---

### Nested config sweep (dotted path)

Use dotted paths for nested config fields like `dataset.n_prompts` or `dataset.source`:

```yaml
name: dataset-size-sweep
model: gpt2
engine: transformers

sweep:
  dataset.n_prompts: [50, 100, 200]

study_execution:
  n_cycles: 3
```

Produces 3 configs × 3 cycles = 9 runs. The dotted path `dataset.n_prompts` expands to
`dataset: { n_prompts: N }` in each generated experiment config.

> **Note:** Dotted paths starting with an engine name (e.g. `transformers.batch_size`,
> `vllm.engine.max_num_seqs`) are treated as engine-scoped parameters. All other dotted
> paths (e.g. `dataset.n_prompts`, `dataset.order`) are treated as nested config fields.

---

### Explicit experiments list

Use `experiments:` when the configurations are not a regular grid:

```yaml
# 2 explicit configs
name: compare-engines

dataset:
  n_prompts: 50

experiments:
  - model: gpt2
    engine: transformers
    dtype: bfloat16
  - model: gpt2
    engine: vllm
    runners:
      vllm: docker

study_execution:
  n_cycles: 3
  experiment_order: interleave
```

Each entry is merged with any top-level shared fields (`dataset.n_prompts: 50` here).

---

### Base inheritance

Use `base:` to load a base experiment config file and sweep on top of it:

```yaml
# base-experiment.yaml is a normal experiment YAML
name: dtype-sweep
base: base-experiment.yaml

sweep:
  dtype: [float32, float16, bfloat16]

study_execution:
  n_cycles: 3
```

The base file is loaded, study-only keys (`sweep`, `experiments`, `study_execution`, `base`, `name`,
`runners`) are stripped, and the remaining fields become the starting point for all generated
configs. Inline fields in the study YAML override base fields. Path is resolved relative to
the study YAML file's directory.

---

### Mixed sweep and explicit

Both `sweep:` and `experiments:` can appear in the same study. Sweep-generated configs come
first, then explicit entries are appended:

```yaml
# 2 sweep configs + 1 explicit config = 3 total
name: mixed-study
model: gpt2

sweep:
  dtype: [float16, bfloat16]

experiments:
  - model: gpt2
    engine: transformers
    dtype: float32
    transformers:
      load_in_4bit: true

study_execution:
  n_cycles: 3
  experiment_order: shuffle
```

---

### Dependent groups (sweep groups)

Some parameters have constraints: `torch_compile_mode` only makes sense when
`torch_compile: true`, quantisation sub-params like `bnb_4bit_quant_type` require
`load_in_4bit: true`, and beam search requires `do_sample: false`. A plain Cartesian
sweep would produce invalid combinations. **Dependent groups** solve this by bundling
constrained parameters into named groups of alternative variants.

**Type-based disambiguation:** a list of scalars is an independent axis; a list of dicts
is a dependent group. Groups are crossed with each other and with independent axes, but
entries *within* a group are alternatives (unioned, not crossed).

```yaml
# 6 configs: 2 dtype × 3 compilation variants
name: compile-sweep
model: gpt2
engine: transformers

sweep:
  dtype: [float16, bfloat16]                     # independent axis (2 values)
  pytorch.compilation:                            # dependent group (3 variants)
    - pytorch.torch_compile: false
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: default
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: max-autotune
```

The group name (`transformers.compilation`) is an abstract label - it doesn't map to a config
field. Keys within each variant dict are fully-qualified dotted paths, routed the same way
as independent axis keys.

---

### Baseline variant (`{}`)

Use an empty dict `{}` as a group entry to include a "no override" baseline:

```yaml
sweep:
  pytorch.quantization:
    - {}                              # baseline: no quantisation
    - pytorch.load_in_8bit: true
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_quant_type: nf4
```

Produces 3 variants: unquantised baseline, 8-bit, and 4-bit.

---

### Mini-grids within group entries

A group entry can contain list-valued fields (list of scalars), which expand as a
mini Cartesian product within that entry:

```yaml
sweep:
  pytorch.caching:
    - {}                              # baseline
    - pytorch.use_cache: true
      pytorch.cache_implementation: [static, offloaded_static, sliding_window]
```

Produces 4 variants: 1 baseline + 3 cache implementations (all with `use_cache: true`).

---

### Cross-section overrides

Group entries can override fields outside their engine section, such as decoder settings:

```yaml
sweep:
  pytorch.decoding:
    - {}                              # baseline: use shared decoder settings
    - decoder.do_sample: false
      decoder.temperature: 0.0
      pytorch.num_beams: 4
      pytorch.early_stopping: true
```

---

### YAML anchors for repetition reduction

Use YAML anchors (`&name`) and merge keys (`<<: *name`) to avoid repeating shared fields
across group variants:

```yaml
sweep:
  tensorrt.quant_config:
    - {}                              # baseline: no quantisation
    - &trt_int8
      tensorrt.quant.quant_algo: INT8
    - <<: *trt_int8
      tensorrt.quant.kv_cache_quant_algo: INT8
    - tensorrt.quant.quant_algo: W4A16_AWQ
```

---

### Multiple groups crossed

When a sweep has multiple groups, they are crossed with each other (and with independent
axes):

```yaml
# 2 dtype × 2 compilation × 3 quantisation = 12 configs
sweep:
  dtype: [float16, bfloat16]
  pytorch.compilation:
    - pytorch.torch_compile: false
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: default
  pytorch.quantization:
    - {}
    - pytorch.load_in_8bit: true
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_quant_type: nf4
```

---

### Engine-scoped groups

Groups with an engine-prefixed name (e.g. `transformers.compilation`, `vllm.decoding`) only
apply to that engine's experiments. Universal groups (no engine prefix) apply to all
engines.
> **Collision rule:** A group name must not match an independent axis key. Use abstract
> names like `transformers.compilation` (not `transformers.torch_compile`) to avoid collisions.

---

## Execution Configuration

The `study_execution:` section controls cycle repetition and ordering:

```yaml
study_execution:
  n_cycles: 3
  experiment_order: shuffle   # sequential | interleave | shuffle | reverse | latin_square
```

**`n_cycles`** — how many times the full experiment list is repeated. Repeated execution
reduces measurement variance.

**`experiment_order`** — controls execution order across cycles. For experiments A and B with 3
cycles each:

| Order | Sequence | When to use |
|-------|----------|-------------|
| `sequential` | A, A, A, B, B, B | Thermal isolation between experiments |
| `interleave` | A, B, A, B, A, B | Reduces temporal bias; fair comparison |
| `shuffle` | random per-cycle, seeded | Publication-quality; eliminates ordering bias |
| `reverse` | A, B, B, A, A, B | Detects ordering effects via counterbalancing |
| `latin_square` | Williams design rows | Balances first-order carryover effects |

`shuffle` order is seeded from the study design hash, so the same study always shuffles
identically — reruns are reproducible. Override with an explicit `shuffle_seed`:

```yaml
study_execution:
  experiment_order: shuffle
  shuffle_seed: 123  # null = derived from study_design_hash
```

> **Note:** `shuffle_seed` (study-level scheduling) and `random_seed` (per-experiment
> inference/dataset RNG) are independent by design. Changing one does not affect the
> other. See [Methodology — Seeding model](methodology.md#seeding-model) for details.

**CLI effective defaults** when running `llem run study.yaml` (if not set in YAML):

- `n_cycles = 3`
- `experiment_order = shuffle`

Override with `llem run study.yaml --cycles 5 --order interleave`.

### Robustness Controls

The `study_execution:` section also supports circuit breaker and timeout fields:

```yaml
study_execution:
  max_consecutive_failures: 10      # 0 = disabled, 1 = fail-fast
  circuit_breaker_cooldown_seconds: 60.0
  wall_clock_timeout_hours: 24      # null = no limit
```

**`max_consecutive_failures`** - circuit breaker threshold. After N consecutive experiment
failures, the study aborts and marks remaining experiments as skipped. The circuit breaker
follows a 3-state pattern (closed/open/half-open): after tripping, it pauses for the
cooldown period, then runs one probe experiment. If the probe succeeds, normal execution
resumes; if it fails, the study aborts.

- `0` disables the circuit breaker entirely (equivalent to `--no-circuit-breaker`)
- `1` aborts on the first failure with no cooldown (equivalent to `--fail-fast`)
- Default: `10`

**`circuit_breaker_cooldown_seconds`** - pause duration before the half-open probe
experiment. Allows transient issues (GPU thermal throttling, OOM recovery) to resolve
before retrying. Default: `60.0`.

**`wall_clock_timeout_hours`** - hard time limit for the entire study. When the timeout
expires, remaining experiments are marked as `skipped` and the study status is set to
`timed_out`. The manifest preserves all completed results. Default: `null` (no limit).

CLI flags `--fail-fast`, `--no-circuit-breaker`, and `--timeout` override these settings.

### GPU Lock Files

Studies acquire advisory file locks on GPU devices to prevent concurrent studies from
competing for the same GPUs. Locks are acquired atomically (all-or-none) and released
automatically on process exit, including after SIGKILL. Disable with `--no-lock` if your
environment handles GPU scheduling externally.

### Study Resume

Interrupted studies (Ctrl-C, timeout, circuit breaker) can be resumed:

```bash
llem run study.yaml --resume           # auto-detect most recent
llem run study.yaml --resume-dir PATH  # specific study directory
```

Resume skips completed experiments and re-runs failed, skipped, pending, and interrupted
ones. Config drift (changed sweep axes or model) raises a hard error to prevent mixing
results from different configurations.

---

## Runner Configuration

The `runners:` section determines how each engine executes:

```yaml
runners:
  transformers: local                                          # run on host
  vllm: docker                                            # use default image
  vllm: "docker:ghcr.io/custom/vllm:latest"               # explicit image
```

| Value | Behaviour |
|-------|-----------|
| `local` | Run directly on the host (all dependencies must be installed) |
| `docker` | Run in a container using the default image for that engine |
| `docker:<image>` | Run in a container using the specified image |

When `docker` is used without an explicit image tag, the image is resolved from the installed
package version using the template `ghcr.io/henrycgbaker/llenergymeasure/{engine}:v{version}`.
For example, with `llenergymeasure==0.9.0` and `engine=vllm`, the image
`ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0` is pulled automatically.

See [Docker Setup](docker-setup.md#image-management) for image pull behaviour and pre-fetching.

---

## Config Reference

<!-- BEGIN CONFIG REFERENCE — Auto-generated by scripts/generate_config_docs.py -->

<!-- Auto-generated by scripts/generate_config_docs.py -- do not edit manually -->

## Configuration Reference

Full reference for all `ExperimentConfig` fields.
All fields except `model` are optional and have sensible defaults.

**Sections:**
- [Top-Level Fields](#top-level-fields)
- [Dataset (`dataset:`)](#dataset-dataset)
- [Decoder / Sampling (`decoder:`)](#decoder-sampling-decoder)
- [Warmup (`warmup:`)](#warmup-warmup)
- [Baseline (`baseline:`)](#baseline-baseline)
- [Energy Sampler (`energy_sampler:`)](#energy-sampler-energy_sampler)
- [GPU Telemetry (`gpu_telemetry:`)](#gpu-telemetry-gpu_telemetry)
- [PyTorch Engine (`pytorch:`)](#pytorch-engine-pytorch)
- [vLLM Engine (`vllm.engine:`)](#vllm-engine-vllm-engine)
- [vLLM Sampling (`vllm.sampling:`)](#vllm-sampling-vllm-sampling)
- [vLLM Beam Search (`vllm.beam_search:`)](#vllm-beam-search-vllm-beam_search)
- [vLLM Attention (`vllm.engine.attention:`)](#vllm-attention-vllm-engine-attention)
- [TensorRT-LLM Engine (`tensorrt:`)](#tensorrt-llm-engine-tensorrt)

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | *(required)* | HuggingFace model ID or local path |
| `engine` | 'pytorch' | 'vllm' | 'tensorrt' | `pytorch` | Inference engine |
| `dataset` | DatasetConfig | *(see below)* | Dataset configuration (nested sub-object) |
| `dtype` | 'float32' | 'float16' | 'bfloat16' | `bfloat16` | Model dtype for inference |
| `random_seed` | integer | `42` | Per-experiment seed: inference RNG and dataset ordering |
| `max_input_tokens` | integer | None | `256` | Max input token length for truncation. Keeps computation workload constant across experiments for fair comparison. `null` disables truncation. |
| `max_output_tokens` | integer | None | `256` | Max output tokens to generate. Keeps computation workload constant across experiments for fair comparison. `null` disables generation cap. |
| `decoder` | DecoderConfig | *(see section)* | Universal decoder/generation configuration |
| `warmup` | WarmupConfig | *(see section)* | Warmup phase configuration |
| `baseline` | BaselineConfig | *(see section)* | Baseline power measurement configuration |
| `energy_sampler` | 'auto' | 'nvml' | 'zeus' | 'codecarbon' | None | `auto` | Energy measurement backend. auto=best available (Zeus>NVML>CodeCarbon). null disables. |
| `gpu_telemetry` | boolean | `true` | Persist GPU power/thermal/memory timeseries to Parquet sidecar. NVML always runs for throttle detection; this controls disk output. |
| `pytorch` | TransformersConfig | None | `null` | PyTorch-specific configuration (only used when engine=transformers) |
| `vllm` | VLLMConfig | None | `null` | vLLM-specific configuration (only used when engine=vllm) |
| `tensorrt` | TensorRTConfig | None | `null` | TensorRT-LLM configuration (only used when engine=tensorrt) |
| `lora` | LoRAConfig | None | `null` | LoRA adapter configuration |
| `passthrough_kwargs` | dict | None | `null` | Extra kwargs passed through to engine at execution time. Keys must not collide with ExperimentConfig top-level fields. |
| `output_dir` | string | None | `null` | Per-experiment output directory override |

### Dataset (`dataset:`)

The `dataset:` section configures which prompts to use and how they are loaded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | string | `aienergyscore` | Dataset source: built-in alias (e.g. `aienergyscore`) or `.jsonl` file path |
| `n_prompts` | integer | `100` | Number of prompts to load |
| `order` | 'interleaved' | 'grouped' | 'shuffled' | `interleaved` | Prompt ordering: interleaved (file order), grouped (sorted by source field), shuffled (seeded random) |

**Examples:**

```yaml
# Built-in dataset (default)
dataset:
  source: aienergyscore
  n_prompts: 100

# Custom JSONL file
dataset:
  source: ./my-prompts.jsonl
  n_prompts: 500
  order: shuffled
```

### Decoder / Sampling (`decoder:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | number | `1.0` | Sampling temperature (0=greedy) |
| `do_sample` | boolean | `true` | Enable sampling (ignored if temp=0) |
| `top_k` | integer | `50` | Top-k sampling (0=disabled) |
| `top_p` | number | `1.0` | Top-p nucleus sampling (1.0=disabled) |
| `repetition_penalty` | number | `1.0` | Repetition penalty (1.0=no penalty) |
| `min_p` | number | None | `null` | Min probability filter (None -> disabled) |
| `min_new_tokens` | integer | None | `null` | Minimum output token count (None -> no minimum) |
| `preset` | 'deterministic' | 'standard' | 'creative' | 'factual' | None | `null` | Sampling preset (expands to preset values, overrides apply on top) |

### Warmup (`warmup:`)

Two modes: **fixed** (default) runs exactly `n_warmup` prompts; **CV convergence** (opt-in via `convergence_detection: true`) runs until latency CV drops below threshold. CV mode replaces `n_warmup` - they are alternative modes, not additive. After warmup, `thermal_floor_seconds` wait lets GPU temperature plateau before measurement.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable warmup phase |
| `n_warmup` | integer | `5` | Number of warmup prompts in fixed mode (ignored when convergence_detection=true) |
| `thermal_floor_seconds` | number | `60.0` | Post-warmup thermal stabilisation wait in seconds. Minimum 30s enforced. |
| `convergence_detection` | boolean | `false` | Enable CV-based adaptive warmup (replaces fixed n_warmup) |
| `cv_threshold` | number | `0.05` | CV target for convergence (stop when CV < this value) |
| `max_prompts` | integer | `20` | Safety cap on warmup prompts in CV mode |
| `window_size` | integer | `3` | Sliding window size for CV calculation |
| `min_prompts` | integer | `5` | Minimum prompts before checking convergence |

### Baseline (`baseline:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable baseline power measurement |
| `duration_seconds` | number | `30.0` | Baseline measurement duration in seconds (5-120s) |
| `strategy` | string | `"validated"` | Caching strategy: `validated` (cached with periodic spot-check, default), `cached` (disk-persisted TTL), `fresh` (measure every experiment) |
| `cache_ttl_seconds` | number | `7200.0` | How long a cached baseline remains valid before re-measurement, in seconds. Min 60s. Used with `cached`/`validated` strategies. |
| `validation_interval` | integer | `5` | Re-validate baseline every N experiments. Used with `validated` strategy only. |
| `drift_threshold` | number | `0.10` | Power drift fraction (0.01-0.50) to trigger re-measurement. Used with `validated` strategy only. |

### Energy Sampler (`energy_sampler:`)

`energy_sampler` is a flat top-level field (not a nested section). See [Energy Measurement](energy-measurement.md) for full details on backends, accuracy, and what the harness resolves internally.

| Value | Description |
|-------|-------------|
| `auto` (default) | Best available: Zeus > NVML > CodeCarbon |
| `nvml` | NVML power polling at 100ms intervals |
| `zeus` | Hardware energy counters (Volta+ GPUs). Most accurate. Install: `pip install "llenergymeasure[zeus]"` |
| `codecarbon` | System-level (GPU+CPU+RAM). Install: `pip install "llenergymeasure[codecarbon]"` |
| `null` | Disable energy measurement (throughput-only mode) |

### GPU Telemetry (`gpu_telemetry:`)

`gpu_telemetry` controls whether the NVML power/thermal/memory timeseries is persisted to a
Parquet sidecar file alongside the result JSON.

```yaml
gpu_telemetry: true    # default: write timeseries.parquet
gpu_telemetry: false   # skip parquet output (saves disk for large studies)
```

**What it controls:** Whether `timeseries.parquet` is written to the output directory.

**What it does not control:** NVML telemetry is always collected during inference for
throttle detection and measurement quality warnings, regardless of this setting. Setting
`gpu_telemetry: false` only suppresses disk output.

The Parquet sidecar contains 1Hz downsampled data with 8 columns: `timestamp_s`,
`gpu_index`, `power_w`, `temperature_c`, `memory_used_mb`, `memory_total_mb`,
`sm_utilisation_pct`, `throttle_reasons`. File sizes are typically < 5KB per minute of
inference per GPU.

See [Energy Measurement](energy-measurement.md) for details on how NVML telemetry
relates to energy measurement.

### Transformers Engine (`pytorch:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | integer | None | `null` | Batch size (None -> 1) |
| `attn_implementation` | 'sdpa' | 'flash_attention_2' | 'flash_attention_3' | 'eager' | None | `null` | Attention implementation (None -> sdpa) |
| `torch_compile` | boolean | None | `null` | Enable torch.compile (None -> False) |
| `torch_compile_mode` | string | None | `null` | torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default') |
| `torch_compile_backend` | string | None | `null` | torch.compile backend (None -> 'inductor') |
| `load_in_4bit` | boolean | None | `null` | BitsAndBytes 4-bit quantization |
| `load_in_8bit` | boolean | None | `null` | BitsAndBytes 8-bit quantization |
| `bnb_4bit_compute_dtype` | 'float16' | 'bfloat16' | 'float32' | None | `null` | Compute dtype for 4-bit (None -> float32, usually want bfloat16) |
| `bnb_4bit_quant_type` | 'nf4' | 'fp4' | None | `null` | 4-bit quantization type (None -> 'nf4') |
| `bnb_4bit_use_double_quant` | boolean | None | `null` | Double quantization saves ~0.4 bits/param (None -> False) |
| `use_cache` | boolean | None | `null` | Use KV cache during generation (None -> True) |
| `cache_implementation` | 'static' | 'offloaded_static' | 'sliding_window' | None | `null` | KV cache strategy; 'static' enables CUDA graphs (None -> dynamic) |
| `num_beams` | integer | None | `null` | Beam search width (None -> 1, greedy/sampling) |
| `early_stopping` | boolean | None | `null` | Stop beam search when all beams hit EOS (None -> False) |
| `length_penalty` | number | None | `null` | Beam length penalty: >1 shorter, <1 longer (None -> 1.0) |
| `no_repeat_ngram_size` | integer | None | `null` | Prevent n-gram repetition (None -> 0, disabled) |
| `prompt_lookup_num_tokens` | integer | None | `null` | Prompt-lookup speculative decoding tokens (None -> disabled) |
| `device_map` | string | None | `null` | Device placement strategy (None -> 'auto') |
| `max_memory` | dict | None | `null` | Per-device memory limits, e.g. {0: '10GiB', 'cpu': '50GiB'} |
| `revision` | string | None | `null` | Model revision/commit hash for reproducibility |
| `trust_remote_code` | boolean | None | `null` | Trust remote code in model repo (None -> True) |
| `tp_plan` | string | None | `null` | Tensor parallelism plan for native HF TP (None -> disabled). Only 'auto' is currently supported by Transformers. Mutually exclusive with device_map. Requires torchrun launch. |
| `tp_size` | integer | None | `null` | Number of tensor parallel ranks (None -> WORLD_SIZE). Only used when tp_plan is set. |

### vLLM Engine (`vllm.engine:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_memory_utilization` | number | None | `null` | GPU memory fraction for KV cache (None -> 0.9). Higher = more KV cache, less headroom. |
| `swap_space` | number | None | `null` | CPU swap space in GiB for KV cache offloading (None -> 4). Enables model weight offload to prevent OOM. |
| `cpu_offload_gb` | number | None | `null` | CPU RAM in GiB to offload model weights to (None -> 0, disabled). Reduces VRAM pressure at throughput cost. |
| `block_size` | 8 | 16 | 32 | None | `null` | KV cache block size in tokens (None -> 16). Affects KV cache fragmentation and memory efficiency. |
| `kv_cache_dtype` | 'auto' | 'fp8' | 'fp8_e5m2' | 'fp8_e4m3' | None | `null` | KV cache storage dtype (None -> auto = model dtype). fp8 variants halve KV cache VRAM on Ampere+. |
| `enforce_eager` | boolean | None | `null` | Disable CUDA graphs, always use eager mode (None -> False). Eager mode: predictable latency, no graph compilation overhead. |
| `enable_chunked_prefill` | boolean | None | `null` | Chunk large prefills across multiple scheduler iterations (None -> False). Affects scheduling latency and throughput. |
| `max_num_seqs` | integer | None | `null` | Max concurrent sequences per scheduler iteration (None -> 256). Affects batch size and KV cache usage. |
| `max_num_batched_tokens` | integer | None | `null` | Max tokens processed per scheduler iteration (None -> auto). Controls per-step compute budget. |
| `max_model_len` | integer | None | `null` | Max sequence length in tokens (input + output). Overrides model config (None -> model default). Caps KV cache allocation. |
| `tensor_parallel_size` | integer | None | `null` | Tensor parallel degree — number of GPUs to shard the model across (None -> 1). |
| `pipeline_parallel_size` | integer | None | `null` | Pipeline parallel stages — memory per GPU changes with PP (None -> 1). |
| `enable_prefix_caching` | boolean | None | `null` | Automatic prefix caching for repeated shared prompts (None -> False). |
| `quantization` | 'awq' | 'gptq' | 'fp8' | 'fp8_e5m2' | 'fp8_e4m3' | 'marlin' | 'bitsandbytes' | None | `null` | Quantization method. Requires pre-quantized model checkpoint. |
| `speculative_model` | string | None | `null` | HF model name or path of draft model for speculative decoding (None -> disabled). Requires num_speculative_tokens. |
| `num_speculative_tokens` | integer | None | `null` | Tokens to draft per speculative step (None -> required if speculative_model is set). |
| `offload_group_size` | integer | None | `null` | Groups of layers for CPU offloading (None -> 0). |
| `offload_num_in_group` | integer | None | `null` | Number of layers offloaded per group (None -> 1). |
| `offload_prefetch_step` | integer | None | `null` | Prefetch steps ahead for CPU offload (None -> 1). |
| `offload_params` | list[string] | None | `null` | Specific parameter names to offload to CPU (None -> all eligible). |
| `disable_custom_all_reduce` | boolean | None | `null` | Disable custom all-reduce for multi-GPU (None -> False). |
| `kv_cache_memory_bytes` | integer | None | `null` | Absolute KV cache size in bytes (None -> use gpu_memory_utilization). Mutually exclusive with gpu_memory_utilization. |
| `compilation_config` | dict | None | `null` | Full passthrough to vLLM CompilationConfig (~30 fields). No validation — passed directly. |
| `attention` | VLLMAttentionConfig | None | `null` | Attention implementation configuration. |

### vLLM Sampling (`vllm.sampling:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | integer | None | `null` | Max output tokens. Overrides ExperimentConfig.max_output_tokens for vLLM sweeps (None -> uses max_output_tokens). Use for Engine-specific max_tokens sweeps. |
| `min_tokens` | integer | None | `null` | Minimum output tokens before EOS is allowed (None -> 0, no minimum). |
| `presence_penalty` | number | None | `null` | Presence penalty: penalises tokens that appear at all (None -> 0.0). Affects generation diversity. |
| `frequency_penalty` | number | None | `null` | Frequency penalty: penalises tokens proportional to frequency (None -> 0.0). Affects repetition. |
| `ignore_eos` | boolean | None | `null` | Continue generating past EOS token (None -> False). Forces max_tokens generation every time — affects total token count. |
| `n` | integer | None | `null` | Number of output sequences per prompt (None -> 1). |

### vLLM Beam Search (`vllm.beam_search:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beam_width` | integer | None | `null` | Number of beams (ge=1). |
| `length_penalty` | number | None | `null` | Length penalty: >1 favours shorter, <1 longer (None -> 1.0). |
| `early_stopping` | boolean | None | `null` | Stop when beam_width complete sequences found (None -> False). |
| `max_tokens` | integer | None | `null` | Max output tokens for beam search (None -> max_output_tokens). |

### vLLM Attention (`vllm.engine.attention:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `engine` | string | None | `null` | Attention backend: flash_attn, flashinfer, etc. (None -> auto). |
| `flash_attn_version` | integer | None | `null` | Flash attention version (None -> auto). |
| `flash_attn_max_num_splits_for_cuda_graph` | integer | None | `null` | Max splits for CUDA graph with flash attention (None -> auto). |
| `use_prefill_decode_attention` | boolean | None | `null` | Use prefill-decode attention (None -> True). |
| `use_prefill_query_quantization` | boolean | None | `null` | Quantize queries during prefill (None -> False). |
| `use_cudnn_prefill` | boolean | None | `null` | Use cuDNN for prefill (None -> False). |
| `disable_flashinfer_prefill` | boolean | None | `null` | Disable FlashInfer for prefill (None -> False). |
| `disable_flashinfer_q_quantization` | boolean | None | `null` | Disable FlashInfer query quantization (None -> False). |
| `use_trtllm_attention` | boolean | None | `null` | Use TensorRT-LLM attention backend (None -> False). |
| `use_trtllm_ragged_deepseek_prefill` | boolean | None | `null` | Use TRT-LLM ragged DeepSeek prefill (None -> False). |

### TensorRT-LLM Engine (`tensorrt:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | integer | None | `null` | Max batch size (compile-time constant, None -> 8) |
| `tensor_parallel_size` | integer | None | `null` | Tensor parallel size (None -> 1) |
| `max_input_len` | integer | None | `null` | Max input sequence length (compile-time constant, None -> 1024) |
| `max_seq_len` | integer | None | `null` | Max total sequence length (input + output, compile-time constant, None -> 2048) |
| `dtype` | 'float16' | 'bfloat16' | None | `null` | Model dtype (None -> auto). TRT-LLM is optimised for fp16/bf16; fp32 not supported. |
| `fast_build` | boolean | None | `null` | Enable fast engine build mode (reduced optimisation, None -> False) |
| `engine` | string | None | `null` | TRT-LLM internal backend: 'trt' for TensorRT engine (None -> 'trt'). This is the TRT-LLM LLM(backend=...) parameter, not the llem engine field. |
| `engine_path` | string | None | `null` | Pre-compiled engine path (skip compilation) |
| `quant` | TensorRTQuantConfig | None | `null` | Quantisation configuration (QuantConfig) |
| `kv_cache` | TensorRTKvCacheConfig | None | `null` | KV cache configuration |
| `scheduler` | TensorRTSchedulerConfig | None | `null` | Scheduler configuration |
| `calib` | TensorRTCalibConfig | None | `null` | PTQ calibration configuration (CalibConfig) |
| `build_cache` | TensorRTBuildCacheConfig | None | `null` | Engine build cache configuration (BuildCacheConfig) |
| `sampling` | TensorRTSamplingConfig | None | `null` | Sampling configuration (TRT-LLM-specific SamplingParams extensions) |

<!-- END CONFIG REFERENCE — Auto-generated by scripts/generate_config_docs.py -->

---

## User Config File

llenergymeasure reads per-user defaults from `~/.config/llenergymeasure/config.yaml`
(XDG base directory, detected via `platformdirs`). This file is optional — all settings
have sensible defaults.

```yaml
# ~/.config/llenergymeasure/config.yaml
output:
  results_dir: ./results
  model_cache_dir: ~/.cache/huggingface

runners:
  transformers: local
  vllm: docker         # always use Docker for vLLM
  tensorrt: docker     # TensorRT-LLM requires Docker

measurement:
  datacenter_pue: 1.0
  carbon_intensity_gco2_kwh: 0.233
```

Run `llem config` to display the current effective configuration and check which engines
are installed. Use `llem config --verbose` for detailed environment information.
