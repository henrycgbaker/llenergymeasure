# Study and Experiment Configuration

llenergymeasure uses YAML files to define experiments and studies. `llem run` auto-detects
whether a YAML file is an **experiment** (single run) or a **study** (sweep or multi-experiment
run) by inspecting its top-level keys. Files with a `sweep:` or `experiments:` key are loaded
as studies; all others are loaded as single experiments.

## Single Experiment

The minimal experiment YAML requires only `model:`:

```yaml
model: gpt2
backend: pytorch
n: 100
```

A fuller example with sub-configs:

```yaml
model: gpt2
backend: pytorch
n: 100
precision: bf16
max_input_tokens: 512
max_output_tokens: 128

decoder:
  preset: deterministic   # greedy decoding

pytorch:
  batch_size: 4
  attn_implementation: sdpa

warmup:
  enabled: true
  n_warmup: 5

baseline:
  enabled: true
  duration_seconds: 30

energy:
  backend: auto
```

A vLLM experiment (requires Docker runner):

```yaml
model: gpt2
backend: vllm
n: 100

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

- **`sweep:`** — defines a grid of parameter values. llenergymeasure takes the Cartesian
  product of all sweep dimensions to produce every combination.
- **`experiments:`** — explicit list of experiment configs. Each entry is merged with any
  top-level shared fields.

Both can be combined: sweep configs are produced first, then explicit entries are appended.

The study YAML also accepts a `base:` key pointing to a base experiment config file, and an
`execution:` block controlling how many cycles to run and in what order.

## Sweep Grammar

### Single dimension sweep

```yaml
# 2 configs: precision=fp16 and precision=bf16
name: precision-sweep
model: gpt2
backend: pytorch
n: 100

sweep:
  precision: [fp16, bf16]

execution:
  n_cycles: 3
  cycle_order: shuffled
```

Run with `llem run precision-sweep.yaml`. Produces 2 configs × 3 cycles = 6 runs.

---

### Multi-dimension sweep (Cartesian product)

```yaml
# 4 configs: fp16+50, fp16+100, bf16+50, bf16+100
name: precision-n-sweep
model: gpt2
backend: pytorch

sweep:
  precision: [fp16, bf16]
  n: [50, 100]

execution:
  n_cycles: 3
  cycle_order: shuffled
```

Produces 4 configs × 3 cycles = 12 runs.

---

### Backend-scoped sweep (2-segment path)

Use dotted paths (`backend.param`) to sweep a backend-specific parameter:

```yaml
# 4 configs: batch_size 1, 2, 4, 8 — all with backend=pytorch
name: batch-size-sweep
model: gpt2
backend: pytorch

sweep:
  pytorch.batch_size: [1, 2, 4, 8]

execution:
  n_cycles: 3
  cycle_order: shuffled
```

Produces 4 configs × 3 cycles = 12 runs. The `pytorch.batch_size` path expands to a
`pytorch: { batch_size: N }` section in each generated experiment config.

---

### Backend-scoped sweep (3-segment path)

For nested backend configs (e.g. vLLM's `engine` sub-section):

```yaml
# 6 configs: 3 block_sizes × 2 kv_cache_dtypes
name: kv-cache-sweep
model: gpt2
backend: vllm

sweep:
  vllm.engine.block_size: [8, 16, 32]
  vllm.engine.kv_cache_dtype: [auto, fp8]

runners:
  vllm: docker

execution:
  n_cycles: 3
```

Produces 6 configs × 3 cycles = 18 runs. The path `vllm.engine.block_size` expands to
`vllm: { engine: { block_size: N } }`.

---

### Explicit experiments list

Use `experiments:` when the configurations are not a regular grid:

```yaml
# 2 explicit configs
name: compare-backends
n: 50

experiments:
  - model: gpt2
    backend: pytorch
    precision: bf16
  - model: gpt2
    backend: vllm
    runners:
      vllm: docker

execution:
  n_cycles: 3
  cycle_order: interleaved
```

Each entry is merged with any top-level shared fields (`n: 50` here).

---

### Base inheritance

Use `base:` to load a base experiment config file and sweep on top of it:

```yaml
# base-experiment.yaml is a normal experiment YAML
name: precision-sweep
base: base-experiment.yaml

sweep:
  precision: [fp32, fp16, bf16]

execution:
  n_cycles: 3
```

The base file is loaded, study-only keys (`sweep`, `experiments`, `execution`, `base`, `name`,
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
  precision: [fp16, bf16]

experiments:
  - model: gpt2
    backend: pytorch
    precision: fp32
    pytorch:
      load_in_4bit: true

execution:
  n_cycles: 3
  cycle_order: shuffled
```

---

## Execution Configuration

The `execution:` section controls cycle repetition and ordering:

```yaml
execution:
  n_cycles: 3
  cycle_order: shuffled   # sequential | interleaved | shuffled
```

**`n_cycles`** — how many times the full experiment list is repeated. Repeated execution
reduces measurement variance.

**`cycle_order`** — controls execution order across cycles. For experiments A and B with 3
cycles each:

| Order | Sequence | When to use |
|-------|----------|-------------|
| `sequential` | A, A, A, B, B, B | Thermal isolation between experiments |
| `interleaved` | A, B, A, B, A, B | Reduces temporal bias; fair comparison |
| `shuffled` | random per-cycle, seeded | Publication-quality; eliminates ordering bias |

`shuffled` order is seeded from the study design hash, so the same study always shuffles
identically — reruns are reproducible.

**CLI effective defaults** when running `llem run study.yaml` (if not set in YAML):

- `n_cycles = 3`
- `cycle_order = shuffled`

Override with `llem run study.yaml --cycles 5 --order interleaved`.

---

## Config Reference

<!-- BEGIN CONFIG REFERENCE — Auto-generated by scripts/generate_config_docs.py -->

## Configuration Reference

Full reference for all `ExperimentConfig` fields.
All fields except `model` are optional and have sensible defaults.

**Sections:**
- [Top-Level Fields](#top-level-fields)
- [Decoder / Sampling (`decoder:`)](#decoder-sampling-decoder)
- [Warmup (`warmup:`)](#warmup-warmup)
- [Baseline (`baseline:`)](#baseline-baseline)
- [Energy (`energy:`)](#energy-energy)
- [PyTorch Backend (`pytorch:`)](#pytorch-backend-pytorch)
- [vLLM Engine (`vllm.engine:`)](#vllm-engine-vllm-engine)
- [vLLM Sampling (`vllm.sampling:`)](#vllm-sampling-vllm-sampling)
- [vLLM Beam Search (`vllm.beam_search:`)](#vllm-beam-search-vllm-beam_search)
- [vLLM Attention (`vllm.engine.attention:`)](#vllm-attention-vllm-engine-attention)
- [TensorRT-LLM Backend (`tensorrt:`)](#tensorrt-llm-backend-tensorrt)

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | *(required)* | HuggingFace model ID or local path |
| `backend` | `pytorch` \| `vllm` \| `tensorrt` | `pytorch` | Inference backend |
| `n` | integer | `100` | Number of prompts from dataset |
| `dataset` | string \| SyntheticDatasetConfig | `aienergyscore` | Dataset name (built-in alias) or synthetic dataset config |
| `dataset_order` | `interleaved` \| `grouped` \| `shuffled` | `interleaved` | Prompt ordering: interleaved (round-robin by source, file order), grouped (sorted by source), shuffled (seed-based random) |
| `precision` | `fp32` \| `fp16` \| `bf16` | `bf16` | Floating point precision |
| `random_seed` | integer | `42` | Random seed for reproducibility |
| `max_input_tokens` | integer | `512` | Max input tokens |
| `max_output_tokens` | integer | `128` | Max output tokens |
| `decoder` | DecoderConfig | *(see section)* | Universal decoder/generation configuration |
| `warmup` | WarmupConfig | *(see section)* | Warmup phase configuration |
| `baseline` | BaselineConfig | *(see section)* | Baseline power measurement configuration |
| `energy` | EnergyConfig | *(see section)* | Energy measurement backend configuration |
| `pytorch` | PyTorchConfig \| null | `null` | PyTorch-specific configuration (only used when backend=pytorch) |
| `vllm` | VLLMConfig \| null | `null` | vLLM-specific configuration (only used when backend=vllm) |
| `tensorrt` | TensorRTConfig \| null | `null` | TensorRT-LLM configuration (only used when backend=tensorrt) |
| `lora` | LoRAConfig \| null | `null` | LoRA adapter configuration |
| `passthrough_kwargs` | dict \| null | `null` | Extra kwargs passed through to backend at execution time. Keys must not collide with ExperimentConfig top-level fields. |
| `output_dir` | string \| null | `null` | Per-experiment output directory override |

### Decoder / Sampling (`decoder:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | number | `1.0` | Sampling temperature (0=greedy) |
| `do_sample` | boolean | `true` | Enable sampling (ignored if temp=0) |
| `top_k` | integer | `50` | Top-k sampling (0=disabled) |
| `top_p` | number | `1.0` | Top-p nucleus sampling (1.0=disabled) |
| `repetition_penalty` | number | `1.0` | Repetition penalty (1.0=no penalty) |
| `min_p` | number \| null | `null` | Min probability filter (None -> disabled) |
| `min_new_tokens` | integer \| null | `null` | Minimum output token count (None -> no minimum) |
| `preset` | `deterministic` \| `standard` \| `creative` \| `factual` \| null | `null` | Sampling preset (expands to preset values, overrides apply on top) |

**Presets:** `deterministic` (temp=0, greedy), `standard` (temp=1.0, top_p=0.95),
`creative` (temp=0.8, top_p=0.9, repetition_penalty=1.1), `factual` (temp=0.3).

### Warmup (`warmup:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable warmup phase |
| `n_warmup` | integer | `5` | Number of full-length warmup prompts before measurement |
| `thermal_floor_seconds` | number | `60.0` | Minimum seconds to wait after warmup before measuring (thermal stabilisation). Minimum 30s enforced. |
| `convergence_detection` | boolean | `false` | Enable CV-based convergence detection (additive to n_warmup) |
| `cv_threshold` | number | `0.05` | CV target for convergence (only used when convergence_detection=True) |
| `max_prompts` | integer | `20` | Maximum warmup prompts when CV mode is on (safety cap) |
| `window_size` | integer | `5` | Window size for CV calculation |
| `min_prompts` | integer | `5` | Minimum prompts before checking convergence (warm start) |

### Baseline (`baseline:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable baseline power measurement |
| `duration_seconds` | number | `30.0` | Baseline measurement duration in seconds |

### Energy (`energy:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `auto` \| `nvml` \| `zeus` \| `codecarbon` \| null | `auto` | Energy measurement backend. `null` disables energy measurement. |

### PyTorch Backend (`pytorch:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | integer \| null | `null` | Batch size (None -> 1) |
| `attn_implementation` | `sdpa` \| `flash_attention_2` \| `flash_attention_3` \| `eager` \| null | `null` | Attention implementation (None -> sdpa) |
| `torch_compile` | boolean \| null | `null` | Enable torch.compile (None -> False) |
| `torch_compile_mode` | string \| null | `null` | torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default') |
| `torch_compile_backend` | string \| null | `null` | torch.compile backend (None -> 'inductor') |
| `load_in_4bit` | boolean \| null | `null` | BitsAndBytes 4-bit quantization |
| `load_in_8bit` | boolean \| null | `null` | BitsAndBytes 8-bit quantization |
| `bnb_4bit_compute_dtype` | `float16` \| `bfloat16` \| `float32` \| null | `null` | Compute dtype for 4-bit (None -> float32, usually want bfloat16) |
| `bnb_4bit_quant_type` | `nf4` \| `fp4` \| null | `null` | 4-bit quantization type (None -> 'nf4') |
| `bnb_4bit_use_double_quant` | boolean \| null | `null` | Double quantization saves ~0.4 bits/param (None -> False) |
| `use_cache` | boolean \| null | `null` | Use KV cache during generation (None -> True) |
| `cache_implementation` | `static` \| `offloaded_static` \| `sliding_window` \| null | `null` | KV cache strategy; 'static' enables CUDA graphs (None -> dynamic) |
| `num_beams` | integer \| null | `null` | Beam search width (None -> 1, greedy/sampling) |
| `early_stopping` | boolean \| null | `null` | Stop beam search when all beams hit EOS (None -> False) |
| `length_penalty` | number \| null | `null` | Beam length penalty: >1 shorter, <1 longer (None -> 1.0) |
| `no_repeat_ngram_size` | integer \| null | `null` | Prevent n-gram repetition (None -> 0, disabled) |
| `prompt_lookup_num_tokens` | integer \| null | `null` | Prompt-lookup speculative decoding tokens (None -> disabled) |
| `device_map` | string \| null | `null` | Device placement strategy (None -> 'auto') |
| `max_memory` | dict \| null | `null` | Per-device memory limits, e.g. {0: '10GiB', 'cpu': '50GiB'} |
| `revision` | string \| null | `null` | Model revision/commit hash for reproducibility |
| `trust_remote_code` | boolean \| null | `null` | Trust remote code in model repo (None -> True) |
| `num_processes` | integer \| null | `null` | Data parallel processes via Accelerate (None -> 1) |

### vLLM Engine (`vllm.engine:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_memory_utilization` | number \| null | `null` | GPU memory fraction for KV cache (None -> 0.9). Higher = more KV cache, less headroom. |
| `swap_space` | number \| null | `null` | CPU swap space in GiB for KV cache offloading (None -> 4). |
| `cpu_offload_gb` | number \| null | `null` | CPU RAM in GiB to offload model weights to (None -> 0, disabled). |
| `block_size` | `8` \| `16` \| `32` \| null | `null` | KV cache block size in tokens (None -> 16). |
| `kv_cache_dtype` | `auto` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` \| null | `null` | KV cache storage dtype (None -> auto = model dtype). fp8 variants halve KV cache VRAM on Ampere+. |
| `enforce_eager` | boolean \| null | `null` | Disable CUDA graphs, always use eager mode (None -> False). |
| `enable_chunked_prefill` | boolean \| null | `null` | Chunk large prefills across multiple scheduler iterations (None -> False). |
| `max_num_seqs` | integer \| null | `null` | Max concurrent sequences per scheduler iteration (None -> 256). |
| `max_num_batched_tokens` | integer \| null | `null` | Max tokens processed per scheduler iteration (None -> auto). |
| `max_model_len` | integer \| null | `null` | Max sequence length in tokens (input + output). Overrides model config (None -> model default). |
| `tensor_parallel_size` | integer \| null | `null` | Tensor parallel degree — number of GPUs to shard model across (None -> 1). |
| `pipeline_parallel_size` | integer \| null | `null` | Pipeline parallel stages (None -> 1). |
| `enable_prefix_caching` | boolean \| null | `null` | Automatic prefix caching for repeated shared prompts (None -> False). |
| `quantization` | `awq` \| `gptq` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` \| `marlin` \| `bitsandbytes` \| null | `null` | Quantization method. Requires pre-quantized model checkpoint. |
| `speculative_model` | string \| null | `null` | Draft model for speculative decoding (None -> disabled). Requires `num_speculative_tokens`. |
| `num_speculative_tokens` | integer \| null | `null` | Tokens to draft per speculative step. |
| `kv_cache_memory_bytes` | integer \| null | `null` | Absolute KV cache size in bytes. Mutually exclusive with `gpu_memory_utilization`. |
| `compilation_config` | dict \| null | `null` | Full passthrough to vLLM CompilationConfig. No validation — passed directly. |
| `attention` | VLLMAttentionConfig \| null | `null` | Attention implementation configuration. |

### vLLM Sampling (`vllm.sampling:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | integer \| null | `null` | Max output tokens. Overrides `max_output_tokens` for vLLM sweeps. |
| `min_tokens` | integer \| null | `null` | Minimum output tokens before EOS is allowed (None -> 0). |
| `presence_penalty` | number \| null | `null` | Presence penalty: penalises tokens that appear at all (None -> 0.0). |
| `frequency_penalty` | number \| null | `null` | Frequency penalty: penalises tokens proportional to frequency (None -> 0.0). |
| `ignore_eos` | boolean \| null | `null` | Continue generating past EOS token (None -> False). |
| `n` | integer \| null | `null` | Number of output sequences per prompt (None -> 1). |

### vLLM Beam Search (`vllm.beam_search:`)

Mutually exclusive with `vllm.sampling:`. When set, uses `BeamSearchParams` instead of
`SamplingParams`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beam_width` | integer \| null | `null` | Number of beams. |
| `length_penalty` | number \| null | `null` | Length penalty: >1 favours shorter, <1 longer (None -> 1.0). |
| `early_stopping` | boolean \| null | `null` | Stop when beam_width complete sequences found (None -> False). |
| `max_tokens` | integer \| null | `null` | Max output tokens for beam search (None -> `max_output_tokens`). |

### vLLM Attention (`vllm.engine.attention:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string \| null | `null` | Attention backend: `flash_attn`, `flashinfer`, etc. (None -> auto). |
| `flash_attn_version` | integer \| null | `null` | Flash attention version (None -> auto). |
| `use_prefill_decode_attention` | boolean \| null | `null` | Use prefill-decode attention (None -> True). |

### TensorRT-LLM Backend (`tensorrt:`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | integer \| null | `null` | Max batch size (compile-time constant, None -> 8) |
| `tp_size` | integer \| null | `null` | Tensor parallel size (None -> 1) |
| `quantization` | `int8_sq` \| `int4_awq` \| `fp8` \| null | `null` | Quantization method |
| `engine_path` | string \| null | `null` | Pre-compiled engine path (skip compilation) |

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
  pytorch: local
  vllm: docker         # always use Docker for vLLM
  tensorrt: local

measurement:
  datacenter_pue: 1.0
  grid_carbon_intensity: 0.233
```

Run `llem config` to display the current effective configuration and check which backends
are installed. Use `llem config --verbose` for detailed environment information.
