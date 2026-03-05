# Backend Configuration

`llenergymeasure` supports multiple inference backends. Each backend uses a different runtime
and requires different setup. Currently active: **PyTorch** (local) and **vLLM** (Docker).
Planned: TensorRT-LLM (M4), SGLang (M5).

---

## Backend Overview

| Backend | Runner | GPU Required | Status |
|---------|--------|--------------|--------|
| PyTorch | local | Yes | Active |
| vLLM | docker | Yes | Active |
| TensorRT-LLM | docker | Yes | Planned (M4) |
| SGLang | docker | Yes | Planned (M5) |

**Runner** determines where the backend executes: `local` runs directly on the host,
`docker` launches an isolated container.

---

## PyTorch (Local)

The default backend. Runs the HuggingFace `transformers` AutoModelForCausalLM stack directly
on the host. No Docker required.

**Minimal config:**

```yaml
model: gpt2
backend: pytorch
```

**With PyTorch-specific options:**

```yaml
model: gpt2
backend: pytorch
n: 100
precision: bf16
pytorch:
  batch_size: 4
  attn_implementation: sdpa
  torch_compile: false
  load_in_4bit: false
```

### PyTorch Parameters

All `pytorch:` fields default to `null` — `null` means "use the backend's own default".
Unknown fields under `pytorch:` are forwarded to HuggingFace APIs.

**Batching:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1 | Number of prompts processed per forward pass |

**Attention:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attn_implementation` | `sdpa` \| `flash_attention_2` \| `flash_attention_3` \| `eager` | `sdpa` | Attention kernel |

**Compilation:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `torch_compile` | bool | false | Enable `torch.compile` |
| `torch_compile_mode` | str | `default` | Compile mode: `default`, `reduce-overhead`, `max-autotune` |
| `torch_compile_backend` | str | `inductor` | Compile backend |

Note: `torch_compile_mode` and `torch_compile_backend` require `torch_compile: true`.

**Quantization (BitsAndBytes):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_4bit` | bool | false | BitsAndBytes 4-bit quantization (NF4) |
| `load_in_8bit` | bool | false | BitsAndBytes 8-bit quantization |
| `bnb_4bit_compute_dtype` | `float16` \| `bfloat16` \| `float32` | `float32` | Compute dtype for 4-bit (usually set to `bfloat16`) |
| `bnb_4bit_quant_type` | `nf4` \| `fp4` | `nf4` | 4-bit quantization type |
| `bnb_4bit_use_double_quant` | bool | false | Double quantization (saves ~0.4 bits/param) |

Note: `load_in_4bit` and `load_in_8bit` are mutually exclusive.
`bnb_4bit_*` fields require `load_in_4bit: true`.

**KV Cache:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cache` | bool | true | Enable KV cache during generation |
| `cache_implementation` | `static` \| `offloaded_static` \| `sliding_window` | dynamic | KV cache strategy; `static` enables CUDA graphs |

**Model Loading:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_map` | str | `auto` | Device placement strategy |
| `max_memory` | dict | null | Per-device memory limits, e.g. `{0: "10GiB", cpu: "50GiB"}` |
| `revision` | str | null | Model commit hash for reproducibility |
| `trust_remote_code` | bool | true | Allow executing remote code in model repo |

**Beam Search:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_beams` | int | 1 | Beam search width (1 = greedy/sampling) |
| `early_stopping` | bool | false | Stop when all beams hit EOS |
| `length_penalty` | float | 1.0 | Length penalty: >1 shorter, <1 longer |

**Speculative Decoding:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_lookup_num_tokens` | int | null | Prompt-lookup speculative decoding (disabled when null) |

---

## vLLM (Docker)

A high-throughput inference backend using PagedAttention and continuous batching. Requires
Docker with NVIDIA Container Toolkit. See [Docker Setup Guide](docker-setup.md) for
installation instructions.

**Minimal config:**

```yaml
model: gpt2
backend: vllm
runners:
  vllm: docker
```

**With vLLM-specific options:**

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
    block_size: 16
    kv_cache_dtype: auto
  sampling:
    max_tokens: 128
    presence_penalty: 0.0
```

> **Nested structure required.** vLLM config uses a nested `engine:` / `sampling:` structure
> that mirrors vLLM's own API separation. Flat `vllm:` configs (from pre-M3 versions) are
> not supported.

### vLLM Engine Parameters

`vllm.engine:` fields map to `vllm.LLM()` constructor arguments. These are set at model
initialisation time. All fields default to `null` (use vLLM's own default).

**Memory Management:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_memory_utilization` | float [0.0, 1.0) | 0.9 | GPU memory fraction reserved for KV cache |
| `swap_space` | float | 4 GiB | CPU swap space in GiB for KV cache offloading |
| `cpu_offload_gb` | float | 0 | CPU RAM in GiB to offload model weights to |

**KV Cache:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_size` | `8` \| `16` \| `32` | 16 | KV cache block size in tokens |
| `kv_cache_dtype` | `auto` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` | `auto` | KV cache storage dtype; `fp8` halves VRAM on Ampere+ |
| `kv_cache_memory_bytes` | int | null | Absolute KV cache size in bytes (mutually exclusive with `gpu_memory_utilization`) |

**Execution Mode:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enforce_eager` | bool | false | Disable CUDA graphs, always use eager mode |
| `enable_chunked_prefill` | bool | false | Chunk large prefills across scheduler iterations |

**Scheduler / Batching:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_seqs` | int | 256 | Max concurrent sequences per scheduler iteration |
| `max_num_batched_tokens` | int | auto | Max tokens processed per scheduler iteration |
| `max_model_len` | int | model default | Max sequence length (input + output tokens) |

**Parallelism:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | int | 1 | Number of GPUs to shard the model across |
| `pipeline_parallel_size` | int | 1 | Pipeline parallel stages |

**Quantization and Prefix Caching:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization` | `awq` \| `gptq` \| `fp8` \| `marlin` \| `bitsandbytes` \| ... | null | Quantization method (requires pre-quantised checkpoint) |
| `enable_prefix_caching` | bool | false | Automatic prefix caching for shared prompt prefixes |

**Speculative Decoding:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speculative_model` | str | null | HF model name for draft model (speculative decoding) |
| `num_speculative_tokens` | int | null | Tokens to draft per step (required when `speculative_model` set) |

### vLLM Sampling Parameters

`vllm.sampling:` fields map to vLLM-specific `SamplingParams` extensions.
Universal sampling parameters (temperature, top_p, top_k, repetition_penalty) live in
`decoder:` and are shared across all backends.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | uses `max_output_tokens` | Max output tokens (overrides `max_output_tokens` for vLLM sweeps) |
| `min_tokens` | int | 0 | Minimum output tokens before EOS is allowed |
| `presence_penalty` | float [-2.0, 2.0] | 0.0 | Penalises tokens that appear at all |
| `frequency_penalty` | float [-2.0, 2.0] | 0.0 | Penalises tokens proportional to their frequency |
| `ignore_eos` | bool | false | Continue generating past EOS (forces `max_tokens` generation) |
| `n` | int | 1 | Number of output sequences per prompt |

### vLLM Beam Search

When using beam search with vLLM, use `vllm.beam_search:` instead of `vllm.sampling:`.
The two sections are mutually exclusive.

```yaml
vllm:
  engine:
    enforce_eager: false
  beam_search:
    beam_width: 4
    length_penalty: 1.0
    early_stopping: false
    max_tokens: 128
```

### Passthrough for New vLLM Parameters

Unknown fields under `vllm.engine:` or `vllm.sampling:` are forwarded to vLLM's native
`LLM()` and `SamplingParams()` APIs. This lets you use new vLLM parameters without waiting
for `llenergymeasure` to add explicit support:

```yaml
vllm:
  engine:
    some_new_vllm_param: true   # forwarded directly to vllm.LLM()
  sampling:
    another_new_param: 0.5      # forwarded directly to vllm.SamplingParams()
```

---

## Switching Between Backends

Change the `backend:` field and add the required runner config. The model and measurement
parameters stay the same.

```yaml
# Same experiment — PyTorch (local)
model: gpt2
backend: pytorch
n: 100
precision: bf16
```

```yaml
# Same experiment — vLLM (Docker)
model: gpt2
backend: vllm
n: 100
precision: bf16
runners:
  vllm: docker
```

Changing `backend:` switches the inference engine. Backend-specific sections (`pytorch:`,
`vllm:`) are ignored when not running that backend. Universal parameters (`n`, `precision`,
`decoder:`, etc.) apply to all backends.

---

## Runner Configuration

The runner determines where each backend executes. Configure runners in three ways
(listed in precedence order, highest first):

**1. Environment variable** — overrides all other settings:

```bash
LLEM_RUNNER_VLLM=docker llem run study.yaml
LLEM_RUNNER_VLLM=docker:my-registry/llem-vllm:custom llem run study.yaml
```

**2. Per-study YAML** — applies to that study only:

```yaml
runners:
  vllm: docker                             # use built-in default image
  vllm: docker:my-registry/llem:custom     # explicit image override
```

**3. User config file** — applies to all runs for that user:

```yaml
# ~/.config/llenergymeasure/config.yaml
runners:
  pytorch: local
  vllm: docker       # always use Docker for vLLM
```

**4. Default** — `local` for all backends if no runner is configured.

**Auto-elevation.** A study that mixes backends (e.g., `pytorch` local + `vllm` local) is
automatically elevated to use Docker for vLLM when Docker is available. This is logged at
info level and requires no user action.

**Multi-backend without Docker is an error.** If a study requires Docker runners and Docker
is not set up, `llem` fails at pre-flight with a clear error before running any experiments.

---

## Parameter Support Matrix

<!-- Parameter matrix — regenerate from GPU test results: uv run python scripts/generate_param_matrix.py -->

This matrix shows which parameters are supported by each backend. Derived from the Pydantic
config models. Full runtime verification (with GPU test results) can be generated with
`uv run python scripts/generate_param_matrix.py` after running GPU tests.

### Universal Parameters (all backends)

These parameters live in `ExperimentConfig` and are shared across all backends:

| Parameter | PyTorch | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `model` | Yes | Yes | Yes | HuggingFace model ID or local path |
| `backend` | Yes | Yes | Yes | Selects the inference engine |
| `n` | Yes | Yes | Yes | Number of prompts |
| `precision` | Yes | Yes | Yes | `fp32`, `fp16`, `bf16` |
| `dataset` | Yes | Yes | Yes | Dataset name or synthetic config |
| `max_input_tokens` | Yes | Yes | Yes | Input sequence length cap |
| `max_output_tokens` | Yes | Yes | Yes | Output token budget |
| `random_seed` | Yes | Yes | Yes | Reproducibility seed |
| `decoder.temperature` | Yes | Yes | Yes | Sampling temperature |
| `decoder.top_p` | Yes | Yes | Yes | Nucleus sampling threshold |
| `decoder.top_k` | Yes | Yes | Yes | Top-k sampling (0 = disabled) |
| `decoder.repetition_penalty` | Yes | Yes | Yes | Repetition penalty |
| `decoder.preset` | Yes | Yes | Yes | `deterministic`, `creative`, `balanced` |

### PyTorch-Specific Parameters

| Parameter | PyTorch | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `pytorch.batch_size` | Yes | N/A | N/A | Transformers batching |
| `pytorch.attn_implementation` | Yes | N/A | N/A | Attention kernel selection |
| `pytorch.torch_compile` | Yes | N/A | N/A | `torch.compile` acceleration |
| `pytorch.load_in_4bit` | Yes | N/A | N/A | BitsAndBytes 4-bit quantization |
| `pytorch.load_in_8bit` | Yes | N/A | N/A | BitsAndBytes 8-bit quantization |
| `pytorch.device_map` | Yes | N/A | N/A | Device placement strategy |
| `pytorch.num_beams` | Yes | N/A | N/A | Beam search width |
| `pytorch.prompt_lookup_num_tokens` | Yes | N/A | N/A | Prompt-lookup speculative decoding |

### vLLM-Specific Parameters

| Parameter | PyTorch | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `vllm.engine.gpu_memory_utilization` | N/A | Yes | N/A | KV cache memory fraction |
| `vllm.engine.block_size` | N/A | Yes | N/A | KV cache block size |
| `vllm.engine.kv_cache_dtype` | N/A | Yes | N/A | fp8 KV cache on Ampere+ |
| `vllm.engine.enforce_eager` | N/A | Yes | N/A | Disable CUDA graphs |
| `vllm.engine.tensor_parallel_size` | N/A | Yes | N/A | Multi-GPU sharding |
| `vllm.engine.quantization` | N/A | Yes | N/A | AWQ, GPTQ, FP8, etc. |
| `vllm.engine.speculative_model` | N/A | Yes | N/A | Draft model for spec. decoding |
| `vllm.sampling.max_tokens` | N/A | Yes | N/A | vLLM-specific max output tokens |
| `vllm.sampling.presence_penalty` | N/A | Yes | N/A | Presence penalty |
| `vllm.sampling.frequency_penalty` | N/A | Yes | N/A | Frequency penalty |
| `vllm.beam_search.beam_width` | N/A | Yes | N/A | vLLM beam search |

### TensorRT-LLM Parameters (Planned — M4)

| Parameter | PyTorch | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `tensorrt.max_batch_size` | N/A | N/A | Yes | Compile-time constant |
| `tensorrt.tp_size` | N/A | N/A | Yes | Tensor parallel size |
| `tensorrt.quantization` | N/A | N/A | Yes | `int8_sq`, `int4_awq`, `fp8` |
| `tensorrt.engine_path` | N/A | N/A | Yes | Pre-compiled engine path |
