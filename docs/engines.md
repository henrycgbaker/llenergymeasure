# Engine Configuration

`llenergymeasure` supports multiple inference engines. each engine uses a different runtime
and requires different setup. Currently active: **Transformers** (local), **vLLM** (Docker), and
**TensorRT-LLM** (Docker). Planned: SGLang (M5).

---

## Engine Overview

| Engine | Runner | GPU Required | Status |
|---------|--------|--------------|--------|
| Transformers | local | Yes | Active |
| vLLM | docker | Yes | Active |
| TensorRT-LLM | docker | Yes | Active |
| SGLang | docker | Yes | Planned (M5) |

**Runner** determines where the engine executes: `local` runs directly on the host,
`docker` launches an isolated container.

---

## Transformers (Local)

The default engine. Runs the HuggingFace `transformers` AutoModelForCausalLM stack directly
on the host. No Docker required.

**Minimal config:**

```yaml
model: gpt2
engine: transformers
```

**With Transformers-specific options:**

```yaml
model: gpt2
engine: transformers
n: 100
dtype: bfloat16
transformers:
  batch_size: 4
  attn_implementation: sdpa
  torch_compile: false
  load_in_4bit: false
```

### Transformers Parameters

All `pytorch:` fields default to `null` — `null` means "use the engine's own default".
Unknown fields under `pytorch:` are forwarded to HuggingFace APIs.

**Batching:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1 | Number of prompts processed per forward pass |

**Attention:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attn_implementation` | `sdpa` \| `flash_attention_2` \| `flash_attention_3` \| `eager` | `sdpa` | Attention kernel |

Note: `flash_attention_3` requires the `flash_attn_3` package (built separately from the
flash-attn repo's `hopper/` directory) and an Ampere+ GPU (SM80+, e.g. A100 or H100).
The Docker Transformers image includes FA3 by default. To skip it (e.g. for faster CI builds),
rebuild with `--build-arg INSTALL_FA3=false`. See
[Installation - FlashAttention-3](installation.md#flashattention-3) for details.

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

**N-gram Repetition:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `no_repeat_ngram_size` | int | 0 | Prevent n-gram repetition (0 = disabled) |

**Speculative Decoding:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_lookup_num_tokens` | int | null | Prompt-lookup speculative decoding (disabled when null) |

**Tensor Parallelism (HF Transformers >= 4.50):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tp_plan` | `auto` \| null | null | Native HF tensor parallelism plan. Mutually exclusive with `device_map`. Requires `torchrun` launch. |
| `tp_size` | int | WORLD_SIZE | Number of tensor parallel ranks. Only used when `tp_plan` is set. |

Note: `tp_plan` and `device_map` are mutually exclusive — tensor parallelism handles its own
device placement. When `tp_plan='auto'`, `device_map` is automatically omitted.

---

## vLLM (Docker)

A high-throughput inference engine using PagedAttention and continuous batching. Requires
Docker with NVIDIA Container Toolkit. See [Docker Setup Guide](docker-setup.md) for
installation instructions.

**Minimal config:**

```yaml
model: gpt2
engine: vllm
runners:
  vllm: docker
```

**With vLLM-specific options:**

```yaml
model: gpt2
engine: vllm
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
`decoder:` and are shared across all engines.

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

## TensorRT-LLM (Docker)

A maximum-performance inference engine using NVIDIA TensorRT engine compilation. TRT-LLM
compiles a model into an optimised TensorRT engine on first use, then runs inference
against that engine. Engines are cached on disk so subsequent runs skip compilation.
Requires Docker with NVIDIA Container Toolkit. See [Docker Setup Guide](docker-setup.md)
for installation instructions.

**Minimal config:**

```yaml
model: meta-llama/Llama-2-7b-hf
engine: tensorrt
runners:
  tensorrt: docker
```

**With TensorRT-LLM-specific options:**

```yaml
model: meta-llama/Llama-2-7b-hf
engine: tensorrt
n: 50
dtype: bfloat16
runners:
  tensorrt: docker
tensorrt:
  tp_size: 2
  max_batch_size: 8
  dtype: bfloat16
  quant:
    quant_algo: W4A16_AWQ
  build_cache:
    max_cache_storage_gb: 256
```

> **Engine compilation on first run.** The first run with a given config will compile a
> TensorRT engine (which may take several minutes). Subsequent runs with the same config
> use the cached engine and start much faster.

### Compile-Time Parameters

These parameters define the engine shape and cannot be changed without recompiling.
Changing any **[recompile]** field invalidates the cached engine and triggers a new build.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_batch_size` | int | 8 | Maximum batch size the engine accepts. **[recompile]** |
| `tp_size` | int | 1 | Tensor parallel degree (number of GPUs). **[recompile]** |
| `max_input_len` | int | 1024 | Maximum input sequence length in tokens. **[recompile]** |
| `max_seq_len` | int | 2048 | Maximum total sequence length (input + output). **[recompile]** |
| `dtype` | `float16` \| `bfloat16` | auto | Model compute dtype. TRT-LLM is optimised for fp16/bf16; fp32 is not supported. **[recompile]** |
| `fast_build` | bool | false | Enable fast engine build mode (reduced optimisation, faster compilation). **[recompile]** |
| `engine` | `trt` | `trt` | TRT-LLM internal backend selector. This is the `LLM(backend=...)` parameter, not the `llem` engine field. Leave unset unless you have a specific reason to override. |
| `engine_path` | str | null | Path to a pre-compiled engine directory. When set, skips compilation and loads the engine directly. All compile-time parameters and `build_cache` are ignored. See [Pre-Compiled Engine Loading](#pre-compiled-engine-loading) below. |

### tensorrt.quant: Quantization

Quantization is applied at engine compile time — changing `quant_algo` triggers a recompile.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quant_algo` | see below | null (no quantization) | Quantization algorithm (native QuantAlgo enum name) |
| `kv_cache_quant_algo` | `FP8` \| `INT8` | null | KV cache quantization algorithm |

**Valid `quant_algo` values:**

| Value | Description |
|-------|-------------|
| `FP8` | FP8 weight and activation quantization. Requires SM >= 8.9 (Ada Lovelace or Hopper). **Not supported on A100 (SM 8.0).** |
| `INT8` | INT8 smooth quantization |
| `W4A16_AWQ` | 4-bit AWQ weight quantization, FP16 activations |
| `W4A16_GPTQ` | 4-bit GPTQ weight quantization, FP16 activations |
| `W8A16` | 8-bit weight quantization, FP16 activations |
| `W8A16_GPTQ` | 8-bit GPTQ weight quantization, FP16 activations |
| `W4A8_AWQ` | 4-bit AWQ weight, INT8 activations |
| `NO_QUANT` | Explicitly disable quantization |

> **A100 note:** A100 (SM 8.0) does not support FP8. Valid A100 quantization options:
> `INT8`, `W4A16_AWQ`, `W4A16_GPTQ`, `W8A16`, `W8A16_GPTQ`, `W4A8_AWQ`, `NO_QUANT`.

### tensorrt.calib: PTQ Calibration

Only relevant when `quant_algo` requires post-training quantization (PTQ) calibration
(e.g. `INT8`, `W4A16_AWQ`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calib_batches` | int | 512 | Number of calibration batches |
| `calib_dataset` | str | `cnn_dailymail` | Calibration dataset name or HuggingFace path |
| `calib_max_seq_length` | int | 512 | Max sequence length for calibration samples |

### tensorrt.kv_cache: KV Cache

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_block_reuse` | bool | false | Enable KV cache block reuse across requests |
| `free_gpu_memory_fraction` | float [0.0, 1.0] | 0.9 | Fraction of free GPU memory to allocate for KV cache |
| `max_tokens` | int | auto | Maximum total tokens in the KV cache |
| `host_cache_size` | int | 0 | Host (CPU) cache size in bytes for KV cache offloading (0 = disabled) |

### tensorrt.scheduler: Scheduler

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity_scheduling_policy` | `GUARANTEED_NO_EVICT` \| `MAX_UTILIZATION` \| `STATIC_BATCH` | `GUARANTEED_NO_EVICT` | Scheduling capacity policy |

**Policy descriptions:**
- `GUARANTEED_NO_EVICT` — guarantees no request eviction; may reduce throughput
- `MAX_UTILIZATION` — maximises GPU utilisation; may evict requests under memory pressure
- `STATIC_BATCH` — fixed batch size; useful for reproducible benchmarking

### tensorrt.build_cache: Engine Build Cache

Engine caching is enabled by default even without an explicit `build_cache:` section.
Compiled engines are stored in `~/.cache/tensorrt_llm` and reused across runs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_root` | str | `~/.cache/tensorrt_llm` | Root directory for engine cache storage |
| `max_records` | int | 10 | Maximum number of cached engine records |
| `max_cache_storage_gb` | float | 256 | Maximum total cache size in GB |

### Pre-Compiled Engine Loading

When `tensorrt.engine_path` is set, llem loads a pre-compiled TensorRT-LLM engine
directly, skipping engine compilation entirely. This is useful when:

- You have pre-built engines from a previous `llm.save()` call or a build pipeline
- You want deterministic, fast experiment startup (no compilation overhead)
- You are running sweeps where only runtime parameters vary (not engine shape)

**Engine directory structure:**

```
/path/to/engine/
  config.json          # Must contain 'pretrained_config' and 'build_config' keys
  rank0.engine         # Compiled engine binary for GPU 0
  rank1.engine         # Required if tp_size=2 (one file per rank)
  tokenizer.json       # Saved by llm.save() - enables self-contained engines
  tokenizer_config.json
```

**Validation checks (run before loading):**

1. Directory exists
2. `config.json` exists and is valid JSON
3. `tp_size` in `config.json` matches `tensorrt.tp_size` (if detectable)
4. `rank{N}.engine` files exist for each rank (0 to tp_size-1)

**What happens when `engine_path` is set:**

- `model` field is still required but used only as a fallback tokeniser source
  (if the engine directory lacks tokeniser files)
- All compile-time parameters (`max_batch_size`, `tp_size`, `max_input_len`,
  `max_seq_len`, `dtype`, `fast_build`) are **ignored** - they are baked into
  the engine
- `build_cache` is **ignored** - no compilation occurs, so caching is irrelevant
- Runtime parameters (`kv_cache`, `scheduler`, `sampling`) still apply
- `build_metadata.engine_path` in the result indicates which engine was loaded

**Example:**

```yaml
tensorrt:
  engine_path: /engines/llama-7b-fp16-tp1
  tp_size: 1  # Must match the engine's tp_size
```

> **Note:** Engines are not portable across GPU architectures. An engine compiled
> on A100 (SM 8.0) will not load on H100 (SM 9.0) or vice versa. TRT-LLM will
> raise a clear error at load time if there is an architecture mismatch.

> **Tokeniser note:** Engines built via `llm.save()` include tokeniser files and
> are self-contained. Engines built via the `trtllm-build` CLI may lack tokeniser
> files - in that case, the `model` field is used as a fallback tokeniser source.

### tensorrt.sampling: TRT-LLM-Specific Sampling

These are TRT-LLM-specific extensions to `SamplingParams`. Universal sampling parameters
(temperature, top_p, top_k, repetition_penalty) live in `decoder:` and are shared across
all engines.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_tokens` | int | 0 | Minimum output tokens before EOS is allowed |
| `n` | int | 1 | Number of output sequences per prompt |
| `ignore_eos` | bool | false | Continue generating past EOS token (forces full `max_output_tokens` generation) |
| `return_perf_metrics` | bool | false | Return TRT-LLM internal performance metrics in output |

For advanced TRT-LLM parameters, see the [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/).

---

## Switching Between Engines

Change the `engine:` field and add the required runner config. The model and measurement
parameters stay the same.

```yaml
# Same experiment — Transformers (local)
model: gpt2
engine: transformers
n: 100
dtype: bfloat16
```

```yaml
# Same experiment — vLLM (Docker)
model: gpt2
engine: vllm
n: 100
dtype: bfloat16
runners:
  vllm: docker
```

```yaml
# Same experiment — TensorRT-LLM (Docker)
model: gpt2
engine: tensorrt
n: 100
dtype: bfloat16
runners:
  tensorrt: docker
```

Changing `engine:` switches the inference engine. Engine-specific sections (`pytorch:`,
`vllm:`, `tensorrt:`) are ignored when not running that engine. Universal parameters (`n`,
`dtype`, `decoder:`, etc.) apply to all engines.
---

## Runner Configuration

The runner determines where each engine executes. Configure runners in three ways
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
  transformers: local
  vllm: docker       # always use Docker for vLLM
```

**4. Default** — `local` for all engines if no runner is configured.

**Auto-elevation.** A study that mixes engines (e.g., `pytorch` local + `vllm` local) is
automatically elevated to use Docker for vLLM when Docker is available. This is logged at
info level and requires no user action.

**Multi-engine without Docker is an error.** If a study requires Docker runners and Docker
is not set up, `llem` fails at pre-flight with a clear error before running any experiments.

---

## Parameter Support Matrix

<!-- Parameter matrix — regenerate from GPU test results: uv run python scripts/generate_param_matrix.py -->

This matrix shows which parameters are supported by each engine. Derived from the Pydantic
config models. Full runtime verification (with GPU test results) can be generated with
`uv run python scripts/generate_param_matrix.py` after running GPU tests.

### Universal Parameters (all engines)

These parameters live in `ExperimentConfig` and are shared across all engines:

| Parameter | Transformers | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `model` | Yes | Yes | Yes | HuggingFace model ID or local path |
| `engine` | Yes | Yes | Yes | Selects the inference engine |
| `dataset.n_prompts` | Yes | Yes | Yes | Number of prompts |
| `dtype` | Yes | Yes | Yes | `fp32`, `fp16`, `bf16` |
| `dataset.source` | Yes | Yes | Yes | Dataset source (built-in alias or .jsonl path) |
| `max_input_tokens` | Yes | Yes | Yes | Input sequence length cap |
| `max_output_tokens` | Yes | Yes | Yes | Output token budget |
| `random_seed` | Yes | Yes | Yes | Per-experiment seed: inference RNG, dataset ordering |
| `decoder.temperature` | Yes | Yes | Yes | Sampling temperature |
| `decoder.top_p` | Yes | Yes | Yes | Nucleus sampling threshold |
| `decoder.top_k` | Yes | Yes | Yes | Top-k sampling (0 = disabled) |
| `decoder.repetition_penalty` | Yes | Yes | Yes | Repetition penalty |
| `decoder.preset` | Yes | Yes | Yes | `deterministic`, `creative`, `balanced` |

### Transformers-Specific Parameters

| Parameter | Transformers | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `transformers.batch_size` | Yes | N/A | N/A | Transformers batching |
| `transformers.attn_implementation` | Yes | N/A | N/A | Attention kernel selection |
| `transformers.torch_compile` | Yes | N/A | N/A | `torch.compile` acceleration |
| `transformers.load_in_4bit` | Yes | N/A | N/A | BitsAndBytes 4-bit quantization |
| `transformers.load_in_8bit` | Yes | N/A | N/A | BitsAndBytes 8-bit quantization |
| `transformers.device_map` | Yes | N/A | N/A | Device placement strategy |
| `transformers.num_beams` | Yes | N/A | N/A | Beam search width |
| `transformers.no_repeat_ngram_size` | Yes | N/A | N/A | Prevent n-gram repetition |
| `transformers.prompt_lookup_num_tokens` | Yes | N/A | N/A | Prompt-lookup speculative decoding |
| `transformers.tp_plan` | Yes | N/A | N/A | Native HF tensor parallelism plan |
| `transformers.tp_size` | Yes | N/A | N/A | Tensor parallel rank count |

### vLLM-Specific Parameters

| Parameter | Transformers | vLLM | TensorRT-LLM | Notes |
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

### TensorRT-LLM-Specific Parameters

| Parameter | Transformers | vLLM | TensorRT-LLM | Notes |
|-----------|---------|------|--------------|-------|
| `tensorrt.max_batch_size` | N/A | N/A | Yes | Compile-time constant |
| `tensorrt.tp_size` | N/A | N/A | Yes | Tensor parallel size (compile-time) |
| `tensorrt.max_input_len` | N/A | N/A | Yes | Max input tokens (compile-time) |
| `tensorrt.max_seq_len` | N/A | N/A | Yes | Max total sequence length (compile-time) |
| `tensorrt.dtype` | N/A | N/A | Yes | Model compute dtype (compile-time) |
| `tensorrt.fast_build` | N/A | N/A | Yes | Fast build mode (compile-time) |
| `tensorrt.engine_path` | N/A | N/A | Yes | Pre-compiled engine path |
| `tensorrt.quant.quant_algo` | N/A | N/A | Yes | FP8, INT8, W4A16_AWQ, W4A16_GPTQ, W8A16, etc. |
| `tensorrt.quant.kv_cache_quant_algo` | N/A | N/A | Yes | KV cache quantization: FP8 or INT8 |
| `tensorrt.kv_cache.free_gpu_memory_fraction` | N/A | N/A | Yes | KV cache memory fraction |
| `tensorrt.kv_cache.enable_block_reuse` | N/A | N/A | Yes | KV cache block reuse |
| `tensorrt.scheduler.capacity_scheduling_policy` | N/A | N/A | Yes | GUARANTEED_NO_EVICT / MAX_UTILIZATION / STATIC_BATCH |
| `tensorrt.build_cache.max_cache_storage_gb` | N/A | N/A | Yes | Engine cache size limit |
| `tensorrt.build_cache.cache_root` | N/A | N/A | Yes | Engine cache directory |
| `tensorrt.sampling.min_tokens` | N/A | N/A | Yes | Minimum output tokens |
| `tensorrt.sampling.ignore_eos` | N/A | N/A | Yes | Force full generation past EOS |
| `tensorrt.sampling.return_perf_metrics` | N/A | N/A | Yes | TRT-LLM internal perf metrics |
