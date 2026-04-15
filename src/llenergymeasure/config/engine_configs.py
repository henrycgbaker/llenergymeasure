"""Engine-specific configuration models (v2.0 schema).

Each engine section uses None-as-default: all fields default to None, meaning
"use the engine's own default at execution time". This makes it explicit when
a researcher has set a value versus when the engine's built-in default applies.

Usage in YAML:
    engine: transformers
    transformers:
      batch_size: 4
      load_in_4bit: true

    engine: vllm
    vllm:
      engine:
        enforce_eager: false
        gpu_memory_utilization: 0.9
        kv_cache_dtype: fp8
      sampling:
        presence_penalty: 0.0

    engine: tensorrt
    tensorrt:
      tensor_parallel_size: 2
      dtype: bfloat16
      quant:
        quant_algo: W4A16_AWQ
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Transformers Engine Configuration (v2.0)
# =============================================================================


class TransformersConfig(BaseModel):
    """HuggingFace Transformers engine configuration.

    All fields default to None — None means "use the engine's own default".
    This distinguishes explicit researcher choices from engine defaults,
    which is important for result reproducibility and experiment attribution.

    Fields cover the complete researcher-useful parameter space for
    AutoModelForCausalLM.from_pretrained() and model.generate().
    Unknown fields are forwarded to HuggingFace/transformers APIs via extra="allow".
    """

    model_config = {"extra": "allow"}

    # -------------------------------------------------------------------------
    # Batching
    # -------------------------------------------------------------------------

    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Batch size (None -> 1)",
    )

    # -------------------------------------------------------------------------
    # Attention implementation
    # -------------------------------------------------------------------------

    attn_implementation: (
        Literal["sdpa", "flash_attention_2", "flash_attention_3", "eager"] | None
    ) = Field(
        default=None,
        description="Attention implementation (None -> sdpa)",
    )

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    torch_compile: bool | None = Field(
        default=None,
        description="Enable torch.compile (None -> False)",
    )
    torch_compile_mode: str | None = Field(
        default=None,
        description="torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default')",
    )
    torch_compile_backend: str | None = Field(
        default=None,
        description="torch.compile backend (None -> 'inductor')",
    )

    # -------------------------------------------------------------------------
    # BitsAndBytes quantization
    # -------------------------------------------------------------------------

    load_in_4bit: bool | None = Field(
        default=None,
        description="BitsAndBytes 4-bit quantization",
    )
    load_in_8bit: bool | None = Field(
        default=None,
        description="BitsAndBytes 8-bit quantization",
    )
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] | None = Field(
        default=None,
        description="Compute dtype for 4-bit (None -> float32, usually want bfloat16)",
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] | None = Field(
        default=None,
        description="4-bit quantization type (None -> 'nf4')",
    )
    bnb_4bit_use_double_quant: bool | None = Field(
        default=None,
        description="Double quantization saves ~0.4 bits/param (None -> False)",
    )

    # -------------------------------------------------------------------------
    # KV caching
    # -------------------------------------------------------------------------

    use_cache: bool | None = Field(
        default=None,
        description="Use KV cache during generation (None -> True)",
    )
    cache_implementation: Literal["static", "offloaded_static", "sliding_window"] | None = Field(
        default=None,
        description="KV cache strategy; 'static' enables CUDA graphs (None -> dynamic)",
    )

    # -------------------------------------------------------------------------
    # Beam search
    # -------------------------------------------------------------------------

    num_beams: int | None = Field(
        default=None,
        ge=1,
        description="Beam search width (None -> 1, greedy/sampling)",
    )
    early_stopping: bool | None = Field(
        default=None,
        description="Stop beam search when all beams hit EOS (None -> False)",
    )
    length_penalty: float | None = Field(
        default=None,
        description="Beam length penalty: >1 shorter, <1 longer (None -> 1.0)",
    )

    # -------------------------------------------------------------------------
    # N-gram repetition
    # -------------------------------------------------------------------------

    no_repeat_ngram_size: int | None = Field(
        default=None,
        ge=0,
        description="Prevent n-gram repetition (None -> 0, disabled)",
    )

    # -------------------------------------------------------------------------
    # Speculative decoding (prompt-lookup — draft model via passthrough_kwargs)
    # -------------------------------------------------------------------------

    prompt_lookup_num_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Prompt-lookup speculative decoding tokens (None -> disabled)",
    )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    device_map: str | None = Field(
        default=None,
        description="Device placement strategy (None -> 'auto')",
    )
    max_memory: dict[str | int, str] | None = Field(
        default=None,
        description="Per-device memory limits, e.g. {0: '10GiB', 'cpu': '50GiB'}",
    )

    # -------------------------------------------------------------------------
    # Mixed precision
    # -------------------------------------------------------------------------

    allow_tf32: bool | None = Field(
        default=None,
        description="Allow TF32 on Ampere GPUs (None -> PyTorch default)",
    )
    autocast_enabled: bool | None = Field(
        default=None,
        description="Enable torch.autocast mixed precision (None -> False)",
    )
    autocast_dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description="torch.autocast dtype (None -> bfloat16 on Ampere)",
    )
    low_cpu_mem_usage: bool | None = Field(
        default=None,
        description="Low CPU memory usage during model loading (None -> False)",
    )

    # -------------------------------------------------------------------------
    # Tensor parallelism (HF Transformers >=4.50)
    # -------------------------------------------------------------------------

    tp_plan: Literal["auto"] | None = Field(
        default=None,
        description=(
            "Tensor parallelism plan for native HF TP (None -> disabled). "
            "Only 'auto' is currently supported by Transformers. "
            "Mutually exclusive with device_map. Requires torchrun launch."
        ),
    )
    tp_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of tensor parallel ranks (None -> WORLD_SIZE). Only used when tp_plan is set. "
            "Field name preserved to match HuggingFace accelerate convention "
            "(distinct from TensorRTConfig.tensor_parallel_size which aligns with TrtLlmArgs)."
        ),
    )

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_quantization(self) -> TransformersConfig:
        """4-bit and 8-bit quantization are mutually exclusive."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "Cannot use both load_in_4bit=True and load_in_8bit=True simultaneously"
            )
        return self

    @model_validator(mode="after")
    def validate_torch_compile_options(self) -> TransformersConfig:
        """torch_compile_mode/torch_compile_backend require torch_compile=True."""
        if (
            self.torch_compile_mode is not None or self.torch_compile_backend is not None
        ) and self.torch_compile is not True:
            raise ValueError("torch_compile_mode/torch_compile_backend requires torch_compile=True")
        return self

    @model_validator(mode="after")
    def validate_bnb_4bit_options(self) -> TransformersConfig:
        """bnb_4bit_* fields require load_in_4bit=True."""
        if (
            self.bnb_4bit_compute_dtype is not None
            or self.bnb_4bit_quant_type is not None
            or self.bnb_4bit_use_double_quant is not None
        ) and self.load_in_4bit is not True:
            raise ValueError("bnb_4bit_* fields require load_in_4bit=True")
        return self

    @model_validator(mode="after")
    def validate_cache_options(self) -> TransformersConfig:
        """cache_implementation requires use_cache to be True or None (not explicitly False)."""
        if self.cache_implementation is not None and self.use_cache is False:
            raise ValueError(
                "cache_implementation requires use_cache to be True or None (not explicitly False)"
            )
        return self

    @model_validator(mode="after")
    def validate_tp_device_map_exclusive(self) -> TransformersConfig:
        """tp_plan and device_map are mutually exclusive."""
        if self.tp_plan is not None and self.device_map is not None:
            raise ValueError(
                "tp_plan and device_map are mutually exclusive. "
                "Tensor parallelism uses its own device placement; remove device_map."
            )
        return self


# =============================================================================
# vLLM Engine Configuration
# =============================================================================


class VLLMSpeculativeConfig(BaseModel):
    """vLLM speculative-decoding configuration.

    Replaces flat ``speculative_model`` + ``num_speculative_tokens`` fields.
    Mirrors vLLM's native ``speculative_config`` dict shape.
    Unknown fields are forwarded via extra="allow" to vLLM.
    """

    model_config = {"extra": "allow"}  # mirror native shape; CI diff catches drift

    model: str | None = Field(
        default=None,
        description="Draft model name/path for speculative decoding.",
    )
    num_speculative_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Tokens to draft per speculative step.",
    )
    method: str | None = Field(
        default=None,
        description=(
            "Speculative-decoding method (e.g. 'draft_model', 'ngram', 'medusa', 'eagle'). "
            "Kept as str because the Literal has drifted across vLLM releases — verify against "
            "EngineArgs.speculative_config.method in the vendored schema before narrowing."
        ),
    )


class VLLMAttentionConfig(BaseModel):
    """vLLM attention implementation configuration.

    Nested under VLLMEngineConfig.attention. Mirrors vLLM's AttentionConfig.
    All fields default to None — None means "use vLLM's own default".
    Uses extra="allow" for forward compatibility with new vLLM attention options.
    """

    model_config = {"extra": "allow"}

    backend: str | None = Field(
        default=None,
        description="Attention backend: flash_attn, flashinfer, etc. (None -> auto).",
    )
    flash_attn_version: int | None = Field(
        default=None,
        description="Flash attention version (None -> auto).",
    )
    flash_attn_max_num_splits_for_cuda_graph: int | None = Field(
        default=None,
        description="Max splits for CUDA graph with flash attention (None -> auto).",
    )
    use_prefill_decode_attention: bool | None = Field(
        default=None,
        description="Use prefill-decode attention (None -> True).",
    )
    use_prefill_query_quantization: bool | None = Field(
        default=None,
        description="Quantize queries during prefill (None -> False).",
    )
    use_cudnn_prefill: bool | None = Field(
        default=None,
        description="Use cuDNN for prefill (None -> False).",
    )
    disable_flashinfer_prefill: bool | None = Field(
        default=None,
        description="Disable FlashInfer for prefill (None -> False).",
    )
    disable_flashinfer_q_quantization: bool | None = Field(
        default=None,
        description="Disable FlashInfer query quantization (None -> False).",
    )
    use_trtllm_attention: bool | None = Field(
        default=None,
        description="Use TensorRT-LLM attention backend (None -> False).",
    )
    use_trtllm_ragged_deepseek_prefill: bool | None = Field(
        default=None,
        description="Use TRT-LLM ragged DeepSeek prefill (None -> False).",
    )


class VLLMEngineConfig(BaseModel):
    """vLLM engine-level configuration (vllm.LLM() constructor arguments).

    All fields default to None — None means "use vLLM's own default".
    These parameters are loaded once at model initialisation time.
    Unknown fields are forwarded to vllm.LLM() via extra="allow".
    """

    model_config = {"extra": "allow"}

    # -------------------------------------------------------------------------
    # Memory management
    # -------------------------------------------------------------------------

    gpu_memory_utilization: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "GPU memory fraction for KV cache (None -> 0.9). Higher = more KV cache, less headroom."
        ),
    )
    swap_space: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "CPU swap space in GiB for KV cache offloading (None -> 4). "
            "Enables model weight offload to prevent OOM."
        ),
    )
    cpu_offload_gb: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "CPU RAM in GiB to offload model weights to (None -> 0, disabled). "
            "Reduces VRAM pressure at throughput cost."
        ),
    )

    # -------------------------------------------------------------------------
    # KV cache
    # -------------------------------------------------------------------------

    block_size: Literal[8, 16, 32] | None = Field(
        default=None,
        description=(
            "KV cache block size in tokens (None -> 16). "
            "Affects KV cache fragmentation and memory efficiency."
        ),
    )
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"] | None = Field(
        default=None,
        description=(
            "KV cache storage dtype (None -> auto = model dtype). "
            "fp8 variants halve KV cache VRAM on Ampere+."
        ),
    )

    # -------------------------------------------------------------------------
    # Execution mode
    # -------------------------------------------------------------------------

    enforce_eager: bool | None = Field(
        default=None,
        description=(
            "Disable CUDA graphs, always use eager mode (None -> False). "
            "Eager mode: predictable latency, no graph compilation overhead."
        ),
    )
    enable_chunked_prefill: bool | None = Field(
        default=None,
        description=(
            "Chunk large prefills across multiple scheduler iterations (None -> False). "
            "Affects scheduling latency and throughput."
        ),
    )

    # -------------------------------------------------------------------------
    # Scheduler / batching
    # -------------------------------------------------------------------------

    max_num_seqs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max concurrent sequences per scheduler iteration (None -> 256). "
            "Affects batch size and KV cache usage."
        ),
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max tokens processed per scheduler iteration (None -> auto). "
            "Controls per-step compute budget."
        ),
    )
    max_model_len: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max sequence length in tokens (input + output). "
            "Overrides model config (None -> model default). Caps KV cache allocation."
        ),
    )
    num_scheduler_steps: int | None = Field(
        default=None,
        ge=1,
        description="Number of scheduler steps per iteration (multi-step scheduling, None -> 1).",
    )

    # -------------------------------------------------------------------------
    # Parallelism
    # -------------------------------------------------------------------------

    tensor_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Tensor parallel degree — number of GPUs to shard the model across (None -> 1)."
        ),
    )
    pipeline_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=("Pipeline parallel stages — memory per GPU changes with PP (None -> 1)."),
    )
    distributed_executor_backend: Literal["mp", "ray"] | None = Field(
        default=None,
        description="Multi-GPU executor backend: 'mp' (multiprocessing) or 'ray' (None -> mp).",
    )

    # -------------------------------------------------------------------------
    # Prefix caching and quantization
    # -------------------------------------------------------------------------

    enable_prefix_caching: bool | None = Field(
        default=None,
        description="Automatic prefix caching for repeated shared prompts (None -> False).",
    )
    quantization: (
        Literal["awq", "gptq", "fp8", "fp8_e5m2", "fp8_e4m3", "marlin", "bitsandbytes"] | None
    ) = Field(
        default=None,
        description="Quantization method. Requires pre-quantized model checkpoint.",
    )

    # -------------------------------------------------------------------------
    # CUDA graphs
    # -------------------------------------------------------------------------

    max_seq_len_to_capture: int | None = Field(
        default=None,
        ge=1,
        description="Maximum sequence length for CUDA graph capture (None -> 8192).",
    )

    # -------------------------------------------------------------------------
    # Speculative decoding (typed nested sub-config)
    # -------------------------------------------------------------------------

    speculative: VLLMSpeculativeConfig | None = Field(
        default=None,
        description=(
            "Speculative decoding configuration. Replaces flat speculative_model / "
            "num_speculative_tokens fields. Mirrors vLLM native speculative_config shape."
        ),
    )

    # -------------------------------------------------------------------------
    # CPU offload
    # -------------------------------------------------------------------------

    offload_group_size: int | None = Field(
        default=None,
        ge=0,
        description="Groups of layers for CPU offloading (None -> 0).",
    )
    offload_num_in_group: int | None = Field(
        default=None,
        ge=1,
        description="Number of layers offloaded per group (None -> 1).",
    )
    offload_prefetch_step: int | None = Field(
        default=None,
        ge=0,
        description="Prefetch steps ahead for CPU offload (None -> 1).",
    )
    offload_params: list[str] | None = Field(
        default=None,
        description="Specific parameter names to offload to CPU (None -> all eligible).",
    )

    # -------------------------------------------------------------------------
    # Multi-GPU
    # -------------------------------------------------------------------------

    disable_custom_all_reduce: bool | None = Field(
        default=None,
        description="Disable custom all-reduce for multi-GPU (None -> False).",
    )

    # -------------------------------------------------------------------------
    # KV cache (absolute)
    # -------------------------------------------------------------------------

    kv_cache_memory_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Absolute KV cache size in bytes (None -> use gpu_memory_utilization). "
            "Mutually exclusive with gpu_memory_utilization."
        ),
    )

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    compilation_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Full passthrough to vLLM CompilationConfig (~30 fields). "
            "No validation — passed directly."
        ),
    )

    # -------------------------------------------------------------------------
    # Attention
    # -------------------------------------------------------------------------

    attention: VLLMAttentionConfig | None = Field(
        default=None,
        description="Attention implementation configuration.",
    )

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_kv_cache_memory(self) -> VLLMEngineConfig:
        """kv_cache_memory_bytes and gpu_memory_utilization are mutually exclusive."""
        if self.kv_cache_memory_bytes is not None and self.gpu_memory_utilization is not None:
            raise ValueError(
                "kv_cache_memory_bytes and gpu_memory_utilization are mutually exclusive. "
                "Use one or the other to control KV cache memory."
            )
        return self

    @model_validator(mode="after")
    def validate_batched_tokens_vs_model_len(self) -> VLLMEngineConfig:
        """max_num_batched_tokens must be >= max_model_len when both are set."""
        if (
            self.max_num_batched_tokens is not None
            and self.max_model_len is not None
            and self.max_num_batched_tokens < self.max_model_len
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_model_len ({self.max_model_len}). "
                "vLLM requires the batched token budget to accommodate at least one full sequence."
            )
        return self


class VLLMSamplingConfig(BaseModel):
    """vLLM sampling-level configuration (vllm.SamplingParams extensions).

    Only vLLM-specific sampling parameters are included here.
    Universal sampling params (temperature, top_p, top_k, repetition_penalty)
    live in DecoderConfig and are shared across all engines.

    All fields default to None — None means "use vLLM's own default".
    Unknown fields are forwarded to vllm.SamplingParams() via extra="allow".

    Note: max_tokens is intentionally absent — it is bridged from
    ExperimentConfig.max_output_tokens in _build_sampling_kwargs().
    """

    model_config = {"extra": "allow"}

    min_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Minimum output tokens before EOS is allowed (None -> 0, no minimum).",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Presence penalty: penalises tokens that appear at all (None -> 0.0). "
            "Affects generation diversity."
        ),
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Frequency penalty: penalises tokens proportional to frequency (None -> 0.0). "
            "Affects repetition."
        ),
    )
    ignore_eos: bool | None = Field(
        default=None,
        description=(
            "Continue generating past EOS token (None -> False). "
            "Forces max_tokens generation every time — affects total token count."
        ),
    )
    n: int | None = Field(
        default=None,
        ge=1,
        description="Number of output sequences per prompt (None -> 1).",
    )


class VLLMBeamSearchConfig(BaseModel):
    """vLLM beam search configuration.

    When set, the engine uses BeamSearchParams instead of SamplingParams.
    Nested under VLLMConfig.beam_search.
    All fields default to None — None means "use vLLM's own default".
    Uses extra="allow" for forward compatibility with new vLLM beam search options.

    Note: max_tokens is intentionally absent — it is bridged from
    ExperimentConfig.max_output_tokens in _build_beam_search_kwargs().
    """

    model_config = {"extra": "allow"}

    beam_width: int | None = Field(
        default=None,
        ge=1,
        description="Number of beams (ge=1).",
    )
    length_penalty: float | None = Field(
        default=None,
        description="Length penalty: >1 favours shorter, <1 longer (None -> 1.0).",
    )
    early_stopping: bool | None = Field(
        default=None,
        description="Stop when beam_width complete sequences found (None -> False).",
    )


class VLLMConfig(BaseModel):
    """vLLM engine configuration.

    Nested structure mirrors vLLM's own two-API separation:
    - engine: vllm.LLM() constructor arguments (engine-level, loaded at model init)
    - sampling: vllm.SamplingParams arguments (vLLM-specific extensions only)

    Universal sampling params (temperature, top_p, top_k, repetition_penalty)
    live in DecoderConfig and are shared across all engines.

    Example YAML:
        engine: vllm
        vllm:
          engine:
            enforce_eager: false
            gpu_memory_utilization: 0.9
            kv_cache_dtype: fp8
            speculative:
              model: gpt2
              num_speculative_tokens: 3
          sampling:
            presence_penalty: 0.0
    """

    model_config = {"extra": "allow"}

    engine: VLLMEngineConfig | None = Field(
        default=None,
        description="Engine-level configuration: vllm.LLM() constructor args",
    )
    sampling: VLLMSamplingConfig | None = Field(
        default=None,
        description=(
            "Sampling-level configuration: vllm.SamplingParams extensions (vLLM-specific only)"
        ),
    )
    beam_search: VLLMBeamSearchConfig | None = Field(
        default=None,
        description=(
            "Beam search configuration. When set, uses BeamSearchParams instead of SamplingParams. "
            "Mutually exclusive with sampling section."
        ),
    )

    @model_validator(mode="after")
    def validate_beam_search_exclusive(self) -> VLLMConfig:
        """beam_search and sampling sections are mutually exclusive."""
        if self.beam_search is not None and self.sampling is not None:
            raise ValueError(
                "Cannot use both beam_search and sampling sections simultaneously. "
                "beam_search uses BeamSearchParams; sampling uses SamplingParams."
            )
        return self


# =============================================================================
# TensorRT-LLM Engine Configuration
# =============================================================================


class TensorRTQuantConfig(BaseModel):
    """TRT-LLM quantisation configuration.

    Maps to tensorrt_llm.llmapi.QuantConfig. Uses native QuantAlgo enum names
    as Literal values (not custom string aliases).
    All fields default to None - None means "use TRT-LLM's own default".
    """

    model_config = {"extra": "allow"}

    quant_algo: (
        Literal[
            "FP8",
            "INT8",
            "W4A16_AWQ",
            "W4A16_GPTQ",
            "W8A16",
            "W8A16_GPTQ",
            "W4A8_AWQ",
            "NO_QUANT",
        ]
        | None
    ) = Field(
        default=None,
        description="Quantisation algorithm (native QuantAlgo enum name)",
    )
    kv_cache_quant_algo: Literal["FP8", "INT8"] | None = Field(
        default=None,
        description="KV cache quantisation algorithm (None -> no KV cache quantisation)",
    )


class TensorRTKvCacheConfig(BaseModel):
    """TRT-LLM KV cache configuration."""

    model_config = {"extra": "allow"}

    enable_block_reuse: bool | None = Field(
        default=None,
        description="Enable KV cache block reuse (None -> False)",
    )
    free_gpu_memory_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of free GPU memory for KV cache (None -> 0.9)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens in KV cache (None -> auto)",
    )
    host_cache_size: int | None = Field(
        default=None,
        ge=0,
        description="Host (CPU) cache size in bytes for KV cache offloading (None -> 0, disabled)",
    )


class TensorRTSchedulerConfig(BaseModel):
    """TRT-LLM scheduler configuration."""

    model_config = {"extra": "allow"}

    capacity_scheduling_policy: (
        Literal[
            "GUARANTEED_NO_EVICT",
            "MAX_UTILIZATION",
            "STATIC_BATCH",
        ]
        | None
    ) = Field(
        default=None,
        description="Scheduling capacity policy (None -> GUARANTEED_NO_EVICT)",
    )


class TensorRTSamplingConfig(BaseModel):
    """TRT-LLM sampling configuration.

    Maps to tensorrt_llm.SamplingParams (TRT-LLM-specific extensions only).
    Universal sampling params (temperature, top_p, top_k, repetition_penalty)
    live in DecoderConfig and are shared across all engines.

    Note: return_perf_metrics dropped (D1 observability flag).
    """

    model_config = {"extra": "allow"}

    min_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Minimum output tokens before EOS allowed (None -> 0)",
    )
    n: int | None = Field(
        default=None,
        ge=1,
        description="Number of output sequences per prompt (None -> 1)",
    )
    ignore_eos: bool | None = Field(
        default=None,
        description="Continue generating past EOS token (None -> False)",
    )


class TensorRTConfig(BaseModel):
    """TensorRT-LLM engine configuration.

    All fields default to None - None means "use TRT-LLM's own default".
    TensorRT requires engine compilation; max_batch_size, max_input_len,
    max_seq_len are compile-time constants that affect engine shape.

    Nested structure mirrors TRT-LLM's own API separation:
    - quant: QuantConfig (quantisation algorithm + KV cache quantisation)
    - kv_cache: KvCacheConfig (block reuse, memory fraction)
    - scheduler: SchedulerConfig (capacity policy)
    - sampling: SamplingParams (TRT-LLM-specific extensions only)

    Universal sampling params (temperature, top_p, top_k, repetition_penalty)
    live in DecoderConfig and are shared across all engines.

    Dropped (falls through extra="allow"):
    - engine_path — D1 deployment path, not a measurement axis
    - calib sub-config — D3 build-only PTQ calibration (we consume pre-quantised checkpoints)
    - build_cache sub-config — D1 engine-cache housekeeping

    Re-added after audit:
    - backend: Literal["trt","pytorch","_autodeploy"] — measurement-relevant
      axis in TRT-LLM >=0.13. "trt" is the AOT-compiled engine; "pytorch" is
      TRT-LLM's eager runtime (same scheduler/KV cache, no compile); "_autodeploy"
      is the experimental model-porter. Original drop rubric (D2 single-option
      enum) was incorrect for contemporary TRT-LLM.

    Example YAML:
        engine: tensorrt
        tensorrt:
          tensor_parallel_size: 2
          max_batch_size: 8
          dtype: bfloat16
          quant:
            quant_algo: W4A16_AWQ
    """

    model_config = {"extra": "allow"}

    # -------------------------------------------------------------------------
    # Compile-time parameters (LLM() constructor)
    # -------------------------------------------------------------------------

    max_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Max batch size (compile-time constant, None -> 8)",
    )
    tensor_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Tensor parallel size — number of GPUs to shard across (None -> 1). "
            "Aligns with TrtLlmArgs.tensor_parallel_size. "
            "Note: TransformersConfig.tp_size follows accelerate convention and is preserved."
        ),
    )
    pipeline_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description="Pipeline parallel stages (None -> 1).",
    )
    max_input_len: int | None = Field(
        default=None,
        ge=1,
        description="Max input sequence length (compile-time constant, None -> 1024)",
    )
    max_seq_len: int | None = Field(
        default=None,
        ge=1,
        description="Max total sequence length (input + output, compile-time constant, None -> 2048)",
    )
    max_num_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens the engine can handle per iteration (None -> auto).",
    )
    dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description=(
            "Model dtype (None -> auto). TRT-LLM is optimised for fp16/bf16; fp32 not supported."
        ),
    )
    fast_build: bool | None = Field(
        default=None,
        description="Enable fast engine build mode (reduced optimisation, None -> False)",
    )
    backend: Literal["trt", "pytorch", "_autodeploy"] | None = Field(
        default=None,
        description=(
            "TRT-LLM runtime backend — a measurement axis, not a per-host knob. "
            "'trt' = AOT-compiled TensorRT engine (best steady-state, minutes-hours "
            "compile); 'pytorch' = TRT-LLM's eager runtime (same scheduler/KV cache, "
            "no compile, supports newer model archs without hand-written converters); "
            "'_autodeploy' = experimental autoporter. None -> TRT-LLM auto-picks "
            "(respects TLLM_USE_TRT_ENGINE env)."
        ),
    )

    # -------------------------------------------------------------------------
    # Nested sub-configs
    # -------------------------------------------------------------------------

    quant: TensorRTQuantConfig | None = Field(
        default=None,
        description="Quantisation configuration (QuantConfig)",
    )
    kv_cache: TensorRTKvCacheConfig | None = Field(
        default=None,
        description="KV cache configuration",
    )
    scheduler: TensorRTSchedulerConfig | None = Field(
        default=None,
        description="Scheduler configuration",
    )
    sampling: TensorRTSamplingConfig | None = Field(
        default=None,
        description="Sampling configuration (TRT-LLM-specific SamplingParams extensions)",
    )
