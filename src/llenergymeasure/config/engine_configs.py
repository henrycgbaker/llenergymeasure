"""Engine-specific configuration models (v2.0 schema).

Each engine section uses None-as-default: all fields default to None, meaning
"use the engine's own default at execution time". This makes it explicit when
a researcher has set a value versus when the engine's built-in default applies.

Every typed field carries ``CurationMetadata`` via ``json_schema_extra``.
This is the machine-readable rubric verdict used by the generated
inclusion/exclusion table (Plan E). The rubric is defined in
``.product/research/parameter-curation-rubric.md``.

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

from llenergymeasure.config.curation import CurationMetadata

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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="Universal measurement axis; MLPerf, Optimum, LLM-Perf all split by it.",
            native_mapping="AutoModelForCausalLM.generate batch dimension",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Attention implementation
    # -------------------------------------------------------------------------

    attn_implementation: (
        Literal["sdpa", "flash_attention_2", "flash_attention_3", "eager"] | None
    ) = Field(
        default=None,
        description="Attention implementation (None -> sdpa)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2", "E3"),
            rationale="Kernel selection is a latency/energy regime switch; Optimum types it.",
            native_mapping="AutoModelForCausalLM.from_pretrained(attn_implementation=...)",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    torch_compile: bool | None = Field(
        default=None,
        description="Enable torch.compile (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Compile vs eager is a regime switch; Optimum/Databricks report splits.",
            native_mapping="torch.compile(model)",
        ).to_schema_extra(),
    )
    torch_compile_mode: str | None = Field(
        default=None,
        description="torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default')",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E3"),
            rationale="3-option enum with meaningful throughput deltas (default vs reduce-overhead vs max-autotune).",
            native_mapping="torch.compile(model, mode=...)",
        ).to_schema_extra(),
    )
    torch_compile_backend: str | None = Field(
        default=None,
        description="torch.compile backend (None -> 'inductor')",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E3"),
            rationale="Compile-backend selects different codegen → different kernels → measurable energy delta.",
            native_mapping="torch.compile(model, backend=...)",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # BitsAndBytes quantization
    # -------------------------------------------------------------------------

    load_in_4bit: bool | None = Field(
        default=None,
        description="BitsAndBytes 4-bit quantization",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Regime switch — 4-bit is its own measurement mode.",
            native_mapping="BitsAndBytesConfig(load_in_4bit=True)",
        ).to_schema_extra(),
    )
    load_in_8bit: bool | None = Field(
        default=None,
        description="BitsAndBytes 8-bit quantization",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Regime switch — 8-bit is its own measurement mode.",
            native_mapping="BitsAndBytesConfig(load_in_8bit=True)",
        ).to_schema_extra(),
    )
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] | None = Field(
        default=None,
        description="Compute dtype for 4-bit (None -> float32, usually want bfloat16)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E3"),
            rationale="Compute-path selection inside 4-bit; affects energy directly.",
            native_mapping="BitsAndBytesConfig(bnb_4bit_compute_dtype=...)",
        ).to_schema_extra(),
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] | None = Field(
        default=None,
        description="4-bit quantization type (None -> 'nf4')",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E3"),
            rationale="nf4 vs fp4 selects different dequantisation kernels → different energy/token.",
            native_mapping="BitsAndBytesConfig(bnb_4bit_quant_type=...)",
        ).to_schema_extra(),
    )
    bnb_4bit_use_double_quant: bool | None = Field(
        default=None,
        description="Double quantization saves ~0.4 bits/param (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Double quantisation changes memory footprint and dequant arithmetic — regime toggle.",
            native_mapping="BitsAndBytesConfig(bnb_4bit_use_double_quant=...)",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # KV caching
    # -------------------------------------------------------------------------

    use_cache: bool | None = Field(
        default=None,
        description="Use KV cache during generation (None -> True)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Turning it off changes inference from AR to prefill-only (~100× latency delta).",
            native_mapping="GenerationConfig.use_cache",
        ).to_schema_extra(),
    )
    cache_implementation: Literal["static", "offloaded_static", "sliding_window"] | None = Field(
        default=None,
        description="KV cache strategy; 'static' enables CUDA graphs (None -> dynamic)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "E3"),
            rationale="'static' enables CUDA graphs — regime-changing; stable 3-value enum.",
            native_mapping="AutoModelForCausalLM.from_pretrained(cache_implementation=...)",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Beam search
    # -------------------------------------------------------------------------

    num_beams: int | None = Field(
        default=None,
        ge=1,
        description="Beam search width (None -> 1, greedy/sampling)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1", "E2"),
            rationale="Linear compute scaling; beam search is its own measurement regime.",
            native_mapping="GenerationConfig.num_beams",
        ).to_schema_extra(),
    )
    early_stopping: bool | None = Field(
        default=None,
        description="Stop beam search when all beams hit EOS (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Couples to beam search; affects final token count.",
            native_mapping="GenerationConfig.early_stopping",
        ).to_schema_extra(),
    )
    length_penalty: float | None = Field(
        default=None,
        description="Beam length penalty: >1 shorter, <1 longer (None -> 1.0)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Tunes beam-search sequence length → FLOPs.",
            native_mapping="GenerationConfig.length_penalty",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # N-gram repetition
    # -------------------------------------------------------------------------

    no_repeat_ngram_size: int | None = Field(
        default=None,
        ge=0,
        description="Prevent n-gram repetition (None -> 0, disabled)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Affects generation length via repetition prevention → indirect throughput/energy delta.",
            native_mapping="GenerationConfig.no_repeat_ngram_size",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Speculative decoding (prompt-lookup — draft model via passthrough_kwargs)
    # -------------------------------------------------------------------------

    prompt_lookup_num_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Prompt-lookup speculative decoding tokens (None -> disabled)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="HF's speculative-decoding flavour (prompt-lookup); regime switch.",
            native_mapping="GenerationConfig.prompt_lookup_num_tokens",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    device_map: str | None = Field(
        default=None,
        description="Device placement strategy (None -> 'auto')",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Offload strategy; Optimum types it. Mutually exclusive with tp_plan.",
            native_mapping="AutoModelForCausalLM.from_pretrained(device_map=...)",
        ).to_schema_extra(),
    )
    max_memory: dict[str | int, str] | None = Field(
        default=None,
        description="Per-device memory limits, e.g. {0: '10GiB', 'cpu': '50GiB'}",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "T4"),
            rationale=(
                "Sets per-device memory budget; once exceeded forces weight offload CPU↔GPU "
                "— major energy/latency regime change. Dict passthrough (T4)."
            ),
            native_mapping="AutoModelForCausalLM.from_pretrained(max_memory=...)",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Mixed precision
    # -------------------------------------------------------------------------

    allow_tf32: bool | None = Field(
        default=None,
        description="Allow TF32 on Ampere GPUs (None -> PyTorch default)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Ampere TF32 toggle; accuracy/throughput tradeoff affecting GEMM energy.",
            native_mapping="torch.backends.cuda.matmul.allow_tf32",
        ).to_schema_extra(),
    )
    autocast_enabled: bool | None = Field(
        default=None,
        description="Enable torch.autocast mixed precision (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Mixed-precision runtime switch; determines whether AMP is applied during generation.",
            native_mapping="torch.autocast(enabled=...)",
        ).to_schema_extra(),
    )
    autocast_dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description="torch.autocast dtype (None -> bfloat16 on Ampere)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E3"),
            rationale="AMP compute dtype selector; different dtypes have distinct GEMM throughput.",
            native_mapping="torch.autocast(dtype=...)",
        ).to_schema_extra(),
    )
    low_cpu_mem_usage: bool | None = Field(
        default=None,
        description="Low CPU memory usage during model loading (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Affects load-time memory footprint; can trigger weight offload regime.",
            native_mapping="AutoModelForCausalLM.from_pretrained(low_cpu_mem_usage=...)",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Tensor parallelism enable; engine-native HF TP regime toggle.",
            native_mapping="AutoModelForCausalLM.from_pretrained(tp_plan=...)",
        ).to_schema_extra(),
    )
    tp_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of tensor parallel ranks (None -> WORLD_SIZE). Only used when tp_plan is set. "
            "Follows accelerate convention; HF has no single native 'tensor_parallel_size' equivalent."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="TP degree — universal measurement axis. Preserved as tp_size (accelerate convention).",
            native_mapping="accelerate tp_size",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Draft model identity — primary axis for speculative-decoding measurement.",
            native_mapping="EngineArgs.speculative_config.model",
        ).to_schema_extra(),
    )
    num_speculative_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Tokens to draft per speculative step.",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1", "E2"),
            rationale="Draft-token count determines speculation depth; strongly peer-validated.",
            native_mapping="EngineArgs.speculative_config.num_speculative_tokens",
        ).to_schema_extra(),
    )
    method: str | None = Field(
        default=None,
        description=(
            "Speculative-decoding method (e.g. 'draft_model', 'ngram', 'medusa', 'eagle'). "
            "Kept as str because the Literal has drifted across vLLM releases — verify against "
            "EngineArgs.speculative_config.method in the vendored schema before narrowing."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "E3"),
            rationale="Method enum selects qualitatively different speculative-decoding regimes.",
            native_mapping="EngineArgs.speculative_config.method",
            notes="Kept str | None; narrow to Literal once vendored vLLM schema confirms stable values.",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1", "E3"),
            rationale="Attention kernel selection (flash_attn, flashinfer, triton, xformers); peer-validated.",
            native_mapping="AttentionConfig.backend",
            notes="Field name 'backend' preserved — sub-config is an abstraction layer, not a 1:1 EngineArgs mirror.",
        ).to_schema_extra(),
    )
    flash_attn_version: int | None = Field(
        default=None,
        description="Flash attention version (None -> auto).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="v2 vs v3 differ in kernel structure → measurable energy delta.",
            native_mapping="AttentionConfig.flash_attn_version",
        ).to_schema_extra(),
    )
    flash_attn_max_num_splits_for_cuda_graph: int | None = Field(
        default=None,
        description="Max splits for CUDA graph with flash attention (None -> auto).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Controls CUDA-graph tiling of attention → throughput/energy tradeoff for long sequences.",
            native_mapping="AttentionConfig.flash_attn_max_num_splits_for_cuda_graph",
        ).to_schema_extra(),
    )
    use_prefill_decode_attention: bool | None = Field(
        default=None,
        description="Use prefill-decode attention (None -> True).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Routes prefill through decode kernel — different compute path, regime toggle.",
            native_mapping="AttentionConfig.use_prefill_decode_attention",
        ).to_schema_extra(),
    )
    use_prefill_query_quantization: bool | None = Field(
        default=None,
        description="Quantize queries during prefill (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Quantises query tensor during prefill → energy/memory delta, regime toggle.",
            native_mapping="AttentionConfig.use_prefill_query_quantization",
        ).to_schema_extra(),
    )
    use_cudnn_prefill: bool | None = Field(
        default=None,
        description="Use cuDNN for prefill (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Selects cuDNN vs FlashAttention for prefill — distinct kernel paths.",
            native_mapping="AttentionConfig.use_cudnn_prefill",
        ).to_schema_extra(),
    )
    disable_flashinfer_prefill: bool | None = Field(
        default=None,
        description="Disable FlashInfer for prefill (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Forces fallback when flashinfer prefill is active — changes kernel used.",
            native_mapping="AttentionConfig.disable_flashinfer_prefill",
        ).to_schema_extra(),
    )
    disable_flashinfer_q_quantization: bool | None = Field(
        default=None,
        description="Disable FlashInfer query quantization (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Disables an optimisation; straightforward energy/latency axis.",
            native_mapping="AttentionConfig.disable_flashinfer_q_quantization",
        ).to_schema_extra(),
    )
    use_trtllm_attention: bool | None = Field(
        default=None,
        description="Use TensorRT-LLM attention backend (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Routes attention through TRT-LLM kernel — regime-changing kernel path switch.",
            native_mapping="AttentionConfig.use_trtllm_attention",
        ).to_schema_extra(),
    )
    use_trtllm_ragged_deepseek_prefill: bool | None = Field(
        default=None,
        description="Use TRT-LLM ragged DeepSeek prefill (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="TRT-LLM ragged prefill for DeepSeek architectures — distinct kernel path.",
            native_mapping="AttentionConfig.use_trtllm_ragged_deepseek_prefill",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="Primary KV-cache sizing knob; MLPerf/AMD flag it. Mutually exclusive with kv_cache_memory_bytes.",
            native_mapping="EngineArgs.gpu_memory_utilization",
        ).to_schema_extra(),
    )
    swap_space: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "CPU swap space in GiB for KV cache offloading (None -> 4). "
            "Enables model weight offload to prevent OOM."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="CPU-swap budget for KV cache — when exercised shifts KV blocks CPU↔GPU, real energy/latency regime.",
            native_mapping="EngineArgs.swap_space",
        ).to_schema_extra(),
    )
    cpu_offload_gb: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "CPU RAM in GiB to offload model weights to (None -> 0, disabled). "
            "Reduces VRAM pressure at throughput cost."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Weight-offload regime — Optimum/Zhou flag offloading as a class.",
            native_mapping="EngineArgs.cpu_offload_gb",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1", "E3"),
            rationale="vLLM paper + KV survey; tractable 3-value enum.",
            native_mapping="EngineArgs.block_size",
        ).to_schema_extra(),
    )
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"] | None = Field(
        default=None,
        description=(
            "KV cache storage dtype (None -> auto = model dtype). "
            "fp8 variants halve KV cache VRAM on Ampere+."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "T3", "E3"),
            rationale=(
                "Energy-NVML research flags as bias source (T3); MLPerf/AMD/vLLM/TRT all report it."
            ),
            native_mapping="EngineArgs.kv_cache_dtype",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="CUDA-graph regime switch.",
            native_mapping="EngineArgs.enforce_eager",
        ).to_schema_extra(),
    )
    enable_chunked_prefill: bool | None = Field(
        default=None,
        description=(
            "Chunk large prefills across multiple scheduler iterations (None -> False). "
            "Affects scheduling latency and throughput."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Scheduling regime switch; strong academic coverage (Yu/Orca, vLLM, Databricks).",
            native_mapping="EngineArgs.enable_chunked_prefill",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="Primary scheduler axis in AMD-MLPerf vLLM submissions.",
            native_mapping="EngineArgs.max_num_seqs",
        ).to_schema_extra(),
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max tokens processed per scheduler iteration (None -> auto). "
            "Controls per-step compute budget."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="Primary scheduler axis; controls per-step compute budget.",
            native_mapping="EngineArgs.max_num_batched_tokens",
        ).to_schema_extra(),
    )
    max_model_len: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max sequence length in tokens (input + output). "
            "Overrides model config (None -> model default). Caps KV cache allocation."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="MLPerf core knob; caps KV-cache pre-allocation.",
            native_mapping="EngineArgs.max_model_len",
        ).to_schema_extra(),
    )
    num_scheduler_steps: int | None = Field(
        default=None,
        ge=1,
        description="Number of scheduler steps per iteration (multi-step scheduling, None -> 1).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="AMD MLPerf v5.1 multi-step scheduling axis.",
            native_mapping="EngineArgs.num_scheduler_steps",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "T3"),
            rationale="Universal + energy-sampler-relevant (T3: must appear in results metadata).",
            native_mapping="EngineArgs.tensor_parallel_size",
        ).to_schema_extra(),
    )
    pipeline_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=("Pipeline parallel stages — memory per GPU changes with PP (None -> 1)."),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="NVIDIA MLPerf uses it; academic coverage.",
            native_mapping="EngineArgs.pipeline_parallel_size",
        ).to_schema_extra(),
    )
    distributed_executor_backend: Literal["mp", "ray"] | None = Field(
        default=None,
        description="Multi-GPU executor backend: 'mp' (multiprocessing) or 'ray' (None -> mp).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Multi-GPU dispatch regime toggle (mp vs ray) — different overhead profiles.",
            native_mapping="EngineArgs.distributed_executor_backend",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Prefix caching and quantization
    # -------------------------------------------------------------------------

    enable_prefix_caching: bool | None = Field(
        default=None,
        description="Automatic prefix caching for repeated shared prompts (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="Strong academic + vLLM-paper evidence (SGLang RadixAttention parallel).",
            native_mapping="EngineArgs.enable_prefix_caching",
        ).to_schema_extra(),
    )
    quantization: (
        Literal["awq", "gptq", "fp8", "fp8_e5m2", "fp8_e4m3", "marlin", "bitsandbytes"] | None
    ) = Field(
        default=None,
        description="Quantization method. Requires pre-quantized model checkpoint.",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2", "E3"),
            rationale="Universal measurement axis; enum-validated.",
            native_mapping="EngineArgs.quantization",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # CUDA graphs
    # -------------------------------------------------------------------------

    max_seq_len_to_capture: int | None = Field(
        default=None,
        ge=1,
        description="Maximum sequence length for CUDA graph capture (None -> 8192).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="CUDA-graph budget for long sequences; AMD MLPerf-derived.",
            native_mapping="EngineArgs.max_seq_len_to_capture",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1", "E2", "T5"),
            rationale="Regime switch; typed sub-config for inner-field sweep discoverability (T5).",
            native_mapping="EngineArgs.speculative_config",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # CPU offload
    # -------------------------------------------------------------------------

    offload_group_size: int | None = Field(
        default=None,
        ge=0,
        description="Groups of layers for CPU offloading (None -> 0).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="CPU-offload grouping granularity — larger groups trade latency for throughput.",
            native_mapping="EngineArgs.offload_group_size",
        ).to_schema_extra(),
    )
    offload_num_in_group: int | None = Field(
        default=None,
        ge=1,
        description="Number of layers offloaded per group (None -> 1).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Controls how many params are offloaded per group — shifts memory/compute boundary.",
            native_mapping="EngineArgs.offload_num_in_group",
        ).to_schema_extra(),
    )
    offload_prefetch_step: int | None = Field(
        default=None,
        ge=0,
        description="Prefetch steps ahead for CPU offload (None -> 1).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Prefetch lookahead — directly affects hide-latency effectiveness of CPU offload.",
            native_mapping="EngineArgs.offload_prefetch_step",
        ).to_schema_extra(),
    )
    offload_params: list[str] | None = Field(
        default=None,
        description="Specific parameter names to offload to CPU (None -> all eligible).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Selects which parameter classes are offloaded — shifts the memory/compute boundary.",
            native_mapping="EngineArgs.offload_params",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Multi-GPU
    # -------------------------------------------------------------------------

    disable_custom_all_reduce: bool | None = Field(
        default=None,
        description="Disable custom all-reduce for multi-GPU (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Forces fallback to NCCL-native all-reduce — affects collective-comm latency/energy on multi-GPU.",
            native_mapping="EngineArgs.disable_custom_all_reduce",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E4"),
            rationale=(
                "Alternative to gpu_memory_utilization; parse-time mutex already wired. "
                "⚠ Verify validate_kv_cache_memory still correct after Phase 50."
            ),
            native_mapping=None,
            notes="Phase 50 boundary: do not move/modify validate_kv_cache_memory in C.2.",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "T4"),
            rationale=(
                "~30-field dict controlling CUDA-graph capture, fusions, what gets compiled "
                "— major energy implications when varied. Opaque dict (T4)."
            ),
            native_mapping="EngineArgs.compilation_config",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Attention
    # -------------------------------------------------------------------------

    attention: VLLMAttentionConfig | None = Field(
        default=None,
        description="Attention implementation configuration.",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="Attention backend and kernel toggles; nested for future flashinfer/cuDNN additions.",
            native_mapping="EngineArgs.attention_backend (via sub-config mapping)",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale=(
                "Minimum output tokens before EOS; future native slot for DecoderConfig.min_new_tokens "
                "once Phase 49 migrates decoder per-engine."
            ),
            native_mapping="SamplingParams.min_tokens",
        ).to_schema_extra(),
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Presence penalty: penalises tokens that appear at all (None -> 0.0). "
            "Affects generation diversity."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="vLLM/TRT-native; not in DecoderConfig; routine in vLLM sampling studies.",
            native_mapping="SamplingParams.presence_penalty",
        ).to_schema_extra(),
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description=(
            "Frequency penalty: penalises tokens proportional to frequency (None -> 0.0). "
            "Affects repetition."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="vLLM/TRT-native; not in DecoderConfig.",
            native_mapping="SamplingParams.frequency_penalty",
        ).to_schema_extra(),
    )
    ignore_eos: bool | None = Field(
        default=None,
        description=(
            "Continue generating past EOS token (None -> False). "
            "Forces max_tokens generation every time — affects total token count."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Forces fixed output length — measurement workload control.",
            native_mapping="SamplingParams.ignore_eos",
        ).to_schema_extra(),
    )
    n: int | None = Field(
        default=None,
        ge=1,
        description="Number of output sequences per prompt (None -> 1).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="n-sequences multiplier; vLLM-bench first-class.",
            native_mapping="SamplingParams.n",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="Linear compute scaling; beam search is its own measurement regime.",
            native_mapping="BeamSearchParams.beam_width",
        ).to_schema_extra(),
    )
    length_penalty: float | None = Field(
        default=None,
        description="Length penalty: >1 favours shorter, <1 longer (None -> 1.0).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Tunes beam-search output length.",
            native_mapping="BeamSearchParams.length_penalty",
        ).to_schema_extra(),
    )
    early_stopping: bool | None = Field(
        default=None,
        description="Stop when beam_width complete sequences found (None -> False).",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="Affects final token count.",
            native_mapping="BeamSearchParams.early_stopping",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2", "E3", "E4"),
            rationale="Universal axis + FP8-on-A100 hardware rail (E4); enum-validated.",
            native_mapping="QuantConfig.quant_algo",
            notes="FP8 requires SM >= 8.9 (Ada Lovelace+). A100 (SM80): use INT8/W4A16_AWQ/W8A16.",
        ).to_schema_extra(),
    )
    kv_cache_quant_algo: Literal["FP8", "INT8"] | None = Field(
        default=None,
        description="KV cache quantisation algorithm (None -> no KV cache quantisation)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "T3", "E3"),
            rationale="KV-cache quantisation; aligns with vLLM's kv_cache_dtype. Energy-measurement-biasing (T3).",
            native_mapping="QuantConfig.kv_cache_quant_algo",
        ).to_schema_extra(),
    )


class TensorRTKvCacheConfig(BaseModel):
    """TRT-LLM KV cache configuration."""

    model_config = {"extra": "allow"}

    enable_block_reuse: bool | None = Field(
        default=None,
        description="Enable KV cache block reuse (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E2"),
            rationale="TRT's prefix-caching analog — regime toggle.",
            native_mapping="KvCacheConfig.enable_block_reuse",
        ).to_schema_extra(),
    )
    free_gpu_memory_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of free GPU memory for KV cache (None -> 0.9)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="TRT's gpu_memory_utilization analog.",
            native_mapping="KvCacheConfig.free_gpu_memory_fraction",
        ).to_schema_extra(),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens in KV cache (None -> auto)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="KV-cache token cap → concurrency ceiling.",
            native_mapping="KvCacheConfig.max_tokens",
        ).to_schema_extra(),
    )
    host_cache_size: int | None = Field(
        default=None,
        ge=0,
        description="Host (CPU) cache size in bytes for KV cache offloading (None -> 0, disabled)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="CPU KV-offload regime — cross-engine analog to vLLM swap_space.",
            native_mapping="KvCacheConfig.host_cache_size",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "E3"),
            rationale="Scheduling regime (NO_EVICT vs MAX_UTIL vs STATIC_BATCH); stable 3-value enum.",
            native_mapping="SchedulerConfig.capacity_scheduling_policy",
        ).to_schema_extra(),
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
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale=(
                "Minimum output tokens before EOS; future native slot for DecoderConfig.min_new_tokens "
                "once Phase 49 migrates decoder per-engine."
            ),
            native_mapping="SamplingParams.min_tokens",
        ).to_schema_extra(),
    )
    n: int | None = Field(
        default=None,
        ge=1,
        description="Number of output sequences per prompt (None -> 1)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="n-sequences multiplier.",
            native_mapping="SamplingParams.n",
        ).to_schema_extra(),
    )
    ignore_eos: bool | None = Field(
        default=None,
        description="Continue generating past EOS token (None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2"),
            rationale="Forces fixed output length — measurement workload control.",
            native_mapping="SamplingParams.ignore_eos",
        ).to_schema_extra(),
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
    - backend: Literal["trt"] — D2 single-option enum, no information content
    - engine_path — D1 deployment path, not a measurement axis
    - calib sub-config — D3 build-only PTQ calibration (we consume pre-quantised checkpoints)
    - build_cache sub-config — D1 engine-cache housekeeping

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
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "E1"),
            rationale="MLPerf + trtllm-bench first-class; compile-time shape constant.",
            native_mapping="TrtLlmArgs.max_batch_size",
        ).to_schema_extra(),
    )
    tensor_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Tensor parallel size — number of GPUs to shard across (None -> 1). "
            "Aligns with TrtLlmArgs.tensor_parallel_size. "
            "Note: TransformersConfig.tp_size follows accelerate convention and is preserved."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2", "T3"),
            rationale="Universal + energy-sampler-relevant (T3: must appear in results metadata).",
            native_mapping="TrtLlmArgs.tensor_parallel_size",
        ).to_schema_extra(),
    )
    pipeline_parallel_size: int | None = Field(
        default=None,
        ge=1,
        description="Pipeline parallel stages (None -> 1).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="NVIDIA MLPerf v5.0 Blackwell submissions surface PP size as a primary knob.",
            native_mapping="TrtLlmArgs.pipeline_parallel_size",
        ).to_schema_extra(),
    )
    max_input_len: int | None = Field(
        default=None,
        ge=1,
        description="Max input sequence length (compile-time constant, None -> 1024)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="Compile-time shape; MLPerf core knob.",
            native_mapping="TrtLlmArgs.max_input_len",
        ).to_schema_extra(),
    )
    max_seq_len: int | None = Field(
        default=None,
        ge=1,
        description="Max total sequence length (input + output, compile-time constant, None -> 2048)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "R2"),
            rationale="Compile-time shape; MLPerf core knob.",
            native_mapping="TrtLlmArgs.max_seq_len",
        ).to_schema_extra(),
    )
    max_num_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens the engine can handle per iteration (None -> auto).",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E1"),
            rationale="trtllm-bench scheduler axis alongside max_batch_size.",
            native_mapping="TrtLlmArgs.max_num_tokens",
        ).to_schema_extra(),
    )
    dtype: Literal["float16", "bfloat16"] | None = Field(
        default=None,
        description=(
            "Model dtype (None -> auto). TRT-LLM is optimised for fp16/bf16; fp32 not supported."
        ),
        json_schema_extra=CurationMetadata(
            clauses=("E4",),
            rationale="TRT-LLM rejects fp32 — narrowed Literal is a parse-time safety rail (E4).",
            native_mapping="TrtLlmArgs.dtype",
        ).to_schema_extra(),
    )
    fast_build: bool | None = Field(
        default=None,
        description="Enable fast engine build mode (reduced optimisation, None -> False)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale=(
                "Skips some build-time optimisations → the produced engine may be less efficient "
                "at runtime → measurable energy/throughput delta."
            ),
            native_mapping="TrtLlmArgs.fast_build",
        ).to_schema_extra(),
    )

    # -------------------------------------------------------------------------
    # Nested sub-configs
    # -------------------------------------------------------------------------

    quant: TensorRTQuantConfig | None = Field(
        default=None,
        description="Quantisation configuration (QuantConfig)",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "E3"),
            rationale="Quantisation algorithm and KV-cache quantisation sub-config.",
            native_mapping="TrtLlmArgs.quant_config → QuantConfig",
        ).to_schema_extra(),
    )
    kv_cache: TensorRTKvCacheConfig | None = Field(
        default=None,
        description="KV cache configuration",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="KV-cache configuration (memory fraction, block reuse, host offload).",
            native_mapping="TrtLlmArgs.kv_cache_config → KvCacheConfig",
        ).to_schema_extra(),
    )
    scheduler: TensorRTSchedulerConfig | None = Field(
        default=None,
        description="Scheduler configuration",
        json_schema_extra=CurationMetadata(
            clauses=("R1", "E2", "E3"),
            rationale="Scheduling capacity policy sub-config.",
            native_mapping="TrtLlmArgs.scheduler_config → SchedulerConfig",
        ).to_schema_extra(),
    )
    sampling: TensorRTSamplingConfig | None = Field(
        default=None,
        description="Sampling configuration (TRT-LLM-specific SamplingParams extensions)",
        json_schema_extra=CurationMetadata(
            clauses=("R1",),
            rationale="TRT-LLM-specific sampling parameters sub-config.",
            native_mapping="TrtLlmArgs → SamplingParams",
        ).to_schema_extra(),
    )
