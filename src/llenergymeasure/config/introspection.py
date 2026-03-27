"""Configuration introspection for SSOT architecture.

This module provides the Single Source of Truth (SSOT) for parameter metadata
by introspecting Pydantic models. All downstream consumers (tests, CLI, docs)
should use these functions to derive parameter information rather than
maintaining separate parameter lists.

Usage:
    from llenergymeasure.config.introspection import (
        get_backend_params,
        get_shared_params,
        get_all_params,
        get_param_test_values,
        get_experiment_config_schema,
    )

    # Get all params for a backend
    pytorch_params = get_backend_params("pytorch")

    # Get test values for a param
    values = get_param_test_values("pytorch.batch_size")

    # Get full JSON schema
    schema = get_experiment_config_schema()
"""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _extract_param_metadata(
    field_name: str,
    field_info: FieldInfo,
    prefix: str = "",
) -> dict[str, Any]:
    """Extract metadata from a Pydantic field.

    Returns:
        Dict with keys: type, default, description, optional, constraints,
        options (for Literal), test_values.
    """
    param_path = f"{prefix}.{field_name}" if prefix else field_name
    annotation = field_info.annotation

    # Handle Optional types (X | None)
    origin = get_origin(annotation)
    args = get_args(annotation)

    is_optional = False
    if origin is type(None) or (args and type(None) in args):
        is_optional = True
        actual_types = [a for a in args if a is not type(None)]
        if actual_types:
            annotation = actual_types[0]
            origin = get_origin(annotation)
            args = get_args(annotation)

    # Build metadata
    meta: dict[str, Any] = {
        "path": param_path,
        "name": field_name,
        "default": field_info.default if field_info.default is not ... else None,
        "description": field_info.description or "",
        "optional": is_optional,
        "constraints": {},
        "options": None,
        "test_values": [],
        "type_str": "unknown",
    }

    # Extract constraints from field metadata
    if hasattr(field_info, "metadata"):
        for constraint in field_info.metadata:
            if hasattr(constraint, "ge"):
                meta["constraints"]["ge"] = constraint.ge
            if hasattr(constraint, "le"):
                meta["constraints"]["le"] = constraint.le
            if hasattr(constraint, "gt"):
                meta["constraints"]["gt"] = constraint.gt
            if hasattr(constraint, "lt"):
                meta["constraints"]["lt"] = constraint.lt

    # Determine type and generate test values
    if origin is Literal:
        meta["type_str"] = "literal"
        meta["options"] = list(args)
        meta["test_values"] = list(args)  # Test ALL Literal values

    elif annotation is bool:
        meta["type_str"] = "bool"
        meta["test_values"] = [False, True]

    elif annotation is int:
        meta["type_str"] = "int"
        ge = meta["constraints"].get("ge")
        le = meta["constraints"].get("le")
        default = meta["default"]

        if ge is not None and le is not None:
            # Test min, mid, max
            meta["test_values"] = sorted(set([ge, (ge + le) // 2, le]))
        elif ge is not None:
            meta["test_values"] = [ge, ge * 2, ge * 4]
        elif default is not None and isinstance(default, int):
            meta["test_values"] = [max(1, default // 2), default, default * 2]
        else:
            meta["test_values"] = [1, 4, 8]

    elif annotation is float:
        meta["type_str"] = "float"
        default = meta["default"]
        if default is not None and isinstance(default, int | float):
            meta["test_values"] = [
                round(default * 0.5, 2),
                default,
                round(default * 1.5, 2),
            ]
        else:
            meta["test_values"] = [0.5, 0.7, 0.9]

    elif annotation is str:
        meta["type_str"] = "str"
        meta["test_values"] = []  # Strings need context-specific values

    else:
        meta["type_str"] = str(annotation)

    return meta


def get_params_from_model(
    model_class: type[BaseModel],
    prefix: str = "",
    include_nested: bool = True,
) -> dict[str, dict[str, Any]]:
    """Extract all parameters from a Pydantic model.

    Args:
        model_class: Pydantic model to introspect.
        prefix: Prefix for param paths (e.g., "pytorch").
        include_nested: Whether to recurse into nested models.

    Returns:
        Dict mapping param paths to metadata dicts.
    """
    params: dict[str, dict[str, Any]] = {}

    for field_name, field_info in model_class.model_fields.items():
        annotation = field_info.annotation

        # Handle Optional wrapper
        args = get_args(annotation)
        if args and type(None) in args:
            actual_types = [a for a in args if a is not type(None)]
            if actual_types:
                annotation = actual_types[0]

        # Check if nested Pydantic model
        if include_nested and hasattr(annotation, "model_fields"):
            nested_prefix = f"{prefix}.{field_name}" if prefix else field_name
            # Cast annotation to BaseModel subclass (we know it is from model_fields check)
            nested_params = get_params_from_model(
                annotation,  # type: ignore[arg-type]
                prefix=nested_prefix,
                include_nested=True,
            )
            params.update(nested_params)
        else:
            meta = _extract_param_metadata(field_name, field_info, prefix)
            params[meta["path"]] = meta

    return params


def _get_custom_test_values() -> dict[str, list[Any]]:
    """Get custom test value overrides for params that need special handling.

    Returns known-invalid values for constrained fields — used by runtime
    parameter tests to verify validation rejects out-of-range inputs.
    One invalid value per constrained field (the simplest violation).
    """
    return {
        # VLLMEngineConfig: one known-invalid value per constrained field
        "vllm.engine.gpu_memory_utilization": [1.5],  # ge=0.0, lt=1.0: 1.5 violates lt
        "vllm.engine.swap_space": [-1.0],  # ge=0.0: negative violates ge
        "vllm.engine.cpu_offload_gb": [-0.5],  # ge=0.0: negative violates ge
        "vllm.engine.max_num_seqs": [0],  # ge=1: 0 violates ge
        "vllm.engine.max_num_batched_tokens": [0],  # ge=1: 0 violates ge
        "vllm.engine.max_model_len": [0],  # ge=1: 0 violates ge
        "vllm.engine.tensor_parallel_size": [0],  # ge=1: 0 violates ge
        "vllm.engine.pipeline_parallel_size": [0],  # ge=1: 0 violates ge
        "vllm.engine.num_speculative_tokens": [0],  # ge=1: 0 violates ge
        # VLLMSamplingConfig: one known-invalid value per constrained field
        "vllm.sampling.max_tokens": [0],  # ge=1: 0 violates ge
        "vllm.sampling.presence_penalty": [3.0],  # ge=-2.0, le=2.0: 3.0 violates le
        "vllm.sampling.frequency_penalty": [-3.0],  # ge=-2.0, le=2.0: -3.0 violates ge
        # VLLMEngineConfig: constrained fields
        "vllm.engine.offload_num_in_group": [0],  # ge=1: 0 violates ge
        "vllm.engine.kv_cache_memory_bytes": [0],  # ge=1: 0 violates ge
        # VLLMSamplingConfig: constrained field
        "vllm.sampling.n": [0],  # ge=1: 0 violates ge
        # VLLMBeamSearchConfig: constrained fields
        "vllm.beam_search.beam_width": [0],  # ge=1: 0 violates ge
        "vllm.beam_search.max_tokens": [0],  # ge=1: 0 violates ge
        # TensorRTConfig: compile-time params
        "tensorrt.max_batch_size": [0],  # ge=1: 0 violates ge
        "tensorrt.tp_size": [0],  # ge=1: 0 violates ge
        "tensorrt.max_input_len": [0],  # ge=1: 0 violates ge
        "tensorrt.max_seq_len": [0],  # ge=1: 0 violates ge
        # TensorRTCalibConfig: calibration params
        "tensorrt.calib.calib_batches": [0],  # ge=1: 0 violates ge
        "tensorrt.calib.calib_max_seq_length": [0],  # ge=1: 0 violates ge
        # TensorRTKvCacheConfig: cache params
        "tensorrt.kv_cache.max_tokens": [0],  # ge=1: 0 violates ge
        # TensorRTBuildCacheConfig: engine cache params
        "tensorrt.build_cache.max_records": [0],  # ge=1: 0 violates ge
        # TensorRTSamplingConfig: sampling params
        "tensorrt.sampling.n": [0],  # ge=1: 0 violates ge
    }


def get_backend_params(backend: str) -> dict[str, dict[str, Any]]:
    """Get all parameters for a backend from its Pydantic model.

    Args:
        backend: One of "pytorch", "vllm", "tensorrt".

    Returns:
        Dict mapping param paths to metadata. Each param includes
        ``backend_support: list[str]`` indicating which backends expose it.
    """
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    backend_models = {
        "pytorch": PyTorchConfig,
        "vllm": VLLMConfig,
        "tensorrt": TensorRTConfig,
    }

    if backend not in backend_models:
        raise ValueError(
            f"Unknown backend: {backend}. Must be one of {list(backend_models.keys())}"
        )

    model_class = backend_models[backend]
    # All values are Pydantic BaseModel subclasses, mypy can't infer this from dict
    params = get_params_from_model(model_class, prefix=backend)  # type: ignore[arg-type]

    # Add backend_support to every param
    for param in params.values():
        param["backend_support"] = [backend]

    # Apply custom test value overrides
    custom_values = _get_custom_test_values()
    for param_path, values in custom_values.items():
        if param_path in params:
            params[param_path]["test_values"] = values

    return params


def get_shared_params() -> dict[str, dict[str, Any]]:
    """Get shared/universal parameters from ExperimentConfig and DecoderConfig.

    Returns params that are universal across all backends:
    - Top-level: dtype, n, max_input_tokens, max_output_tokens, random_seed
    - Decoder: temperature, do_sample, top_p, top_k, repetition_penalty, preset

    Each param includes ``backend_support: list[str]`` indicating which backends
    expose each parameter.
    """
    from llenergymeasure.config.models import DecoderConfig

    shared: dict[str, dict[str, Any]] = {}

    # Decoder params (introspected from model)
    decoder_params = get_params_from_model(DecoderConfig, prefix="decoder")
    # Add backend_support to decoder params
    for param in decoder_params.values():
        param["backend_support"] = ["pytorch", "vllm", "tensorrt"]
    shared.update(decoder_params)

    # Top-level universal params — defined manually for explicit backend_support
    shared["dtype"] = {
        "path": "dtype",
        "name": "dtype",
        "type_str": "literal",
        "default": "bfloat16",
        "description": "Model dtype for inference",
        "options": ["float32", "float16", "bfloat16"],
        "test_values": ["float32", "float16", "bfloat16"],
        "constraints": {},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["dataset.source"] = {
        "path": "dataset.source",
        "name": "source",
        "type_str": "str",
        "default": "aienergyscore",
        "description": "Dataset source: built-in alias or .jsonl file path",
        "options": None,
        "test_values": ["aienergyscore"],
        "constraints": {"min_length": 1},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["dataset.n_prompts"] = {
        "path": "dataset.n_prompts",
        "name": "n_prompts",
        "type_str": "int",
        "default": 100,
        "description": "Number of prompts to load",
        "options": None,
        "test_values": [10, 100, 500],
        "constraints": {"ge": 1},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["dataset.order"] = {
        "path": "dataset.order",
        "name": "order",
        "type_str": "str",
        "default": "interleaved",
        "description": "Prompt ordering: interleaved, grouped, or shuffled",
        "options": ["interleaved", "grouped", "shuffled"],
        "test_values": ["interleaved", "grouped", "shuffled"],
        "constraints": {},
        "optional": False,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["max_input_tokens"] = {
        "path": "max_input_tokens",
        "name": "max_input_tokens",
        "type_str": "int | None",
        "default": 256,
        "description": (
            "Max input token length for truncation. Keeps computation workload "
            "constant across experiments for fair comparison. None = no truncation."
        ),
        "options": None,
        "test_values": [64, 128, 256, None],
        "constraints": {"ge": 1},
        "optional": True,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }
    shared["max_output_tokens"] = {
        "path": "max_output_tokens",
        "name": "max_output_tokens",
        "type_str": "int | None",
        "default": 256,
        "description": (
            "Max output tokens (max_new_tokens for generation). "
            "None = generate until EOS or model context limit."
        ),
        "options": None,
        "test_values": [32, 128, 256, None],
        "constraints": {"ge": 1},
        "optional": True,
        "backend_support": ["pytorch", "vllm", "tensorrt"],
    }

    return shared


def get_experiment_config_schema() -> dict[str, Any]:
    """Return the full ExperimentConfig JSON schema (Pydantic v2 schema).

    Returns:
        JSON-serialisable dict with the complete schema including all
        properties, types, constraints, and nested model schemas.
        Uses Pydantic's built-in model_json_schema() — always in sync
        with the actual model definition.
    """
    from llenergymeasure.config.models import ExperimentConfig

    return ExperimentConfig.model_json_schema()


def get_all_params() -> dict[str, dict[str, dict[str, Any]]]:
    """Get all parameters organised by backend + shared.

    Returns:
        {
            "shared": {...},
            "pytorch": {...},
            "vllm": {...},
            "tensorrt": {...},
        }
    """
    return {
        "shared": get_shared_params(),
        "pytorch": get_backend_params("pytorch"),
        "vllm": get_backend_params("vllm"),
        "tensorrt": get_backend_params("tensorrt"),
    }


def get_param_test_values(param_path: str) -> list[Any]:
    """Get test values for a specific parameter.

    Args:
        param_path: Full param path, e.g., "pytorch.batch_size" or "decoder.temperature".

    Returns:
        List of test values.
    """
    all_params = get_all_params()

    for section in all_params.values():
        if param_path in section:
            test_values: list[Any] = section[param_path].get("test_values", [])
            return test_values

    return []


def get_param_options(param_path: str) -> list[Any] | None:
    """Get valid options for a Literal-typed parameter.

    Args:
        param_path: Full param path.

    Returns:
        List of options for Literal types, None otherwise.
    """
    all_params = get_all_params()

    for section in all_params.values():
        if param_path in section:
            return section[param_path].get("options")

    return None


def list_all_param_paths(backend: str | None = None) -> list[str]:
    """List all parameter paths, optionally filtered by backend.

    Args:
        backend: Optional backend filter ("pytorch", "vllm", "tensorrt", "shared").

    Returns:
        Sorted list of param paths.
    """
    all_params = get_all_params()

    if backend:
        if backend not in all_params:
            raise ValueError(f"Unknown backend: {backend}")
        return sorted(all_params[backend].keys())

    paths: list[str] = []
    for section in all_params.values():
        paths.extend(section.keys())
    return sorted(set(paths))


# =============================================================================
# Constraint Metadata for SSOT Architecture Hardening
# =============================================================================


def get_mutual_exclusions() -> dict[str, list[str]]:
    """Get parameters that are mutually exclusive.

    Returns:
        Dict mapping param path to list of params it cannot be used with.
        These combinations should be skipped during runtime testing.
    """
    return {
        # PyTorch: can't use both 4-bit and 8-bit quantization
        "pytorch.load_in_4bit": ["pytorch.load_in_8bit"],
        "pytorch.load_in_8bit": ["pytorch.load_in_4bit"],
        # torch_compile sub-options require torch_compile=True
        "pytorch.torch_compile_mode": ["pytorch.torch_compile=None|False"],
        "pytorch.torch_compile_backend": ["pytorch.torch_compile=None|False"],
        # BitsAndBytes 4-bit sub-options require load_in_4bit=True
        "pytorch.bnb_4bit_compute_dtype": ["pytorch.load_in_4bit=None|False"],
        "pytorch.bnb_4bit_quant_type": ["pytorch.load_in_4bit=None|False"],
        "pytorch.bnb_4bit_use_double_quant": ["pytorch.load_in_4bit=None|False"],
        # cache_implementation contradicts use_cache=False
        "pytorch.cache_implementation": ["pytorch.use_cache=False"],
        # vLLM speculative decoding: speculative_model requires num_speculative_tokens
        "vllm.engine.speculative_model": ["vllm.engine.num_speculative_tokens=None"],
        # vLLM kv_cache_memory_bytes vs gpu_memory_utilization
        "vllm.engine.kv_cache_memory_bytes": ["vllm.engine.gpu_memory_utilization"],
        "vllm.engine.gpu_memory_utilization": ["vllm.engine.kv_cache_memory_bytes"],
        # vLLM beam_search vs sampling sections (cross-section mutual exclusion)
        "vllm.beam_search": ["vllm.sampling"],
        "vllm.sampling": ["vllm.beam_search"],
        # TensorRT: quantisation method is exclusive
        "tensorrt.quant.quant_algo": [],  # Handled by Literal type constraint
    }


def get_backend_specific_params() -> dict[str, list[str]]:
    """Get params that are only valid for specific backends.

    Returns:
        Dict mapping backend name to list of exclusive param paths.
        Derived from v2.0 minimal backend config fields.
    """
    return {
        "pytorch": [
            "pytorch.batch_size",
            "pytorch.attn_implementation",
            "pytorch.torch_compile",
            "pytorch.torch_compile_mode",
            "pytorch.torch_compile_backend",
            "pytorch.load_in_4bit",
            "pytorch.load_in_8bit",
            "pytorch.bnb_4bit_compute_dtype",
            "pytorch.bnb_4bit_quant_type",
            "pytorch.bnb_4bit_use_double_quant",
            "pytorch.use_cache",
            "pytorch.cache_implementation",
            "pytorch.num_beams",
            "pytorch.early_stopping",
            "pytorch.length_penalty",
            "pytorch.no_repeat_ngram_size",
            "pytorch.prompt_lookup_num_tokens",
            "pytorch.device_map",
            "pytorch.max_memory",
            "pytorch.revision",
            "pytorch.trust_remote_code",
        ],
        "vllm": [
            # Engine-level params (vllm.LLM() constructor args)
            "vllm.engine.gpu_memory_utilization",
            "vllm.engine.swap_space",
            "vllm.engine.cpu_offload_gb",
            "vllm.engine.block_size",
            "vllm.engine.kv_cache_dtype",
            "vllm.engine.enforce_eager",
            "vllm.engine.enable_chunked_prefill",
            "vllm.engine.max_num_seqs",
            "vllm.engine.max_num_batched_tokens",
            "vllm.engine.max_model_len",
            "vllm.engine.tensor_parallel_size",
            "vllm.engine.pipeline_parallel_size",
            "vllm.engine.enable_prefix_caching",
            "vllm.engine.quantization",
            "vllm.engine.speculative_model",
            "vllm.engine.num_speculative_tokens",
            # Engine-level offloading + memory params
            "vllm.engine.offload_group_size",
            "vllm.engine.offload_num_in_group",
            "vllm.engine.offload_prefetch_step",
            "vllm.engine.offload_params",
            "vllm.engine.disable_custom_all_reduce",
            "vllm.engine.kv_cache_memory_bytes",
            "vllm.engine.compilation_config",
            # Attention sub-model
            "vllm.engine.attention.backend",
            "vllm.engine.attention.flash_attn_version",
            "vllm.engine.attention.flash_attn_max_num_splits_for_cuda_graph",
            "vllm.engine.attention.use_prefill_decode_attention",
            "vllm.engine.attention.use_prefill_query_quantization",
            "vllm.engine.attention.use_cudnn_prefill",
            "vllm.engine.attention.disable_flashinfer_prefill",
            "vllm.engine.attention.disable_flashinfer_q_quantization",
            "vllm.engine.attention.use_trtllm_attention",
            "vllm.engine.attention.use_trtllm_ragged_deepseek_prefill",
            # Sampling-level params (vllm.SamplingParams args, vLLM-specific only)
            "vllm.sampling.max_tokens",
            "vllm.sampling.min_tokens",
            "vllm.sampling.presence_penalty",
            "vllm.sampling.frequency_penalty",
            "vllm.sampling.ignore_eos",
            "vllm.sampling.n",
            # Beam search section (all 4 fields)
            "vllm.beam_search.beam_width",
            "vllm.beam_search.length_penalty",
            "vllm.beam_search.early_stopping",
            "vllm.beam_search.max_tokens",
        ],
        "tensorrt": [
            # Compile-time parameters (LLM() constructor)
            "tensorrt.max_batch_size",
            "tensorrt.tp_size",
            "tensorrt.max_input_len",
            "tensorrt.max_seq_len",
            "tensorrt.dtype",
            "tensorrt.fast_build",
            # TRT-LLM internal backend selection
            "tensorrt.backend",
            # Engine path
            "tensorrt.engine_path",
            # Quantisation sub-config
            "tensorrt.quant.quant_algo",
            "tensorrt.quant.kv_cache_quant_algo",
            # Calibration sub-config
            "tensorrt.calib.calib_batches",
            "tensorrt.calib.calib_dataset",
            "tensorrt.calib.calib_max_seq_length",
            # KV cache sub-config
            "tensorrt.kv_cache.enable_block_reuse",
            "tensorrt.kv_cache.free_gpu_memory_fraction",
            "tensorrt.kv_cache.max_tokens",
            "tensorrt.kv_cache.host_cache_size",
            # Scheduler sub-config
            "tensorrt.scheduler.capacity_scheduling_policy",
            # Build cache sub-config
            "tensorrt.build_cache.cache_root",
            "tensorrt.build_cache.max_records",
            "tensorrt.build_cache.max_cache_storage_gb",
            # Sampling sub-config
            "tensorrt.sampling.min_tokens",
            "tensorrt.sampling.n",
            "tensorrt.sampling.ignore_eos",
            "tensorrt.sampling.return_perf_metrics",
        ],
    }


def get_special_test_models() -> dict[str, str]:
    """Get parameters that require special pre-quantized test models.

    Some parameters (like AWQ/GPTQ quantization) require models that have
    been pre-quantized with that method. Using a non-quantized model will fail.

    Returns:
        Dict mapping param value patterns to appropriate test model names.
    """
    return {
        # vLLM quantization methods requiring pre-quantized models
        "vllm.engine.quantization=awq": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "vllm.engine.quantization=gptq": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        # TensorRT quantisation methods requiring pre-quantized models
        "tensorrt.quant.quant_algo=W4A16_AWQ": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "tensorrt.quant.quant_algo=W4A16_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    }


def get_params_requiring_gpu_capability(min_compute_capability: float = 8.0) -> list[str]:
    """Get params that require specific GPU compute capabilities.

    Args:
        min_compute_capability: Minimum compute capability (default 8.0 = Ampere).

    Returns:
        List of param paths that require the specified compute capability.
    """
    # These features require Ampere (8.0) or newer GPUs
    ampere_required = [
        "vllm.engine.quantization=fp8",
        "tensorrt.quant.quant_algo=FP8",
        "pytorch.attn_implementation=flash_attention_2",
    ]

    # These features require Hopper (9.0) or newer GPUs
    hopper_required = [
        "pytorch.attn_implementation=flash_attention_3",
    ]

    if min_compute_capability >= 9.0:
        return ampere_required + hopper_required
    return ampere_required


def get_param_skip_conditions() -> dict[str, str]:
    """Get conditions under which params should be skipped during testing.

    Returns:
        Dict mapping param paths to skip reasons for documentation/logging.
    """
    return {
        # Multi-GPU params - skip if single GPU
        "vllm.engine.tensor_parallel_size>1": "Requires 2+ GPUs",
        "tensorrt.tp_size>1": "Requires 2+ GPUs",
        # Flash Attention 2 - requires flash-attn package
        "pytorch.attn_implementation=flash_attention_2": "Requires flash-attn package",
        # Flash Attention 3 - requires Hopper+ GPU (H100)
        "pytorch.attn_implementation=flash_attention_3": "Requires Hopper+ GPU (compute capability 9.0+)",
        # torch.compile - may not work on all model architectures
        "pytorch.torch_compile=True": "May fail on some model architectures (non-fatal fallback)",
        # FP8 - Ampere or newer
        "vllm.engine.quantization=fp8": "Requires Ampere+ GPU",
        "tensorrt.quant.quant_algo=FP8": "Requires Ada Lovelace+ GPU (SM >= 8.9)",
        # TensorRT quantisation - requires pre-quantized models
        "tensorrt.quant.quant_algo=W4A16_AWQ": "Requires AWQ-quantized model",
        "tensorrt.quant.quant_algo=W4A16_GPTQ": "Requires GPTQ-quantized model",
        # Quantization - requires pre-quantized models (see get_special_test_models)
        "vllm.engine.quantization=awq": "Requires AWQ-quantized model",
        "vllm.engine.quantization=gptq": "Requires GPTQ-quantized model",
        # PyTorch optional dependencies
        "pytorch.load_in_4bit": "Requires compatible bitsandbytes version",
        "pytorch.load_in_8bit": "Requires compatible bitsandbytes version",
        # BitsAndBytes 4-bit sub-options
        "pytorch.bnb_4bit_compute_dtype": "Requires load_in_4bit=True and bitsandbytes package",
        "pytorch.bnb_4bit_quant_type": "Requires load_in_4bit=True and bitsandbytes package",
        "pytorch.bnb_4bit_use_double_quant": "Requires load_in_4bit=True and bitsandbytes package",
        # Prompt lookup speculative decoding
        "pytorch.prompt_lookup_num_tokens": "Requires compatible model and sufficient prompt overlap",
    }


def get_streaming_constraints() -> dict[str, str]:
    """Streaming constraints (not yet implemented).

    Returns:
        Empty dict — streaming parameters are not in scope.
    """
    return {}


def get_streaming_incompatible_tests() -> list[tuple[str, str]]:
    """Streaming incompatible tests (not yet implemented).

    Returns:
        Empty list — streaming parameters are not in scope.
    """
    return []


# =============================================================================
# SSOT Backend Capability Matrix
# =============================================================================


def get_backend_capabilities() -> dict[str, dict[str, bool | str]]:
    """Derive backend capability matrix from Pydantic model structure.

    This is the SSOT for the capability matrix shown in documentation.
    Capabilities are inferred by checking which fields exist in each
    backend config and their allowed values.

    Returns:
        Dict mapping capability names to per-backend support status.
        Values are True/False for simple support, or str for notes.
    """
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        TensorRTQuantConfig,
        VLLMEngineConfig,
    )

    # Get field names for each backend
    # VLLMConfig is nested: engine fields are in VLLMEngineConfig
    pytorch_fields = set(PyTorchConfig.model_fields.keys())
    vllm_fields = set(VLLMEngineConfig.model_fields.keys())
    tensorrt_fields = set(TensorRTConfig.model_fields.keys())

    # Get quantization Literal values for vLLM and TensorRT
    vllm_quant_field = VLLMEngineConfig.model_fields.get("quantization")
    vllm_quant_options: list[str] = []
    if vllm_quant_field and vllm_quant_field.annotation:
        args = get_args(vllm_quant_field.annotation)
        # Filter out None from Optional[Literal[...]]
        for arg in args:
            if arg is not type(None):
                inner_args = get_args(arg)
                if inner_args:
                    vllm_quant_options = [a for a in inner_args if a is not None]

    trt_quant_field = TensorRTQuantConfig.model_fields.get("quant_algo")
    trt_quant_options: list[str] = []
    if trt_quant_field and trt_quant_field.annotation:
        args = get_args(trt_quant_field.annotation)
        # Filter out None from Optional[Literal[...]]
        for arg in args:
            if arg is not type(None):
                inner_args = get_args(arg)
                if inner_args:
                    trt_quant_options = [a for a in inner_args if a is not None]

    return {
        "tensor_parallel": {
            # PyTorch does NOT support tensor parallelism for HuggingFace models
            "pytorch": False,
            "vllm": "tensor_parallel_size" in vllm_fields,
            "tensorrt": "tp_size" in tensorrt_fields,
        },
        "data_parallel": {
            # PyTorch data parallelism via Accelerate is not supported in this version
            "pytorch": False,
            # vLLM/TensorRT manage parallelism internally
            "vllm": False,
            "tensorrt": False,
        },
        "bitsandbytes_4bit": {
            "pytorch": "load_in_4bit" in pytorch_fields,
            "vllm": False,  # vLLM uses native quantization, not bitsandbytes
            "tensorrt": False,  # TensorRT uses native quantization
        },
        "bitsandbytes_8bit": {
            "pytorch": "load_in_8bit" in pytorch_fields,
            "vllm": False,
            "tensorrt": False,
        },
        "native_quantization": {
            "pytorch": False,  # PyTorch relies on bitsandbytes, not native
            "vllm": "AWQ/GPTQ/FP8" if vllm_quant_options else False,
            "tensorrt": "INT8/W4A16_AWQ/W4A16_GPTQ/FP8" if trt_quant_options else False,
        },
        "float32_precision": {
            "pytorch": True,
            "vllm": True,
            # TensorRT-LLM is optimised for lower precision
            "tensorrt": False,
        },
        "float16_precision": {
            "pytorch": True,
            "vllm": True,
            "tensorrt": True,
        },
        "bfloat16_precision": {
            "pytorch": True,
            "vllm": True,
            "tensorrt": True,
        },
        "prefix_caching": {
            "pytorch": False,
            "vllm": "enable_prefix_caching" in vllm_fields,
            "tensorrt": False,
        },
        "lora_adapters": {
            "pytorch": True,  # Via peft library
            "vllm": False,  # Not in v2.0 minimal VLLMConfig
            "tensorrt": False,  # Not in v2.0 minimal TensorRTConfig
        },
        "torch_compile": {
            "pytorch": "torch_compile" in pytorch_fields,
            "vllm": False,
            "tensorrt": False,
        },
        "beam_search": {
            "pytorch": "num_beams" in pytorch_fields,
            "vllm": True,
            "tensorrt": False,
        },
        "speculative_decoding": {
            "pytorch": "prompt_lookup_num_tokens" in pytorch_fields,
            "vllm": "speculative_model" in vllm_fields,
            "tensorrt": False,
        },
        "static_kv_cache": {
            "pytorch": "cache_implementation" in pytorch_fields,
            "vllm": False,
            "tensorrt": False,
        },
    }


def get_capability_matrix_markdown() -> str:
    """Generate the capability matrix as a markdown table.

    This is used by doc generation scripts to create the capability
    matrix section in documentation files.

    Returns:
        Markdown table string.
    """
    capabilities = get_backend_capabilities()

    # Define display names
    display_names = {
        "tensor_parallel": "Tensor Parallel",
        "data_parallel": "Data Parallel",
        "bitsandbytes_4bit": "BitsAndBytes (4-bit)",
        "bitsandbytes_8bit": "BitsAndBytes (8-bit)",
        "native_quantization": "Native Quantization",
        "float32_precision": "float32 precision",
        "float16_precision": "float16 precision",
        "bfloat16_precision": "bfloat16 precision",
        "prefix_caching": "Prefix Caching",
        "lora_adapters": "LoRA Adapters",
        "torch_compile": "torch.compile",
        "beam_search": "Beam Search",
        "speculative_decoding": "Speculative Decoding",
        "static_kv_cache": "Static KV Cache",
    }

    lines = [
        "| Feature | PyTorch | vLLM | TensorRT |",
        "|---------|---------|------|----------|",
    ]

    for cap_key, cap_values in capabilities.items():
        display_name = display_names.get(cap_key, cap_key)
        cells = []

        for backend in ["pytorch", "vllm", "tensorrt"]:
            value = cap_values.get(backend, False)
            if value is True:
                cells.append("Yes")
            elif value is False:
                cells.append("No")
            elif isinstance(value, str):
                cells.append(value)
            else:
                cells.append("No")

        lines.append(f"| {display_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines.append("")
    lines.append("**Notes:**")
    lines.append("- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes")
    lines.append("- TensorRT-LLM is optimised for FP16/BF16/INT8, not FP32")

    return "\n".join(lines)


def get_validation_rules() -> list[dict[str, str]]:
    """Get validation rules from config validators for documentation.

    Extracts cross-backend validation rules that are enforced at config
    load time. These rules are the SSOT for the "Config Validation Errors"
    section in invalid-combos.md.

    Returns:
        List of dicts with keys: backend, combination, reason, resolution.
    """
    return [
        {
            "backend": "pytorch",
            "combination": "load_in_4bit=True + load_in_8bit=True",
            "reason": "Cannot use both 4-bit and 8-bit quantization simultaneously",
            "resolution": "Choose one: pytorch.load_in_4bit=true OR pytorch.load_in_8bit=true",
        },
        {
            "backend": "pytorch",
            "combination": "torch_compile_mode without torch_compile=True",
            "reason": "torch_compile_mode/torch_compile_backend only take effect when torch_compile=True",
            "resolution": "Set pytorch.torch_compile=true when using torch_compile_mode or torch_compile_backend",
        },
        {
            "backend": "pytorch",
            "combination": "bnb_4bit_* without load_in_4bit=True",
            "reason": "BitsAndBytes 4-bit options require 4-bit quantization to be enabled",
            "resolution": "Set pytorch.load_in_4bit=true when using bnb_4bit_compute_dtype, bnb_4bit_quant_type, or bnb_4bit_use_double_quant",
        },
        {
            "backend": "pytorch",
            "combination": "cache_implementation with use_cache=False",
            "reason": "Cannot specify a cache strategy when caching is explicitly disabled",
            "resolution": "Remove use_cache=false or remove cache_implementation",
        },
        {
            "backend": "all",
            "combination": "backend section mismatch",
            "reason": "Backend section must match the backend field",
            "resolution": "Ensure pytorch:/vllm:/tensorrt: section matches backend: field",
        },
        {
            "backend": "all",
            "combination": "passthrough_kwargs key collision",
            "reason": "passthrough_kwargs keys must not collide with ExperimentConfig fields",
            "resolution": "Use named fields directly instead of passthrough_kwargs",
        },
        {
            "backend": "tensorrt",
            "combination": "dtype=float32",
            "reason": "TensorRT-LLM is optimised for lower-precision inference",
            "resolution": "Use dtype='float16' or 'bfloat16'",
        },
        {
            "backend": "vllm",
            "combination": "load_in_4bit or load_in_8bit",
            "reason": "vLLM does not support bitsandbytes quantization",
            "resolution": "Use vllm.quantization (awq, gptq, fp8) for quantized inference",
        },
    ]
