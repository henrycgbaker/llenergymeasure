"""Single source of truth for backend capability constants.

These dicts define which precision modes and decoding strategies each backend
supports. They are consumed by:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (backend capability metadata)
- CLI help generation

Do not inline these values in validators — always import from here.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Backend name constants — use these instead of raw string literals
# ---------------------------------------------------------------------------

BACKEND_PYTORCH = "pytorch"
BACKEND_VLLM = "vllm"
BACKEND_TENSORRT = "tensorrt"

BackendName = Literal["pytorch", "vllm", "tensorrt"]

# ---------------------------------------------------------------------------
# Runner mode constants
# ---------------------------------------------------------------------------

RUNNER_LOCAL = "local"
RUNNER_DOCKER = "docker"

RunnerMode = Literal["local", "docker"]

# ---------------------------------------------------------------------------
# Backend capability dicts
# ---------------------------------------------------------------------------

# Precision modes supported by each backend.
# "fp32" = full precision, "fp16" = half, "bf16" = bfloat16.
# Note: fp16/bf16 require GPU. The cpu backend (future) would be fp32-only.
# GPU detection and cpu-precision cross-validation is handled at pre-flight.
PRECISION_SUPPORT: dict[str, list[str]] = {
    BACKEND_PYTORCH: ["fp32", "fp16", "bf16"],
    BACKEND_VLLM: ["fp16", "bf16"],  # vLLM does not support fp32 inference
    BACKEND_TENSORRT: ["fp16", "bf16"],  # TRT-LLM does not support fp32 inference
}

# Decoding strategies supported by each backend.
# "sampling" = do_sample=True path; "greedy" = do_sample=False path.
DECODING_SUPPORT: dict[str, list[str]] = {
    BACKEND_PYTORCH: ["greedy", "sampling"],  # full HuggingFace generate() support
    BACKEND_VLLM: ["greedy", "sampling"],  # vLLM supports both via SamplingParams
    BACKEND_TENSORRT: ["greedy", "sampling"],  # TRT-LLM supports both
}

# Backends that support the full DecoderConfig temperature/top_k/top_p fields.
# All current backends support these — this dict exists to make future
# backend additions explicit rather than implicit.
# min_p and min_new_tokens: pytorch and vLLM support them (identical semantics).
# min_p/min_new_tokens: vLLM supports; TensorRT support varies by version.
DECODER_PARAM_SUPPORT: dict[str, list[str]] = {
    BACKEND_PYTORCH: [
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "min_p",
        "min_new_tokens",
    ],
    BACKEND_VLLM: ["temperature", "top_k", "top_p", "repetition_penalty"],
    BACKEND_TENSORRT: [
        "temperature",
        "top_k",
        "top_p",
    ],  # TRT-LLM: repetition_penalty support varies
}

# Map from backend name to the Python package that provides it.
# Used by preflight checks and CLI to verify backend availability.
BACKEND_PACKAGES: dict[str, str] = {
    BACKEND_PYTORCH: "transformers",
    BACKEND_VLLM: "vllm",
    BACKEND_TENSORRT: "tensorrt_llm",
}

__all__ = [
    "BACKEND_PACKAGES",
    "BACKEND_PYTORCH",
    "BACKEND_TENSORRT",
    "BACKEND_VLLM",
    "DECODER_PARAM_SUPPORT",
    "DECODING_SUPPORT",
    "PRECISION_SUPPORT",
    "RUNNER_DOCKER",
    "RUNNER_LOCAL",
    "BackendName",
    "RunnerMode",
]
