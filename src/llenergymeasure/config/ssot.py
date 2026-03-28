"""Single source of truth for backend capability constants.

These dicts define which dtype modes and decoding strategies each backend
supports. They are consumed by:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (backend capability metadata)
- CLI help generation

Do not inline these values in validators — always import from here.
"""

from __future__ import annotations

from typing import Final, Literal

# ---------------------------------------------------------------------------
# Backend name constants — use these instead of raw string literals
# ---------------------------------------------------------------------------

BACKEND_PYTORCH: Final = "pytorch"
BACKEND_VLLM: Final = "vllm"
BACKEND_TENSORRT: Final = "tensorrt"

BackendName = Literal["pytorch", "vllm", "tensorrt"]

# ---------------------------------------------------------------------------
# Runner mode constants
# ---------------------------------------------------------------------------

RUNNER_LOCAL: Final = "local"
RUNNER_DOCKER: Final = "docker"
CONTAINER_EXCHANGE_DIR: Final = "/run/llem"
"""Mount point inside Docker containers for config/result exchange."""

SOURCE_MULTI_BACKEND_ELEVATION: Final = "multi_backend_elevation"
"""RunnerSpec source tag when a backend is auto-elevated to Docker for multi-backend isolation."""

RunnerMode = Literal["local", "docker"]

DOCKER_PULL_TIMEOUT: Final = 1800
"""Maximum seconds to wait for ``docker pull`` (30 min — generous for large images like TensorRT ~10 GB)."""

# ---------------------------------------------------------------------------
# Backend capability dicts
# ---------------------------------------------------------------------------

# Precision modes supported by each backend.
# "float32" = full precision, "float16" = half, "bfloat16" = brain float16.
# Note: fp16/bf16 require GPU. The cpu backend (future) would be fp32-only.
# GPU detection and cpu-dtype cross-validation is handled at pre-flight.
DTYPE_SUPPORT: dict[str, list[str]] = {
    BACKEND_PYTORCH: ["float32", "float16", "bfloat16"],
    BACKEND_VLLM: ["float16", "bfloat16"],  # vLLM does not support fp32 inference
    BACKEND_TENSORRT: ["float16", "bfloat16"],  # TRT-LLM does not support fp32 inference
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
    "DOCKER_PULL_TIMEOUT",
    "DTYPE_SUPPORT",
    "RUNNER_DOCKER",
    "RUNNER_LOCAL",
    "BackendName",
    "RunnerMode",
]
