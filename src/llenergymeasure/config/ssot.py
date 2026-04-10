"""Single source of truth for project-wide constants.

Centralises backend capabilities, runner modes, environment variable names,
temp file prefixes, and infrastructure timeout values. Consumers include:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (backend capability metadata)
- CLI help generation
- Infrastructure modules (Docker runner, runner resolution, image registry)
- Study runner and container lifecycle management

Do not inline these values in validators or infrastructure code — always
import from here.
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
# Environment variable name constants
# ---------------------------------------------------------------------------

ENV_RUNNER_PREFIX: Final = "LLEM_RUNNER_"
"""Prefix for per-backend runner override env vars (e.g. ``LLEM_RUNNER_PYTORCH=docker``)."""

ENV_IMAGE_PREFIX: Final = "LLEM_IMAGE_"
"""Prefix for per-backend image override env vars (e.g. ``LLEM_IMAGE_VLLM=custom:tag``)."""

ENV_CARBON_INTENSITY: Final = "LLEM_CARBON_INTENSITY"
ENV_DATACENTER_PUE: Final = "LLEM_DATACENTER_PUE"
ENV_NO_PROMPT: Final = "LLEM_NO_PROMPT"
ENV_HF_TOKEN: Final = "HF_TOKEN"
ENV_OUTPUT_DIR: Final = "LLEM_OUTPUT_DIR"
ENV_SAVE_TIMESERIES: Final = "LLEM_SAVE_TIMESERIES"
ENV_CONFIG_PATH: Final = "LLEM_CONFIG_PATH"
ENV_LOG_LEVEL: Final = "LLEM_LOG_LEVEL"
ENV_TABLE_ROWS: Final = "LLEM_TABLE_ROWS"

# ---------------------------------------------------------------------------
# Temp file/directory prefixes
# ---------------------------------------------------------------------------

TEMP_PREFIX_EXCHANGE: Final = "llem-"
"""Prefix for exchange directory created by DockerRunner."""

TEMP_PREFIX_ENV_FILE: Final = "llem-env"
"""Prefix for env-file temp files used to pass secrets to Docker."""

TEMP_PREFIX_TIMESERIES: Final = "llem-ts-"
"""Prefix for temp directories holding timeseries parquet files."""

# ---------------------------------------------------------------------------
# Subprocess / thread timeout constants (seconds)
# ---------------------------------------------------------------------------

# Docker CLI subprocess timeouts
TIMEOUT_DOCKER_CLI: Final = 5
"""Quick Docker CLI calls: ``docker ps``, ``docker image inspect`` (cache check)."""

TIMEOUT_DOCKER_INSPECT: Final = 10
"""``docker image inspect`` in ensure_image / study runner image preparation."""

TIMEOUT_DOCKER_STOP: Final = 10
"""``docker stop`` graceful shutdown."""

# NVIDIA tool subprocess timeouts
TIMEOUT_NVCC: Final = 5
"""``nvcc --version`` subprocess."""

TIMEOUT_NVIDIA_SMI: Final = 10
"""``nvidia-smi`` query subprocess."""

# Background task timeouts
TIMEOUT_ENV_SNAPSHOT: Final = 10
"""Environment snapshot collection future."""

# Thread / process lifecycle timeouts
TIMEOUT_THREAD_JOIN: Final = 5
"""Thread joins, process teardown."""

TIMEOUT_SIGTERM_GRACE: Final = 2
"""Grace period after SIGTERM before SIGKILL."""

TIMEOUT_INTERRUPT_POLL: Final = 1
"""Interrupt event wait loop tick."""

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
    "CONTAINER_EXCHANGE_DIR",
    "DECODER_PARAM_SUPPORT",
    "DECODING_SUPPORT",
    "DOCKER_PULL_TIMEOUT",
    "DTYPE_SUPPORT",
    "ENV_CARBON_INTENSITY",
    "ENV_CONFIG_PATH",
    "ENV_DATACENTER_PUE",
    "ENV_HF_TOKEN",
    "ENV_IMAGE_PREFIX",
    "ENV_LOG_LEVEL",
    "ENV_NO_PROMPT",
    "ENV_OUTPUT_DIR",
    "ENV_RUNNER_PREFIX",
    "ENV_SAVE_TIMESERIES",
    "ENV_TABLE_ROWS",
    "RUNNER_DOCKER",
    "RUNNER_LOCAL",
    "SOURCE_MULTI_BACKEND_ELEVATION",
    "TEMP_PREFIX_ENV_FILE",
    "TEMP_PREFIX_EXCHANGE",
    "TEMP_PREFIX_TIMESERIES",
    "TIMEOUT_DOCKER_CLI",
    "TIMEOUT_DOCKER_INSPECT",
    "TIMEOUT_DOCKER_STOP",
    "TIMEOUT_ENV_SNAPSHOT",
    "TIMEOUT_INTERRUPT_POLL",
    "TIMEOUT_NVCC",
    "TIMEOUT_NVIDIA_SMI",
    "TIMEOUT_SIGTERM_GRACE",
    "TIMEOUT_THREAD_JOIN",
    "BackendName",
    "RunnerMode",
]
