"""Single source of truth for project-wide constants.

Centralises engine capabilities, runner modes, environment variable names,
temp file prefixes, and infrastructure timeout values. Consumers include:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (engine capability metadata)
- CLI help generation
- Infrastructure modules (Docker runner, runner resolution, image registry)
- Study runner and container lifecycle management

Do not inline these values in validators or infrastructure code — always
import from here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Literal

# ---------------------------------------------------------------------------
# Sampling presets (shared across engines)
# ---------------------------------------------------------------------------
# Preset values are aligned with industry conventions (vLLM, OpenAI, MLPerf).
# do_sample is intentionally absent from presets — it's HF-specific; the
# transformers engine builder infers do_sample=False from temperature=0.

SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "deterministic": {"temperature": 0.0},
    "standard": {"temperature": 1.0, "top_p": 0.95},
    "creative": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3},
}

# ---------------------------------------------------------------------------
# Engine enum — the ONE source of truth for backend engine identifiers.
# Add a new backend by adding a member here; everything else derives from it
# (Pydantic validation, membership checks, ordered iteration, CLI choices).
# ---------------------------------------------------------------------------


class Engine(str, Enum):
    """Supported inference backends.

    Uses (str, Enum) so members compare and hash equal to their string values
    on Python 3.10+. The __str__ override ensures str(Engine.X) returns the
    raw value (e.g. "transformers"), not "Engine.TRANSFORMERS".
    StrEnum (3.11+) is intentionally avoided to keep requires-python = ">=3.10".
    """

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TENSORRT = "tensorrt"

    def __str__(self) -> str:
        return self.value


ALL_ENGINES: Final[frozenset[Engine]] = frozenset(Engine)
"""Unordered engine set — use for O(1) membership checks (``engine in ALL_ENGINES``)."""

# ---------------------------------------------------------------------------
# Runner mode constants
# ---------------------------------------------------------------------------

RUNNER_LOCAL: Final = "local"
RUNNER_DOCKER: Final = "docker"
CONTAINER_EXCHANGE_DIR: Final = "/run/llem"
"""Mount point inside Docker containers for config/result exchange."""

SOURCE_MULTI_ENGINE_ELEVATION: Final = "multi_engine_elevation"
"""RunnerSpec source tag when an engine is auto-elevated to Docker for multi-engine isolation."""

RunnerMode = Literal["local", "docker"]

DOCKER_PULL_TIMEOUT: Final = 1800
"""Maximum seconds to wait for ``docker pull`` (30 min — generous for large images like TensorRT ~10 GB)."""

# ---------------------------------------------------------------------------
# Environment variable name constants
# ---------------------------------------------------------------------------

ENV_RUNNER_PREFIX: Final = "LLEM_RUNNER_"
"""Prefix for per-engine runner override env vars (e.g. ``LLEM_RUNNER_TRANSFORMERS=docker``)."""

ENV_IMAGE_PREFIX: Final = "LLEM_IMAGE_"
"""Prefix for per-engine image override env vars (e.g. ``LLEM_IMAGE_VLLM=custom:tag``)."""

ENV_CARBON_INTENSITY: Final = "LLEM_CARBON_INTENSITY"
ENV_DATACENTER_PUE: Final = "LLEM_DATACENTER_PUE"
ENV_NO_PROMPT: Final = "LLEM_NO_PROMPT"
ENV_HF_TOKEN: Final = "HF_TOKEN"
ENV_OUTPUT_DIR: Final = "LLEM_OUTPUT_DIR"
ENV_SAVE_TIMESERIES: Final = "LLEM_SAVE_TIMESERIES"
ENV_CONFIG_PATH: Final = "LLEM_CONFIG_PATH"
ENV_BASELINE_SPEC_PATH: Final = "LLEM_BASELINE_SPEC_PATH"
"""Path inside a baseline container where the entrypoint reads its spec JSON."""
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
# Engine capability dicts
# ---------------------------------------------------------------------------

# Precision modes supported by each engine.
# "float32" = full precision, "float16" = half, "bfloat16" = brain float16.
# Note: fp16/bf16 require GPU. The cpu engine (future) would be fp32-only.
# GPU detection and cpu-dtype cross-validation is handled at pre-flight.
DTYPE_SUPPORT: dict[Engine, list[str]] = {
    Engine.TRANSFORMERS: ["float32", "float16", "bfloat16"],
    Engine.VLLM: ["float16", "bfloat16"],  # vLLM does not support fp32 inference
    Engine.TENSORRT: ["float16", "bfloat16"],  # TRT-LLM does not support fp32 inference
}

# Decoding strategies supported by each engine.
# "sampling" = do_sample=True path; "greedy" = do_sample=False path.
DECODING_SUPPORT: dict[Engine, list[str]] = {
    Engine.TRANSFORMERS: ["greedy", "sampling"],  # full HuggingFace generate() support
    Engine.VLLM: ["greedy", "sampling"],  # vLLM supports both via SamplingParams
    Engine.TENSORRT: ["greedy", "sampling"],  # TRT-LLM supports both
}

# Sampling parameter support is now derivable via Pydantic introspection of
# each engine's <Engine>SamplingConfig model (get_engine_params), so the
# DECODER_PARAM_SUPPORT constant is no longer needed.

# Map from engine name to the Python package that provides it.
# Used by preflight checks and CLI to verify engine availability.
ENGINE_PACKAGES: dict[Engine, str] = {
    Engine.TRANSFORMERS: "transformers",
    Engine.VLLM: "vllm",
    Engine.TENSORRT: "tensorrt_llm",
}

__all__ = [
    "ALL_ENGINES",
    "CONTAINER_EXCHANGE_DIR",
    "DECODING_SUPPORT",
    "DOCKER_PULL_TIMEOUT",
    "DTYPE_SUPPORT",
    "ENGINE_PACKAGES",
    "ENV_BASELINE_SPEC_PATH",
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
    "SAMPLING_PRESETS",
    "SOURCE_MULTI_ENGINE_ELEVATION",
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
    "Engine",
    "RunnerMode",
]
