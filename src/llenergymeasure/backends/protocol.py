"""Backend protocol contracts for inference plugins and the harness interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.metrics import WarmupResult


@dataclass
class InferenceOutput:
    """Minimal output from one backend inference run.

    Backend-specific data (e.g. vLLM RequestOutput objects) goes in extras.
    The harness uses these fields to assemble the full ExperimentResult.
    """

    elapsed_time_sec: float
    input_tokens: int
    output_tokens: int
    peak_memory_mb: float
    model_memory_mb: float
    batch_times: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    inference_time_sec: float = 0.0  # Set by harness after perf_counter brackets (H1)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class BackendPlugin(Protocol):
    """Contract for thin inference plugins.

    MeasurementHarness owns the full measurement lifecycle (energy tracking,
    CUDA sync, FLOPs estimation, result assembly). Plugins own only inference.
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g. 'pytorch', 'vllm', 'tensorrt')."""
        ...

    def load_model(
        self,
        config: ExperimentConfig,
        on_substep: Callable[[str, float], None] | None = None,
    ) -> Any:
        """Load model into memory. Returns opaque model object passed to warmup/run_inference/cleanup.

        Args:
            config: Experiment configuration.
            on_substep: Optional callback ``(text, elapsed_sec)`` for reporting
                sub-operation progress (e.g. tokenizer loaded, engine compiled).
        """
        ...

    def warmup(self, config: ExperimentConfig, model: Any, prompts: list[str]) -> WarmupResult:
        """Run warmup iterations.

        Args:
            config: Experiment configuration.
            model: Opaque model object from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            WarmupResult (thermal_floor_wait_s set by harness).
        """
        ...

    def run_warmup_prompt(self, config: ExperimentConfig, model: Any, prompt: str) -> float:
        """Run one warmup prompt and return latency in ms.

        Returns 0.0 to signal the harness should skip CV-based convergence
        (e.g. vLLM/TRT-LLM use single-token kernel warmup instead).

        Args:
            config: Experiment configuration.
            model: Opaque model object from load_model().
            prompt: Single warmup prompt text.

        Returns:
            Latency in milliseconds, or 0.0 to opt out of convergence loop.
        """
        ...

    def run_inference(
        self, config: ExperimentConfig, model: Any, prompts: list[str]
    ) -> InferenceOutput:
        """Run inference over all prompts.

        Args:
            config: Experiment configuration.
            model: Opaque model object from load_model().
            prompts: Pre-loaded prompts (loaded by harness before measurement window).

        Returns:
            InferenceOutput with token counts, timing, and memory stats.
        """
        ...

    def cleanup(self, model: Any) -> None:
        """Release model from memory and clear CUDA cache."""
        ...

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """Validate config against hardware capabilities. Returns error strings (empty = valid)."""
        ...
