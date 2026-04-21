"""Engine protocol contracts for inference plugins and the harness interface."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llenergymeasure.config.models import ExperimentConfig


@dataclass
class InferenceOutput:
    """Minimal output from one engine inference run.

    Engine-specific data (e.g. vLLM RequestOutput objects) goes in extras.
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


@dataclass(frozen=True)
class DormantField:
    """A user-declared config field that the engine ignored or overrode.

    Distinguishes two shapes of dormancy:
      - Stripped: declared_value set, effective_value is None (field absent from
        effective kwargs, e.g. temperature under greedy decoding).
      - Overridden: effective_value != declared_value (engine remapped it).
    """

    declared_value: Any
    effective_value: Any | None
    reason: str | None = None


@dataclass(frozen=True)
class ConfigProbe:
    """Outcome of probing an ExperimentConfig against an engine.

    The probe observes what the engine would do with this config without
    loading weights, allocating GPU memory, or initialising engine contexts.

    Attributes:
        effective_engine_params: Kwargs that would be passed to the engine
            constructor (vllm.LLM, AutoModelForCausalLM, tensorrt_llm.LLM).
        effective_sampling_params: Kwargs that would be passed to the
            sampling-params constructor after any greedy stripping.
        dormant_fields: Keyed by dotted path (e.g. ``"vllm.sampling.top_p"``)
            — fields the user declared that the engine will silently ignore
            or override.
        errors: Engine-reported framework errors (T1/T2 construction,
            hardware checks). Non-empty means the config will not run as-is.
        warnings: Non-fatal observations.
    """

    effective_engine_params: dict[str, Any]
    effective_sampling_params: dict[str, Any]
    dormant_fields: dict[str, DormantField]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the probe captured no framework errors."""
        return len(self.errors) == 0


@runtime_checkable
class EnginePlugin(Protocol):
    """Contract for thin inference plugins.

    MeasurementHarness owns the full measurement lifecycle (energy tracking,
    CUDA sync, FLOPs estimation, result assembly). Plugins own only inference.
    """

    @property
    def name(self) -> str:
        """Engine identifier (e.g. 'transformers', 'vllm', 'tensorrt')."""
        ...

    @property
    def version(self) -> str:
        """Engine library version string for reproducibility."""
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

    def probe_config(self, config: ExperimentConfig) -> ConfigProbe:
        """Probe *config* for dormancy, framework errors, and effective params.

        Observes what the engine would do with the configuration without side
        effects: never loads model weights, allocates GPU memory, initialises
        engine contexts, or runs inference. Implementations MAY import engine
        libraries, download small metadata files (e.g. HF ``config.json``),
        construct meta-device models, and query NVML.

        Contract: this method never raises. All exceptions are caught and
        appended to :attr:`ConfigProbe.errors`.
        """
        ...
