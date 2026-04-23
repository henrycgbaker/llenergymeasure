"""Configuration models for LLM efficiency measurement experiments (v2.0 schema).

This module defines the Tier 1 (Universal) configuration that applies identically
across all engines. Engine-specific parameters live in engine_configs.py.

v2.0 field renames from v1.x:
    model_name         -> model
    fp_precision       -> dtype
    num_input_prompts  -> n
    extra_metadata     -> passthrough_kwargs

v2.0 removals:
    config_name, schema_version, TrafficSimulation, ScheduleConfig, IOConfig,
    query_rate, streaming, streaming_warmup_requests, save_outputs,
    decode_token_to_text, extra_metadata, gpus, min_output_tokens
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from llenergymeasure.config.ssot import SAMPLING_PRESETS, Engine
from llenergymeasure.config.warnings import ConfigValidationWarning

logger = logging.getLogger(__name__)

#: Valid energy sampler names for ``energy_sampler`` fields.
EnergySamplerName = Literal["auto", "nvml", "zeus", "codecarbon"]

#: Literal type of supported sampling presets (derived from SAMPLING_PRESETS keys).
SamplingPreset = Literal["deterministic", "standard", "creative", "factual"]

if TYPE_CHECKING:
    from llenergymeasure.config.engine_configs import (
        TensorRTConfig,
        TransformersConfig,
        VLLMConfig,
    )
    from llenergymeasure.engines.protocol import DormantField
    from llenergymeasure.engines.vendored_rules.loader import VendoredRulesLoader


#: Module-level per-process cache. Populated lazily on first
#: :meth:`ExperimentConfig._apply_vendored_rules` call so that module import
#: never runs the loader (which lives under ``llenergymeasure.engines`` and
#: would trigger a circular import with :mod:`engines.protocol`).
#:
#: Tests that need a fresh loader (e.g. to pick up monkeypatched rules on
#: disk) can reassign this binding directly or call
#: :func:`_reset_rules_loader_cache`.
_RULES_LOADER: VendoredRulesLoader | None = None


def _get_rules_loader() -> VendoredRulesLoader:
    """Return the shared loader, constructing it on first call."""
    global _RULES_LOADER
    if _RULES_LOADER is None:
        # Import via the submodule (not ``engines.__init__``) to avoid the
        # engines-package import cycle; the package root pulls in
        # :mod:`engines.protocol`, which imports :class:`ExperimentConfig`
        # from *this* module.
        from llenergymeasure.engines.vendored_rules.loader import VendoredRulesLoader

        _RULES_LOADER = VendoredRulesLoader()
    return _RULES_LOADER


def _reset_rules_loader_cache() -> None:
    """Reset the module-level loader; primarily used by tests."""
    global _RULES_LOADER
    _RULES_LOADER = None


# =============================================================================
# Warmup Configuration
# =============================================================================


class WarmupConfig(BaseModel):
    """Warmup configuration for the measurement phase.

    Controls the warmup phase before measurement begins. Default uses fixed
    iteration count plus a thermal floor wait. Set convergence_detection=True
    to enable adaptive CV-based convergence (additive to n_warmup).

    # Confidence: n_warmup=5 HIGH (DeepSpeed 5-10, Zeus 10, AIEnergyScore 10)
    # Confidence: thermal_floor_seconds=60 HIGH (MLPerf Power mandates 60s minimum)
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable warmup phase")

    n_warmup: int = Field(
        default=5,
        ge=1,
        description="Number of full-length warmup prompts before measurement",
    )
    thermal_floor_seconds: float = Field(
        default=60.0,
        ge=30.0,
        description="Minimum seconds to wait after warmup before measuring (thermal stabilisation). Minimum 30s enforced.",
    )

    # CV convergence detection (opt-in, additive to n_warmup)
    convergence_detection: bool = Field(
        default=False,
        description="Enable CV-based convergence detection (additive to n_warmup)",
    )
    cv_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="CV target for convergence (only used when convergence_detection=True)",
    )
    max_prompts: int = Field(
        default=20,
        ge=5,
        description="Maximum warmup prompts when CV mode is on (safety cap)",
    )
    window_size: int = Field(
        default=3,
        ge=3,
        description="Sliding window size for CV calculation (3 balances responsiveness and stability)",
    )
    min_prompts: int = Field(
        default=5,
        ge=1,
        description="Minimum prompts before checking convergence (warm start)",
    )


# =============================================================================
# Baseline Configuration
# =============================================================================


class BaselineConfig(BaseModel):
    """Baseline power measurement configuration.

    Controls whether and how idle GPU power is measured before experiments,
    enabling baseline-adjusted energy attribution.

    Strategies:
        cached: Disk-persisted baseline with configurable TTL (default).
            Host measures once, writes to JSON, mounts into Docker containers.
        validated: Same as cached but periodically spot-checks for drift.
            If drift exceeds threshold, re-measures full baseline.
        fresh: Every experiment measures its own baseline. Most accurate
            but wastes ~30s per experiment.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable baseline power measurement")
    duration_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Baseline measurement duration in seconds",
    )
    strategy: Literal["cached", "validated", "fresh"] = Field(
        default="validated",
        description=(
            "Baseline caching strategy: 'cached' (disk-persisted TTL), "
            "'validated' (cached with periodic spot-check), "
            "'fresh' (measure every experiment)"
        ),
    )
    cache_ttl_seconds: float = Field(
        default=7200.0,
        ge=60.0,
        description=(
            "How long a cached baseline remains valid before re-measurement, in seconds. "
            "Only used with strategy='cached' or 'validated'."
        ),
    )
    validation_interval: int = Field(
        default=5,
        ge=1,
        description=(
            "Re-validate baseline every N experiments. Only used with strategy='validated'."
        ),
    )
    drift_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description=(
            "Power drift threshold (fraction) to trigger re-measurement. "
            "Only used with strategy='validated'."
        ),
    )


# =============================================================================
# Dataset Configuration
# =============================================================================


class DatasetConfig(BaseModel):
    """Dataset configuration for experiment prompts.

    source is one of:
    - Built-in alias (e.g. "aienergyscore")
    - Path to a .jsonl file
    """

    model_config = {"extra": "forbid"}

    source: str = Field(
        default="aienergyscore",
        min_length=1,
        description="Dataset source: built-in alias or .jsonl file path",
        json_schema_extra={"display_label": "Dataset Source", "role": "workload"},
    )
    n_prompts: int = Field(
        default=100,
        ge=1,
        description="Number of prompts to load or generate",
        json_schema_extra={"display_label": "Prompts", "role": "workload"},
    )
    order: Literal["interleaved", "grouped", "shuffled"] = Field(
        default="interleaved",
        description=(
            "Prompt ordering: interleaved (round-robin by source, file order), "
            "grouped (sorted by source), shuffled (seed-based random)"
        ),
    )


# =============================================================================
# LoRA Configuration
# =============================================================================


class LoRAConfig(BaseModel):
    """LoRA adapter configuration.

    Exactly one of adapter_id or adapter_path must be set.
    """

    model_config = {"extra": "forbid"}

    adapter_id: str | None = Field(default=None, description="HuggingFace Hub adapter ID")
    adapter_path: str | None = Field(default=None, description="Local path to adapter weights")
    merge_weights: bool = Field(
        default=False, description="Merge adapter weights into base model at load time"
    )

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> LoRAConfig:
        """Exactly one of adapter_id or adapter_path must be set."""
        has_id = self.adapter_id is not None
        has_path = self.adapter_path is not None
        if has_id == has_path:  # both set or neither set
            raise ValueError(
                "LoRAConfig requires exactly one of adapter_id or adapter_path. "
                f"Got: adapter_id={self.adapter_id!r}, adapter_path={self.adapter_path!r}"
            )
        return self


# =============================================================================
# Task Configuration (what to measure)
# =============================================================================


class TaskConfig(BaseModel):
    """What to measure: model identity, dataset, and workload shape.

    These fields define the scientific workload — changing any of them means
    you're measuring a fundamentally different task.
    """

    model_config = {"extra": "forbid"}

    model: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model ID or local path",
        json_schema_extra={"display_label": "Model"},
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration",
        json_schema_extra={"display_label": "Dataset"},
    )
    max_input_tokens: int | None = Field(
        default=256,
        ge=1,
        description=(
            "Max input token length for truncation. Keeps computation workload "
            "constant across experiments for fair comparison. None = no truncation."
        ),
        json_schema_extra={"display_label": "Max Input Tokens"},
    )
    max_output_tokens: int | None = Field(
        default=256,
        ge=1,
        description=(
            "Max output tokens (max_new_tokens for generation). "
            "None = generate until EOS or model context limit."
        ),
        json_schema_extra={"display_label": "Max Output Tokens"},
    )
    random_seed: int = Field(
        default=42,
        description="Per-experiment seed for all stochasticity: inference RNG and dataset ordering.",
    )


# =============================================================================
# Measurement Configuration (how to measure)
# =============================================================================


class MeasurementConfig(BaseModel):
    """How to measure: warmup, baseline, and energy sampling strategy.

    These fields control the measurement methodology — changing them affects
    measurement quality/accuracy but not the workload itself.
    """

    model_config = {"extra": "forbid"}

    warmup: WarmupConfig = Field(
        default_factory=WarmupConfig, description="Warmup phase configuration"
    )
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig, description="Baseline power measurement configuration"
    )
    energy_sampler: EnergySamplerName | None = Field(
        default="auto",
        description=(
            "Energy measurement backend. "
            "auto=best available (Zeus>NVML>CodeCarbon). null disables energy measurement."
        ),
        json_schema_extra={"display_label": "Sampler"},
    )


# =============================================================================
# Main Experiment Configuration (v2.0)
# =============================================================================


class ExperimentConfig(BaseModel):
    """v2.0 experiment configuration.

    Central configuration object controlling all aspects of a single LLM inference
    efficiency measurement. Organised into semantic groups:

    - task: What to measure (model, dataset, token limits, seed)
    - measurement: How to measure (warmup, baseline, energy sampler)
    - Engine sections (transformers:, vllm:, tensorrt:): How to execute

    The engine section must match the engine field. Providing a transformers:
    section when engine=vllm is a configuration error.
    """

    model_config = {"extra": "forbid"}

    # Task — what to measure
    task: TaskConfig = Field(..., description="Task configuration: model, dataset, workload shape")

    # Engine selection
    engine: Engine = Field(
        default=Engine.TRANSFORMERS,
        description="Inference engine",
        json_schema_extra={"display_label": "Engine"},
    )

    # Measurement — how to measure
    measurement: MeasurementConfig = Field(
        default_factory=MeasurementConfig,
        description="Measurement methodology: warmup, baseline, energy sampling",
    )

    # Sampling preset — expands into the active engine's sampling section
    sampling_preset: SamplingPreset | None = Field(
        default=None,
        description=(
            "Sampling preset. When set, preset values are merged into the active "
            "engine's sampling section at parse time; explicit YAML values take "
            "precedence over preset values."
        ),
    )

    # Engine sections (None = use engine's own defaults)
    transformers: TransformersConfig | None = Field(
        default=None,
        description="HuggingFace Transformers engine configuration (only used when engine=transformers)",
    )
    vllm: VLLMConfig | None = Field(
        default=None,
        description="vLLM-specific configuration (only used when engine=vllm)",
    )
    tensorrt: TensorRTConfig | None = Field(
        default=None,
        description="TensorRT-LLM configuration (only used when engine=tensorrt)",
    )

    # LoRA adapter (optional)
    lora: LoRAConfig | None = Field(default=None, description="LoRA adapter configuration")

    # Escape hatch — explicitly declared for extra="forbid" compatibility
    passthrough_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra kwargs passed through to engine at execution time. "
        "Keys must not collide with ExperimentConfig top-level fields.",
    )

    # -------------------------------------------------------------------------
    # Pre-validators (run before field parsing)
    # -------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def expand_sampling_preset(cls, data: Any) -> Any:
        """Merge ``sampling_preset`` values into the active engine's sampling section.

        Explicit YAML values take precedence over preset values (each preset key
        is applied via ``setdefault``). The preset name itself stays on the
        top-level model so it remains inspectable after parsing.
        """
        if not isinstance(data, dict):
            return data
        preset_name = data.get("sampling_preset")
        if not preset_name or preset_name not in SAMPLING_PRESETS:
            return data
        engine = data.get("engine", Engine.TRANSFORMERS)
        engine_key = engine.value if hasattr(engine, "value") else str(engine)
        # Ensure the engine section and its sampling sub-dict exist as dicts
        # (an explicit ``engine: null`` in YAML would otherwise leave None here).
        engine_section = data.get(engine_key)
        if not isinstance(engine_section, dict):
            engine_section = {}
            data[engine_key] = engine_section
        sampling_section = engine_section.get("sampling")
        if not isinstance(sampling_section, dict):
            sampling_section = {}
            engine_section["sampling"] = sampling_section
        for key, value in SAMPLING_PRESETS[preset_name].items():
            sampling_section.setdefault(key, value)
        return data

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_engine_section_match(self) -> ExperimentConfig:
        """Engine section must match the engine field.

        A transformers: section with engine=vllm is a configuration error — it indicates
        the researcher copied the wrong config block. Fail explicitly rather than
        silently ignoring the mismatched section.
        """
        if self.transformers is not None and self.engine != "transformers":
            raise ValueError(
                f"transformers: config section provided but engine={self.engine!r}. "
                "Remove the transformers: section or set engine: transformers."
            )
        if self.vllm is not None and self.engine != "vllm":
            raise ValueError(
                f"vllm: config section provided but engine={self.engine!r}. "
                "Remove the vllm: section or set engine: vllm."
            )
        if self.tensorrt is not None and self.engine != "tensorrt":
            raise ValueError(
                f"tensorrt: config section provided but engine={self.engine!r}. "
                "Remove the tensorrt: section or set engine: tensorrt."
            )
        return self

    @model_validator(mode="after")
    def validate_passthrough_kwargs_no_collision(self) -> ExperimentConfig:
        """passthrough_kwargs keys must not collide with ExperimentConfig fields.

        If a researcher writes passthrough_kwargs: {model: gpt2}, they intended to
        set the model field directly. Collisions are always a misconfiguration.
        """
        if self.passthrough_kwargs:
            top_level_fields = set(ExperimentConfig.model_fields.keys())
            collisions = set(self.passthrough_kwargs.keys()) & top_level_fields
            if collisions:
                raise ValueError(
                    f"passthrough_kwargs keys collide with ExperimentConfig fields: "
                    f"{sorted(collisions)}. Use the named fields instead."
                )
        return self

    # vLLM fp8 + float32 and TRT FP8 + float32 are rejected by the respective
    # VLLMConfig.dtype / TensorRTConfig.dtype Literal types at field validation
    # (neither engine accepts float32). No separate cross-validator needed.

    # Framework-rule validators (e.g. FlashAttention + float32 rejection) now
    # live in ``configs/validation_rules/{engine}.yaml`` and are applied by
    # ``_apply_vendored_rules`` below — never hand-written here.

    @model_validator(mode="after")
    def _apply_vendored_rules(self) -> ExperimentConfig:
        """Apply the vendored validation-rules corpus for ``self.engine``.

        Each matching rule dispatches by severity:

        - ``error`` → raise ``ValueError`` with ``[rule_id] <message>``
          (Pydantic wraps this into a ``ValidationError``).
        - ``warn`` → emit ``ConfigValidationWarning`` via ``warnings.warn``.
        - ``dormant`` / ``dormant_silent`` → append a ``DormantField`` entry
          to ``_dormant_observations`` for downstream preflight display and
          the sidecar audit trail.

        The observation list is attached via ``object.__setattr__`` because
        Pydantic's ``extra='forbid'`` rejects unknown attributes through the
        normal assignment path. Consumers read via ``getattr`` with a default
        of ``[]`` so the attribute's absence is a non-error.

        Missing corpus for an engine is not fatal — the rules layer is
        additive. A ``FileNotFoundError`` from the loader (e.g. running
        against an engine without a seeded corpus) is logged at debug level
        and skipped.
        """
        from llenergymeasure.engines.protocol import DormantField

        engine_name = self.engine.value if hasattr(self.engine, "value") else str(self.engine)

        try:
            rules = _get_rules_loader().load_rules(engine_name).rules
        except FileNotFoundError:
            logger.debug("No vendored rules corpus for engine %r; skipping.", engine_name)
            object.__setattr__(self, "_dormant_observations", [])
            return self

        dormant_observations: list[DormantField] = []
        for rule in rules:
            match = rule.try_match(self)
            if match is None:
                continue
            message = rule.render_message(match)
            annotated = f"[{rule.id}] {message}"
            if rule.severity == "error":
                raise ValueError(annotated)
            if rule.severity == "warn":
                warnings.warn(annotated, ConfigValidationWarning, stacklevel=2)
                continue
            if rule.severity in ("dormant", "dormant_silent"):
                dormant_observations.append(
                    DormantField(
                        declared_value=match.declared_value,
                        effective_value=match.effective_value,
                        reason=annotated,
                    )
                )
                continue
            # Unknown severity — treat as warning so a typo in the corpus
            # surfaces at runtime rather than failing silently.
            warnings.warn(
                f"[{rule.id}] unknown severity {rule.severity!r}: {message}",
                ConfigValidationWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "_dormant_observations", dormant_observations)
        return self


# Rebuild to resolve forward references for engine configs
def _rebuild_experiment_config() -> None:
    """Rebuild ExperimentConfig to resolve forward references."""
    from llenergymeasure.config.engine_configs import (
        TensorRTConfig,
        TransformersConfig,
        VLLMConfig,
    )

    ExperimentConfig.model_rebuild(
        _types_namespace={
            "VLLMConfig": VLLMConfig,
            "TransformersConfig": TransformersConfig,
            "TensorRTConfig": TensorRTConfig,
        }
    )


_rebuild_experiment_config()


class OutputConfig(BaseModel):
    """Study-level output configuration.

    Controls where results are written and what auxiliary artefacts are persisted.
    Lives on StudyConfig only - experiments don't own output config because output
    is an operational concern, not part of the scientific specification.

    Resolution chain (highest wins):
        study YAML output.results_dir > user_config.output.results_dir > "./results"
    """

    model_config = {"extra": "forbid"}

    results_dir: str | None = Field(
        default=None,
        description=(
            "Base directory for study results. A timestamped study subdirectory "
            "is created within this path. Resolved identically for local and "
            "Docker runs (Docker results are always written back to host). "
            "None = defer to user config or built-in default (./results)."
        ),
    )
    save_timeseries: bool = Field(
        default=True,
        description=(
            "Persist GPU power/thermal/memory timeseries as Parquet sidecar. "
            "NVML telemetry is always collected for throttle detection; this "
            "controls whether the full timeseries is written to disk."
        ),
    )


class ExecutionConfig(BaseModel):
    """Execution controls for a study (cycle repetition, ordering, gaps).

    Controls how many times the experiment list is repeated (n_cycles), the order
    in which experiments are executed across cycles (experiment_order), optional gaps
    between configs and cycles for thermal stabilisation, and an explicit shuffle
    seed override (default: derived from study_design_hash for reproducibility).

    Pydantic defaults are conservative (1 cycle, sequential, no gaps). The CLI
    will apply research-appropriate effective defaults (e.g. 3 cycles, shuffle).
    """

    model_config = {"extra": "forbid"}

    n_cycles: int = Field(
        default=1, ge=1, description="Number of times to repeat the experiment list"
    )
    experiment_order: Literal["sequential", "interleave", "shuffle", "reverse", "latin_square"] = (
        Field(
            default="sequential",
            description=(
                "Ordering strategy across cycles. "
                "sequential: [A,A,A,B,B,B], interleave: [A,B,A,B,A,B], "
                "shuffle: random per-cycle order."
            ),
        )
    )
    experiment_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds to wait between individual experiments. "
            "None = use machine default from user config."
        ),
    )
    cycle_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds to wait between full cycles. None = use machine default from user config."
        ),
    )
    shuffle_seed: int | None = Field(
        default=None,
        description=(
            "Explicit seed for shuffle experiment_order. "
            "None = derived from study_design_hash (same study always shuffles identically)."
        ),
    )
    skip_preflight: bool = Field(
        default=False,
        description=(
            "Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility). "
            "Useful for remote Docker daemon setups or CI environments. "
            "The --skip-preflight CLI flag always overrides this setting."
        ),
    )
    max_consecutive_failures: int = Field(
        default=10,
        ge=0,
        description=(
            "Circuit breaker threshold: abort after N consecutive failures. "
            "0 = disabled. 1 = fail-fast (no cooldown)."
        ),
    )
    circuit_breaker_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Cooldown pause before half-open probe experiment.",
    )
    wall_clock_timeout_hours: float | None = Field(
        default=None,
        gt=0.0,
        description="Study wall-clock timeout in hours. null = no limit.",
    )
    experiment_timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        description=(
            "Per-experiment wall-clock timeout in seconds. Applies to both the "
            "local subprocess path and the Docker container path. Experiments "
            "that exceed this budget are killed and recorded as TimeoutError; "
            "the circuit breaker counts them toward max_consecutive_failures."
        ),
    )


class StudyConfig(BaseModel):
    """Thin resolved container for a study (list of experiments + execution config).

    Populated by the study loader after sweep expansion. The experiments list
    contains fully-validated ExperimentConfig objects ready for execution.
    skipped_configs records any grid points that failed Pydantic validation so
    they can be displayed to the researcher in pre-flight output.
    """

    model_config = {"extra": "forbid"}

    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Resolved list of experiments to run"
    )
    study_name: str | None = Field(
        default=None, description="Study name (used in output directory naming)"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Study-level output configuration (results_dir, format, save_timeseries)",
    )
    study_execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Cycle repetition and ordering controls",
    )
    runners: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-engine runner configuration. Keys are engine names "
            "('transformers', 'vllm', 'tensorrt'), values are runner strings "
            "('local', 'docker', or 'docker:<image>'). "
            "None = use user config / auto-detection. "
            "Runner is metadata — not part of the experiment config hash."
        ),
    )
    images: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-engine Docker image overrides (orthogonal to runners). "
            "Keys are engine names, values are image references "
            "(e.g. 'ghcr.io/org/img:tag'). None = use smart default "
            "(local build → registry fallback). "
            "Image is metadata — not part of the experiment config hash."
        ),
    )
    study_design_hash: str | None = Field(
        default=None,
        description=(
            "16-char SHA-256 hex of the resolved experiment list (execution block excluded). "
            "Set by the loader after expansion; None before expansion."
        ),
    )
    skipped_configs: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Grid points that failed Pydantic validation during expansion. "
            "Persisted for post-hoc review and pre-flight display."
        ),
    )
