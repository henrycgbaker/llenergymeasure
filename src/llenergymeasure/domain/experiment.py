"""Experiment and study result domain models."""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from llenergymeasure.domain.environment import EnvironmentSnapshot
from llenergymeasure.domain.metrics import (
    ComputeMetrics,
    EnergyBreakdown,
    EnergyMetrics,
    ExtendedEfficiencyMetrics,
    InferenceMetrics,
    LatencyStatistics,
    MultiGPUMetrics,
    ThermalThrottleInfo,
    WarmupResult,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


@functools.lru_cache(maxsize=128)
def _hash_canonical(canonical: str) -> str:
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def compute_measurement_config_hash(config: ExperimentConfig) -> str:
    """SHA-256[:16] of ExperimentConfig. Layer 3 fields excluded by design.

    Layer 3 fields (datacenter_pue, grid_carbon_intensity) are not in
    ExperimentConfig (they live in user config only), so model_dump()
    naturally excludes them. No special exclusion logic needed.
    """
    canonical = json.dumps(config.model_dump(), sort_keys=True)
    return _hash_canonical(canonical)


class Timestamps(BaseModel):
    """Timing information for an experiment run."""

    start: datetime = Field(..., description="Experiment start time")
    end: datetime = Field(..., description="Experiment end time")
    duration_sec: float = Field(..., description="Duration in seconds")

    @classmethod
    def from_times(cls, start: datetime, end: datetime) -> Timestamps:
        """Create Timestamps from start and end times."""
        duration = (end - start).total_seconds()
        return cls(start=start, end=end, duration_sec=duration)


class RawProcessResult(BaseModel):
    """Raw metrics from a single process - never aggregated inline.

    This represents the output from one GPU/process during a distributed
    experiment. Raw results are saved individually and aggregated separately.
    """

    schema_version: str = Field(default="2.0", description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    backend: str = Field(default="pytorch", description="Inference backend used")
    backend_version: str | None = Field(
        default=None, description="Backend version string for reproducibility"
    )
    process_index: int = Field(..., description="Process rank in distributed setup")
    gpu_id: int = Field(..., description="GPU device index")
    gpu_name: str = Field(default="", description="GPU model name")
    gpu_is_mig: bool = Field(default=False, description="Whether GPU is a MIG instance")
    gpu_mig_profile: str | None = Field(default=None, description="MIG profile if applicable")
    energy_measurement_warning: str | None = Field(
        default=None,
        description="Warning about energy measurement accuracy (e.g., MIG limitations)",
    )
    config_name: str = Field(..., description="Configuration name for this experiment")
    model_name: str = Field(..., description="Model name/path used")
    timestamps: Timestamps = Field(..., description="Timing information")
    inference_metrics: InferenceMetrics = Field(..., description="Inference performance metrics")
    energy_metrics: EnergyMetrics = Field(..., description="Energy consumption metrics")
    compute_metrics: ComputeMetrics = Field(..., description="Computational metrics")

    # Effective configuration (for reproducibility)
    effective_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Full resolved config",
    )

    # Extended efficiency metrics (always present, fields null when not computable)
    extended_metrics: ExtendedEfficiencyMetrics = Field(
        default_factory=ExtendedEfficiencyMetrics,
        description="Extended efficiency metrics (TPOT, memory, GPU utilisation, etc.)",
    )

    # Raw data for late aggregation of extended metrics
    per_request_latencies_ms: list[float] = Field(
        default_factory=list,
        description="Per-request E2E latencies for late aggregation",
    )
    gpu_utilisation_samples: list[float] = Field(
        default_factory=list,
        description="GPU utilisation samples for late aggregation",
    )

    # Energy breakdown, thermal, warmup
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None,
        description="Detailed energy breakdown with baseline adjustment",
    )
    thermal_throttle: ThermalThrottleInfo | None = Field(
        default=None,
        description="GPU thermal and power throttling information",
    )
    warmup_result: WarmupResult | None = Field(
        default=None,
        description="Warmup convergence detection result",
    )

    model_config = {"frozen": True}


class AggregationMetadata(BaseModel):
    """Metadata about the aggregation process."""

    method: str = Field(
        default="sum_energy_avg_throughput",
        description="Aggregation method used",
    )
    num_processes: int = Field(..., description="Number of processes aggregated")
    temporal_overlap_verified: bool = Field(
        default=False, description="Whether process timestamps overlapped"
    )
    gpu_attribution_verified: bool = Field(
        default=False, description="Whether GPU IDs were unique (no double counting)"
    )
    warnings: list[str] = Field(default_factory=list, description="Aggregation warnings")


class ExperimentResult(BaseModel):
    """Experiment result — the user-visible output of a measurement run.

    Combines raw results from all processes into a single result with proper
    aggregation (sum energy, average throughput). For single-GPU experiments,
    process_results has exactly one item.

    v2.0 schema: all fields ship together (decision #50).
    """

    # Identity
    schema_version: str = Field(default="2.0", description="Result schema version")
    experiment_id: str = Field(..., description="Unique experiment identifier")
    measurement_config_hash: str = Field(
        ..., description="SHA-256[:16] of ExperimentConfig (environment excluded)"
    )

    # Backend
    backend: str = Field(default="pytorch", description="Inference backend used")
    backend_version: str | None = Field(
        default=None, description="Backend version string for reproducibility"
    )

    # Methodology
    measurement_methodology: Literal["total", "steady_state", "windowed"] = Field(
        ..., description="What was measured — total run, steady-state window, or explicit window"
    )
    steady_state_window: tuple[float, float] | None = Field(
        default=None,
        description="(start_sec, end_sec) of measurement window relative to experiment start",
    )

    # Core metrics
    total_tokens: int = Field(..., description="Total tokens across all processes")
    total_energy_j: float = Field(..., description="Total energy (sum across processes)")
    total_inference_time_sec: float = Field(..., description="Total inference time")
    avg_tokens_per_second: float = Field(..., description="Average throughput")
    avg_energy_per_token_j: float = Field(..., description="Average energy per token")
    total_flops: float = Field(..., description="Total FLOPs (reference metadata)")

    # FLOPs derived fields (computed from total_flops + token/time denominators)
    flops_per_output_token: float | None = Field(
        default=None,
        description="FLOPs per output (decode) token. None if total_flops=0 or output_tokens=0.",
    )
    flops_per_input_token: float | None = Field(
        default=None,
        description="FLOPs per input (prefill) token. None if total_flops=0 or input_tokens=0.",
    )
    flops_per_second: float | None = Field(
        default=None,
        description="FLOPs throughput (total_flops / inference_time_sec). None if time=0 or flops=0.",
    )

    # Energy detail
    baseline_power_w: float | None = Field(
        default=None, description="Idle GPU power (W) measured before experiment"
    )
    energy_adjusted_j: float | None = Field(
        default=None, description="Baseline-subtracted energy attributable to inference"
    )
    energy_per_device_j: list[float] | None = Field(
        default=None, description="Per-GPU energy breakdown (Zeus backend only)"
    )
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None, description="Detailed energy breakdown with baseline adjustment"
    )

    # Multi-GPU (from result-schema.md design)
    multi_gpu: MultiGPUMetrics | None = Field(
        default=None, description="Multi-GPU metrics. None for single-GPU runs."
    )

    # Environment
    environment_snapshot: EnvironmentSnapshot | None = Field(
        default=None, description="Full software+hardware environment snapshot"
    )

    # Quality
    measurement_warnings: list[str] = Field(
        default_factory=list,
        description="Measurement quality warnings (e.g., short duration, thermal drift)",
    )
    warmup_excluded_samples: int | None = Field(
        default=None,
        description="Number of prompts excluded during warmup. None when methodology=total.",
    )
    reproducibility_notes: str = Field(
        default=(
            "Energy measured via NVML polling. Accuracy +/-5%. "
            "Results may vary with thermal state and system load."
        ),
        description="Fixed disclaimer about measurement accuracy",
    )

    # Timeseries sidecar reference
    timeseries: str | None = Field(
        default=None,
        description="Relative filename of timeseries sidecar (e.g. 'timeseries.parquet')",
    )

    # Timestamps
    start_time: datetime = Field(..., description="Earliest process start time")
    end_time: datetime = Field(..., description="Latest process end time")

    # Effective configuration (for reproducibility)
    effective_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Full resolved config",
    )

    # Per-process breakdown (embedded, not separate files per CONTEXT.md)
    process_results: list[RawProcessResult] = Field(
        default_factory=list, description="Original per-process results"
    )
    aggregation: AggregationMetadata | None = Field(
        default=None, description="Aggregation metadata (populated for multi-GPU)"
    )

    # Carry-forward fields (v1.x compat, used by aggregation/CLI)
    thermal_throttle: ThermalThrottleInfo | None = Field(
        default=None, description="GPU thermal and power throttling information"
    )
    warmup_result: WarmupResult | None = Field(
        default=None, description="Warmup convergence detection result"
    )
    latency_stats: LatencyStatistics | None = Field(
        default=None,
        description="Computed TTFT/ITL statistics from streaming inference",
    )
    extended_metrics: ExtendedEfficiencyMetrics | None = Field(
        default=None, description="Extended efficiency metrics (when computed)"
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def duration_sec(self) -> float:
        """Total experiment duration."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def tokens_per_joule(self) -> float:
        """Overall energy efficiency."""
        if self.total_energy_j > 0:
            return self.total_tokens / self.total_energy_j
        return 0.0


class StudySummary(BaseModel):
    """Aggregate summary statistics for a completed study."""

    total_experiments: int = Field(..., description="Total experiments in the study")
    completed: int = Field(default=0, description="Number of successfully completed experiments")
    failed: int = Field(default=0, description="Number of failed experiments")
    total_wall_time_s: float = Field(default=0.0, description="Total wall-clock time in seconds")
    total_energy_j: float = Field(default=0.0, description="Total energy consumed in joules")
    unique_configurations: int | None = Field(
        default=None,
        description="Number of distinct experiment configurations (total_experiments / n_cycles)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Runtime warnings (CLI narrowing, failures, etc.)",
    )


class StudyResult(BaseModel):
    """Final return value of a study run.

    Distinct from StudyManifest (the in-progress checkpoint). StudyResult is
    assembled once after all experiments complete (or after interrupt) and returned
    to the caller.
    """

    experiments: list[ExperimentResult] = Field(
        default_factory=list, description="Results for each experiment in the study"
    )
    name: str | None = Field(default=None, description="Study name")
    study_design_hash: str | None = Field(
        default=None, description="16-char SHA-256 hex of experiment list"
    )
    measurement_protocol: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flat dict from ExecutionConfig: n_cycles, cycle_order, experiment_gap_seconds, "
            "cycle_gap_seconds, shuffle_seed, experiment_timeout_seconds"
        ),
    )
    result_files: list[str] = Field(
        default_factory=list,
        description="Paths to per-experiment result.json files (paths, not embedded)",
    )
    summary: StudySummary | None = Field(default=None, description="Aggregate summary statistics")
