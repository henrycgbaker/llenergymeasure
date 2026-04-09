"""Metrics domain models for LLM Bench."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

# =============================================================================
# Precision Metadata - For cross-backend comparisons
# =============================================================================


class PrecisionMetadata(BaseModel):
    """Precision metadata for cross-backend efficiency comparisons.

    Tracks the actual precision used for weights, activations, and compute
    operations. This enables normalised efficiency comparisons across backends
    using different quantization methods.

    Precision factors for effective FLOPs:
        FP32 = 1.0
        FP16/BF16 = 1.0 (same effective ops)
        FP8 = 0.5 (half precision)
        INT8 = 0.5 (half precision)
        INT4 = 0.25 (quarter precision)
    """

    weights: Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4", "mixed"] = Field(
        default="fp16",
        description="Weight storage precision",
    )
    activations: Literal["fp32", "fp16", "bf16", "fp8", "int8"] = Field(
        default="fp16",
        description="Activation precision during inference",
    )
    compute: Literal["fp32", "fp16", "bf16", "fp8", "int8", "tf32"] = Field(
        default="fp16",
        description="Actual compute precision",
    )

    # For mixed precision - breakdown by layer type
    mixed_precision_breakdown: dict[str, float] | None = Field(
        default=None,
        description="Breakdown of precision usage by layer type (for mixed precision)",
    )

    # Quality tracking
    quantization_method: str | None = Field(
        default=None,
        description="Quantization method used (bitsandbytes, gptq, awq, trt_ptq, etc.)",
    )
    perplexity_degradation: float | None = Field(
        default=None,
        description="Estimated perplexity degradation vs FP16 baseline (0.0 = no degradation)",
    )

    @property
    def precision_factor(self) -> float:
        """Compute effective FLOPs factor based on precision.

        Used to calculate effective_flops = theoretical_flops * precision_factor.
        Lower precision = lower effective FLOPs (less computational work).
        """
        precision_factors = {
            "fp32": 1.0,
            "fp16": 1.0,
            "bf16": 1.0,
            "tf32": 1.0,
            "fp8": 0.5,
            "int8": 0.5,
            "int4": 0.25,
            "mixed": 0.75,  # Conservative estimate for mixed
        }
        return precision_factors.get(self.compute, 1.0)


# =============================================================================
# FLOPs Result
# =============================================================================


class FlopsResult(BaseModel):
    """FLOPs estimation result with provenance tracking.

    Tracks both the estimated value and the method used to obtain it,
    allowing downstream consumers to understand confidence levels.

    Note: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
    computation happens at FP16 after dequantization.
    """

    value: float = Field(..., description="Estimated FLOPs count")
    method: Literal["calflops", "architecture", "parameter_estimate", "palm_formula"] = Field(
        ..., description="Estimation method used"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level of the estimate"
    )
    precision: str = Field(..., description="Compute precision (e.g., fp16, fp32)")
    notes: str | None = Field(default=None, description="Additional context or warnings")

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid (non-zero) estimate."""
        return self.value > 0


class InferenceMetrics(BaseModel):
    """Metrics from model inference."""

    total_tokens: int = Field(..., description="Total tokens generated")
    input_tokens: int = Field(..., description="Number of input/prompt tokens")
    output_tokens: int = Field(..., description="Number of output/generated tokens")
    inference_time_sec: float = Field(..., description="Total inference time in seconds")
    tokens_per_second: float = Field(..., description="Throughput in tokens/second")
    latency_per_token_ms: float = Field(..., description="Average latency per token in ms")
    time_to_first_token_ms: float | None = Field(
        default=None, description="Average time to first token in ms (if available)"
    )
    # Raw latency measurements for streaming mode (late aggregation)
    # Forward reference to LatencyMeasurements (defined below in this module)
    latency_measurements: Any | None = Field(
        default=None,
        description="Raw TTFT/ITL samples from streaming inference",
    )

    @property
    def throughput(self) -> float:
        """Alias for tokens_per_second."""
        return self.tokens_per_second


class EnergyMetrics(BaseModel):
    """Energy consumption metrics."""

    total_energy_j: float = Field(..., description="Total energy consumed in Joules")
    gpu_energy_j: float = Field(0.0, description="GPU energy in Joules")
    cpu_energy_j: float = Field(0.0, description="CPU energy in Joules")
    ram_energy_j: float = Field(0.0, description="RAM energy in Joules")
    gpu_power_w: float = Field(0.0, description="Average GPU power in Watts")
    cpu_power_w: float = Field(0.0, description="Average CPU power in Watts")
    duration_sec: float = Field(..., description="Measurement duration in seconds")
    emissions_kg_co2: float = Field(0.0, description="Carbon emissions in kg CO2")
    energy_per_token_j: float = Field(0.0, description="Energy per token in Joules")

    @classmethod
    def placeholder(cls, duration_sec: float = 0.0) -> "EnergyMetrics":
        """Create placeholder metrics when energy measurement is unavailable.

        Args:
            duration_sec: Optional duration to record (e.g. from inference time).

        Returns:
            EnergyMetrics with all values zeroed.
        """
        return cls(
            total_energy_j=0.0,
            gpu_energy_j=0.0,
            cpu_energy_j=0.0,
            ram_energy_j=0.0,
            gpu_power_w=0.0,
            cpu_power_w=0.0,
            duration_sec=duration_sec,
            emissions_kg_co2=0.0,
            energy_per_token_j=0.0,
        )

    @property
    def total_power_w(self) -> float:
        """Total average power consumption."""
        return self.gpu_power_w + self.cpu_power_w


# =============================================================================
# Schema v3: Energy Breakdown, Thermal Throttle, Warmup Result
# =============================================================================


class MultiGPUMetrics(BaseModel):
    """Per-device energy breakdown for multi-GPU experiments."""

    num_gpus: int = Field(..., description="Number of GPUs used")
    energy_per_gpu_j: list[float] = Field(..., description="Per-device energy in joules")
    energy_total_j: float = Field(..., description="Sum of energy across all devices")
    energy_per_output_token_j: float = Field(
        ..., description="Primary cross-configuration efficiency metric"
    )


class EnergyBreakdown(BaseModel):
    """Detailed energy breakdown with baseline adjustment.

    Separates raw measured energy from baseline-adjusted values to enable
    accurate attribution of energy to inference work (not idle power).
    """

    raw_j: float = Field(..., description="Total measured energy in Joules")
    adjusted_j: float | None = Field(
        default=None,
        description="Baseline-adjusted energy (raw - baseline * duration) in Joules",
    )
    baseline_power_w: float | None = Field(
        default=None,
        description="Measured baseline idle power in Watts",
    )
    baseline_method: str | None = Field(
        default=None,
        description="How baseline was obtained ('cached', 'validated', 'fresh', 'unavailable')",
    )
    baseline_timestamp: datetime | None = Field(
        default=None,
        description="When baseline power was measured",
    )
    baseline_cache_age_sec: float | None = Field(
        default=None,
        description="Age of cached baseline measurement in seconds",
    )


class ThermalThrottleInfo(BaseModel):
    """GPU thermal and power throttling information.

    Tracks whether any throttling occurred during an experiment, which
    can invalidate energy and performance measurements.
    """

    thermal: bool = Field(
        default=False,
        description="GPU thermal throttling detected",
    )
    power: bool = Field(
        default=False,
        description="Power brake throttling detected",
    )
    sw_thermal: bool = Field(
        default=False,
        description="Software thermal slowdown detected",
    )
    hw_thermal: bool = Field(
        default=False,
        description="Hardware thermal slowdown detected",
    )
    hw_power: bool = Field(
        default=False,
        description="Hardware power brake slowdown detected",
    )
    throttle_duration_sec: float = Field(
        default=0.0,
        description="Estimated duration of throttling in seconds",
    )
    max_temperature_c: float | None = Field(
        default=None,
        description="Peak GPU temperature during experiment in Celsius",
    )
    throttle_timestamps: list[float] = Field(
        default_factory=list,
        description="Timestamps (seconds from start) when throttle was detected",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def detected(self) -> bool:
        """Whether any throttling occurred during experiment."""
        return any((self.thermal, self.power, self.sw_thermal, self.hw_thermal, self.hw_power))


class WarmupResult(BaseModel):
    """Result of warmup convergence detection.

    Records whether the warmup phase achieved stable latency (low CV)
    before the measurement phase began.
    """

    converged: bool = Field(..., description="Whether convergence was achieved")
    final_cv: float = Field(..., description="Final coefficient of variation")
    iterations_completed: int = Field(..., description="Number of warmup prompts run")
    target_cv: float = Field(..., description="Configured CV threshold")
    max_prompts: int = Field(..., description="Configured maximum warmup iterations")
    latencies_ms: list[float] = Field(
        default_factory=list,
        description="Warmup latencies in ms (for debugging)",
    )
    thermal_floor_wait_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds spent in thermal floor wait after warmup. Set by caller, not by warmup_until_converged().",
    )


class ComputeMetrics(BaseModel):
    """Computational metrics (FLOPs, memory)."""

    flops_total: float = Field(..., description="Total FLOPs for the inference")
    flops_per_token: float = Field(0.0, description="FLOPs per token")
    flops_per_second: float = Field(0.0, description="FLOPs throughput")
    peak_memory_mb: float = Field(
        0.0,
        description=(
            "Peak GPU memory allocated during the inference measurement window (MB). "
            "Captured via torch.cuda.max_memory_allocated() after resetting stats at "
            "measurement start (post-warmup). Reflects KV cache + activations + "
            "batch buffers, NOT model weights. 0.0 = not measured."
        ),
    )
    model_memory_mb: float = Field(
        0.0,
        description=(
            "GPU memory after model load, before inference (MB). "
            "Represents model weights + framework overhead. "
            "Captured via torch.cuda.max_memory_allocated() immediately after from_pretrained(). "
            "0.0 = not measured."
        ),
    )

    flops_method: str = Field(
        "unknown", description="Method used to estimate FLOPs (calflops, architecture, parameter)"
    )
    flops_confidence: str = Field("unknown", description="Confidence level (high, medium, low)")
    compute_precision: str = Field("fp16", description="Compute precision used")


class CombinedMetrics(BaseModel):
    """All metrics combined for a single measurement."""

    inference: InferenceMetrics
    energy: EnergyMetrics
    compute: ComputeMetrics

    @property
    def efficiency_tokens_per_joule(self) -> float:
        """Tokens generated per Joule of energy."""
        if self.energy.total_energy_j > 0:
            return self.inference.total_tokens / self.energy.total_energy_j
        return 0.0

    @property
    def efficiency_flops_per_watt(self) -> float:
        """FLOPs per Watt (computational efficiency)."""
        if self.energy.total_power_w > 0:
            return self.compute.flops_per_second / self.energy.total_power_w
        return 0.0


# =============================================================================
# Extended Efficiency Metrics - Consistent schema with conditional computation
# =============================================================================


class MemoryEfficiencyMetrics(BaseModel):
    """Memory efficiency metrics.

    All fields always present in schema. Values are null when not computable.
    """

    # Raw memory values (always available)
    total_vram_mb: float = Field(default=0.0, description="Total GPU VRAM in MB")
    model_memory_mb: float = Field(default=0.0, description="Model weights memory in MB")
    peak_memory_mb: float = Field(
        default=0.0,
        description=(
            "Peak GPU memory during inference measurement window (MB). "
            "See ComputeMetrics.peak_memory_mb for full semantics. 0.0 = not measured."
        ),
    )
    inference_memory_mb: float = Field(
        default=0.0,
        description=(
            "Inference-only memory: peak minus model baseline (MB). "
            "Derived: max(0.0, peak_memory_mb - model_memory_mb). "
            "Represents KV cache + activations + batch buffers allocated during inference. "
            "Computed by backend _build_result(), not by the caller. "
            "0.0 = not measured or not computable."
        ),
    )
    kv_cache_mb: float | None = Field(default=None, description="KV cache memory in MB (vLLM only)")

    # Derived efficiency metrics (null if not computable)
    tokens_per_gb_vram: float | None = Field(
        default=None, description="Output tokens per GB of VRAM used"
    )
    model_memory_utilisation: float | None = Field(
        default=None, description="Model memory / total VRAM (0.0-1.0)"
    )
    kv_cache_memory_ratio: float | None = Field(
        default=None, description="KV cache / peak memory (vLLM only)"
    )


class GPUUtilisationMetrics(BaseModel):
    """GPU utilisation during inference.

    Collected via pynvml background sampling. Null if pynvml unavailable.
    """

    sm_utilisation_mean: float | None = Field(
        default=None, description="Mean SM utilisation (0-100%)"
    )
    sm_utilisation_samples: int = Field(default=0, description="Number of samples collected")
    memory_bandwidth_utilisation: float | None = Field(
        default=None, description="Memory bandwidth utilisation (0-100%)"
    )


class BatchEfficiencyMetrics(BaseModel):
    """Batch processing efficiency.

    Only applicable for static batching (PyTorch, TensorRT). Null for vLLM
    continuous batching.
    """

    effective_batch_size: float | None = Field(
        default=None, description="Average actual batch size"
    )
    batch_utilisation: float | None = Field(
        default=None, description="Actual / configured batch size (0.0-1.0)"
    )
    padding_overhead: float | None = Field(
        default=None, description="Padding tokens / total tokens (0.0-1.0)"
    )
    num_batches: int | None = Field(default=None, description="Number of batches processed")


class KVCacheEfficiencyMetrics(BaseModel):
    """KV cache efficiency metrics.

    vLLM-specific. Always null for PyTorch/TensorRT backends.
    """

    kv_cache_hit_rate: float | None = Field(
        default=None, description="Prefix cache hit rate (0.0-1.0, vLLM only)"
    )
    kv_cache_blocks_used: int | None = Field(default=None, description="KV cache blocks used")
    kv_cache_blocks_total: int | None = Field(
        default=None, description="Total KV cache blocks available"
    )


class RequestLatencyMetrics(BaseModel):
    """Per-request end-to-end latency statistics.

    E2E latency = total time from request submission to completion.
    """

    e2e_latency_mean_ms: float | None = Field(
        default=None, description="Mean E2E latency per request"
    )
    e2e_latency_median_ms: float | None = Field(default=None, description="Median E2E latency")
    e2e_latency_p95_ms: float | None = Field(
        default=None, description="95th percentile E2E latency"
    )
    e2e_latency_p99_ms: float | None = Field(
        default=None, description="99th percentile E2E latency"
    )
    e2e_latency_samples: int = Field(default=0, description="Number of request samples")


class ExtendedEfficiencyMetrics(BaseModel):
    """Extended efficiency metrics container.

    Consistent schema - all fields always present in results.
    Individual values are null when not computable for the configuration.

    Design principles:
    - Graceful degradation: null values, never errors
    - Backend-agnostic where possible
    - Late aggregation: raw samples stored, stats computed at aggregation
    """

    # Core efficiency metrics
    tpot_ms: float | None = Field(
        default=None,
        description="Time Per Output Token in ms (ITL mean, streaming only)",
    )
    token_efficiency_index: float | None = Field(
        default=None,
        description="Composite: throughput * tokens_per_joule * precision_factor",
    )

    # Grouped metrics (always present as objects, internal fields may be null)
    memory: MemoryEfficiencyMetrics = Field(
        default_factory=MemoryEfficiencyMetrics,
        description="Memory efficiency metrics",
    )
    gpu_utilisation: GPUUtilisationMetrics = Field(
        default_factory=GPUUtilisationMetrics,
        description="GPU utilisation during inference",
    )
    batch: BatchEfficiencyMetrics = Field(
        default_factory=BatchEfficiencyMetrics,
        description="Batch efficiency (static batching only)",
    )
    kv_cache: KVCacheEfficiencyMetrics = Field(
        default_factory=KVCacheEfficiencyMetrics,
        description="KV cache efficiency (vLLM only)",
    )
    request_latency: RequestLatencyMetrics = Field(
        default_factory=RequestLatencyMetrics,
        description="Per-request E2E latency statistics",
    )


# =============================================================================
# Latency Measurement Types - For streaming inference metrics
# =============================================================================


class LatencyMeasurementMode(Enum):
    """How latency measurements were obtained.

    Different backends have different latency measurement capabilities:
    - PyTorch: True per-token timestamps via TextIteratorStreamer
    - vLLM/TensorRT: May use proportional estimation from total time

    This enum makes the measurement semantics explicit in results.
    """

    TRUE_STREAMING = "true_streaming"
    """Actual per-token timestamps captured via streaming API.

    Most accurate method. Each token timestamp represents when that
    specific token was generated. PyTorch backend achieves this via
    TextIteratorStreamer callback.
    """

    PER_REQUEST_BATCH = "per_request_batch"
    """Per-request timing without streaming.

    Measures total request latency but may estimate ITL by dividing
    total time by token count. Less accurate than true streaming.
    """

    PROPORTIONAL_ESTIMATE = "proportional"
    """Estimated from total inference time.

    ITL calculated by distributing total time proportionally across
    tokens. Least accurate - used as fallback when streaming not available.
    """


@dataclass
class LatencyMeasurements:
    """Raw latency measurements for late aggregation.

    Stores raw samples from streaming inference. Statistics are computed
    at aggregation time, enabling correct multi-process aggregation
    (concatenate samples first, then compute percentiles).

    Attributes:
        ttft_ms: Per-request time-to-first-token in milliseconds.
        itl_full_ms: All inter-token latencies (includes all intervals).
        itl_trimmed_ms: Trimmed ITL excluding first/last per request
            (first token is TTFT, last may have EOS anomalies).
        request_count: Number of requests measured.
        total_output_tokens: Total tokens generated across all requests.
        excluded_tokens: Count of first+last tokens excluded from trimmed ITL.
        streaming_mode: Whether streaming API was used for measurement.
        warmup_requests_excluded: Number of warmup requests not included.
        measurement_mode: How latency was measured (see LatencyMeasurementMode).
    """

    ttft_ms: list[float]
    itl_full_ms: list[float]
    itl_trimmed_ms: list[float]
    request_count: int
    total_output_tokens: int
    excluded_tokens: int
    streaming_mode: bool
    warmup_requests_excluded: int
    measurement_mode: LatencyMeasurementMode = LatencyMeasurementMode.TRUE_STREAMING


@dataclass
class LatencyStatistics:
    """Computed statistics from raw latency measurements.

    Created at aggregation time from LatencyMeasurements. This is the final
    form stored in AggregatedResult and displayed in CLI output.

    Primary metrics use trimmed ITL (excluding first/last tokens per request).
    Full ITL stats are provided for comparison/debugging.
    """

    # TTFT statistics
    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_min_ms: float
    ttft_max_ms: float
    ttft_samples: int

    # ITL statistics (trimmed - primary metric)
    itl_mean_ms: float | None = None
    itl_median_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    itl_samples: int = 0

    # ITL statistics (full - for comparison)
    itl_full_mean_ms: float | None = None
    itl_full_p99_ms: float | None = None


def collect_itl_measurements(
    token_timestamps_per_request: list[list[float]],
) -> tuple[list[float], list[float], int]:
    """Calculate ITL metrics from per-token timestamps.

    Standard implementation used by all backends for consistent ITL calculation.
    Extracts inter-token latencies from timestamp lists, optionally trimming
    first/last intervals per request for cleaner statistics.

    Args:
        token_timestamps_per_request: Per-request list of token arrival times (ms).
            Each inner list contains cumulative timestamps for one request.

    Returns:
        Tuple of (itl_full, itl_trimmed, excluded_count):
            - itl_full: All inter-token intervals
            - itl_trimmed: Excluding first/last per request (cleaner for percentiles)
            - excluded_count: Number of excluded intervals
    """
    import numpy as np

    itl_full: list[float] = []
    itl_trimmed: list[float] = []
    excluded = 0

    for timestamps in token_timestamps_per_request:
        if len(timestamps) < 2:
            continue

        # Calculate inter-token intervals
        intervals = list(np.diff(timestamps))
        itl_full.extend(intervals)

        # Trim first and last intervals for cleaner statistics
        # First interval may include warmup effects, last may have EOS anomalies
        if len(intervals) >= 3:
            itl_trimmed.extend(intervals[1:-1])
            excluded += 2
        elif len(intervals) >= 1:
            # Too short to trim meaningfully
            excluded += len(intervals)

    return itl_full, itl_trimmed, excluded
