# domain/ - Domain Models

Pydantic models for metrics, experiment results, and model metadata.

## Purpose

Defines the data structures used throughout the framework for metrics collection and result persistence. All models are immutable Pydantic BaseModels.

## Key Files

### metrics.py
Metrics collected during experiments.

**InferenceMetrics** - Throughput and latency:
```python
InferenceMetrics(
    total_tokens=1024,
    input_tokens=512,
    output_tokens=512,
    inference_time_sec=2.5,
    tokens_per_second=204.8,
    latency_per_token_ms=4.88,
)
```

**EnergyMetrics** - Energy consumption:
```python
EnergyMetrics(
    total_energy_j=150.0,
    gpu_energy_j=140.0,
    cpu_energy_j=10.0,
    gpu_power_w=280.0,
    duration_sec=2.5,
    emissions_kg_co2=0.00015,
)
```

**ComputeMetrics** - FLOPs and memory:
```python
ComputeMetrics(
    flops_total=1.5e12,
    flops_per_token=1.5e9,
    flops_per_second=6e11,
    peak_memory_mb=14000,
    flops_method="calflops",
    flops_confidence="high",
)
```

**FlopsResult** - FLOPs estimation with provenance:
```python
FlopsResult(
    value=1.5e12,
    method="calflops",  # calflops | architecture | parameter_estimate
    confidence="high",   # high | medium | low
    dtype="float16",
)
```

**CombinedMetrics** - All metrics together:
```python
combined = CombinedMetrics(
    inference=inference_metrics,
    energy=energy_metrics,
    compute=compute_metrics,
)
# Properties:
# combined.efficiency_tokens_per_joule
# combined.efficiency_flops_per_watt
```

**LatencyMeasurements** - Streaming latency data:
```python
LatencyMeasurements(
    ttft_ms=[45.2, 48.1, 42.8, ...],      # Time to first token samples
    itl_full_ms=[12.1, 11.8, 13.2, ...],  # All inter-token latencies
    itl_trimmed_ms=[12.1, 11.8, ...],     # ITL with first/last tokens removed
    request_count=95,
    total_output_tokens=12350,
    excluded_tokens=190,                   # First/last tokens excluded from ITL
    streaming_mode=True,
    warmup_requests_excluded=5,
    measurement_method="streaming",        # streaming | per_request_batch | proportional_estimate
)
```

**LatencyStatistics** - Computed percentiles:
```python
LatencyStatistics(
    mean_ms=46.5,
    median_ms=45.8,
    p95_ms=52.1,
    p99_ms=58.3,
    min_ms=38.2,
    max_ms=62.1,
    sample_count=95,
)
```

### experiment.py
Experiment result models.

**RawProcessResult** - Single GPU/process output:
```python
RawProcessResult(
    experiment_id="exp_20240115_123456",
    process_index=0,
    gpu_id=0,
    model_name="meta-llama/Llama-2-7b-hf",
    timestamps=Timestamps(...),
    inference_metrics=InferenceMetrics(...),
    energy_metrics=EnergyMetrics(...),
    compute_metrics=ComputeMetrics(...),
)
```

**AggregatedResult** - Combined multi-GPU result:
```python
AggregatedResult(
    experiment_id="exp_20240115_123456",
    aggregation=AggregationMetadata(
        method="sum_energy_avg_throughput",
        num_processes=4,
        temporal_overlap_verified=True,
    ),
    total_tokens=4096,
    total_energy_j=600.0,
    avg_tokens_per_second=819.2,
    total_flops=6e12,
    process_results=[...],  # Original raw results
)
```

**Timestamps** - Timing info:
```python
Timestamps.from_times(start_datetime, end_datetime)
# .start, .end, .duration_sec
```

### model_info.py
Model metadata.

**ModelInfo** - Model characteristics:
```python
ModelInfo(
    name="meta-llama/Llama-2-7b-hf",
    num_parameters=7_000_000_000,
    architecture="LlamaForCausalLM",
    context_length=4096,
)
```

**QuantizationSpec** - Quantization details:
```python
QuantizationSpec(
    enabled=True,
    bits=4,
    quant_type="nf4",
)
```

## Schema Version

Results include `schema_version` (currently "2.0.0") for forward compatibility:
```python
from llenergymeasure.utils.constants import SCHEMA_VERSION
```

## Related

- See `../results/README.md` for result persistence
- See `../results/aggregation.py` for aggregation logic
