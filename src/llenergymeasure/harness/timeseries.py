"""Parquet timeseries writer for energy measurement telemetry.

Downsamples 100ms NVML power/thermal samples to 1 Hz and writes
a Parquet sidecar file alongside the result JSON. Uses pyarrow
directly (no pandas dependency).

For multi-GPU experiments, each GPU gets its own row per second bucket,
grouped by (bucket_idx, gpu_index).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.device.power_thermal import PowerThermalSample


def _timeseries_schema() -> Any:
    """Return the locked Parquet schema for timeseries data.

    Schema is locked per CONTEXT.md — do not change column names or types
    without a schema version bump.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("timestamp_s", pa.float64()),
            pa.field("gpu_index", pa.int32()),
            pa.field("power_w", pa.float32()),
            pa.field("temperature_c", pa.float32()),
            pa.field("memory_used_mb", pa.float32()),
            pa.field("memory_total_mb", pa.float32()),
            pa.field("sm_utilisation_pct", pa.float32()),
            pa.field("throttle_reasons", pa.int64()),
        ]
    )


def write_timeseries_parquet(
    samples: list[PowerThermalSample],
    output_path: Path,
    gpu_index: int = 0,
) -> Path:
    """Write 1 Hz downsampled timeseries to a Parquet file.

    Groups 100ms NVML power/thermal samples into 1-second buckets per GPU and
    writes the mean of each metric per (bucket, gpu_index) pair. The schema is
    locked (see CONTEXT.md).

    For multi-GPU experiments, samples carry a gpu_index field; each GPU produces
    its own row per second. The ``gpu_index`` parameter is kept for backward
    compatibility but is only used as a fallback when samples lack the attribute.

    Args:
        samples: Raw PowerThermalSamples from PowerThermalSampler.
        output_path: Destination path for the Parquet file.
        gpu_index: Fallback GPU device index when samples have no gpu_index attribute.

    Returns:
        The output_path after writing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = _timeseries_schema()

    if not samples:
        # Write empty Parquet with correct schema
        empty_table = pa.table(
            {field.name: pa.array([], type=field.type) for field in schema},
            schema=schema,
        )
        pq.write_table(empty_table, output_path)
        return output_path

    # Group samples into (1-second bucket, gpu_index) pairs
    base_ts = samples[0].timestamp
    # buckets key: (bucket_idx, gpu_idx)
    buckets: dict[tuple[int, int], list[PowerThermalSample]] = {}
    for s in samples:
        bucket = int(s.timestamp - base_ts)
        gpu_idx = getattr(s, "gpu_index", gpu_index)
        key = (bucket, gpu_idx)
        buckets.setdefault(key, []).append(s)

    rows = []
    for bucket_idx, gpu_idx in sorted(buckets.keys()):
        bucket_samples = buckets[(bucket_idx, gpu_idx)]

        def _mean(values: list[float | None]) -> float | None:
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else None

        power_w = _mean([s.power_w for s in bucket_samples])
        temperature_c = _mean([s.temperature_c for s in bucket_samples])
        memory_used_mb = _mean([s.memory_used_mb for s in bucket_samples])
        memory_total_mb = _mean([s.memory_total_mb for s in bucket_samples])
        sm_utilisation_pct = _mean([s.sm_utilisation for s in bucket_samples])

        # OR all throttle_reasons bitmasks for the bucket
        throttle_reasons = 0
        for s in bucket_samples:
            throttle_reasons |= s.throttle_reasons

        rows.append(
            {
                "timestamp_s": float(bucket_idx),
                "gpu_index": gpu_idx,
                "power_w": float(power_w) if power_w is not None else None,
                "temperature_c": float(temperature_c) if temperature_c is not None else None,
                "memory_used_mb": float(memory_used_mb) if memory_used_mb is not None else None,
                "memory_total_mb": float(memory_total_mb) if memory_total_mb is not None else None,
                "sm_utilisation_pct": (
                    float(sm_utilisation_pct) if sm_utilisation_pct is not None else None
                ),
                "throttle_reasons": throttle_reasons,
            }
        )

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path)
    return output_path
