# harness/ - Measurement Harness

Measurement lifecycle orchestration for any inference backend. Layer 3 in the six-layer architecture.

## Purpose

`MeasurementHarness` extracts the measurement infrastructure that was previously duplicated across backends into a single location. Backends are thin plugins (4 methods); the harness owns everything else: environment snapshot, baseline power, energy tracking, CUDA sync, thermal floor wait, FLOPs estimation, timeseries, warnings, and result assembly.

## Modules

| Module | Description |
|--------|-------------|
| `__init__.py` | `MeasurementHarness` class, re-exports |
| `warmup.py` | `warmup_until_converged()` — CV-based adaptive warmup; `thermal_floor_wait()` |
| `baseline.py` | `measure_baseline_power()` — idle GPU power baseline with session cache |
| `flops.py` | `estimate_flops_palm()`, `estimate_flops_palm_from_config()` — FLOPs estimation |
| `extended_metrics.py` | `compute_extended_metrics()` — derived efficiency metrics |
| `timeseries.py` | `write_timeseries_parquet()` — power/thermal timeseries sidecar |
| `measurement_warnings.py` | `collect_measurement_warnings()` — quality flag generation |
| `state.py` | `ExperimentState`, `ExperimentPhase` — experiment lifecycle state machine |

## MeasurementHarness

```python
from llenergymeasure.harness import MeasurementHarness
from llenergymeasure.backends import get_backend

harness = MeasurementHarness()
backend = get_backend("pytorch")
result = harness.run(backend, config, gpu_indices=[0])
```

### Measurement lifecycle (in order)

1. Collect environment snapshot (background thread, hidden behind model load)
2. Measure baseline power (before model load)
3. Load model via `backend.load_model(config)`
4. Capture model memory (torch.cuda.max_memory_allocated)
5. Run warmup via `backend.warmup(config, model)`
6. Thermal floor wait (let GPU cool after warmup)
7. Select energy sampler
8. CUDA sync (before inference — Zeus best practice)
9. Start energy tracking
10. Run inference via `backend.run_inference(config, model)`
11. CUDA sync (after inference, before stopping energy)
12. Stop energy tracking
13. Estimate FLOPs (AutoConfig path, then hf_model fallback)
14. Clean up model (always, even on exception)
15. Write timeseries Parquet sidecar (if output_dir set)
16. Collect measurement quality warnings
17. Assemble `ExperimentResult`

## Warmup convergence (warmup.py)

CV-based adaptive warmup replaces fixed-iteration warmup. Warmup continues until the latency coefficient of variation (CV) drops below a configured threshold, or until the safety cap is reached:

```python
from llenergymeasure.harness.warmup import warmup_until_converged

result = warmup_until_converged(
    run_single_inference=lambda: backend_latency_ms,
    config=warmup_config,
)
```

## Baseline power (baseline.py)

Measures idle GPU power before model load. Session-level cache keyed by GPU indices avoids repeated measurement within a study:

```python
from llenergymeasure.harness.baseline import measure_baseline_power

baseline = measure_baseline_power(duration_sec=10.0, gpu_indices=[0])
# baseline.power_w — idle power in watts
```

Energy breakdown uses baseline adjustment: `adjusted_j = total_j - baseline_j` where `baseline_j = baseline.power_w * inference_duration_sec`.

## FLOPs estimation (flops.py)

PaLM/Chinchilla formula: `FLOPs = 2 * N_non_embedding_params * total_tokens`

Two paths:
1. `estimate_flops_palm_from_config(model_name, ...)` — uses HuggingFace `AutoConfig` (no weights loaded, works for all backends)
2. `estimate_flops_palm(model, ...)` — uses loaded model object (higher confidence, PyTorch only)

## State machine (state.py)

Three-phase lifecycle: `INITIALISING → MEASURING → DONE`, with orthogonal `failed` flag. Used for resume/deduplication:

```python
from llenergymeasure.harness.state import ExperimentPhase, ExperimentState
```

## Layer constraints

- Layer 3 — may import from layers 0–2
- Can import from: `config/`, `domain/`, `device/`, `utils/`, `energy/`, `backends/`, `datasets/`, `infra/`
- Cannot import from: `study/`, `api/`, `cli/`, `results/`

## Related

- See `../backends/` for the BackendPlugin protocol backends implement
- See `../energy/` for energy sampler selection
- See `../device/` for PowerThermalSampler used during inference
