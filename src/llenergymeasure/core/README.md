# core/ - Measurement Engine

Core measurement infrastructure for LLM inference efficiency experiments.

## Architecture

The `MeasurementHarness` orchestrates the full measurement lifecycle. Backends are thin inference plugins that implement the `BackendPlugin` protocol - they own only inference, while the harness owns energy tracking, CUDA sync, FLOPs estimation, and result assembly.

```
MeasurementHarness.run(backend, config)
    ├── backend.load_model()
    ├── backend.warmup()
    ├── energy_backend.start_tracking()
    ├── backend.run_inference()     ← measurement window
    ├── energy_backend.stop_tracking()
    ├── estimate_flops()
    └── assemble ExperimentResult
```

## Modules

### harness.py
`MeasurementHarness` - orchestrates backend lifecycle, energy measurement, and result assembly.

### backends/
Thin inference plugins implementing `BackendPlugin` protocol:
- `protocol.py` - `BackendPlugin` protocol + `InferenceOutput` dataclass
- `pytorch.py` - HuggingFace Transformers backend
- `vllm.py` - vLLM offline batch inference backend
- `tensorrt.py` - TensorRT-LLM compiled engine backend
- `_helpers.py` - shared cleanup and warmup utilities
- `__init__.py` - `get_backend()` factory + `detect_default_backend()`

### energy_backends/
Energy measurement backends:
- `base.py` - `EnergyBackend` protocol (canonical definition)
- `nvml.py` - NVML power sampling backend
- `zeus.py` - Zeus hardware energy register backend
- `codecarbon.py` - CodeCarbon software estimation backend
- `__init__.py` - `select_energy_backend()` auto-selection API

### Other Modules
- `flops.py` - FLOPs estimation (PaLM formula, calflops, config-based fallback)
- `warmup.py` - warmup convergence detection (CV threshold)
- `baseline.py` - idle GPU power measurement
- `power_thermal.py` - NVML power/thermal sampling during inference
- `timeseries.py` - Parquet timeseries writer (1 Hz downsampled telemetry)
- `extended_metrics.py` - tokens/sec, J/token, FLOPs/J, latency breakdown
- `measurement_warnings.py` - measurement quality warnings
- `environment.py` - environment snapshot capture
- `gpu_info.py` - GPU topology detection with MIG support
- `state.py` - experiment state machine

## Dependencies

- `torch` - CUDA memory tracking (lazy import)
- `transformers` - model/tokenizer loading (lazy import)
- `pynvml` - GPU power/thermal sampling
- `pyarrow` - timeseries Parquet output
- `calflops` - FLOPs estimation (optional)
