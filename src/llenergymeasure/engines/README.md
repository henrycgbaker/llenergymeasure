# backends/ - Inference Backend Plugins

Thin inference backend plugins implementing the `BackendPlugin` protocol. Layer 2 in the six-layer architecture.

## Purpose

Each backend owns only inference: load model, run warmup, run inference, clean up. The `MeasurementHarness` (layer 3) owns everything else: energy tracking, FLOPs estimation, CUDA sync, result assembly.

## Modules

| Module | Description |
|--------|-------------|
| `protocol.py` | `BackendPlugin` protocol and `InferenceOutput` dataclass |
| `pytorch.py` | HuggingFace Transformers backend |
| `vllm.py` | vLLM backend (Docker-only in production) |
| `tensorrt.py` | TensorRT-LLM backend (NGC Docker) |
| `_helpers.py` | Shared helpers (dataset loading, token counting) |
| `__init__.py` | `get_backend(name)` factory, `detect_default_backend()` |

## BackendPlugin protocol

```python
from llenergymeasure.backends.protocol import BackendPlugin, InferenceOutput

class MyBackend:
    @property
    def name(self) -> str: ...

    def load_model(self, config: ExperimentConfig) -> Any: ...
    def warmup(self, config: ExperimentConfig, model: Any) -> WarmupResult: ...
    def run_inference(self, config: ExperimentConfig, model: Any) -> InferenceOutput: ...
    def cleanup(self, model: Any) -> None: ...
    def validate_config(self, config: ExperimentConfig) -> list[str]: ...
```

`validate_config()` returns a list of error strings (empty means valid). Called by pre-flight before GPU allocation to catch hardware-config mismatches (e.g., FP8 on A100, unsupported quantisation).

`InferenceOutput` carries the minimal data the harness needs:

```python
InferenceOutput(
    elapsed_time_sec=...,  # backend-measured (overridden by harness perf_counter)
    input_tokens=512,
    output_tokens=256,
    peak_memory_mb=14000.0,
    model_memory_mb=12000.0,
    batch_times=[...],
    extras={"hf_model": model},  # optional, e.g. for FLOPs estimation
)
```

## Backend factory

```python
from llenergymeasure.backends import get_backend, detect_default_backend

backend = get_backend("pytorch")   # PyTorchBackend
backend = get_backend("vllm")      # VLLMBackend
backend = get_backend("tensorrt")  # TensorRTBackend

default = detect_default_backend()  # "pytorch" if transformers installed, etc.
```

Priority for auto-detection: pytorch > tensorrt > vllm.

## Installation extras

Zero backend deps at base. Each backend requires an install extra:

| Backend | Install extra | Required package |
|---------|--------------|-----------------|
| `pytorch` | `pip install llenergymeasure[pytorch]` | `transformers` |
| `vllm` | `pip install llenergymeasure[vllm]` | `vllm` |
| `tensorrt` | `pip install llenergymeasure[tensorrt]` | `tensorrt_llm` |

## Layer constraints

- Layer 2 — may import from layers 0–1 only
- Can import from: `config/`, `domain/`, `device/`, `utils/`, `energy/`, `datasets/`, `infra/`
- Cannot import from: `harness/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../harness/` for the measurement lifecycle that drives these backends
- See `../config/README.md` for `PyTorchConfig`, `VLLMConfig`, `TensorRTConfig`
- See `../api/preflight.py` for backend pre-flight checks
