# engines/ - Inference Engine Plugins

Thin inference engine plugins implementing the `EnginePlugin` protocol. Layer 2 in the six-layer architecture.

## Purpose

Each engine owns only inference: load model, run warmup, run inference, clean up. The `MeasurementHarness` (layer 3) owns everything else: energy tracking, FLOPs estimation, CUDA sync, result assembly.

## Modules

| Module | Description |
|--------|-------------|
| `protocol.py` | `EnginePlugin` protocol and `InferenceOutput` dataclass |
| `transformers.py` | HuggingFace Transformers engine |
| `vllm.py` | vLLM engine (Docker-only in production) |
| `tensorrt.py` | TensorRT-LLM engine (NGC Docker) |
| `_helpers.py` | Shared helpers (dataset loading, token counting) |
| `__init__.py` | `get_engine(name)` factory, `detect_default_engine()` |

## EnginePlugin protocol

```python
from llenergymeasure.engines.protocol import EnginePlugin, InferenceOutput

class MyEngine:
    @property
    def name(self) -> str: ...

    def load_model(self, config: ExperimentConfig) -> Any: ...
    def warmup(self, config: ExperimentConfig, model: Any) -> WarmupResult: ...
    def run_inference(self, config: ExperimentConfig, model: Any) -> InferenceOutput: ...
    def cleanup(self, model: Any) -> None: ...
    def check_hardware(self, config: ExperimentConfig) -> list[str]: ...
```

`check_hardware()` returns a list of error strings (empty means compatible). Called via `engines.probe_adapter.build_config_probe()` at preflight to catch host-GPU mismatches (e.g., FP8 on A100, SM below the engine's floor). Framework-rule validation (library-semantics) lives in the vendored-rules corpus consumed by `ExperimentConfig._apply_vendored_rules`.

`InferenceOutput` carries the minimal data the harness needs:

```python
InferenceOutput(
    elapsed_time_sec=...,  # engine-measured (overridden by harness perf_counter)
    input_tokens=512,
    output_tokens=256,
    peak_memory_mb=14000.0,
    model_memory_mb=12000.0,
    batch_times=[...],
    extras={"hf_model": model},  # optional, e.g. for FLOPs estimation
)
```

## Engine factory

```python
from llenergymeasure.engines import get_engine, detect_default_engine

engine = get_engine("pytorch")   # PyTorchEngine
engine = get_engine("vllm")      # VLLMEngine
engine = get_engine("tensorrt")  # TensorRTEngine

default = detect_default_engine()  # "pytorch" if transformers installed, etc.
```

Priority for auto-detection: pytorch > tensorrt > vllm.

## Installation extras

Zero engine deps at base. Each engine requires an install extra:

| Engine | Install extra | Required package |
|---------|--------------|-----------------|
| `pytorch` | `pip install llenergymeasure[transformers]` | `transformers` |
| `vllm` | `pip install llenergymeasure[vllm]` | `vllm` |
| `tensorrt` | `pip install llenergymeasure[tensorrt]` | `tensorrt_llm` |

## Layer constraints

- Layer 2 — may import from layers 0–1 only
- Can import from: `config/`, `domain/`, `device/`, `utils/`, `energy/`, `datasets/`, `infra/`
- Cannot import from: `harness/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../harness/` for the measurement lifecycle that drives these engines
- See `../config/README.md` for `TransformersConfig`, `VLLMConfig`, `TensorRTConfig`
- See `../api/preflight.py` for engine pre-flight checks
