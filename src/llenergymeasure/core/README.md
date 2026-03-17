# core/ - Inference Engine

Core functionality for running LLM inference and collecting metrics.

## Purpose

Provides the inference engine, model loading, FLOPs estimation, energy tracking backends, and metrics collection. This is where the actual benchmarking work happens.

## Key Files

### inference.py
Main inference engine that processes prompts through the model.

```python
from llenergymeasure.core import run_inference, InferenceResult

result = run_inference(model, config, prompts, tokenizer, accelerator)
# result.metrics contains InferenceMetrics
# result.input_ids contains tokenized inputs
```

Key functions:
- `run_inference()` - Main entry point for batch inference
- `calculate_inference_metrics()` - Compute tokens/sec, latency
- `_create_batches()` - Create fixed or dynamic batches
- `_process_batch()` - Process single batch with timing

### model_loader.py
Model and tokenizer loading with quantization support.

```python
from llenergymeasure.core import load_model_tokenizer, ModelWrapper

model, tokenizer = load_model_tokenizer(config, accelerator)
```

Key exports:
- `load_model_tokenizer()` - Load model with config settings
- `ModelWrapper` - Wrapper for model with metadata
- `QuantizationSupport` - Quantization capabilities detection
- `detect_quantization_support()` - Check BNB availability

### flops.py
FLOPs estimation with multiple fallback methods.

```python
from llenergymeasure.core import FlopsEstimator, estimate_flops

estimator = FlopsEstimator(model, tokenizer, config)
result = estimator.estimate(input_ids)  # Returns FlopsResult
```

Estimation methods (in priority order):
1. `calflops` - Direct measurement (highest confidence)
2. `architecture` - Architecture-based calculation
3. `parameter_estimate` - Parameter count heuristic

### compute_metrics.py
GPU memory and utilization statistics.

```python
from llenergymeasure.core import get_memory_stats, get_utilization_stats

memory = get_memory_stats(device)  # MemoryStats
util = get_utilization_stats(device)  # UtilizationStats
```

### gpu_info.py
GPU topology detection with MIG (Multi-Instance GPU) support.

```python
from llenergymeasure.core.gpu_info import (
    detect_gpu_topology,
    format_gpu_topology,
    validate_gpu_selection,
    get_device_mig_info,
    GPUInfo,
    GPUTopology,
)

# Detect all visible CUDA devices
topology = detect_gpu_topology()
print(format_gpu_topology(topology))  # Human-readable tree

# Check if MIG instances are present
if topology.has_mig:
    print(f"MIG instances: {topology.mig_instances}")

# Validate GPU selection and get warnings
warnings = validate_gpu_selection([0, 1], topology)

# Get MIG metadata for a specific device (respects CUDA_VISIBLE_DEVICES)
mig_info = get_device_mig_info(0)  # {"gpu_is_mig": bool, "gpu_mig_profile": str|None, ...}
```

**Key classes:**
- `GPUInfo` - Information about a GPU or MIG instance
- `GPUTopology` - Collection of all detected devices with helper properties

**MIG detection:** Uses pynvml (preferred) or falls back to PyTorch. MIG instances are identified by UUID prefix (`MIG-`) or device name containing "MIG".

### distributed.py
Distributed training utilities for `accelerate`.

```python
from llenergymeasure.core import get_accelerator, safe_wait, cleanup_distributed

accelerator = get_accelerator(config)
safe_wait(accelerator)  # Barrier with timeout
cleanup_distributed(accelerator)
```

Key functions:
- `get_accelerator()` - Initialize accelerator
- `get_persistent_unique_id()` - Unique experiment ID
- `safe_wait()` - Barrier with configurable timeout

### prompts.py
Prompt processing and batching.

```python
from llenergymeasure.core import create_fixed_batches, create_adaptive_batches

batches = create_fixed_batches(prompts, batch_size=8)
batches = create_adaptive_batches(prompts, tokenizer, max_tokens_per_batch=2048)
```

### dataset_loader.py
Load prompts from files or HuggingFace datasets.

```python
from llenergymeasure.core.dataset_loader import (
    load_prompts_from_source,
    load_prompts_from_file,
    load_prompts_from_hf,
    list_builtin_datasets,
)
from llenergymeasure.config.models import FilePromptSource, HuggingFacePromptSource

# From file
prompts = load_prompts_from_file("prompts.txt")

# From HuggingFace (using config model)
source = HuggingFacePromptSource(dataset="alpaca", sample_size=1000)
prompts = load_prompts_from_source(source)

# List built-in aliases
list_builtin_datasets()  # {"alpaca": {...}, "gsm8k": {...}, ...}
```

**Built-in datasets:** alpaca, sharegpt, gsm8k, mmlu

**Auto-detect columns:** text, prompt, question, instruction, input, content

### implementations.py
Protocol implementations for DI wiring.

```python
from llenergymeasure.core import (
    HuggingFaceModelLoader,
    TransformersInferenceEngine,
    ThroughputMetricsCollector,
)

# These wrap the functions above into protocol-compliant classes
loader = HuggingFaceModelLoader()
model, tokenizer = loader.load(config)

engine = TransformersInferenceEngine(accelerator)
result = engine.run(model, tokenizer, prompts, config)

collector = ThroughputMetricsCollector(accelerator)
metrics = collector.collect(model, result, config)
```

**DI Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│  Protocol (interface)    │  Implementation (wrapper)    │
├─────────────────────────────────────────────────────────┤
│  ModelLoader             │  HuggingFaceModelLoader      │
│  InferenceEngine         │  TransformersInferenceEngine │
│  MetricsCollector        │  ThroughputMetricsCollector  │
│  EnergyBackend           │  CodeCarbonBackend           │
│  ResultsRepository       │  FileSystemRepository        │
└─────────────────────────────────────────────────────────┘
```

### inference_backends/
Pluggable inference backend implementations.

```
inference_backends/
├── __init__.py      # Registry with lazy loading
├── protocols.py     # InferenceBackend protocol, RuntimeCapabilities
├── pytorch.py       # HuggingFace Transformers + Accelerate
├── vllm.py          # vLLM with PagedAttention
└── tensorrt.py      # TensorRT-LLM with compiled engines
```

**Key types:**
- `InferenceBackend` - Protocol for all backends
- `RuntimeCapabilities` - Backend requirements declaration
- `LaunchMode` - How to launch (ACCELERATE, TORCHRUN, DIRECT)
- `CudaManagement` - Who manages CUDA (ORCHESTRATOR, BACKEND)

Usage:
```python
from llenergymeasure.core.inference_backends import (
    get_backend,
    RuntimeCapabilities,
    LaunchMode,
    CudaManagement,
)

# Get a backend
backend = get_backend("vllm")

# Query capabilities (no CUDA initialization)
caps = backend.get_runtime_capabilities()
if caps.orchestrator_may_call_cuda:
    # Safe to call torch.cuda.* before initialize()
    pass

# Initialize and run
backend.initialize(config, runtime)
result = backend.run_inference(prompts, config)
backend.cleanup()
```

**RuntimeCapabilities architecture:**
```
Backend declares capabilities
         ↓
┌─────────────────────────────────────┐
│  RuntimeCapabilities                │
│  - launch_mode: ACCELERATE/DIRECT   │
│  - cuda_management: ORCHESTRATOR/   │
│                     BACKEND         │
│  - supports_tensor_parallel: bool   │
└─────────────────────────────────────┘
         ↓
Orchestration layer respects them
(no hardcoded backend checks)
```

### energy_backends/
Energy tracking implementations.

```
energy_backends/
├── __init__.py
├── base.py          # EnergyBackend protocol
└── codecarbon.py    # CodeCarbon implementation
```

Usage:
```python
from llenergymeasure.core.energy_backends import CodeCarbonBackend

backend = CodeCarbonBackend()
tracker = backend.start_tracking()
# ... run inference ...
metrics = backend.stop_tracking(tracker)  # EnergyMetrics
```

## Data Flow

```
prompts -> tokenize_batch() -> run_inference() -> BatchResult[]
                                    |
                           calculate_inference_metrics()
                                    |
                              InferenceResult
```

## Dependencies

- `torch` - PyTorch tensors and CUDA
- `transformers` - Model/tokenizer loading
- `accelerate` - Distributed inference
- `codecarbon` - Energy tracking
- `calflops` - FLOPs estimation
- `pynvml` - GPU stats

## Related

- See `../domain/README.md` for metric models
- See `../orchestration/README.md` for experiment runner
