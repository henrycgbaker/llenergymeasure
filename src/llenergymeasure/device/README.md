# device/ - GPU Hardware Abstraction

GPU hardware detection and real-time power/thermal sampling via NVML. Layer 0 in the six-layer architecture.

## Purpose

Provides a clean interface over pynvml for GPU topology detection (including MIG), power monitoring, and temperature sampling. All NVML operations go through this layer. Higher layers never call pynvml directly.

## Modules

| Module | Description |
|--------|-------------|
| `gpu_info.py` | `GPUInfo` dataclass, MIG detection, `nvml_context()` |
| `power_thermal.py` | `PowerThermalSampler`, `PowerThermalSample` dataclass |
| `__init__.py` | Package marker |

## gpu_info.py

### nvml_context()

Context manager for safe NVML init/shutdown. Silently handles pynvml unavailability and GPU absence:

```python
from llenergymeasure.device.gpu_info import nvml_context
import pynvml

with nvml_context():
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # ... query GPU ...
# nvmlShutdown() called automatically
```

### GPUInfo

Dataclass describing a physical GPU or MIG instance:

```python
@dataclass
class GPUInfo:
    index: int                    # CUDA device index
    name: str                     # e.g. "NVIDIA A100-PCIE-40GB"
    uuid: str                     # Unique GPU identifier
    memory_mb: int                # Total VRAM in MB
    is_mig_capable: bool          # GPU supports MIG
    is_mig_enabled: bool          # MIG mode currently active
    is_mig_instance: bool         # This is a MIG slice (not parent GPU)
    parent_gpu_index: int | None  # Parent GPU index for MIG slices
    mig_profile: str | None       # e.g. "1g.10gb", "3g.40gb"
    compute_capability: str | None  # e.g. "8.0" for A100
```

## power_thermal.py

### PowerThermalSampler

Background context manager that samples GPU power, memory, temperature, and thermal throttle state during inference:

```python
from llenergymeasure.device.power_thermal import PowerThermalSampler

with PowerThermalSampler(gpu_indices=[0, 1]) as sampler:
    # ... run inference ...

samples = sampler.get_samples()          # list[PowerThermalSample]
throttle_info = sampler.get_thermal_throttle_info()  # ThermalThrottleInfo
```

### PowerThermalSample

```python
@dataclass
class PowerThermalSample:
    timestamp: float
    power_w: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    temperature_c: float | None
    sm_utilisation: float | None
    thermal_throttle: bool
    throttle_reasons: int
    gpu_index: int                # Which GPU this sample is from
```

Sampling is best-effort: if pynvml is unavailable, returns empty sample list without raising.

## NVML availability

All device/ functions degrade gracefully when NVML is unavailable (CPU-only host, pynvml not installed, Docker without `--gpus`). No function raises ImportError or NVMLError — callers receive None or empty collections.

## Layer constraints

- Layer 0 — base layer; no imports from other llenergymeasure layers
- Can be imported by all layers above
- All NVML usage in the codebase must go through this layer

## Related

- See `../energy/` for energy integration (total joules from NVML power samples)
- See `../harness/` for how PowerThermalSampler is used during measurement
