# energy/ - Energy Measurement Samplers

Energy measurement samplers implementing the `EnergySampler` protocol. Layer 2 in the six-layer architecture.

## Purpose

Provides pluggable GPU energy tracking via NVML (default), Zeus, or CodeCarbon. The `select_energy_sampler()` function selects the best available sampler or raises a clear error when an explicit choice is unavailable.

Classes use `*Sampler` suffix to distinguish them from inference backends. The selection function is `select_energy_sampler()`.

## Modules

| Module | Description |
|--------|-------------|
| `base.py` | `EnergySampler` protocol definition |
| `nvml.py` | `NVMLSampler` — NVML power integration (default) |
| `zeus.py` | `ZeusSampler` — Zeus hardware energy registers |
| `codecarbon.py` | `CodeCarbonSampler` — software fallback |
| `__init__.py` | `select_energy_sampler()`, re-exports |

## EnergySampler protocol

```python
class EnergySampler(Protocol):
    @property
    def name(self) -> str: ...

    def start_tracking(self) -> Any: ...      # returns tracker handle
    def stop_tracking(self, tracker: Any) -> Any: ...  # returns measurement
    def is_available(self) -> bool: ...
```

## Primary API

```python
from llenergymeasure.energy import select_energy_sampler

# Auto-select best available sampler (Zeus > NVML > CodeCarbon > None)
sampler = select_energy_sampler("auto")

# Explicit sampler — raises ConfigError if unavailable
sampler = select_energy_sampler("nvml")
sampler = select_energy_sampler("zeus")
sampler = select_energy_sampler("codecarbon")

# Intentional disable — returns None immediately, no warnings
sampler = select_energy_sampler(None)

# With GPU indices
sampler = select_energy_sampler("auto", gpu_indices=[0, 1])

if sampler is not None:
    tracker = sampler.start_tracking()
    # ... run inference ...
    measurement = sampler.stop_tracking(tracker)
```

## Sampler comparison

| Sampler | Source | Accuracy | Requires |
|---------|--------|----------|---------|
| `NVMLSampler` | NVML power draw polling | Good | nvidia-ml-py (base dep) |
| `ZeusSampler` | Zeus hardware energy registers | Best | `pip install llenergymeasure[zeus]` |
| `CodeCarbonSampler` | Software estimation | Approximate | `pip install llenergymeasure[codecarbon]` |

Auto-selection priority: Zeus > NVML > CodeCarbon. NVML is always available on GPU machines (nvidia-ml-py is a base dependency), so `select_energy_sampler("auto")` returns `NVMLSampler` at minimum on any GPU host.

## Configuration

Energy sampler selection is controlled by `config.energy.backend`:

```yaml
energy:
  backend: auto      # default: auto-select
  backend: nvml      # explicit NVML
  backend: zeus      # explicit Zeus
  backend: null      # disable energy measurement
```

## Layer constraints

- Layer 2 — may import from layers 0–1 only
- Can import from: `config/`, `domain/`, `device/`, `utils/`
- Cannot import from: `harness/`, `backends/`, `study/`, `api/`, `cli/`, `results/`

## Related

- See `../harness/` for how the harness calls `select_energy_sampler()` and brackets inference
- See `../device/` for the NVML context manager used by `NVMLSampler`
