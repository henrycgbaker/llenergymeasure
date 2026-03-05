# Energy Measurement

What llenergymeasure measures and how it works.

---

## What We Measure

llenergymeasure collects three categories of metrics during inference:

### Energy (Joules)

GPU power draw integrated over inference time. This is the primary metric.

Energy in Joules = ∫ Power(t) dt over the inference window.

llenergymeasure reports:
- `inference_energy_joules` — total GPU energy during inference
- `adjusted_energy_joules` — inference energy minus baseline idle power (isolates the
  inference-attributable component)
- `baseline_power_watts` — idle GPU power measured before inference

The default backend is NVML polling. Zeus (kernel-level) and CodeCarbon (system-level)
are optional alternatives.

### Throughput

Tokens generated per second.

- `tokens_per_second` — total output tokens divided by total inference time
- Per-prompt throughput is available in the detailed per-prompt result records

Throughput measurements are backend-independent and do not require GPU access — they work
from wall-clock timing.

### FLOPs

Floating point operations estimated from model architecture.

- Based on the standard transformer FLOPs formula: `~6 * N * T` where N is non-embedding
  parameter count and T is the total token count (input + output)
- **Approximate** — uses architecture-based estimation (parameter count, sequence lengths),
  not hardware performance counters
- Useful for efficiency normalisation (energy per FLOP) but not for direct hardware
  utilisation analysis

### Peak Memory

- `inference_memory_mb` — peak GPU memory used during inference only (not model loading)
- Measured as the delta between pre-inference and peak NVML memory readings
- Reflects the KV cache and activation memory cost of the specific inference configuration

---

## Energy Measurement Backends

Configure the energy backend in the `energy:` section of your experiment YAML:

```yaml
energy:
  backend: auto   # auto | nvml | zeus | codecarbon | null
```

### `auto` (default)

Uses NVML if available, falls back gracefully if not. Recommended for most users.

### `nvml`

NVIDIA Management Library. Polls GPU power draw at 100ms intervals during inference.

- Most accurate for GPU-only measurement
- No additional install needed — `pynvml` is bundled with llenergymeasure
- Works on any NVIDIA GPU (Kepler and newer)
- Measures GPU power only (not CPU or RAM)

### `zeus`

Zeus energy monitoring. Kernel-level GPU energy measurement.

- More precise than polling — reads GPU energy counters directly
- Requires NVIDIA Ampere (GA100) or newer GPU
- Requires additional install: `pip install "llenergymeasure[zeus]"`

### `codecarbon`

CodeCarbon tracker. Estimates total system energy including CPU, GPU, and RAM.

- Broader scope than NVML — captures full system power draw
- Adds CO2 emissions estimate based on grid carbon intensity
- Requires additional install: `pip install "llenergymeasure[codecarbon]"`
- Less precise for GPU-only measurement due to system-level aggregation

### `null` (disabled)

Set `energy: { backend: null }` (YAML `null`) to disable energy measurement entirely.
The tool runs in throughput-only mode — no energy metrics, but faster execution.

---

## How Energy is Measured (NVML)

NVML is the default backend. The measurement loop:

1. A `PowerThermalSampler` thread starts immediately before inference begins.
2. The sampler polls `nvmlDeviceGetPowerUsage()` at 100ms intervals (configurable).
3. Each sample records timestamp and power draw in milliwatts, converted to watts.
4. When inference completes, the sampler thread is stopped.
5. Consecutive sample pairs are integrated using the **trapezoidal rule**:
   ```
   energy_j += (power[i] + power[i+1]) / 2 * (time[i+1] - time[i])
   ```
6. The result is total GPU energy in joules over the inference window.
7. If baseline measurement is enabled, `adjusted_energy_joules = total_j - (baseline_power_w * duration_s)`.

The raw power timeseries is available in the result JSON for detailed post-hoc analysis.

For very short inference runs (< 200ms), the 100ms polling interval means only 1-2 samples
may be collected. Energy estimates for sub-200ms runs have higher relative uncertainty.
Use `n >= 50` prompts to ensure sufficient measurement duration.

---

## Baseline Power

Before inference, llenergymeasure measures idle GPU power draw. This baseline represents
the GPU's steady-state power consumption when not doing inference work.

**Why baseline matters:** GPU power draw is never zero. A 300W GPU consuming 50W at idle
means that a 100-second inference consuming 200J of total energy is attributable to only
`200J - (50W * duration_s)` of inference work. Without baseline subtraction, results
conflate background power draw with inference energy.

**How it works:**
- The sampler polls GPU power for `baseline.duration_seconds` (default: 30s) before
  the first experiment
- The mean is stored as `baseline_power_watts`
- Baseline results are cached per-session (1-hour TTL) so subsequent experiments in a
  study reuse the same baseline measurement

Configure via the `baseline:` section:

```yaml
baseline:
  enabled: true
  duration_seconds: 30   # 5–120s accepted
```

---

## Limitations

**NVML measures GPU only.** CPU and RAM power are not included. Use CodeCarbon if you
need total system energy.

**Polling frequency limits precision for short runs.** At 100ms intervals, inference
runs under ~500ms may have only a few samples. For publication-quality results, use
`n >= 100` prompts.

**Multi-GPU measurement requires aggregation.** NVML reports per-device power. For
tensor-parallel runs across multiple GPUs, llenergymeasure sums per-device energy.
Ensure all participating GPUs are monitored (see `device_index` in advanced config).

**Container isolation.** Inside Docker containers, pynvml accesses the same physical GPU
as the host via the NVIDIA Container Toolkit. Energy readings represent the full device
power draw, including any other processes sharing the GPU.

**MIG instances.** Multi-Instance GPU partitions report the parent GPU's power, not
per-instance power. MIG energy readings are approximate.
