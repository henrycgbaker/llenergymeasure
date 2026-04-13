# Measurement Methodology

How llenergymeasure ensures reproducible, reliable energy measurements.

---

## Warmup

**Purpose:** GPU thermal state, driver caches, and JIT compilation warm-up all affect
first-run measurements. The first few inferences with a freshly loaded model are
consistently slower and higher-energy than subsequent inferences. Warmup discards
these initial measurements.

### Two warmup modes

llenergymeasure has two warmup modes, controlled by `convergence_detection` (default: `false`):

#### Fixed mode (default)

Runs exactly `n_warmup` prompts (default 5). The coefficient of variation (CV) is
computed for informational purposes but does not affect iteration count. Always reports
`converged: true`. Simple, predictable, and sufficient for most use cases.

```yaml
warmup:
  enabled: true                    # default: true
  n_warmup: 5                     # default: 5
  thermal_floor_seconds: 60.0     # default: 60.0
```

#### CV convergence mode (opt-in)

Runs warmup prompts until the **coefficient of variation** (CV = std_dev / mean) of
recent latencies drops below `cv_threshold`. This mode replaces fixed iteration
count - `n_warmup` is ignored when convergence detection is active.

```yaml
warmup:
  enabled: true
  convergence_detection: true      # enable CV-based warmup
  cv_threshold: 0.05              # stop when CV < 5% (default: 0.05)
  window_size: 3                  # sliding window for CV calc (default: 3)
  min_prompts: 5                  # minimum prompts before checking CV (default: 5)
  max_prompts: 20                 # safety cap (default: 20)
```

CV convergence mode checks `len(latencies) >= max(min_prompts, window_size)` before
evaluating the threshold. The safety cap (`max_prompts`) prevents infinite loops if
the system never stabilises.

### Execution order

1. **Warmup prompts** - heat the GPU to steady state (fixed or CV mode).
2. **Thermal floor wait** - sleep `thermal_floor_seconds` (default 60s) for GPU
   temperature to plateau after warmup.
3. **Measurement** - energy tracking begins.

The thermal floor wait occurs *after* warmup, not before. This ensures the GPU has
reached operating temperature from warmup but has stabilised before measurement starts.

### Engine-specific behaviour

For **vLLM** and **TRT-LLM** engines, warmup is a single kernel warmup call that
returns `first_latency=0.0`. These engines perform their own internal warmup during
server startup (CUDA graph capture, kernel compilation). The warmup phase for these
engines confirms the engine is ready, rather than iterating multiple inference passes.

### Default values

- `n_warmup: 5` is consistent with DeepSpeed, Zeus, and AI Energy Score benchmarks
  (which use 5-10 warmup rounds)
- `thermal_floor_seconds: 60.0` meets the MLPerf Power minimum (60s mandatory)

**For quick testing:** disable warmup to skip the warmup phase and thermal floor wait:

```yaml
warmup:
  enabled: false
```

This significantly reduces total experiment time at the cost of measurement quality.
Do not use `enabled: false` for published results.

---

## Baseline Power

**Purpose:** Measures idle GPU power draw before inference to enable baseline-adjusted
energy attribution. The adjusted figure isolates the energy cost of the inference work
itself, removing constant background power draw.

Configure via the `baseline:` section:

```yaml
baseline:
  enabled: true           # default: true
  duration_seconds: 30    # default: 30.0, range: 5–120
```

**What happens:**
1. Before the first experiment, the GPU power sampler runs for `duration_seconds` with
   no inference work.
2. The mean power over this period is stored as `baseline_power_watts`.
3. For each subsequent measurement, `adjusted_energy_joules = total_j - (baseline_w * duration_s)`.
4. The baseline result is cached per-session (1-hour TTL) — subsequent experiments in a
   study reuse the cached baseline without measuring again.

**In results:**
- `baseline_power_watts` — measured idle power
- `inference_energy_joules` — total GPU energy during inference
- `adjusted_energy_joules` — inference energy minus baseline (inference-attributable only)

For publication-quality results, always include baseline in your reported energy.
`adjusted_energy_joules` is the preferred metric for comparing configurations.

### Where baselines are measured

Baseline idle power is measured in the same CUDA environment as the inference
work it will be subtracted from. For local (host) runs that is the host
process itself. For Docker runs, the baseline is measured inside a short-lived
container of the same engine image, with the CUDA runtime initialised and
the torch memory pool seeded — matching the state the experiment container
will be in just before inference starts.

**Why this matters:** a host-measured baseline underestimates the container's
idle power by ~8.7 W per A100 because the host has no CUDA context and no GPU
memory pool allocated. On a 4-GPU 120 s A100 experiment this is a ~4.2 kJ
under-subtraction (~19 % of typical adjusted energy). Measuring in the
matching CUDA environment eliminates this bias. See
`.product/research/baseline-measurement-location.md` for the controlled
experiment and statistics.

**Baseline strategies and where they run:**

| Strategy    | Where measured (local runner)     | Where measured (Docker runner)                |
|-------------|-----------------------------------|-----------------------------------------------|
| `fresh`     | Host, per experiment              | Inside experiment container (per-experiment) |
| `cached`    | Host, once per TTL window         | Short-lived baseline container, once          |
| `validated` | Host, once + periodic spot-check  | Short-lived baseline / spot-check containers  |

**Cross-engine comparisons:** each engine image gets its own baseline cache.
If your study mixes engines, each engine's adjusted energy is computed
against a baseline measured in that engine's environment — cross-engine
energy comparisons remain apples-to-apples.

### Multi-engine studies: per-engine scoping

Baseline caches, TTL expiry, and the `validated` strategy's spot-check
counter are all keyed per engine target (``local`` for host runs,
``image:<sanitised-tag>`` for each Docker image). In a mixed-engine
study — for example 300 experiments randomly interleaving PyTorch, vLLM,
and TensorRT-LLM — each engine behaves as if it had its own independent
baseline session:

- **`cached` TTL:** each engine's baseline ages out independently after
  `baseline.cache_ttl_seconds`. A stale pytorch cache does not force a
  re-measure of vllm, and vice versa. Cache files live at
  `{study_dir}/_study-artefacts/baseline_cache_{key}.json`.
- **`validated` interval:** `baseline.validation_interval` counts
  experiments *per engine*, not across the whole study. If the interval
  is 50 and the study interleaves three engines, each engine triggers
  its own drift check after 50 experiments against *that* engine's
  cached baseline — regardless of how many experiments ran against the
  other engines in between.
- **Drift threshold:** a drift detected on one engine only re-measures
  that engine's baseline. The other engines' caches are untouched.

This scoping makes randomised Multi-engine studies safe to run without
baseline interference — interleaving does not corrupt the statistical
independence of each engine's adjusted energy figures.

### Two-container architecture (Docker runs)

For `cached` / `validated` strategies, a Docker experiment is dispatched
as **two sequential containers** of the same engine image:

```text
          ┌────────────────────────────────────────────────┐
          │ Host runner (study/runner.py)                  │
          │                                                │
          │  1. dispatch Container A ─┐                    │
          │                           ▼                    │
          │    ┌────────────────────────────────────────┐  │
          │    │ Container A  (baseline_measure)        │  │
          │    │ · init CUDA runtime                    │  │
          │    │ · seed torch memory pool               │  │
          │    │ · sample NVML power for duration_s     │  │
          │    │ · write BaselineCache to JSON          │  │
          │    │ · exit                                 │  │
          │    └────────────────────────────────────────┘  │
          │                           │                    │
          │  2. bind-mount JSON ◀─────┘                    │
          │                                                │
          │  3. dispatch Container B                       │
          │                           ▼                    │
          │    ┌────────────────────────────────────────┐  │
          │    │ Container B  (experiment harness)      │  │
          │    │ · mount baseline_cache.json read-only  │  │
          │    │ · load baseline via harness Branch A   │  │
          │    │ · run warmup + measurement + save      │  │
          │    └────────────────────────────────────────┘  │
          │                                                │
          └────────────────────────────────────────────────┘
```

**Key properties:**

- **Strictly sequential.** Container A runs to completion (`subprocess.run`
  is blocking) before Container B is started. The two containers never
  execute concurrently, even though the CLI display may briefly show
  overlapping updates during the ~100 ms handover.
- **Same image, same CUDA state.** Using the engine image for Container A
  guarantees the baseline is measured in the same CUDA runtime, same
  Python interpreter, and same torch allocator footprint that Container B
  will inherit. This is what eliminates the ~8.7 W/GPU host-vs-container
  bias documented above.
- **No shared process state.** Information crosses the container boundary
  only through a single JSON file (`baseline_cache.json`) bind-mounted
  into Container B under `/run/llem/`. No stdin pipes, no long-lived
  sidecars, no shared volumes beyond the read-only cache.
- **`fresh` strategy is single-container.** The harness measures its own
  baseline inside Container B (Branch B of `harness/__init__.py`). No
  Container A is dispatched. This is the simplest path but pays the
  baseline cost on every experiment.

**Why not measure baseline inside Container B in all cases?** Doing so
would force every experiment in a cached or validated study to pay the
full `duration_seconds` (typically 30 s) up front, cancelling the main
benefit of caching. The two-container design pays that cost once per
engine per TTL window and then reuses the result — a 300-experiment
mixed-engine study pays ~3 × 30 s of baseline measurement instead of
300 × 30 s.

---

## Multi-Cycle Execution

**Purpose:** Single measurements have variance from thermal drift, system load, and
caching effects. Repeating experiments across multiple cycles produces a distribution
of measurements that supports statistical analysis and confidence intervals.

Configure via the `study_execution:` section in a study YAML:

```yaml
study_execution:
  n_cycles: 3               # default (CLI): 3
  experiment_order: shuffle  # default (CLI): shuffle
```

**CLI effective defaults** for `llem run study.yaml` (if not set in the YAML):
- `n_cycles = 3`
- `experiment_order = shuffle`

### Why n_cycles >= 3?

With 3 or more cycles per experiment, you can report median and inter-quartile range,
detect outliers, and assess measurement stability. A single measurement cannot distinguish
true energy cost from transient effects (thermal spike, background process, cache miss).

For publication, use `n_cycles >= 5` to support confidence interval estimation.

### Experiment Ordering

For experiments A and B with 3 cycles:

**`sequential`** → `A, A, A, B, B, B`

All cycles of each experiment run together. Minimises model-load overhead (model stays
loaded across cycles). May introduce temporal bias if system state changes over time.

**`interleave`** → `A, B, A, B, A, B`

One cycle of each experiment per round, repeated. Balances temporal effects across
configurations — both A and B experience similar system conditions per round.
Good for comparisons where temporal fairness matters.

**`shuffle`** → random per-cycle order, seeded from study design hash

The execution order is randomised independently for each cycle. The seed is derived from
the study design hash (SHA-256 of the resolved experiment list), so the same study YAML
always produces the same shuffle sequence — reruns are reproducible.

`shuffle` is the CLI default. It eliminates systematic ordering bias while maintaining
reproducibility.

**`reverse`** → `A, B, B, A, A, B`

Alternates forward and backward experiment order each cycle. Even-numbered cycles run
experiments in the original order; odd-numbered cycles run them in reverse. Counterbalances
temporal drift (e.g. thermal ramp) without introducing randomness.

**`latin_square`** → Williams balanced design

Uses a Williams balanced latin square where each experiment follows every other experiment
exactly once across rows, cancelling first-order carryover effects (e.g. thermal residue
from the previous model). When `n_cycles > k` (number of experiments), the square rows
repeat; when `n_cycles < k`, the first `n_cycles` rows are used.

Best for studies where carryover effects between experiments are a concern.

Override the experiment order from the CLI:

```bash
llem run study.yaml --cycles 5 --order interleave
```

---

## Thermal Management

**Purpose:** GPU temperature affects power draw and throughput. A GPU running at 85°C
performs differently from one at 60°C. Without thermal gaps between experiments, earlier
experiments heat the GPU, causing later experiments to run at a higher baseline
temperature — introducing a systematic bias across sweep positions.

By default, llenergymeasure inserts thermal gaps between experiments in a study. These
gaps allow the GPU to return toward its baseline temperature before the next experiment
starts.

Disable thermal gaps for speed-oriented testing (at the cost of measurement quality):

```bash
llem run study.yaml --no-gaps
```

llenergymeasure also monitors thermal throttle events during measurement. If the GPU
throttled during an experiment, `thermal_throttle_detected: true` is set in that
experiment's result, and the throttle duration and trigger reason are recorded.

---

## Reproducibility

### Seeding model

llenergymeasure uses two independent seeds that control reproducibility at different
scopes:

**`random_seed`** (ExperimentConfig) — per-experiment stochasticity:

- Engine inference RNG (`torch.manual_seed`, vLLM `seed=`, TRT-LLM `random_seed=`)
- Dataset prompt ordering (when `dataset.order: shuffled`)

**`shuffle_seed`** (ExecutionConfig) — study-level scheduling:

- Cycle shuffle order (which experiment runs when)
- Default: derived from `study_design_hash` (same YAML always produces the same order)

These are orthogonal by design. Changing `random_seed` does not affect experiment
scheduling, and changing `shuffle_seed` does not affect inference outputs. This lets you
test sampling variance (vary `random_seed`) independently from ordering effects (vary
`shuffle_seed`).

### Reproducibility checklist

To maximise reproducibility across runs and machines:

1. **Fix the random seed.** The default `random_seed: 42` controls all per-experiment
   stochasticity — inference RNG and dataset ordering:
   ```yaml
   random_seed: 42
   ```

2. **Use shuffle experiment ordering with n_cycles >= 3.** Shuffle ordering is seeded from
   the study design hash — identical study YAML always produces identical shuffle order.
   To override the shuffle seed explicitly:
   ```yaml
   study_execution:
     experiment_order: shuffle
     shuffle_seed: 123  # null = derived from study_design_hash
   ```

3. **Enable warmup and baseline.** Both are enabled by default. Disabling either reduces
   reproducibility by introducing thermal and background-power variance.

4. **Control system load.** External processes sharing the GPU affect energy readings.
   For the most reproducible results, run on a dedicated GPU with no other CUDA processes.

5. **Report the effective config.** llenergymeasure stores the full resolved experiment
   config in `effective_config` in the result JSON. This captures every parameter value
   used, including engine defaults. Sharing the result JSON is sufficient for full
   reproduction.

6. **Pin model revision.** HuggingFace models update. To ensure the same weights across
   runs, pin the model revision:
   ```yaml
   pytorch:
     revision: "abc1234"   # commit hash or tag from HuggingFace Hub
   ```

### What is stored in results

Each experiment result JSON includes:
- `effective_config` — the fully resolved ExperimentConfig (all defaults filled in)
- `study_design_hash` — SHA-256[:16] of the resolved experiment list (for studies)
- `baseline_power_watts` — measured idle power for this session
- `thermal_throttle_detected` — whether throttling occurred during measurement
- Per-prompt timeseries data (power samples, latency) for detailed analysis

---

## Universal-to-Engine Parameter Mapping

llenergymeasure uses engine-native field names wherever possible. each engine library
(HuggingFace Transformers, vLLM, TensorRT-LLM) has its own naming conventions. A thin
mapping layer translates the handful of universal `ExperimentConfig` and `DecoderConfig`
fields to each engine's native API parameters.

### Design principle

Engine-specific configuration sections (`pytorch:`, `vllm:`, `tensorrt:`) always use the
engine library's native names directly - no translation. The mapping layer only applies to
shared (universal) fields that have identical semantics across all engines but different
API names.

### Complete mapping table

| Universal field | PyTorch native | vLLM native | TensorRT native | Notes |
|---|---|---|---|---|
| `dtype` | `torch_dtype` (torch.float16, etc.) | `dtype` (passthrough) | `dtype` (passthrough) | Direct mapping in PyTorch; passthrough for vLLM/TRT |
| `random_seed` | `torch.manual_seed()` | `seed=` in LLM() | `random_seed=` in SamplingParams | Different API surfaces |
| `max_input_tokens` | `max_length` in tokeniser | (pre-truncated by harness) | `max_input_len` (compile-time) | PyTorch truncates at tokenisation; TRT-LLM uses it as a compile-time engine constraint |
| `max_output_tokens` | `max_new_tokens` | `max_tokens` | `max_new_tokens` | **vLLM uses `max_tokens`**; PyTorch/TRT-LLM use `max_new_tokens` |
| `decoder.temperature` | `temperature` | `temperature` | `temperature` | No rename; conditional stripping in greedy mode |
| `decoder.do_sample` | `do_sample` | (implicit from temperature) | (implicit from temperature) | Only PyTorch has an explicit flag |
| `decoder.top_k` | `top_k` (0=disabled) | `top_k` (**0 → -1**) | `top_k` (0=skipped) | vLLM uses -1 to mean disabled |
| `decoder.top_p` | `top_p` | `top_p` | `top_p` | No rename |
| `decoder.repetition_penalty` | `repetition_penalty` | `repetition_penalty` | `repetition_penalty` | No rename |
| `decoder.min_p` | `min_p` | `min_p` | `min_p` | No rename |
| `decoder.min_new_tokens` | `min_new_tokens` | `min_tokens` | `min_tokens` | **vLLM/TRT-LLM use `min_tokens`** |

### Non-mapped fields

Everything else passes through without translation:

- **Engine-specific configs** (`pytorch.batch_size`, `vllm.engine.max_num_seqs`,
  `tensorrt.max_batch_size`, etc.) use native names - no mapping.
- **Sub-configs** (`warmup`, `baseline`, `energy`) are consumed by the measurement harness,
  not by engines.
- **`lora`** is defined in config but not yet implemented in any engine.

---

## Known Limitations

### NVML thermal throttle subsampling

NVML samples power and thermal state at ~1s intervals. Thermal throttle events shorter
than ~100ms may not be detected. The `throttle_detected` field in results reflects what
NVML observed, not the complete throttle history. This is an inherent limitation of
NVML's polling API and cannot be resolved without kernel-level instrumentation.
