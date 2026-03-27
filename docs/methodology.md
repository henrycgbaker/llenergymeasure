# Measurement Methodology

How llenergymeasure ensures reproducible, reliable energy measurements.

---

## Warmup

**Purpose:** GPU thermal state, driver caches, and JIT compilation warm-up all affect
first-run measurements. The first few inferences with a freshly loaded model are
consistently slower and higher-energy than subsequent inferences. Warmup discards
these initial measurements.

Configure via the `warmup:` section:

```yaml
warmup:
  enabled: true            # default: true
  n_warmup: 5              # default: 5
  thermal_floor_seconds: 60.0   # default: 60.0
```

**What happens:**
1. `n_warmup` inference passes are run with full-length prompts from the dataset.
2. Results from these passes are discarded — they do not appear in output metrics.
3. After warmup passes complete, llenergymeasure waits `thermal_floor_seconds` for GPU
   temperature to stabilise before beginning measurement.

**Default values are calibrated for publication quality:**
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

### Convergence Detection (opt-in)

llenergymeasure supports CV-based adaptive warmup as an alternative to fixed iteration
count. When `convergence_detection: true`, warmup continues until latency coefficient of
variation (CV) drops below `cv_threshold`, up to `max_prompts` total passes.

```yaml
warmup:
  enabled: true
  convergence_detection: true
  cv_threshold: 0.05    # stop when CV < 5%
  max_prompts: 20       # safety cap
  window_size: 5        # rolling window for CV calculation
```

CV detection is additive to `n_warmup` — it adds passes after the fixed warmup phase if
CV has not converged. For most models and GPUs, fixed warmup is sufficient.

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

### Cycle Ordering

For experiments A and B with 3 cycles:

**`sequential`** → `A, A, A, B, B, B`

All cycles of each experiment run together. Minimises model-load overhead (model stays
loaded across cycles). May introduce temporal bias if system state changes over time.

**`interleaved`** → `A, B, A, B, A, B`

One cycle of each experiment per round, repeated. Balances temporal effects across
configurations — both A and B experience similar system conditions per round.
Good for comparisons where temporal fairness matters.

**`shuffled`** → random per-cycle order, seeded from study design hash

The execution order is randomised independently for each cycle. The seed is derived from
the study design hash (SHA-256 of the resolved experiment list), so the same study YAML
always produces the same shuffle sequence — reruns are reproducible.

`shuffled` is the CLI default. It eliminates systematic ordering bias while maintaining
reproducibility.

Override the cycle order from the CLI:

```bash
llem run study.yaml --cycles 5 --order interleaved
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

- Backend inference RNG (`torch.manual_seed`, vLLM `seed=`, TRT-LLM `random_seed=`)
- Dataset prompt ordering (when `dataset_order: shuffled`)
- Synthetic prompt generation

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
   stochasticity — inference RNG, dataset ordering, and synthetic prompt generation:
   ```yaml
   random_seed: 42
   ```

2. **Use shuffled cycle ordering with n_cycles >= 3.** Shuffled ordering is seeded from
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
   used, including backend defaults. Sharing the result JSON is sufficient for full
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
