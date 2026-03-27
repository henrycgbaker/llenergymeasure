# How to Read llenergymeasure Output

When llenergymeasure runs a measurement, it prints a summary to the terminal and saves a detailed result file. This guide explains what each number means — in plain language.

If you have not yet run a measurement, see [Running Your First Measurement](guide-getting-started.md) first.

---

## What You Will See

After running a measurement, the terminal prints output like this:

```
Result: gpt2-pytorch-bf16-20240305-143022

Energy
  Total          847 J
  Baseline       12.3 W
  Adjusted       723 J

Performance
  Throughput     312 tok/s
  FLOPs          4.21e+11 (roofline, medium)

Timing
  Duration       1m 38s
  Warmup         5 prompts excluded
```

A full result file is also saved to the `results/` folder. The sections below explain each metric.

---

## The Experiment ID

```
Result: gpt2-pytorch-bf16-20240305-143022
```

This is a unique identifier for this specific measurement. It encodes:

- `gpt2` — the model measured
- `pytorch` — the inference engine used
- `bfloat16` — the model dtype
- `20240305-143022` — the date and time the experiment ran

The experiment ID helps you trace results back to the exact configuration that produced them.

---

## Energy Metrics

Energy is measured in **joules** (J) — a standard unit of energy. One joule equals one watt of power consumed for one second.

### Total (J) — Raw GPU energy

The total electrical energy drawn by the GPU during the entire measurement period, from first prompt to last.

*In practice:* This includes both the energy for actual inference work and the energy the GPU would have consumed anyway just by being switched on ("idle power").

### Baseline (W) — Idle GPU power

The power the GPU draws when doing nothing — measured before the experiment starts and used as a reference point.

*Analogy:* Like measuring a car's fuel consumption at idle before a test drive, so you can subtract it from the total and isolate the fuel used specifically for driving.

*In practice:* This is reported in **watts** (W), not joules, because it is a power level rather than a total energy amount.

### Adjusted (J) — Net inference energy

The most meaningful energy metric: total energy minus the idle power multiplied by the duration. This represents the energy specifically attributable to running the AI model.

*Formula:* `Adjusted = Total − (Baseline × Duration)`

*In practice:* Use the adjusted figure when comparing models. Two models running on the same hardware for the same task may have different baseline subtractions; the adjusted figure puts them on equal footing.

---

## Performance Metrics

### Throughput (tok/s) — Output speed

How many output tokens the model generated per second, averaged across the entire experiment.

*In practice:* Higher throughput means faster responses. A model producing 312 tok/s completes 100 prompts roughly 3× faster than a model at 100 tok/s.

*Note:* Throughput is measured across all prompts in the experiment. A single short prompt may feel fast even at low throughput; the experiment-level figure reflects sustained performance.

### FLOPs — Computational work

An estimate of the number of floating-point calculations the model performed. Reported in scientific notation (e.g., `4.21e+11` = 421 billion FLOPs).

The result also shows:
- **Method** (e.g., `roofline`) — how the FLOPs were estimated
- **Confidence** (e.g., `medium`) — how reliable the estimate is

*In practice:* FLOPs are most useful for comparing models of different sizes. A larger model will naturally have higher FLOPs. If two models have similar FLOPs but very different energy, that suggests a hardware or configuration efficiency difference rather than a model complexity difference.

---

## Timing Metrics

### Duration — Total experiment time

The wall-clock time from the start of the first prompt to the end of the last, including all processing time.

*In practice:* Duration × Baseline power gives the idle energy component (which is subtracted to produce the Adjusted energy figure).

### Warmup — Excluded prompts

The number of prompts run at the start that were excluded from the reported metrics.

*Why:* GPUs do not immediately run at a stable temperature. The first few prompts run while the hardware is still warming up, which produces unrepresentative measurements. Warmup prompts are run first and then discarded, ensuring the reported metrics reflect steady-state operation.

*In practice:* If the experiment ran 100 prompts and 5 were warmup, the metrics are based on 95 prompts. The `total_prompts` field in the result file shows the total including warmup; the metrics are calculated from the non-warmup prompts only.

---

## The Result File

The full result is saved as a JSON file in the `results/` directory. Key fields:

| Field | What it means |
|-------|---------------|
| `energy_joules` | Total GPU energy (same as "Total" in terminal output) |
| `inference_energy_joules` | Adjusted energy (same as "Adjusted" in terminal output) |
| `throughput_tokens_per_second` | Output tokens per second (same as "Throughput") |
| `latency_seconds` | Total duration in seconds (same as "Duration") |
| `inference_memory_mb` | Peak GPU memory used during inference, in megabytes |
| `total_prompts` | Number of prompts processed (excluding warmup) |
| `total_output_tokens` | Total output tokens generated across all prompts |
| `effective_config` | The exact configuration used (model, dtype, backend, etc.) |

The `effective_config` section is particularly important for reproducibility — it records every setting that influenced the measurement, including defaults that were not explicitly specified.

---

## Comparing Results Meaningfully

Raw numbers only make sense in context. Here is how to compare results fairly:

**Use energy per token, not total energy.** A run with 100 prompts will use roughly twice the energy of a run with 50 prompts. To compare two experiments, divide adjusted energy by total output tokens: this gives joules per token, which is comparable regardless of experiment size.

**Match prompt counts and input lengths.** Output token counts (and therefore energy) vary with input length. Comparing a run with 100 short prompts against a run with 100 long prompts is not a like-for-like comparison.

**Note the hardware.** A result from an A100 GPU is not directly comparable to a result from a consumer GPU. The result file records the GPU model in `effective_config`.

**Check the dtype setting.** Running at `float16` (16-bit) typically uses less energy than `float32` (32-bit). Results should use the same dtype to be comparable.

---

## Order of Magnitude Context

To put energy figures in context (approximate, for orientation only):

| Scenario | Approximate energy |
|----------|--------------------|
| One GPT-2 inference (single prompt) | ~1–10 joules |
| 100 GPT-2 inferences | ~100–1,000 joules |
| One large model (70B) inference | ~500–5,000 joules |
| Smartphone full charge | ~15,000 joules |
| Boiling 1 litre of water | ~330,000 joules |

These figures vary significantly with hardware, prompt length, and configuration. They are intended to give a sense of scale, not precise values.

---

## Further Reading

- [What We Measure and Why It Matters](guide-what-we-measure.md) — conceptual background on the three metrics
- [Running Your First Measurement](guide-getting-started.md) — step-by-step guide to running measurements
- [Comparison with Other Benchmarks](guide-comparison-context.md) — how these results relate to MLPerf, AI Energy Score, and other benchmarks
