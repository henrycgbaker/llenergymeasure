# Comparison with Other Benchmarks

This guide explains how llenergymeasure relates to other AI benchmarking and energy measurement tools. It is intended for policy analysts and researchers who want to contextualise llenergymeasure results within the broader AI measurement landscape.

---

## The AI Energy Measurement Landscape

Several tools and benchmarks exist for measuring AI performance and energy use. They differ in purpose, scope, and methodology.

llenergymeasure focuses specifically on **inference energy efficiency** — measuring how much energy an AI system uses to generate responses, across different model sizes, deployment configurations, and inference engines. It is designed for reproducible, comparable measurements that can inform research, procurement, and policy.

The key distinction from performance benchmarks: most AI benchmarks measure *how capable* or *how fast* a model is. llenergymeasure measures *how efficiently* it uses energy. The two questions are related but different.

---

## MLPerf

**What it is:** MLPerf is the industry standard benchmark suite for machine learning performance, maintained by [MLCommons](https://mlcommons.org/). It covers both model training and inference.

**Who uses it:** Hardware vendors (NVIDIA, Intel, Google, AMD), cloud providers, and enterprise AI teams. MLPerf results are published publicly and widely cited in hardware procurement decisions.

**What it measures:** Primarily throughput — how many inferences per second a system can process — and quality — whether the model meets a minimum accuracy threshold on standardised tasks.

**What it does not measure:** Energy consumption is not a primary MLPerf metric, though some MLPerf Power results are published separately. MLPerf also requires fixed, certified hardware configurations and uses proprietary model weights in some divisions, limiting its applicability to research settings.

**How llenergymeasure complements MLPerf:**

- MLPerf answers "how fast is this hardware?"; llenergymeasure answers "how efficiently does this deployment use energy?"
- llenergymeasure works with any NVIDIA GPU and any HuggingFace model, not just certified configurations
- llenergymeasure's adjusted energy metric provides finer-grained efficiency analysis than MLPerf Power's system-level measurement

If you have MLPerf throughput results, you can run llenergymeasure on the same model and hardware to add the energy dimension to the comparison.

---

## AI Energy Score

**What it is:** AI Energy Score is an emerging benchmark initiative that assigns standardised energy efficiency ratings to AI models, similar in spirit to energy efficiency labels on household appliances.

**What it measures:** Energy per unit of useful output (e.g., joules per token or joules per task completion) on standardised tasks, intended to be comparable across models and hardware.

**How llenergymeasure relates:**

llenergymeasure includes the AI Energy Score benchmark dataset as a built-in option. When you run:

```bash
llem run --model gpt2 --backend pytorch
```

the default dataset used is the AI Energy Score prompt set. This means llenergymeasure results are directly comparable to AI Energy Score benchmarks run on the same hardware.

llenergymeasure is a *measurement tool* that produces the raw energy data; AI Energy Score is a *benchmark standard* that defines the tasks and comparison framework. They are complementary: you use llenergymeasure to generate measurements, and AI Energy Score to interpret and compare them.

**Citing results:** If you run llenergymeasure with the default AI Energy Score dataset and want to compare or contribute to the AI Energy Score leaderboard, ensure you report the hardware configuration (GPU model and memory), the precision setting, and the exact llenergymeasure version — these all affect comparability.

---

## CodeCarbon

**What it is:** [CodeCarbon](https://codecarbon.io/) is a Python library that estimates the carbon emissions of running Python code. It measures or estimates the energy used by the entire computer system (CPU, GPU, RAM) and converts it to a CO₂ equivalent using regional electricity carbon intensity data.

**What it measures:** Full-system energy and estimated carbon emissions. It uses a combination of hardware power monitoring (where available) and regional electricity grid data.

**How llenergymeasure uses it:**

CodeCarbon is one of the optional energy measurement backends in llenergymeasure. You can install it with:

```bash
pip install "llenergymeasure[codecarbon]"
```

When CodeCarbon is active, llenergymeasure reports both GPU-specific energy (via NVML) and CodeCarbon's broader system estimate, including the carbon conversion.

**The difference from the default:** llenergymeasure's default energy measurement (NVML) is GPU-specific and precise. CodeCarbon's estimate includes the whole system but uses estimation methods that are less precise for GPU workloads. For comparing inference efficiency across models, NVML is preferred; for carbon footprint reporting, CodeCarbon adds the CO₂ dimension.

---

## Zeus

**What it is:** [Zeus](https://ml.energy/zeus/) is a Python library for fine-grained GPU energy measurement, developed by the ML.ENERGY group at the University of Michigan. It measures GPU energy at the kernel level and also supports GPU energy optimisation (power capping).

**What it measures:** GPU energy at a finer granularity than NVML polling, with lower measurement overhead.

**How llenergymeasure uses it:**

Zeus is an optional energy measurement backend in llenergymeasure:

```bash
pip install "llenergymeasure[zeus]"
```

When Zeus is installed, llenergymeasure can use it instead of the default NVML polling, providing more accurate energy attribution for short inferences where polling granularity matters.

**The difference from the default:** NVML polling measures GPU power approximately every 100 milliseconds. Zeus measures energy more continuously, reducing the error for experiments that run for only a few seconds. For experiments running many prompts (100+), the difference is small; for single-prompt measurements, Zeus improves accuracy.

---

## Summary Table

| Tool | Type | Measures | Scope |
|------|------|----------|-------|
| **llenergymeasure** | Measurement tool | Inference energy, throughput, FLOPs | Any HF model, any NVIDIA GPU |
| **MLPerf** | Benchmark standard | Inference throughput, accuracy | Certified hardware + model configurations |
| **AI Energy Score** | Benchmark standard | Energy per unit output | Standardised tasks across models |
| **CodeCarbon** | Measurement library | Full-system energy + CO₂ | Any Python code |
| **Zeus** | Measurement library | GPU energy (kernel-level) | GPU workloads |

---

## How to Use llenergymeasure Results in Reports and Policy Documents

**Citing a result:** Include the following with any llenergymeasure result:

- Model name and version (e.g., `gpt2`, `meta-llama/Llama-3-8B`)
- Hardware (GPU model, e.g., NVIDIA A100 80GB)
- Precision setting (e.g., `bf16`)
- Number of prompts
- llenergymeasure version (from `llem --version`)
- Dataset (default: AI Energy Score prompts)

**Comparing across studies:** Results are only comparable if the hardware, dataset, number of prompts, and precision setting are identical. Hardware differences are the largest source of non-comparability — an A100 and a consumer GPU will produce very different energy figures even for the same model.

**Energy per token:** When comparing models of different sizes or with different output lengths, use the derived metric of joules per output token (`inference_energy_joules ÷ total_output_tokens`). This normalises for the amount of useful output produced.

**Reporting for sustainability:** If you are including AI energy use in an organisation's carbon accounting or sustainability report, pair llenergymeasure energy figures with the electricity carbon intensity of the data centre where the model runs, or use the CodeCarbon backend to get CO₂ estimates directly.

---

## Further Reading

- [What We Measure and Why It Matters](guide-what-we-measure.md) — plain-language explanation of energy, throughput, and FLOPs
- [How to Read llenergymeasure Output](guide-interpreting-results.md) — interpreting the numbers
- [Running Your First Measurement](guide-getting-started.md) — getting started with a measurement
- [Energy Measurement](energy-measurement.md) — technical depth on measurement backends (for researchers)
