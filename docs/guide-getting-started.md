# Running Your First Measurement (Policy Maker Guide)

This guide walks you through installing llenergymeasure and running your first energy measurement — step by step, with explanations along the way. It assumes basic familiarity with a terminal (command line) but no programming knowledge.

If you want to understand *what* the measurements mean before you run them, read [What We Measure and Why It Matters](guide-what-we-measure.md) first.

---

## What You Need

Before you start, you need:

| Requirement | What it means |
|-------------|---------------|
| Python 3.10 or later | A programming language runtime — the engine that runs llenergymeasure |
| An NVIDIA GPU | A graphics card for running AI models. Required for energy measurement. |
| Linux operating system | Required for the full measurement stack. macOS/Windows work for PyTorch-only measurements. |
| Terminal access | A command-line interface (Terminal on macOS/Linux, PowerShell or Command Prompt on Windows) |

**Checking Python:** Open a terminal and type `python --version` or `python3 --version`. You should see a version number like `Python 3.11.2`. If Python is not installed, visit [python.org/downloads](https://www.python.org/downloads/).

**Not sure about GPU?** Type `nvidia-smi` in your terminal. If it shows a table of GPU information, you have an NVIDIA GPU. If it says "command not found", you do not have a compatible GPU.

For a detailed system requirements reference, see the [Installation Guide](installation.md).

---

## Step 1: Install llenergymeasure

In your terminal, run:

```bash
pip install "llenergymeasure[pytorch]"
```

**What this does:** Downloads and installs llenergymeasure along with the PyTorch inference engine — the component that actually runs AI models. The `[pytorch]` part tells the installer to include PyTorch as well as the core tool.

**How long it takes:** Several minutes on first install (PyTorch is a large package). Subsequent installs are faster because packages are cached.

**What you should see:** Lines of text as packages download and install, ending with `Successfully installed llenergymeasure-...`.

If you see a `pip: command not found` error, try `pip3` instead of `pip`, or `python -m pip`.

---

## Step 2: Check Your Setup

Run:

```bash
llem config
```

**What this does:** Checks your environment and prints a summary of what llenergymeasure can see — your GPU, which software components are installed, and the energy measurement method it will use.

**Example output:**

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Backends
  pytorch: installed
  vllm: not installed  (pip install llenergymeasure[vllm])
  tensorrt: not installed  (pip install llenergymeasure[tensorrt])
Energy
  Energy: nvml
Config
  Path: /home/user/.config/llenergymeasure/config.yaml
  Status: using defaults (no config file)
Python
  3.12.0
```

**What to look for:**

- **GPU section** shows your graphics card. If it says "No GPU detected", llenergymeasure will not be able to measure energy. Check that your NVIDIA drivers are installed.
- **Backends section** should show `pytorch: installed`. This is the inference engine you just installed.
- **Energy section** shows `nvml` — this is the energy measurement method. NVML reads directly from the GPU hardware and is the default.
- **Python section** confirms your Python version.

If `pytorch` shows "not installed", repeat Step 1. If the GPU is not detected, you may need to install NVIDIA drivers — see the [Installation Guide](installation.md) for guidance.

---

## Step 3: Run Your First Measurement

Run:

```bash
llem run --model gpt2 --backend pytorch
```

**What this does:**

- `llem run` — starts a measurement experiment
- `--model gpt2` — uses GPT-2, a small AI language model made freely available by OpenAI. It is tiny compared to modern AI systems (124 million parameters vs the billions in GPT-4 or Claude), which makes it fast to download and run.
- `--backend pytorch` — uses the PyTorch inference engine (what you installed in Step 1)

**On first run:** The model downloads from HuggingFace (about 500 MB). This happens once; subsequent runs use a local cache.

**How long it takes:** A few minutes on a modern NVIDIA GPU. You will see a progress bar.

**What you will see during the run:**

```
Downloading model gpt2...  [████████████] 100%
Running warmup (5 prompts)...
Running experiment (100 prompts)...  [████████░░] 80%
```

**What you will see when it finishes:**

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

---

## Step 4: Read Your Results

Here is what each section means:

**Energy**

- **Total (J)** — the total electrical energy your GPU consumed during the entire run, in joules. Think of this as the electricity bill for the experiment.
- **Baseline (W)** — how much power the GPU uses when idle (doing nothing). This is subtracted to isolate the energy specifically used for running the AI model.
- **Adjusted (J)** — total energy minus idle power. This is the most useful number for comparing different models: it tells you the energy specifically attributable to running the AI inference.

**Performance**

- **Throughput (tok/s)** — how many tokens (short word-pieces) the model produced per second across all 100 prompts. Higher is faster.
- **FLOPs** — an estimate of the computational work performed. Useful for comparing models of different sizes.

**Timing**

- **Duration** — how long the experiment took, wall-clock time.
- **Warmup** — how many prompts were excluded from the results to let the hardware reach a stable temperature. The metrics are based on the remaining prompts only.

For a detailed explanation of every metric, including what numbers are "normal" and how to compare results across models, see [How to Read llenergymeasure Output](guide-interpreting-results.md).

---

## Where Your Results Are Saved

Results are automatically saved to a `results/` folder in the directory where you ran the command:

```
results/
└── gpt2-pytorch-bf16-20240305-143022/
    └── result.json
```

The `result.json` file contains all metrics, the exact configuration used, and metadata. It is the scientific record of the measurement — keep it if you want to reproduce or reference the result later.

---

## Next Steps

You have run your first energy measurement. From here:

- **Compare models:** Change `--model gpt2` to a different model name (e.g., `--model facebook/opt-125m`) and compare the results. Larger models will use more energy.
- **Compare precisions:** Add `--precision fp32` to run at full precision and compare energy use against the default `bf16`.
- **Run a sweep:** Define a YAML configuration file to automatically run multiple configurations and compare them. See the [Researcher Getting Started Guide](getting-started.md) for the next step up.
- **Understand the numbers:** Read [How to Read llenergymeasure Output](guide-interpreting-results.md) for a deeper explanation of each metric.
- **See how this compares to other benchmarks:** Read [Comparison with Other Benchmarks](guide-comparison-context.md).
