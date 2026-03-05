# Getting Started

**Prerequisites:** Complete [Installation](installation.md) first.

This guide has two tracks. Choose one based on your setup:

- **Quick Start (Local PyTorch)** — No Docker required. Get running in minutes.
- **Recommended Start (Docker + vLLM)** — Full measurement experience with vLLM backend.

---

## Track 1: Quick Start (Local PyTorch)

### Prerequisites

- `llenergymeasure[pytorch]` installed (see [Installation](installation.md))
- NVIDIA GPU available

### 1. Verify your environment

```bash
llem config
```

Check that the output shows `pytorch: installed` under Backends. If it shows "not installed", run `pip install "llenergymeasure[pytorch]"`.

### 2. Run your first experiment

```bash
llem run --model gpt2 --backend pytorch
```

This runs GPT-2 (124M parameters). On first run, the model downloads from HuggingFace (~500 MB). Subsequent runs use the cache.

Default settings: 100 prompts, `aienergyscore` dataset, `bf16` precision.

You will see a progress indicator on stderr, then results printed to stdout:

```
Result: gpt2-pytorch-bf16-20240305-143022     # ← unique experiment ID

Energy                                          # ← GPU energy consumed
  Total          847 J                          # ← total joules for all 100 prompts
  Baseline       12.3 W                         # ← idle GPU power (subtracted from total)
  Adjusted       723 J                          # ← energy minus baseline × duration

Performance                                     # ← throughput and compute
  Throughput     312 tok/s                      # ← output tokens per second (all 100 prompts)
  FLOPs          4.21e+11 (roofline, medium)   # ← estimated FLOPs (method, confidence)

Timing                                          # ← wall-clock time
  Duration       1m 38s                         # ← total experiment wall time
  Warmup         5 prompts excluded             # ← thermal stabilisation prompts (not in metrics)
```

### Reading the results

| Field | What it measures |
|-------|-----------------|
| `Total` (J) | Raw GPU energy consumed during the experiment |
| `Baseline` (W) | Idle GPU power measured before the run |
| `Adjusted` (J) | Energy minus `Baseline × Duration` — net inference energy |
| `Throughput` (tok/s) | Output tokens generated per second across all prompts |
| `FLOPs` | Estimated floating-point operations (method and confidence shown) |
| `Duration` | Wall-clock time for the full experiment |
| `Warmup` | Number of prompts run for thermal stabilisation and excluded from metrics |

### 3. Output files

Results are saved to `results/` in the current directory by default:

```
results/
└── gpt2-pytorch-bf16-20240305-143022/
    └── result.json        # full result record (all metrics, config, metadata)
```

The JSON file is the scientific record — it contains all raw metrics, the resolved config, timestamps, and measurement warnings.

Specify a different output directory with `--output`:

```bash
llem run --model gpt2 --backend pytorch --output /data/experiments
```

---

## Track 2: Recommended Start (Docker + vLLM)

### Prerequisites

- `llenergymeasure[pytorch]` installed (base + pytorch needed even for Docker dispatch)
- Docker + NVIDIA Container Toolkit installed — see [Docker Setup](docker-setup.md)

### 1. Create a config file

Create `experiment.yaml`:

```yaml
model: gpt2
backend: vllm
n: 50
runners:
  vllm: docker
```

### 2. Run the experiment

```bash
llem run experiment.yaml
```

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU visibility inside container, CUDA/driver compatibility.
2. The vLLM Docker image is pulled on first run (`ghcr.io/henrycgbaker/llenergymeasure/vllm:0.8.0-cuda12`).
3. The container launches, runs the experiment, and streams results back.
4. Results are printed to stdout and saved to `results/`.

### 3. Read the results

The output format is the same as the PyTorch track. The key difference is `backend: vllm` in the experiment ID and result file.

---

## Using a Config File

For repeatability, store your experiment configuration in a YAML file.

Minimal config:

```yaml
# experiment.yaml
model: gpt2
backend: pytorch
n: 100
```

Run it:

```bash
llem run experiment.yaml
```

This is equivalent to `llem run --model gpt2 --backend pytorch -n 100`. CLI flags override YAML values when both are provided.

For study sweeps (running multiple configurations), see the [Study Configuration](study-config.md) reference.

---

## Next Steps

- [Study Configuration](study-config.md) — run parameter sweeps across models, backends, and configurations
- [Docker Setup](docker-setup.md) — set up Docker + NVIDIA Container Toolkit for vLLM/TensorRT-LLM
- [CLI Reference](cli-reference.md) — all `llem run` and `llem config` flags
