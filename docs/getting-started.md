# Getting Started

**Prerequisites:** Complete [Installation](installation.md) first.

This guide has three tracks. Choose one based on your setup:

- **Quick Start (Local PyTorch)** — No Docker required. Get running in minutes.
- **Recommended Start (Docker + vLLM)** — Full measurement experience with vLLM backend.
- **TensorRT-LLM Start (Docker)** — Maximum performance with TensorRT engine compilation.

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
2. The vLLM Docker image is pulled on first run (`ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0`).
3. The container launches, runs the experiment, and streams results back.
4. Results are printed to stdout and saved to `results/`.

### 3. Read the results

The output format is the same as the PyTorch track. The key difference is `backend: vllm` in the experiment ID and result file.

---

## Track 3: TensorRT-LLM (Docker)

TensorRT-LLM compiles models into optimised TensorRT engines, then runs inference against
those engines. The first run compiles the engine (which may take several minutes); subsequent
runs with the same config load the cached engine and are much faster.

### Prerequisites

- Docker + NVIDIA Container Toolkit installed — see [Docker Setup](docker-setup.md)
- `llenergymeasure[pytorch]` installed (base + pytorch needed even for Docker dispatch)
- NVIDIA GPU with SM >= 7.5 (Turing or newer; e.g. RTX 2000-series, A100, H100)

### 1. Create a config file

Create `experiment.yaml`:

```yaml
model: meta-llama/Llama-2-7b-hf
backend: tensorrt
n: 50
runners:
  tensorrt: docker
```

For a quantized run with engine caching configured explicitly:

```yaml
model: meta-llama/Llama-2-7b-hf
backend: tensorrt
n: 50
runners:
  tensorrt: docker
tensorrt:
  max_batch_size: 8
  dtype: bfloat16
  quant:
    quant_algo: W4A16_AWQ
  build_cache:
    max_cache_storage_gb: 100
```

### 2. Run the experiment

```bash
llem run experiment.yaml
```

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU visibility, SM version check.
2. The TensorRT-LLM Docker image is pulled on first run (`ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0`).
3. The container compiles the TensorRT engine from the model weights. **First run only — this takes several minutes.** Progress is shown in the terminal.
4. The compiled engine is cached on disk (`~/.cache/tensorrt_llm` inside the container, mounted from the host).
5. Inference runs against the compiled engine.
6. Results are printed to stdout and saved to `results/`.

> **Engine caching.** The compiled engine is keyed to your config (model, dtype, max_batch_size,
> tp_size, etc.). Running the same experiment config again skips compilation and starts
> inference immediately. Changing any compile-time parameter triggers a new build.

### 3. Read the results

The output format is the same as other backends. The result file will include `backend: tensorrt`
and a `build_metadata` section with engine compilation time, GPU architecture, and TRT-LLM version.

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
- [Backend Configuration](backends.md) — configure vLLM, TensorRT-LLM, and switch between backends
- [CLI Reference](cli-reference.md) — all `llem run` and `llem config` flags
