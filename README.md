# LLenergMeasure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Measure the energy efficiency of LLM inference across different implementation configurations.

LLenergyMeasure is a Python framework for measuring the energy consumption, throughput, and computational cost (FLOPs) of LLM inference across different deployment configurations. It helps researchers compare the energy efficiency of different models, inference engines, and a wide range of implementation decisions — reproducibly and at publication quality.

---

## Key Features

- **Multi-backend inference** — PyTorch, vLLM, TensorRT-LLM, SGLang (planned)
- **GPU energy measurement** — NVML, Zeus, CodeCarbon, others 
- **Smart sweep system** — define parameter grids, run Cartesian product experiments automatically; intelligently managed sweep hierarchy scopes available config fields to appropriate backend/component, and ensures invalid combinations are removed
- **Docker isolation** — launches per-experiment containers with full GPU passthrough; latest docker images for each backend in registry with full runnder configurability and local mode also available.
- **Reproducibility** — fixed seeds, cycle ordering, thermal management, environment snapshots, effective config recorded (add others)
- **Built-in datasets** — AI Energy Score benchmark prompts included; synthetic datasets generation also included

---

## Quick Install

```bash
pip install "llenergymeasure[pytorch]"
```

Run your first measurement:

```bash
llem run --model gpt2 --backend pytorch
```

See [Installation](docs/installation.md) for system requirements, Docker setup, and available extras. See [Getting Started](docs/getting-started.md) to run and interpret your first experiment.

---

## Documentation

### Researcher Docs

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | System requirements, pip install, Docker setup path |
| [Getting Started](docs/getting-started.md) | First experiment, PyTorch and Docker tracks |
| [Docker Setup](docs/docker-setup.md) | NVIDIA Container Toolkit walkthrough for vLLM |
| [Backend Configuration](docs/backends.md) | PyTorch vs vLLM, parameter support matrix |
| [Study & Experiment Configuration](docs/study-config.md) | YAML reference, sweeps, config schema |
| [CLI Reference](docs/cli-reference.md) | `llem run` and `llem config` flags and options |
| [Energy Measurement](docs/energy-measurement.md) | NVML, Zeus, CodeCarbon backends, measurement mechanics |
| [Measurement Methodology](docs/methodology.md) | Warmup, baseline, thermal management, reproducibility |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, invalid combinations, getting help |

### Policy Maker Guides

| Guide | Description |
|-------|-------------|
| [What We Measure](docs/guide-what-we-measure.md) | Plain-language explanation of energy, throughput, and FLOPs |
| [Interpreting Results](docs/guide-interpreting-results.md) | How to read llenergymeasure output |
| [Getting Started (Policy Maker)](docs/guide-getting-started.md) | Minimal path to running a measurement |
| [Comparison with Other Benchmarks](docs/guide-comparison-context.md) | MLPerf, AI Energy Score, CodeCarbon, Zeus context |

---

## Contributing

Contributions welcome. See the [development install](docs/installation.md#install-from-source-development) instructions to set up a local environment.

---

## License

[MIT](LICENSE)
