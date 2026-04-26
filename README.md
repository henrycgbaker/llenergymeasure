# LLenergMeasure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Measure the energy efficiency of LLM inference across different implementation configurations.

LLenergyMeasure is a Python framework for measuring the energy consumption, throughput, and computational cost (FLOPs) of LLM inference across different deployment configurations. It helps researchers compare the energy efficiency of different models, inference engines, and a wide range of implementation decisions — reproducibly and at publication quality.

---

## Key Features

- **Multi-engine inference** — Transformers, vLLM, TensorRT-LLM, SGLang (planned)
- **GPU energy measurement** — NVML, Zeus, CodeCarbon, others 
- **Smart sweep system** — define parameter grids, run Cartesian product experiments automatically; intelligently managed sweep hierarchy scopes available config fields to appropriate engine/component, and ensures invalid combinations are removed
- **Docker isolation** — launches per-experiment containers with full GPU passthrough; latest docker images for each engine in registry with full runner configurability and local mode also available. Every study pre-flight now verifies that each image's `ExperimentConfig` schema fingerprint matches the host's, aborting with an actionable rebuild hint on drift (`llem doctor` for a one-shot check).
- **Reproducibility** — fixed seeds, cycle ordering, thermal management, environment snapshots, effective config recorded (add others)
- **Built-in datasets** — AI Energy Score benchmark prompts included; custom JSONL datasets also supported

---

## Quick Install

```bash
pip install "llenergymeasure[transformers]"
```

Run your first measurement:

```bash
llem run --model gpt2 --engine transformers
```

See [Installation](docs/installation.md) for system requirements, Docker setup, and available extras. See [Getting Started](docs/getting-started.md) to run and interpret your first experiment.

---

## Documentation

### Researcher Docs

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | System requirements, pip install, Docker setup path |
| [Getting Started](docs/getting-started.md) | First experiment, Transformers and Docker tracks |
| [Docker Setup](docs/docker-setup.md) | NVIDIA Container Toolkit walkthrough for vLLM |
| [Engine Configuration](docs/engines.md) | Transformers vs vLLM, parameter support matrix |
| [Study & Experiment Configuration](docs/study-config.md) | YAML reference, sweeps, config schema |
| [CLI Reference](docs/cli-reference.md) | `llem run`, `llem config`, and `llem doctor` flags and options |
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

### Architecture and Internals

| Guide | Description |
|-------|-------------|
| **Start here:** [Architecture Overview](docs/architecture-overview.md) | System diagram, pipeline overview, key concepts (entry point to the suite) |
| [Invariant Miner Pipeline](docs/miner-pipeline.md) | How validation rules are extracted from engine library source |
| [Config Validation Pipeline](docs/parameter-discovery.md) | How configs are validated before engine initialisation |
| [Validation Rule Corpus Format](docs/validation-rule-corpus.md) | YAML schema reference for corpus rules |
| [Extending the Miner](docs/extending-miners.md) | How to add a new engine to the invariant miner |
| [Research Context](docs/research-context.md) | Academic positioning: Daikon, Houdini, NeuRI, and what is novel |
| [Schema Refresh Pipeline](docs/schema-refresh.md) | Renovate-driven engine schema refresh |

---

## Contributing

Contributions welcome. See the [development install](docs/installation.md#install-from-source-development) instructions to set up a local environment.

---

## License

[MIT](LICENSE)
