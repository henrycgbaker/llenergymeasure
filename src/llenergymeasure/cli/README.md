# cli/ - Command-line Interface

Typer-based CLI for llenergymeasure. Layer 6 (top layer) in the six-layer architecture.

## Purpose

Three commands: `llem run` for running experiments and studies, `llem config` for environment diagnostics, and `llem doctor` for host/container Docker-image schema verification. The CLI is a thin client over the `api/` layer — it handles argument parsing, user-facing formatting, and error presentation.

## Modules

| Module | Description |
|--------|-------------|
| `__init__.py` | `app` Typer instance, logging setup, command registration |
| `run.py` | `llem run` command |
| `config_cmd.py` | `llem config` command |
| `doctor_cmd.py` | `llem doctor` command (image schema handshake) |
| `_display.py` | Output formatting helpers (headers, result tables, errors) |
| `_vram.py` | VRAM estimation for pre-run model size hints |

## Commands

### llem run

```bash
llem run [CONFIG]           # run from YAML config
llem run --model gpt2       # inline model spec
llem run --model gpt2 --engine transformers --dataset aienergyscore -n 100
llem run --dry-run          # validate config without running
llem run -v                 # verbose logging (INFO)
llem run -vv                # debug logging
```

Key options:

| Option | Description |
|--------|-------------|
| `CONFIG` | Path to experiment or study YAML |
| `--model / -m` | Model name or HuggingFace path |
| `--engine / -e` | Inference engine (`pytorch`, `vllm`, `tensorrt`) |
| `--dataset / -d` | Dataset name or JSONL file path |
| `-n` | Number of prompts |
| `--batch-size` | Batch size (PyTorch engine) |
| `--dry-run` | Validate config, print plan, exit |
| `--skip-preflight` | Skip Docker/CUDA pre-flight checks |
| `-v / -vv` | Verbosity (INFO / DEBUG) |

### llem config

```bash
llem config          # brief environment summary
llem config -v       # verbose: list all GPU properties, energy samplers, etc.
```

Shows GPU hardware (name, VRAM), installed engines, energy sampler availability, and user config path. Always exits 0 — purely informational.

### llem doctor

```bash
llem doctor          # verify Docker images match the host ExperimentConfig schema
```

Reads the `llem.expconf.schema.fingerprint` OCI label from each engine image and compares it to a fingerprint computed from the host's current `ExperimentConfig.model_json_schema()`. Exits non-zero on any `MISMATCH`. Set `LLEM_SKIP_IMAGE_CHECK=1` to bypass the runtime handshake in `llem run` (doctor still reports the true status with a warning footer).

### llem --version

```bash
llem --version
```

## Logging

Verbosity is controlled per-run via `-v` / `-vv`. The `llenergymeasure` logger hierarchy is configured in `__init__.py`:

| Flag | Level |
|------|-------|
| (none) | WARNING (or `LLEM_LOG_LEVEL` env var) |
| `-v` | INFO |
| `-vv` | DEBUG |

## Layer constraints

- Layer 6 — top layer, may import from all layers below
- Nothing imports from `cli/` except the entrypoint defined in `pyproject.toml`
- CLI is a client of `api/`, not the reverse

## Related

- See `../api/` for the library API the CLI delegates to
- See `../config/README.md` for full YAML config reference
