# CLI Reference

`llem` has two commands (`run` and `config`) and one flag (`--version`).

```
llem run [CONFIG] [OPTIONS]   # run an experiment or study
llem config [OPTIONS]         # show environment and configuration status
llem --version                # print version and exit
```

---

<!-- Auto-generated sections from scripts/generate_cli_reference.py — regenerate with: uv run python scripts/generate_cli_reference.py -->

## `llem run`

Run an experiment or study. Detects study mode automatically when the YAML config contains `sweep:` or `experiments:` keys.

**Arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `config` | path | no | Path to experiment or study YAML config |

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | str | — | Model name or HuggingFace path |
| `--backend` | `-b` | str | — | Inference backend (`pytorch`, `vllm`, `tensorrt`) |
| `--dataset` | `-d` | str | — | Dataset source (`aienergyscore` or `.jsonl` file path) |
| `-n` | | int | — | Number of prompts to run (`dataset.n_prompts`) |
| `--batch-size` | | int | — | Batch size (PyTorch backend only) |
| `--dtype` | `-p` | str | — | Model dtype (`float32`, `float16`, `bfloat16`) |
| `--output` | `-o` | str | — | Output directory for results |
| `--dry-run` | | flag | false | Validate config and estimate VRAM without running |
| `--quiet` | `-q` | flag | false | Suppress progress bars |
| `--verbose` | `-v` | flag | false | Show detailed output and tracebacks |
| `--cycles` | | int | — | Number of measurement cycles (study mode) |
| `--order` | | str | — | Cycle ordering: `sequential`, `interleave`, `shuffle` (study mode) |
| `--no-gaps` | | flag | false | Disable thermal gaps between experiments (study mode) |
| `--skip-preflight` | | flag | false | Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility) |
| `--resume` | | flag | false | Resume most recent interrupted study |
| `--resume-dir` | | path | — | Resume a specific study directory |
| `--fail-fast` | | flag | false | Abort study on first failure (circuit breaker threshold=1) |
| `--no-circuit-breaker` | | flag | false | Disable circuit breaker entirely |
| `--timeout` | | float | — | Study wall-clock timeout in hours (e.g. `24`, `1.5`) |
| `--no-lock` | | flag | false | Disable GPU lock files (advanced) |

**CLI effective defaults for study mode** (applied when neither the YAML `study_execution:` block nor a CLI flag specifies the value):

- `--cycles` defaults to `3` (Pydantic model default is `1`)
- `--order` defaults to `shuffle` (Pydantic model default is `sequential`)

These defaults are applied at the CLI layer to give better statistical coverage out of the box. To use the conservative model defaults, set them explicitly in the YAML `study_execution:` block.

**Exit codes:** `0` success, `1` experiment/backend/preflight error, `2` config validation error, `130` interrupted (Ctrl-C).

---

## `llem config`

Show environment and configuration status. Always exits `0` — this command is informational only.

**Options:**

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--verbose` | `-v` | flag | false | Show driver version, backend versions, and full config diff |

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

---

## `llem --version`

Print version and exit.

```bash
llem --version
```

Example output:

```
llem v0.9.0
```

---

## Examples

### Single experiment via flags

```bash
llem run --model gpt2 --backend pytorch
```

### Single experiment via YAML

```bash
llem run experiment.yaml
```

### Dry run (validate config, estimate VRAM)

```bash
llem run experiment.yaml --dry-run
```

### Study with cycle override

```bash
# Run 5 cycles in interleave order instead of the CLI default (3 shuffle)
llem run study.yaml --cycles 5 --order interleave
```

### Suppress thermal gaps (testing only)

```bash
llem run study.yaml --no-gaps
```

### Skip Docker pre-flight (when Docker daemon is on a remote host)

```bash
llem run study.yaml --skip-preflight
```

### Resume an interrupted study

```bash
# Auto-detect most recent resumable study
llem run study.yaml --resume

# Resume a specific study directory
llem run study.yaml --resume-dir results/full-suite-all-backends_20260329_1716/
```

### Fail-fast mode (abort on first failure)

```bash
llem run study.yaml --fail-fast
```

### Set a wall-clock timeout

```bash
# Abort after 24 hours, mark remaining experiments as skipped
llem run study.yaml --timeout 24
```

### Environment check

```bash
llem config
llem config --verbose
```
