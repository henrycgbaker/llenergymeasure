# api/ - Public Library API

Public Python API entry point for llenergymeasure. Layer 5 in the six-layer architecture.

## Purpose

Exposes `run_experiment()` and `run_study()` as the primary library interface. Also owns pre-flight validation and GPU index resolution. All user code and the CLI route through this layer.

## Modules

| Module | Description |
|--------|-------------|
| `_impl.py` | `run_experiment()` and `run_study()` implementations; `_run()` dispatcher |
| `_gpu.py` | `_resolve_gpu_indices()` — per-engine GPU index resolution |
| `preflight.py` | `run_preflight()` and `run_study_preflight()` — pre-experiment checks |
| `__init__.py` | Re-exports `run_experiment`, `run_study` |

## Public API

```python
from llenergymeasure import run_experiment, run_study
```

### run_experiment()

Three call forms:

```python
# YAML file path
result = run_experiment("configs/my-experiment.yaml")

# Config object
result = run_experiment(ExperimentConfig(model="gpt2", ...))

# Keyword convenience
result = run_experiment(model="gpt2", engine="pytorch", n=100)
```

Returns `ExperimentResult`. Side-effect free unless `output_dir` is set in the config.

### run_study()

```python
result = run_study("configs/sweep.yaml")
result = run_study(study_config)
```

Returns `StudyResult`. Always writes `manifest.json` to disk.

### skip_preflight

Both functions accept `skip_preflight=True` to bypass Docker and CUDA checks (useful in tests or environments where the GPU is inside the container):

```python
result = run_experiment(model="gpt2", skip_preflight=True)
```

## pre-flight checks (preflight.py)

`run_preflight(config)` runs before any GPU allocation:

1. CUDA available (`torch.cuda.is_available()`)
2. Engine package installed (`transformers`, `vllm`, or `tensorrt_llm`)
3. Model accessible (local path exists, or HuggingFace Hub reachable)
4. Backend `validate_config()` — hardware-specific checks (e.g., FP8 on non-Ada GPUs)

All failures are collected and raised together as a single `PreFlightError` so the user sees all problems at once.

`run_study_preflight(study)` adds multi-engine isolation enforcement: multi-engine studies require Docker; single-engine studies pass through.

## GPU index resolution (_gpu.py)

`_resolve_gpu_indices(config)` determines which GPUs to monitor for energy measurement:

| Engine | Rule |
|---------|------|
| `vllm` | `tp_size * pp_size` GPUs |
| `tensorrt` | `tp_size` GPUs |
| `pytorch` with `device_map` | All NVML-visible GPUs |
| Otherwise | `[0]` (single-GPU default) |

## Dispatch flow

```
run_experiment(...)
  └─ _to_study_config()       # normalise all input forms to StudyConfig
run_study(...)
  └─ _run(study)              # shared dispatcher
        ├─ run_study_preflight()
        ├─ resolve_study_runners()
        ├─ create_study_dir() + ManifestWriter
        ├─ _run_in_process()  # single experiment: harness or DockerRunner directly
        └─ _run_via_runner()  # multi-experiment: StudyRunner subprocess loop
```

## Layer constraints

- Layer 5 — may import from layers 0–4
- Cannot be imported by: `harness/`, `engines/`, `energy/`, `infra/`, `study/`, `device/`, `utils/`, `config/`, `domain/`, `datasets/`, `results/`
- The `cli/` layer is the only layer above `api/`

## Related

- See `../harness/` for measurement lifecycle
- See `../study/` for multi-experiment sweep execution
- See `../infra/` for Docker runner dispatch
