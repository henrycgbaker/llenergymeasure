# datasets/ - Prompt Datasets

Built-in prompt datasets for LLM efficiency measurement. Layer 0 in the six-layer architecture.

## Purpose

Provides bundled prompt workloads and dataset loading utilities. Backends use `load_prompts(config)` to obtain the prompt list for an experiment. The default dataset is the AI Energy Score benchmark.

## Modules

| Module | Description |
|--------|-------------|
| `loader.py` | `load_prompts(config)` dispatcher, `BUILTIN_DATASETS` registry |
| `builtin/aienergyscore.jsonl` | Bundled AI Energy Score benchmark prompts |
| `__init__.py` | Re-exports `aienergyscore`, `load_prompts`, `BUILTIN_DATASETS` |

## Public API

```python
from llenergymeasure.datasets import load_prompts, BUILTIN_DATASETS, aienergyscore

# Load prompts for an experiment
prompts = load_prompts(config)  # returns list[str] of exactly config.n prompts

# Access bundled dataset path
path = aienergyscore  # pathlib.Path to the bundled JSONL file

# List available built-in datasets
print(BUILTIN_DATASETS)  # {"aienergyscore": Path(...)}
```

## Built-in datasets

| Name | Description |
|------|-------------|
| `aienergyscore` | AI Energy Score benchmark prompts (default dataset) |

## Dataset formats

### Built-in alias (default)

```yaml
dataset: aienergyscore   # default when no dataset specified
n: 100
```

### Custom JSONL file

```yaml
dataset: ./my-prompts.jsonl
n: 500
```

JSONL records must contain a prompt field. Auto-detected column names (in order): `prompt`, `text`, `instruction`, `input`, `question`.

### Synthetic prompts (for testing)

```yaml
dataset:
  type: synthetic
  input_len: 256    # approximate token count per prompt
  seed: 42
n: 50
```

## Dataset ordering

Control how prompts are selected and ordered via `dataset_order`:

| Value | Behaviour |
|-------|-----------|
| `interleaved` | File order (default) — stops reading after config.n records |
| `grouped` | Sort by `source` field (preserves intra-group order) |
| `shuffled` | Random shuffle using `random_seed` |

## Layer constraints

- Layer 0 — base layer; no imports from other llenergymeasure layers except `config/` (for `ExperimentConfig`)
- Can be imported by all layers above

## Related

- See `../config/README.md` for `PromptSourceConfig` and `SyntheticDatasetConfig`
