# Phase 21: Measurement Carried Items — Research

**Researched:** 2026-03-03
**Domain:** Python package data files, PyTorch CUDA memory measurement, dataset module design
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MEAS-03 | `aienergyscore.jsonl` built-in dataset file created (carried from M1) | Dataset module structure, JSONL format, importlib pattern documented below |
| MEAS-04 | `peak_memory_mb` measurement semantics confirmed and documented (carried from M1) | Current code path traced, two measurement sites found with different semantics |
</phase_requirements>

---

## Summary

Phase 21 closes two M1 carry-forward items that have been stubbed in code since v1.17.0. Both tasks are purely additive — no existing behaviour is broken, no existing tests fail. The phase is internally independent (each plan can ship separately).

**Plan 21-01: aienergyscore dataset.** The `llenergymeasure.datasets` module does not exist yet. The `core/dataset_loader.py` file references `BUILTIN_DATASETS` and `AUTO_DETECT_COLUMNS` from `config.models` which do not exist — this is a broken import today. The design docs specify exactly the module layout, JSONL format, and that the dataset ships bundled (not downloaded at runtime). The AIEnergyScore/text_generation HuggingFace dataset has 1,000 rows with a single `text` column. The JSONL file must be downloaded, converted to `{"prompt": "..."}` format, and committed to `src/llenergymeasure/datasets/builtin/aienergyscore.jsonl`. Simultaneously, `core/dataset_loader.py`'s broken import must be fixed and both backends' `_prepare_prompts()` placeholder must be wired to the real loader.

**Plan 21-02: peak_memory_mb semantics.** There are currently two separate measurement paths that both set `peak_memory_mb` but with different semantics. `PyTorchBackend._run_measurement()` calls `torch.cuda.max_memory_allocated()` at the end of the measurement loop — this includes model weights loaded before the loop. `compute_metrics.py:get_memory_stats()` calls `torch.cuda.reset_peak_memory_stats()` then immediately reads `max_memory_allocated()` (zero or near-zero, not useful). The product design intent (from `designs/result-schema.md`) is inference-window-only: reset is called after model load, before first inference. The code does not match this intent. The fix is: call `torch.cuda.reset_peak_memory_stats()` in PyTorchBackend after model load and before `_run_measurement()`, then the existing `max_memory_allocated()` call at end-of-loop correctly captures inference-only peak. Document the semantics in both `ComputeMetrics.peak_memory_mb` and `MemoryEfficiencyMetrics.peak_memory_mb` field descriptions, and in `designs/result-schema.md`.

**Primary recommendation:** Create the datasets module first (21-01), then fix peak_memory_mb semantics (21-02). Both plans are low-risk additive changes.

---

## Standard Stack

### Core (no new dependencies needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `json` | 3.10+ | Parse/write JSONL | Already used throughout codebase |
| Python stdlib `pathlib.Path` | 3.10+ | File paths | Project convention (CLAUDE.md) |
| `importlib.resources` | 3.10+ stdlib | Access package data files | Standard Python mechanism for bundled data |
| `torch.cuda` | existing | Memory stats | Already used in PyTorchBackend |

### No New Dependencies

This phase requires no new `pip install` dependencies. The HuggingFace `datasets` library is only needed once during the one-time download/generation of the JSONL file (developer task, not runtime).

**Installation (for JSONL generation, dev only):**
```bash
pip install datasets  # already in .venv, dev only — not a package dep
```

---

## Architecture Patterns

### Recommended Module Layout

```
src/llenergymeasure/
├── datasets/
│   ├── __init__.py          # exposes: aienergyscore (Path), load_prompts()
│   ├── loader.py            # load_prompts(config), _load_jsonl(), _load_synthetic()
│   └── builtin/
│       ├── __init__.py      # empty, marks as package for importlib.resources
│       └── aienergyscore.jsonl   # 1000 prompts, {"prompt": "..."} per line
```

This follows the `designs/dataset.md` spec exactly.

### Pattern 1: Package Data Access via importlib.resources

**What:** Access bundled data files without hardcoded paths, works in installed packages.
**When to use:** Any data file shipped inside a Python package.

```python
# src/llenergymeasure/datasets/loader.py
from importlib.resources import files
from pathlib import Path

# Access builtin data files — works in editable installs and installed packages
BUILTIN_DIR = files("llenergymeasure.datasets.builtin")

BUILTIN_DATASETS: dict[str, Path] = {
    "aienergyscore": BUILTIN_DIR / "aienergyscore.jsonl",
}
```

Note: `importlib.resources.files()` is the current (Python 3.9+) API. Do NOT use the older `importlib.resources.open_text()` (deprecated 3.11) or `pkgutil.get_data()` (old pattern).

### Pattern 2: datasets/__init__.py Public API

The success criterion `from llenergymeasure.datasets import aienergyscore` requires that `aienergyscore` is a name exported from `datasets/__init__.py`. The cleanest approach: export a `Path` constant.

```python
# src/llenergymeasure/datasets/__init__.py
from pathlib import Path
from llenergymeasure.datasets.loader import BUILTIN_DATASETS, load_prompts

# Expose individual dataset paths for direct import
aienergyscore: Path = BUILTIN_DATASETS["aienergyscore"]

__all__ = ["aienergyscore", "load_prompts", "BUILTIN_DATASETS"]
```

Usage after this:
```python
from llenergymeasure.datasets import aienergyscore  # → Path object
assert aienergyscore.exists()
```

### Pattern 3: JSONL Format

Each line is a JSON object with at least a `prompt` field:

```json
{"prompt": "Du Fu 's work is notable above all for its range ..."}
{"prompt": "Life-saving overdose prevention sites risk being shut down ..."}
```

The `max_new_tokens` field is optional per line (overrides experiment-level setting for that prompt). For the AIEnergyScore dataset, omit it — the dataset is purely prompt text.

### Pattern 4: Config BUILTIN_DATASETS Fix

`core/dataset_loader.py` currently imports `BUILTIN_DATASETS` and `AUTO_DETECT_COLUMNS` from `config.models` — this is wrong and broken. The fix: move these constants out of their phantom location into `datasets/loader.py` and update `dataset_loader.py` to import from there. But `dataset_loader.py` uses the HuggingFace `datasets` library for HF Hub loading — this is a separate concern from built-in file loading. The two modules should remain separate:

- `llenergymeasure.datasets.loader` — built-in JSONL loading (no HF dependency)
- `llenergymeasure.core.dataset_loader` — HuggingFace Hub loading (optional HF dependency)

Fix `core/dataset_loader.py` to define `BUILTIN_DATASETS` and `AUTO_DETECT_COLUMNS` locally (or import from the new `datasets.loader`), and update `_prepare_prompts()` in both backends.

### Pattern 5: peak_memory_mb — Correct Measurement Point

**Current (broken) state:**

PyTorchBackend lifecycle:
1. Model load (`_load_model`)
2. Warmup (`_run_warmup`) ← peak stats include warmup
3. Measurement loop (`_run_measurement`)
4. After loop: `data.peak_memory_mb = torch.cuda.max_memory_allocated()` ← captures everything since process start

`compute_metrics.get_memory_stats()`: calls `reset_peak_memory_stats()` then reads `max_memory_allocated()` — resets right before reading, so always returns near-zero. This is currently called in the old runner path (legacy, not main path).

**Correct (intended) state per `designs/result-schema.md`:**

```
inference window only — torch.cuda.reset_peak_memory_stats() called after
model load, before first inference. Captures KV cache + activations + batch buffers, not
model weights.
```

**Fix location in PyTorchBackend:**

```python
# In _run() or at the start of _run_measurement(), after model is loaded:
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ... then the existing measurement loop runs ...
# At end of _run_measurement():
if torch.cuda.is_available():
    data.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
```

The reset must happen after model load and after warmup (warmup also allocates KV cache buffers). The intent from the design doc is "inference window only" — meaning after warmup, before the measurement loop.

### Anti-Patterns to Avoid

- **Do not use `__file__`-relative paths for package data:** Works in editable installs but may fail in installed packages depending on build backend. Use `importlib.resources.files()` instead.
- **Do not reset peak stats before warmup:** Warmup allocates KV cache; resetting after warmup captures the steady-state inference peak including KV cache, which is meaningful.
- **Do not remove `core/dataset_loader.py`:** It handles HuggingFace Hub loading, which is a separate concern from built-in JSONL files. Only fix its broken import.
- **Do not add HF `datasets` library as a required (non-optional) dependency:** The built-in JSONL approach is precisely to avoid this runtime dependency.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bundled data file access | Hardcoded `Path(__file__).parent / "builtin"` | `importlib.resources.files()` | Survives installation into zip-safe packages, editable installs |
| JSONL parsing | Custom text splitter | `json.loads(line)` per line in stdlib | Standard pattern, handles escaping |
| Downloading dataset | Custom HTTP code | Use `datasets` lib in a one-off script | Already in .venv, correct caching |

**Key insight:** The dataset file ships with the package — no download logic needed at runtime. This is the same approach AIEnergyScore uses (`test_prompts.json` ships with their code).

---

## Common Pitfalls

### Pitfall 1: importlib.resources.files() returns Traversable, not Path

**What goes wrong:** `BUILTIN_DIR / "aienergyscore.jsonl"` returns a `Traversable`, not a `pathlib.Path`. Calling `open(path)` works, but `path.exists()` and `Path(path)` may not work depending on the install mechanism.

**Why it happens:** In installed (non-editable) packages, data files may be inside a zip archive. `Traversable` handles this; `Path` does not.

**How to avoid:** Use `as_file()` context manager when a real `Path` is needed:
```python
from importlib.resources import files, as_file
with as_file(files("llenergymeasure.datasets.builtin") / "aienergyscore.jsonl") as p:
    data = p.read_text()
```
Or read directly via `Traversable.read_text()` / `Traversable.open()`.

For the public API (`aienergyscore: Path`), in practice this is editable-install only during development and src-layout packaging with hatch makes the files real paths. The current `pyproject.toml` uses `hatch` with `packages = ["src/llenergymeasure"]`, so `Path(__file__).parent` works. Use `importlib.resources.files()` for correctness, but accept that the exported `aienergyscore` variable can be a `Path` for the editable-install case.

**Warning signs:** `FileNotFoundError` when running from an installed wheel (vs editable install).

### Pitfall 2: peak_memory_mb Includes Model Weights (Current Bug)

**What goes wrong:** `torch.cuda.max_memory_allocated()` is called at end-of-measurement without a prior reset. This returns the all-time maximum since CUDA was initialised, which includes model weights (~10-40 GB for large models). The result is meaningless for comparing inference behaviour.

**Why it happens:** No `reset_peak_memory_stats()` is called before the measurement loop. The function records the absolute peak since CUDA init.

**How to avoid:** Call `torch.cuda.reset_peak_memory_stats()` after warmup, immediately before the measurement loop. Confirmed by `designs/result-schema.md` intent: "inference window only ... Captures KV cache + activations + batch buffers, not model weights."

**Warning signs:** `peak_memory_mb` values matching total model weight sizes rather than activation sizes.

### Pitfall 3: JSONL file not included in wheel

**What goes wrong:** The `.jsonl` file is committed to `src/llenergymeasure/datasets/builtin/` but not included in the wheel because it's not a `.py` file.

**Why it happens:** Hatch's default wheel packaging includes Python files but not data files unless the package directory structure is marked to include them.

**How to avoid:** The current `pyproject.toml` uses `[tool.hatch.build.targets.wheel] packages = ["src/llenergymeasure"]`. Hatch includes all files under the package directory (not just `.py`), so the `.jsonl` file will be included automatically. Verify with `pip wheel . && zipinfo *.whl | grep jsonl`.

Also ensure `src/llenergymeasure/datasets/builtin/__init__.py` exists (empty file) so `importlib.resources.files()` can traverse the subpackage.

### Pitfall 4: AUTO_DETECT_COLUMNS phantom import

**What goes wrong:** `core/dataset_loader.py` imports `AUTO_DETECT_COLUMNS` from `config.models`, which doesn't exist. This breaks the entire `dataset_loader` module at import time — any import of this module raises `ImportError`.

**Why it happens:** Leftover from an earlier design iteration that was never implemented.

**How to avoid (Plan 21-01):** Define `AUTO_DETECT_COLUMNS` locally in `core/dataset_loader.py` (or remove the import if unused):
```python
# In core/dataset_loader.py, replace the broken import with:
AUTO_DETECT_COLUMNS = ["prompt", "text", "instruction", "input", "question"]

BUILTIN_DATASETS = {
    "aienergyscore": ...,  # import from datasets.loader
}
```

---

## Code Examples

### Generating aienergyscore.jsonl (one-time developer task)

```python
# scripts/generate_aienergyscore_dataset.py
# Run once to generate the bundled dataset file
from datasets import load_dataset
import json
from pathlib import Path

# Pin to specific commit for reproducibility
COMMIT = "a1b2c3d4..."  # TODO: identify correct commit after download

ds = load_dataset(
    "AIEnergyScore/text_generation",
    split="train",
    revision=COMMIT,
)

output_path = Path("src/llenergymeasure/datasets/builtin/aienergyscore.jsonl")
with output_path.open("w") as f:
    for row in ds:
        f.write(json.dumps({"prompt": row["text"]}) + "\n")

print(f"Written {len(ds)} prompts to {output_path}")
# Document the commit hash in the file header or in a provenance file
```

### datasets/loader.py

```python
# src/llenergymeasure/datasets/loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, SyntheticDatasetConfig

_BUILTIN_DIR = Path(__file__).parent / "builtin"

BUILTIN_DATASETS: dict[str, Path] = {
    "aienergyscore": _BUILTIN_DIR / "aienergyscore.jsonl",
}

AUTO_DETECT_COLUMNS = ["prompt", "text", "instruction", "input", "question"]


def load_prompts(config: "ExperimentConfig") -> list[str]:
    """Load prompts from configured dataset source. Returns list of prompt strings."""
    from llenergymeasure.config.models import SyntheticDatasetConfig

    dataset = config.dataset
    n = config.n

    if isinstance(dataset, SyntheticDatasetConfig):
        return _load_synthetic(dataset, n)

    if dataset in BUILTIN_DATASETS:
        return _load_jsonl(BUILTIN_DATASETS[dataset], n, name=dataset)

    path = Path(str(dataset))
    if path.exists() and path.suffix == ".jsonl":
        return _load_jsonl(path, n, name=str(path))

    raise ValueError(
        f"Unknown dataset: {dataset!r}. "
        f"Valid built-ins: {list(BUILTIN_DATASETS)}. "
        "For custom datasets, provide a path to a .jsonl file."
    )


def _load_jsonl(path: Path, n: int, name: str) -> list[str]:
    """Load first n prompts from JSONL file. Deterministic — same order every run."""
    prompts: list[str] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("text", "")
            if prompt:
                prompts.append(prompt.strip())

    if len(prompts) < n:
        raise ValueError(
            f"Dataset {name!r} has {len(prompts)} prompts but n={n} was requested. "
            f"Reduce n or use a larger dataset."
        )
    return prompts


def _load_synthetic(config: "SyntheticDatasetConfig", n: int) -> list[str]:
    """Generate deterministic synthetic prompts from seed."""
    import random
    rng = random.Random(config.seed)
    words_per_prompt = max(1, config.input_len // 4)
    return [("Hello, " * words_per_prompt).strip() for _ in range(n)]
```

### datasets/__init__.py

```python
# src/llenergymeasure/datasets/__init__.py
"""Built-in datasets for LLenergyMeasure.

Usage:
    from llenergymeasure.datasets import aienergyscore
    assert aienergyscore.exists()

    from llenergymeasure.datasets import load_prompts
"""
from pathlib import Path
from llenergymeasure.datasets.loader import BUILTIN_DATASETS, load_prompts

aienergyscore: Path = BUILTIN_DATASETS["aienergyscore"]

__all__ = ["aienergyscore", "load_prompts", "BUILTIN_DATASETS"]
```

### peak_memory_mb Fix in PyTorchBackend

```python
# In PyTorchBackend._run_measurement(), before the batch loop:
import torch

# Reset peak stats after warmup, before measurement window
# Captures inference-only peak (KV cache + activations + batch buffers),
# NOT model weights. Semantics: inference window only.
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

with PowerThermalSampler(device_index=0) as sampler:
    for batch_start in range(0, len(prompts), batch_size):
        # ... existing batch loop ...
        pass

# Read peak at end of measurement window
if torch.cuda.is_available():
    data.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
```

### Updated field docstrings

```python
# In domain/metrics.py ComputeMetrics:
peak_memory_mb: float = Field(
    0.0,
    description=(
        "Peak GPU memory allocated during the inference measurement window (MB). "
        "Captured via torch.cuda.max_memory_allocated() after resetting stats at "
        "measurement start (after model load and warmup). "
        "Reflects KV cache + activations + batch buffers — NOT model weights."
    )
)

# In domain/metrics.py MemoryEfficiencyMetrics:
peak_memory_mb: float = Field(
    default=0.0,
    description=(
        "Peak GPU memory allocated during inference measurement window (MB). "
        "See ComputeMetrics.peak_memory_mb for full semantics."
    )
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `pkgutil.get_data()` | `importlib.resources.files()` | Python 3.9 | Works in zip-safe installs |
| `open_text()` / `open_binary()` | `files().joinpath().open()` | Python 3.9 | More flexible, not deprecated |

**Deprecated/outdated:**
- `importlib.resources.open_text()`: Deprecated in Python 3.11, removed in 3.13. Do not use.
- `pkgutil.get_data()`: Old pattern, returns bytes. Use `importlib.resources.files()`.

---

## Open Questions

1. **Which commit hash should aienergyscore.jsonl pin to?**
   - What we know: The dataset is at `AIEnergyScore/text_generation`, has 1000 rows, single `text` column.
   - What's unclear: The specific commit hash to pin for reproducibility hasn't been identified yet.
   - Recommendation: During Plan 21-01, download with `load_dataset(..., revision="main")` and record the actual commit SHA shown in the dataset card/git log. Document in a provenance comment at top of the JSONL file or in a `builtin/PROVENANCE.txt` file.

2. **Should the datasets module expose a `load_prompts` function at the top level?**
   - What we know: Success criterion only requires `from llenergymeasure.datasets import aienergyscore` to load without error.
   - What's unclear: Whether backends should import from `llenergymeasure.datasets.loader` or via a top-level export.
   - Recommendation: Expose `load_prompts` in `datasets/__init__.py` for clean API. Have backends import from `llenergymeasure.datasets.loader` directly (avoids circular potential).

3. **Should warmup be excluded from peak_memory_mb window?**
   - What we know: `designs/result-schema.md` says "inference window only" and references reset being called "after model load".
   - What's unclear: Does "after model load" include after warmup, or just after `from_pretrained()`?
   - Recommendation: Reset after warmup (not just after model load). Warmup uses the same memory allocation patterns as inference — the first real batch after warmup represents steady-state. Resetting after warmup captures the stable inference peak. Document this decision in the field description.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/llenergymeasure/core/backends/pytorch.py:487` — current `max_memory_allocated()` call site
- Direct codebase inspection — `src/llenergymeasure/core/compute_metrics.py:166` — misplaced `reset_peak_memory_stats()` call
- `.product/designs/dataset.md` — canonical dataset module layout and JSONL format
- `.product/designs/result-schema.md:344-347` — `peak_memory_mb` semantics specification
- `.product/CODEBASE_PROPAGATION_AUDIT.md:40-41` — explicit action item for peak_memory_mb
- `src/llenergymeasure/core/dataset_loader.py:14-20` — confirmed broken import (ImportError verified)
- `src/llenergymeasure/config/models.py` — `ExperimentConfig.dataset` field defaults to `"aienergyscore"`

### Secondary (MEDIUM confidence)
- HuggingFace dataset viewer: `AIEnergyScore/text_generation` — 1000 rows, single `text` column, diverse text content
- Python 3.10 docs: `importlib.resources.files()` — current recommended API for package data

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies, stdlib only
- Architecture: HIGH — design docs specify exact layout; codebase confirms current broken state
- Pitfalls: HIGH — `ImportError` confirmed by running the import; PyTorch memory API is well-documented

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (stable domain — PyTorch CUDA API and Python packaging stable)
