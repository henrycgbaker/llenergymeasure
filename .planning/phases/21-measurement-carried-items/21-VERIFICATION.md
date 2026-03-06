---
phase: 21-measurement-carried-items
status: passed
verified: 2026-03-05T22:28:00Z
retroactive: true
requirements_verified:
  - MEAS-03
  - MEAS-04
evidence_source: "Retroactive verification against codebase on main (PRs #53, #54)"
---

# Phase 21 Verification: Measurement Carried Items

Retroactive verification of MEAS-03 and MEAS-04 against the codebase on `main`.
Evidence gathered by direct inspection of source files on 2026-03-05.

---

## MEAS-03 — aienergyscore.jsonl Built-in Dataset

**Requirement:** A built-in `aienergyscore.jsonl` dataset must be bundled with the
package and loadable via the datasets module.

**Status: PASS**

### Evidence

**Dataset file exists** — `src/llenergymeasure/datasets/builtin/aienergyscore.jsonl`

```
src/llenergymeasure/datasets/builtin/
  aienergyscore.jsonl
  __init__.py
```

**BUILTIN_DATASETS registry** — `src/llenergymeasure/datasets/loader.py`, lines 23-27:

```python
_BUILTIN_DIR: Path = Path(__file__).parent / "builtin"

BUILTIN_DATASETS: dict[str, Path] = {
    "aienergyscore": _BUILTIN_DIR / "aienergyscore.jsonl",
}
```

**`load_prompts()` function** — `src/llenergymeasure/datasets/loader.py`, lines 38-57:

```python
def load_prompts(config: ExperimentConfig) -> list[str]:
    """Load prompts according to the experiment configuration.

    Dispatches based on config.dataset type:
    - SyntheticDatasetConfig -> generate deterministic synthetic prompts
    - str matching a built-in alias -> load from bundled JSONL file
    - str path to an existing .jsonl file -> load from that file
    - anything else -> raise ValueError with valid options
    ...
    """
```

Built-in alias dispatch (lines 67-75):

```python
if config.dataset in BUILTIN_DATASETS:
    path = BUILTIN_DATASETS[config.dataset]
    return _load_jsonl(
        path,
        n=config.n,
        name=config.dataset,
        order=config.dataset_order,
        seed=config.random_seed,
    )
```

**Public API in `datasets/__init__.py`** — `src/llenergymeasure/datasets/__init__.py`:

```python
from llenergymeasure.datasets.loader import BUILTIN_DATASETS, load_prompts

aienergyscore: Path = BUILTIN_DATASETS["aienergyscore"]

__all__ = ["BUILTIN_DATASETS", "aienergyscore", "load_prompts"]
```

Import chain: `from llenergymeasure.datasets import aienergyscore` returns a `Path` to
the bundled JSONL file. `from llenergymeasure.datasets import load_prompts` provides the
loader function used by inference backends.

---

## MEAS-04 — inference_memory_mb Semantics

**Requirement:** The `inference_memory_mb` field must be computed with confirmed
semantics (peak inference window memory minus model baseline), documented, and present
in both PyTorch and vLLM backends.

**Status: PASS**

### Evidence

**PyTorch backend** — `src/llenergymeasure/core/backends/pytorch.py`, lines 777-791:

```python
# Memory metrics: inference-window-only peak (reset before loop) and derived delta.
# inference_memory_mb = peak (inference window) - model baseline (weights).
inference_memory_mb = max(0.0, data.peak_memory_mb - model_memory_mb)
logger.info(
    "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%.1fMB",
    model_memory_mb,
    data.peak_memory_mb,
    inference_memory_mb,
)
extended_metrics = ExtendedEfficiencyMetrics(
    memory=MemoryEfficiencyMetrics(
        model_memory_mb=model_memory_mb,
        peak_memory_mb=data.peak_memory_mb,
        inference_memory_mb=inference_memory_mb,
    )
)
```

**vLLM backend** — `src/llenergymeasure/core/backends/vllm.py`, lines 784-798:

```python
# Memory metrics: inference-window-only peak and derived delta.
# inference_memory_mb = peak (inference window) - model baseline.
inference_memory_mb = max(0.0, data.peak_memory_mb - model_memory_mb)
logger.info(
    "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%.1fMB",
    model_memory_mb,
    data.peak_memory_mb,
    inference_memory_mb,
)
extended_metrics = ExtendedEfficiencyMetrics(
    memory=MemoryEfficiencyMetrics(
        model_memory_mb=model_memory_mb,
        peak_memory_mb=data.peak_memory_mb,
        inference_memory_mb=inference_memory_mb,
    )
)
```

**Semantics:** `inference_memory_mb = max(0.0, peak_memory_mb - model_memory_mb)`.
The peak is taken from the inference window only (NVML peak reset before the inference
loop starts), so the delta isolates KV cache and activation memory from model weights.

**Documentation** — `docs/energy-measurement.md`, line 49:

```markdown
- `inference_memory_mb` — peak GPU memory used during inference only (not model loading)
- Measured as the delta between pre-inference and peak NVML memory readings
- Reflects the KV cache and activation memory cost of the specific inference configuration
```

---

## Summary

| Requirement | Description                                    | Status |
| ----------- | ---------------------------------------------- | ------ |
| MEAS-03     | aienergyscore.jsonl built-in dataset           | PASS   |
| MEAS-04     | inference_memory_mb semantics confirmed + docs | PASS   |

Both requirements are fully implemented and documented. Phase 21 code was merged to
main in PRs #53 (MEAS-04, pytorch backend) and #54 (MEAS-03, datasets module).
