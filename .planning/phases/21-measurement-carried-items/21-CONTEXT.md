# Phase 21: Measurement Carried Items - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Close two long-standing M1 carry-forwards: create the `aienergyscore.jsonl` built-in dataset file and confirm `peak_memory_mb` measurement semantics across all current backends (PyTorch + vLLM). Implements MEAS-03 and MEAS-04.

</domain>

<decisions>
## Implementation Decisions

### Peak Memory Semantics
- Report **both** total peak and inference-only memory for all current backends (PyTorch and vLLM)
- Best-per-backend memory source: PyTorch allocator for PyTorch, torch-first with NVML fallback for vLLM (vLLM pre-allocates memory, so torch stats may reflect the full pre-allocation rather than actual usage)
- Inference baseline captured **after model load, before warmup** — inference memory = KV cache + activations + batch buffers from the full inference window including warmup
- Memory is **context for energy interpretation**, not a primary metric — researchers need it to interpret energy results, but joules remain the primary output

### Memory Field Schema
- `peak_memory_mb: float = 0.0` — Total peak GPU memory (weights + inference), core metric
- `model_memory_mb: float = 0.0` — Memory after model load (already exists)
- `inference_memory_mb: float = 0.0` — Derived: peak minus model baseline
- `total_vram_mb: float = 0.0` — GPU total capacity (already exists, hardware metadata)
- `kv_cache_mb: float | None = None` — KV cache allocation, vLLM only (already exists)
- All core fields default to 0.0 (matching peers: optimum-benchmark, Zeus). 0.0 = not measured.
- When memory capture fails: log a warning AND record it in the result warnings/notes (propagate through results)

### vLLM Memory Capture
- Memory captured **inside the container** by the vLLM backend's `run()` method — the container entrypoint already writes the full ExperimentResult to the shared volume, so no entrypoint changes needed
- Try `torch.cuda.max_memory_allocated()` first (vLLM uses PyTorch internally). If the value looks like a full pre-allocation (matches `gpu_memory_utilization * total_vram`), fall back to NVML queries inside the container
- pynvml is available in the vLLM Docker image

### Dataset File
- Ship all 1,000 prompts from `AIEnergyScore/text_generation` (HuggingFace Hub, pinned commit)
- JSONL format with `source` metadata per prompt: `{"prompt": "...", "max_new_tokens": N, "source": "wikitext"}`
- Provenance header line as first JSONL line: `{"_provenance": "AIEnergyScore/text_generation", "_commit": "<hash>", "_license": "apache-2.0", "_description": "..."}`
- Interleaved ordering by default (round-robin WikiText, OSCAR, UltraChat) — any n gets equal representation from all 3 sources

### Dataset Ordering Config
- New config field: `dataset_order: interleaved | grouped | shuffled` (default: `interleaved`)
- `interleaved` = file order (round-robin by source)
- `grouped` = sorted by source field
- `shuffled` = seed-based random permutation (uses experiment `random_seed`)
- Implementation is ~20 lines in the loader — quick add, update design docs

### Dataset Module API
- Config-only access: `dataset: aienergyscore` in YAML resolves to the built-in file
- Custom JSONL paths supported: `dataset: path/to/my_prompts.jsonl` (already designed)
- Public Path export: `from llenergymeasure.datasets import aienergyscore` returns a `Path` to the JSONL file
- Exported from `__init__.py` as stable public API
- No `load_dataset()` helper — aligns with peers (AIEnergyScore, MLPerf) who use config-only access

### Claude's Discretion
- Exact NVML query implementation for vLLM fallback
- How to detect "full pre-allocation" vs actual usage in torch stats
- Synthetic dataset prompt generation details (tokeniser-aware implementation)
- Loader skip logic for provenance header line
- Exact warning message text for memory capture failures

</decisions>

<specifics>
## Specific Ideas

- optimum-benchmark is the key peer reference for memory fields — they report both `max_allocated` (torch) and `max_global_vram` (NVML) per lifecycle phase. We simplified to a single best-source-per-backend approach.
- "I want to credit AIEnergyScore somehow as we are using their work" — provenance header line in the JSONL file, travelling with the data
- Dataset ordering should be configurable but have a sensible default so users don't need to select

</specifics>

<deferred>
## Deferred Ideas

- `sharegpt` built-in dataset — deferred until user demand (existing decision)
- HuggingFace Datasets integration — deferred to later v2.x (existing decision)
- Dataset shuffling with per-epoch seeds — beyond simple shuffled mode, useful for statistical robustness checks
- Dual torch+NVML reporting (like optimum-benchmark) — if researchers need both numbers, add in a future phase

</deferred>

---

*Phase: 21-measurement-carried-items*
*Context gathered: 2026-03-03*
