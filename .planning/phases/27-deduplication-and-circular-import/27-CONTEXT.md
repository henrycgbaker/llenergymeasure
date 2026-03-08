# Phase 27: Deduplication and Circular Import - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Consolidate duplicated code across backends, config, and infrastructure. Extract shared utilities, break the config-study circular import, and reshape the backend architecture toward a harness/plugin model where backends are thin inference plugins and the harness owns measurement infrastructure.

This phase covers all P2.x and P3.4 items from the simplify audit report. It does NOT include logging migration (Phase 28) or performance fixes (Phase 28).

</domain>

<decisions>
## Implementation Decisions

### Backend Protocol Reshape

- **Start the harness/plugin reshape in Phase 27** (not deferred to M4). Peer research confirmed every tool (lm-eval, MLPerf, Zeus, CodeCarbon, vLLM benchmarks) separates measurement from inference. The architecture is well-evidenced; 3-4 planned backends justify the abstraction now.
- **Lifecycle methods protocol:** Backends implement 4 methods:
  - `load_model(config) -> Any` - backend-specific model loading
  - `warmup(config, model) -> WarmupResult` - backend-specific warmup (differs per framework)
  - `run_inference(config, model) -> InferenceOutput` - the actual inference call
  - `cleanup(model) -> None` - backend-specific cleanup (vLLM needs destroy_model_parallel, etc.)
- **`MeasurementHarness` class in `core/harness.py`** owns the measurement lifecycle: environment capture, baseline power, thermal floor wait, energy tracking start/stop, CUDA sync, FLOPs estimation, timeseries write, warning collection, result assembly. ~200 lines, written once.
- **`InferenceOutput` dataclass** returned by backends: minimal core fields (`elapsed_time_sec`, `input_tokens`, `output_tokens`, `peak_memory_mb`, `model_memory_mb`, `batch_times`) plus `extras: dict[str, Any]` for backend-specific data. Clean interface, extensible.
- **Wire end-to-end:** `_api.py` updated to call `harness.run(backend, config)` instead of `backend.run(config)`. Ship working code, not a dead abstraction.
- **Backends shrink from ~860 lines each to ~200-300 lines.** The harness absorbs all duplicated measurement infrastructure.

### Circular Import Resolution

- **Move `study/grid.py` to `config/grid.py`** - eliminates the cycle entirely. `expand_grid()` is a config transformation (runs at YAML parse time, produces `list[ExperimentConfig]`). This is the vLLM/HF Transformers pattern: config transformation logic lives with config types.
- **Direct imports from `config.grid`** - no re-exports from study/. study/ consumers (runner.py, format_preflight_summary) import from `llenergymeasure.config.grid` directly. Clean, honest dependency direction (study depends on config).
- **`CycleOrder` and `apply_cycles()` stay in `config/grid.py`** - CycleOrder is a user-facing YAML enum; apply_cycles is a config transformation. Both are config semantics, even though called from study/runner.py.
- **Bonus: eliminates duplicated `_unflatten`/`_deep_merge`** - grid.py currently duplicates these with comments saying "defined locally to avoid circular import with config.loader." Once in config/, consolidate into `_dict_utils.py`.

### Shared Module Organisation

- **Harness absorbs result building, warnings, measurement state** - `_build_result()`, `_collect_warnings()`, `_MeasurementData` become harness methods/internals, not standalone modules.
- **2 new files** (down from 5 originally planned):
  - `core/harness.py` (~200 lines) - the measurement orchestrator
  - `config/_dict_utils.py` (~80 lines) - `_unflatten()`, `_deep_merge()` consolidated from 3 copies
- **GPU utilities added to existing `core/gpu_info.py`** (currently 482 lines, will be ~550-600):
  - `_cuda_sync()` - CUDA synchronisation
  - `_check_persistence_mode()` - NVIDIA persistence mode check
  - `nvml_context()` - NVML init/shutdown context manager
  - Research: Zeus's `device/gpu/common.py` (617 lines) and vLLM's `torch_utils.py` (830 lines) both group by domain, not temporal concern.
- **`compute_config_hash()` stays in `domain/experiment.py`** - consolidate the 5 independent SHA-256[:16] implementations to import from this canonical version. One function doesn't need its own module (vLLM created `hashing.py` because they have 7 distinct hash functions).
- **Existing homes for remaining items** (thermal_floor_wait in core/warmup.py, Docker items in existing infra files).

### Testing Strategy

- **New `test_harness.py`** for MeasurementHarness tests. Tests the orchestration lifecycle with mock backends.
- **Existing backend tests rewritten as unit tests** for the thin plugin interface (test `load_model`, `warmup`, `run_inference`, `cleanup` independently).

### Extraction Scope

- **Full reshape + all P2.x items in one phase** - the harness reshape and dedup items are one coherent piece of work.
- **Opportunistic scope with "same extraction, same scope" boundary:**
  - Always include: renaming/type hints in moved code, dead imports, docstring updates
  - Include if same file: discovered 6th copy of a pattern being consolidated
  - Defer: issues in files not being modified, new abstractions not in the audit
- **Logging inconsistencies deferred to Phase 28** - note but don't fix stdlib logging encountered during extraction. Phase 28 explicitly handles loguru migration.
- **Deferred items logged in this CONTEXT.md** (see section below) for Phase 28 planner visibility.

### Claude's Discretion

- Infrastructure dedup placement: Docker error base class `__init__`, NVIDIA toolkit binary list, Docker dispatch consolidation, `_save_and_record()` helper - Claude picks the right existing file for each
- Error handling in the harness (how BackendError propagates through the harness)
- Harness internal structure (private methods, state management)
- Test migration details for existing backend test files
- `is_backend_available` consolidation (P2.4) - if both implementations survived Phase 25

</decisions>

<specifics>
## Specific Ideas

- The harness pattern is modelled after Zeus's `ZeusMonitor` and lm-eval's evaluator - centralised measurement with thin backend plugins
- InferenceOutput's `extras: dict[str, Any]` field allows vLLM-specific data (RequestOutput objects) to pass through without coupling the interface
- The `config/grid.py` move eliminates both the circular import AND the duplicated dict utilities in one step
- Backend protocol is deliberately coarse-grained (4 methods) to leave room for refinement when TensorRT reveals its quirks in M4

## Peer Research Summary

Research covered: lm-eval-harness, MLPerf LoadGen, Zeus, CodeCarbon, vLLM benchmarks (architecture); vLLM, HF Transformers, PyTorch Lightning, Zeus, MLflow (circular imports + module organisation); HF Transformers, PyTorch Lightning, vLLM, FastAPI/Pydantic, Google Engineering Practices, Martin Fowler (extraction scope).

Key findings:
- Every measurement tool separates measurement from inference - no exceptions
- No peer project uses `_shared.py` - they split by concern with descriptive names
- Moving config transformations to config/ is the standard approach (vLLM, HF)
- "Same extraction, same scope" boundary is the sweet spot for audit-driven refactoring

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

## Discovered During Extraction (For Phase 28)

*This section will be populated during Phase 27 execution with logging inconsistencies, performance issues, or other items discovered but deferred.*

</deferred>

---

*Phase: 27-deduplication-and-circular-import*
*Context gathered: 2026-03-07*
