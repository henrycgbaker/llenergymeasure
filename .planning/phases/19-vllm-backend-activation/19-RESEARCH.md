# Phase 19: vLLM Backend Activation - Research

**Researched:** 2026-02-28
**Domain:** vLLM offline batch inference, in-container NVML energy measurement, multi-GPU energy, Docker flags
**Confidence:** HIGH (codebase verified, official docs consulted, peer patterns confirmed)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Inference Mode**
- Offline batch only — `vllm.LLM()` with `llm.generate(prompts)` (single call, all prompts at once)
- Strip existing streaming/traffic-simulation code entirely — rewrite from scratch in the server phase
- Per-token latency metrics (TTFT, ITL) skipped for offline mode

**Output Collection**
- Token counts by default; full output text only when `save_outputs=True`
- Consistent with PyTorch backend pattern

**Container Lifecycle**
- Self-contained container: receives config via volume mount, runs everything, writes result JSON
- Energy measurement happens **inside the container** (in-process NVML)
- `torch.cuda.synchronize()` before energy reads is mandatory
- Host just does `docker run` and collects result files

**Error Handling**
- Generic error passthrough — no special vLLM-specific error translation
- Container writes error JSON on failure (existing entrypoint pattern, keep this)

**Ground-Up Rewrite**
- Do NOT patch the existing 1007-line `core/inference_backends/vllm.py` — that code is pre-M1, unvalidated
- Design from scratch based on peer patterns

### Claude's Discretion
- Whether to tighten the InferenceBackend protocol in Phase 19 or defer to a separate refactor phase
- Container runtime flags beyond `--shm-size 8g` and `--gpus all`
- vLLM model loading strategy (lazy vs eager, download cache location)

### Deferred Ideas (OUT OF SCOPE)
- vLLM online/streaming server mode
- Backend protocol refactor (tighten plugin architecture) — may be deferred
- Sampling param translation layer
- Complete vLLM param exposure (beyond energy-relevant tier)
- ADRs for cross-cutting decisions
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VLLM-01 | vLLM inference backend activated and producing valid ExperimentResult via Docker | `get_backend("vllm")` must return a `VLLMBackend` with `run()` method; `container_entrypoint.py` already wires `get_backend(config.backend).run(config)` — just register vllm backend |
| VLLM-02 | P0 fix: vLLM streaming broken (CM-07) | Offline batch mode: strip streaming entirely from the new backend — `_run_inference_batch()` is the only path |
| VLLM-03 | P0 fix: vLLM `--shm-size 8g` passed to container (CM-09) | Already implemented in `DockerRunner._build_docker_cmd()` — verify not accidentally removed; no code change needed |
</phase_requirements>

---

## Summary

Phase 19 has two main plans: a P0 fix plan (which turns out to be partially done already) and an end-to-end activation plan (the real work). The key insight from codebase investigation is that **the infrastructure is largely complete** — `DockerRunner`, `container_entrypoint`, and the dispatch machinery all exist. The missing piece is:

1. `get_backend("vllm")` returns `BackendError: Unknown backend: 'vllm'` — it only knows `pytorch`
2. There is no `VLLMBackend` in `core/backends/` (only in the legacy `core/inference_backends/vllm.py` which is 1007 lines of pre-M1 unvalidated code)
3. `VLLMConfig` in `backend_configs.py` has only 5 fields (M1 minimal) — needs expansion for Phase 19.1

The plan is to write a new `VLLMBackend` in `core/backends/vllm.py` (matching the PyTorch backend's structure) that owns the full measurement lifecycle using offline batch inference, then register it in `get_backend()`. The new backend:
- Uses `vllm.LLM()` with offline `generate()` — one call with all prompts
- Wraps inference with in-process NVML energy measurement
- Follows the PyTorch backend's lifecycle pattern: snapshot → baseline → load → warmup → sync → energy start → infer → sync → energy stop → result

The streaming P0 bug (CM-07) is resolved by simply not implementing streaming — offline batch only. The `--shm-size 8g` bug (CM-09) is already fixed in `DockerRunner`.

**Primary recommendation:** Write `core/backends/vllm.py` as a focused offline-batch backend, register it in `get_backend()`, expand `VLLMConfig` with energy-relevant params, and verify the Docker wire-up end-to-end.

---

## Codebase State (Critical Context)

### What Already Exists

**`src/llenergymeasure/infra/docker_runner.py` — `DockerRunner._build_docker_cmd()`**
```python
cmd = ["docker", "run", "--rm", "--gpus", "all",
       "-v", f"{exchange_dir}:/run/llem",
       "-e", f"LLEM_CONFIG_PATH=/run/llem/{config_hash}_config.json",
       "--shm-size", "8g"]  # CM-09 already fixed
```
The `--shm-size 8g` flag is already present. VLLM-03 is already satisfied in the host path.

**`src/llenergymeasure/infra/container_entrypoint.py` — `run_container_experiment()`**
```python
run_preflight(config)
backend = get_backend(config.backend)  # <- will call get_backend("vllm")
result = backend.run(config)           # <- InferenceBackend.run() protocol
```
The wire-up is complete. Just need `get_backend("vllm")` to return a valid backend.

**`src/llenergymeasure/core/backends/__init__.py` — `get_backend()`**
```python
def get_backend(name: str) -> InferenceBackend:
    if name == "pytorch":
        from llenergymeasure.core.backends.pytorch import PyTorchBackend
        return PyTorchBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch")
```
Only `pytorch` is registered. Adding `vllm` here is the activation step.

**`src/llenergymeasure/core/backends/protocol.py` — `InferenceBackend`**
```python
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...
    def run(self, config: ExperimentConfig) -> ExperimentResult: ...
```
One method, one property. The contract is minimal and clear.

**`src/llenergymeasure/core/backends/pytorch.py` — PyTorchBackend lifecycle**
The reference implementation. Lifecycle order:
1. `collect_environment_snapshot()` — before model load
2. `measure_baseline_power()` — before model load
3. `_load_model()` — model + tokenizer
4. `_prepare_prompts()` — dataset loading
5. `_run_warmup()` — JIT warmup
6. Thermal floor wait
7. `select_energy_backend()` + `backend.start_tracking()`
8. `_run_measurement()` — inference loop
9. `_cuda_sync()` — sync before stopping energy
10. `backend.stop_tracking()`
11. FLOPs estimation
12. `_build_result()` — assemble ExperimentResult

vLLM backend must follow the same lifecycle. Differences: no tokenizer separate load, vLLM's `LLM()` handles model loading internally, no explicit batch loop (vLLM batches internally).

### What Does NOT Exist (Must Write)

- `src/llenergymeasure/core/backends/vllm.py` — the new backend
- Registration of `"vllm"` in `get_backend()`
- Expanded `VLLMConfig` (Phase 19.1)

### What Exists But Must NOT Be Used

`src/llenergymeasure/core/inference_backends/vllm.py` — 1007-line pre-M1 code. Contains streaming, traffic simulation, TTFT measurement, all of which are out of scope. Ignore entirely. The new backend goes in `core/backends/vllm.py` (matching `pytorch.py` location).

---

## Standard Stack

### Core Libraries (inside container)
| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| `vllm` | ≥0.6.x | Offline batch inference engine | vLLM Docker image |
| `pynvml` | any | NVML energy measurement (`nvmlDeviceGetTotalEnergyConsumption`) | base dep |
| `torch` | bundled with vLLM | `torch.cuda.synchronize()` call | vLLM image |

### Energy Measurement API
The project already has `NVMLBackend` in `core/energy_backends/nvml.py`. It uses `PowerThermalSampler` polling (100ms interval, trapezoidal integration). This works but has limitations for vLLM:

- **Multi-GPU**: `NVMLBackend(device_index=0)` only measures GPU 0. For multi-GPU runs, need to sum across all devices.
- **API choice**: For Volta+ GPUs, `nvmlDeviceGetTotalEnergyConsumption()` (two-point measurement) is lower overhead than polling. Current NVML backend uses polling. Either approach is valid for Phase 19 — polling is already implemented and tested.

**Decision for Phase 19**: Use the existing `NVMLBackend` with `select_energy_backend()` for single-GPU runs. Multi-GPU energy summing is noted as a research question but deferred to when multi-GPU is actually tested (the A100 in the dev environment can be used as single GPU for now).

---

## Architecture Patterns

### Pattern 1: vLLM Offline Batch Inference
**Source:** [vLLM docs](https://docs.vllm.ai/en/stable/serving/offline_inference/) + lm-eval `vllm_causallms.py`

The correct pattern for offline batch inference:
```python
from vllm import LLM, SamplingParams

# Load model once
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    enforce_eager=False,  # CUDA graphs for performance
)

# Create sampling params
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    max_tokens=128,
)

# Single generate() call with ALL prompts
outputs = llm.generate(prompts, sampling_params)

# Process outputs
for output in outputs:
    input_tokens = len(output.prompt_token_ids)
    output_tokens = len(output.outputs[0].token_ids)
    text = output.outputs[0].text  # only if save_outputs=True
```

**Key behaviours:**
- vLLM automatically batches all prompts internally using continuous batching
- Single `generate()` call is correct — do NOT call one-at-a-time
- `output.outputs[0]` is the primary completion (index 0 = best/first beam/sample)
- Token counts come from `len(output.prompt_token_ids)` and `len(output.outputs[0].token_ids)` — correct approach
- Order of outputs matches order of prompts (stable ordering)

### Pattern 2: Energy Measurement Wrapping (Zeus best practice)
**Source:** [ML.Energy best practices blog](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)

```python
# Synchronize before starting measurement
torch.cuda.synchronize()
start_energy_j = pynvml_handle.getEnergyConsumption() / 1000  # mJ -> J

# Run inference
outputs = llm.generate(prompts, sampling_params)

# Synchronize before stopping measurement (mandatory)
torch.cuda.synchronize()
end_energy_j = pynvml_handle.getEnergyConsumption() / 1000

consumed_j = end_energy_j - start_energy_j
```

For Phase 19, we use the existing `select_energy_backend()` + `NVMLBackend` pipeline which handles this pattern via `PowerThermalSampler` (polling). The vLLM backend must call `torch.cuda.synchronize()` before `stop_tracking()` — identical to PyTorchBackend.

### Pattern 3: VLLMConfig Expansion (Phase 19.1)
**Source:** [vLLM Engine Args v0.7.1](https://docs.vllm.ai/en/v0.7.1/serving/engine_args.html)

Energy-relevant parameters (biggest impact on energy consumption):

| Parameter | Default | Energy Impact |
|-----------|---------|---------------|
| `enforce_eager` | False | HIGH — CUDA graphs off = lower throughput = more energy per token |
| `gpu_memory_utilization` | 0.9 | HIGH — controls KV cache size → batch size capacity |
| `quantization` | None | HIGH — int4/fp8 reduces compute significantly |
| `kv_cache_dtype` | auto | MEDIUM — fp8 KV cache halves KV cache memory |
| `dtype` | auto | MEDIUM — bf16 vs fp16 vs fp8 |
| `tensor_parallel_size` | 1 | HIGH — multi-GPU topology |
| `block_size` | 16 | LOW-MEDIUM — affects KV fragmentation |
| `max_num_seqs` | 256 | MEDIUM — bounds continuous batching width |
| `enable_prefix_caching` | False | MEDIUM — reduces repeated prefill compute |
| `swap_space` | 4 (GiB) | LOW — CPU offload when KV full |

Current `VLLMConfig` only has: `max_num_seqs`, `tensor_parallel_size`, `gpu_memory_utilization`, `enable_prefix_caching`, `quantization` — missing `enforce_eager`, `block_size`, `kv_cache_dtype`, `dtype` override, `swap_space`.

### Pattern 4: lm-eval Escape Hatch
**Source:** Verified from lm-eval `vllm_causallms.py`

```python
self.model_args = {
    "model": pretrained,
    "gpu_memory_utilization": ...,
    # ... other explicit params
}
self.model_args.update(kwargs)  # escape hatch: forward unknown kwargs to LLM()
```

We already have `VLLMConfig.extra: dict | None` as escape hatch in the existing (legacy) `core/inference_backends/vllm.py`. The new VLLMConfig should include `extra_kwargs: dict | None` for the same purpose (Phase 19.1).

### Pattern 5: New VLLMBackend Structure
Mirror `core/backends/pytorch.py`:
```
class VLLMBackend:
    @property
    def name(self) -> str: return "vllm"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        # 1. Environment snapshot
        # 2. Baseline power (optional)
        # 3. Load model: LLM(**kwargs) — blocks until ready
        # 4. Prepare prompts (use existing dataset loading)
        # 5. Warmup: llm.generate([warmup_prompt], warmup_params)
        # 6. Thermal floor wait
        # 7. select_energy_backend() + start_tracking()
        # 8. torch.cuda.synchronize() [before measurement]
        # 9. llm.generate(prompts, sampling_params) [single call]
        # 10. torch.cuda.synchronize() [after inference, before stop]
        # 11. stop_tracking()
        # 12. Compute FLOPs (estimate_flops_palm)
        # 13. _build_result()
        # [cleanup via del llm + gc]
```

---

## CM-07 and CM-09 Analysis (P0 Bugs)

### CM-07: Streaming broken
**Root cause**: The existing `core/inference_backends/vllm.py` attempts streaming via `_run_streaming_inference()` which processes prompts one at a time and tries to extract TTFT from vLLM metrics. This is architecturally broken for offline batch mode.

**Fix**: The new `VLLMBackend` simply does not implement streaming. It calls `llm.generate(prompts, sampling_params)` once. `config.streaming` is an old v1.x field that no longer exists on `ExperimentConfig` (confirmed — removed in v2.0 schema). The bug is resolved by the ground-up rewrite.

### CM-09: --shm-size missing
**Root cause**: `DockerRunner._build_docker_cmd()` was missing `--shm-size 8g`.

**Current state**: ALREADY FIXED. Looking at the current `docker_runner.py` (verified above):
```python
cmd = [..., "--shm-size", "8g"]
```
`--shm-size 8g` is already present. VLLM-03 is satisfied. No code change needed for this fix — just verify in testing.

**Why shm-size matters for vLLM**: vLLM uses shared memory for inter-process communication (tensor parallelism worker processes communicate via `/dev/shm`). Without `--shm-size 8g`, the default 64MB is insufficient for large model weights, causing `OSError: [Errno 28] No space left on device`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model loading | Custom HuggingFace loader | `vllm.LLM(model=..., ...)` | vLLM manages its own CUDA context, weights loading, tensor parallelism — interfering breaks things |
| Batching | Manual batch loops | Single `llm.generate(all_prompts)` | vLLM's continuous batching is optimal — splitting into sub-batches degrades throughput |
| Token counting | Re-tokenizing outputs | `len(output.prompt_token_ids)` + `len(output.outputs[0].token_ids)` | vLLM outputs already include exact token counts |
| Energy measurement | Custom NVML poller | `select_energy_backend()` from existing `core/energy_backends/` | Already implemented, tested, covers Zeus/NVML/CodeCarbon |
| Warmup | Custom warmup logic | `llm.generate([warmup_prompt], SamplingParams(max_tokens=1))` | Simple, matches `_perform_warmup()` pattern from old code |
| Cleanup | GPU cache clearing | `del llm` + `gc.collect()` | vLLM spawns sub-processes — `del` signals them to terminate. `torch.cuda.empty_cache()` after is fine. |

---

## Common Pitfalls

### Pitfall 1: CUDA Context Conflict with vLLM
**What goes wrong**: Calling `torch.cuda.is_available()`, `torch.cuda.device_count()`, or any torch.cuda.* before `LLM()` initialises can pre-initialise CUDA and conflict with vLLM's own CUDA context management.
**Why it happens**: vLLM uses `multiprocessing.spawn` for tensor parallelism workers. CUDA must not be initialised in the parent process before forking.
**How to avoid**: All CUDA-touching code (even `is_available()`) must happen AFTER `LLM()` is constructed, or use `os.environ` checks instead. The container runs single-process for single GPU — less risk — but still be careful with pynvml calls.
**Warning signs**: `RuntimeError: Cannot re-initialize CUDA in forked subprocess` or silent hangs.

### Pitfall 2: Forgetting torch.cuda.synchronize() Before Energy Stop
**What goes wrong**: Energy measurement stops while GPU is still computing, under-measuring actual energy.
**Why it happens**: vLLM returns from `generate()` after the last token is scheduled, but CUDA kernels may still be in flight asynchronously.
**How to avoid**: Always call `torch.cuda.synchronize()` between `generate()` return and `stop_tracking()`. See PyTorchBackend for reference — it does this at `_cuda_sync()`.
**Warning signs**: Energy readings significantly lower than expected; varies run-to-run on same config.

### Pitfall 3: One-at-a-Time Prompts
**What goes wrong**: Calling `llm.generate([prompt])` in a loop per prompt defeats continuous batching — vLLM initialises a new scheduler per call.
**Why it happens**: Confusion with PyTorch backend's batch loop pattern.
**How to avoid**: Always pass the FULL prompts list to a single `llm.generate(prompts, sampling_params)` call.
**Warning signs**: Throughput dramatically lower than vLLM benchmarks; linear scaling with prompt count.

### Pitfall 4: VLLMConfig vs. Legacy Config Mismatch
**What goes wrong**: The legacy `core/inference_backends/vllm.py` reads from `config.vllm` fields that don't exist in the current minimal `VLLMConfig` (5 fields). The new backend must only read fields that exist.
**Current VLLMConfig fields**: `max_num_seqs`, `tensor_parallel_size`, `gpu_memory_utilization`, `enable_prefix_caching`, `quantization` — all with `None` defaults.
**How to avoid**: Build kwargs only from fields present in `VLLMConfig`. For Phase 19, add `enforce_eager`, `swap_space`, `kv_cache_dtype` at minimum.

### Pitfall 5: Backend Field Name `model` vs `model_name`
**What goes wrong**: `ExperimentConfig.model` (v2.0 name) — the legacy `vllm.py` uses `config.model_name` which no longer exists.
**How to avoid**: Use `config.model` throughout the new backend. Check all attribute accesses against current `ExperimentConfig` schema.

### Pitfall 6: Energy Measurement Inside Container (NVML Access)
**What goes wrong**: NVML may require specific Docker flags to access GPU energy counters.
**Current Docker flags**: `--gpus all` (already in DockerRunner) — this is sufficient for NVML access on most systems. The `--gpus all` flag enables the NVIDIA Container Runtime which provides full GPU access including NVML.
**No additional `--cap-add` needed** for standard GPU containers. Verified: `--gpus all` exposes `/dev/nvidia*` devices and NVML library.
**Warning signs**: `pynvml.NVMLError_NoPermission` at runtime — would require `--privileged` as fallback (not recommended).

### Pitfall 7: The `config.model_dump_json()` Field Names
**What goes wrong**: `ExperimentConfig.model_dump_json()` uses the v2.0 field names. The container entrypoint reads this JSON and reconstructs via `ExperimentConfig.model_validate(raw)` — this round-trip is already tested and works.
**No issue here** — just be aware the JSON uses `model` not `model_name`.

---

## Code Examples

### Minimal VLLMBackend.run() Skeleton
```python
# core/backends/vllm.py
from __future__ import annotations
import gc
import time
import logging
from datetime import datetime
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.exceptions import BackendError

logger = logging.getLogger(__name__)


class VLLMBackend:
    @property
    def name(self) -> str:
        return "vllm"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        # 1. Environment snapshot
        from llenergymeasure.domain.environment import collect_environment_snapshot
        snapshot = collect_environment_snapshot()

        # 2. Baseline power
        baseline = None
        if config.baseline.enabled:
            from llenergymeasure.core.baseline import measure_baseline_power
            baseline = measure_baseline_power(duration_sec=config.baseline.duration_seconds)

        # 3. Load model (vLLM manages its own CUDA context)
        llm, sampling_params = self._load_model(config)

        try:
            # 4. Prepare prompts
            prompts = self._prepare_prompts(config)

            # 5. Warmup
            from vllm import SamplingParams as SP
            warmup_params = SP(max_tokens=1, temperature=0.0)
            llm.generate(prompts[:1], warmup_params)

            # 6. Thermal floor
            if config.warmup.enabled and config.warmup.thermal_floor_seconds > 0:
                time.sleep(config.warmup.thermal_floor_seconds)

            # 7. Energy backend
            from llenergymeasure.core.energy_backends import select_energy_backend
            energy_backend = select_energy_backend(config.energy.backend)

            # 8. Start energy measurement
            import torch
            torch.cuda.synchronize()
            tracker = energy_backend.start_tracking() if energy_backend else None

            # 9. Inference (single call, all prompts)
            start_time = datetime.now()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            inference_time = time.perf_counter() - t0

            # 10. Sync before stopping energy
            torch.cuda.synchronize()
            end_time = datetime.now()

            # 11. Stop energy measurement
            energy_measurement = None
            if energy_backend and tracker:
                energy_measurement = energy_backend.stop_tracking(tracker)

            # 12. Count tokens
            total_input = sum(len(o.prompt_token_ids) for o in outputs)
            total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
            output_texts = [o.outputs[0].text for o in outputs] if config.save_outputs else None

        finally:
            del llm
            gc.collect()

        # 13. Build result
        return self._build_result(
            config, total_input, total_output, inference_time,
            start_time, end_time, energy_measurement, baseline, snapshot
        )
```

### Building LLM kwargs from VLLMConfig
```python
def _build_llm_kwargs(self, config: ExperimentConfig) -> dict:
    """Build LLM() constructor kwargs from ExperimentConfig."""
    kwargs = {
        "model": config.model,
        "dtype": self._map_precision(config.precision),  # fp16/bf16/fp32 -> vLLM dtype
        "trust_remote_code": True,
        "seed": config.random_seed,
    }
    vllm_cfg = config.vllm
    if vllm_cfg is None:
        return kwargs

    # Apply VLLMConfig fields (all default to None = use vLLM default)
    if vllm_cfg.tensor_parallel_size is not None:
        kwargs["tensor_parallel_size"] = vllm_cfg.tensor_parallel_size
    if vllm_cfg.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = vllm_cfg.gpu_memory_utilization
    if vllm_cfg.max_num_seqs is not None:
        kwargs["max_num_seqs"] = vllm_cfg.max_num_seqs
    if vllm_cfg.enable_prefix_caching is not None:
        kwargs["enable_prefix_caching"] = vllm_cfg.enable_prefix_caching
    if vllm_cfg.quantization is not None:
        kwargs["quantization"] = vllm_cfg.quantization
    # Phase 19.1 additions: enforce_eager, block_size, kv_cache_dtype, swap_space
    return kwargs

def _map_precision(self, precision: str) -> str:
    return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(precision, "auto")
```

### Registering vLLM in get_backend()
```python
# core/backends/__init__.py — add to get_backend()
def get_backend(name: str) -> InferenceBackend:
    if name == "pytorch":
        from llenergymeasure.core.backends.pytorch import PyTorchBackend
        return PyTorchBackend()
    if name == "vllm":
        from llenergymeasure.core.backends.vllm import VLLMBackend
        return VLLMBackend()
    raise BackendError(f"Unknown backend: {name!r}. Available: pytorch, vllm")
```

### VLLMConfig Expansion (Phase 19.1)
```python
# config/backend_configs.py — VLLMConfig additions
class VLLMConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # Existing fields (keep as-is)
    max_num_seqs: int | None = Field(default=None, ge=1, ...)
    tensor_parallel_size: int | None = Field(default=None, ge=1, ...)
    gpu_memory_utilization: float | None = Field(default=None, ge=0.1, le=1.0, ...)
    enable_prefix_caching: bool | None = Field(default=None, ...)
    quantization: Literal["awq", "gptq", "fp8"] | None = Field(default=None, ...)

    # Phase 19.1 additions — energy-relevant tier
    enforce_eager: bool | None = Field(
        default=None,
        description="Disable CUDA graphs; forces eager PyTorch execution (None -> False). "
                    "Reduces throughput but required for some models."
    )
    block_size: int | None = Field(
        default=None,
        description="KV cache block size in tokens (8/16/32/64/128). "
                    "Smaller blocks reduce fragmentation. (None -> 16)"
    )
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e5m2", "fp8_e4m3"] | None = Field(
        default=None,
        description="KV cache storage dtype. fp8 halves KV memory. (None -> auto = model dtype)"
    )
    swap_space: float | None = Field(
        default=None, ge=0.0,
        description="CPU swap space per GPU in GiB for KV cache offload. (None -> 4 GiB)"
    )
    max_model_len: int | None = Field(
        default=None, ge=1,
        description="Max context length. Overrides model's max_position_embeddings. "
                    "Set lower to reduce KV cache memory."
    )
    extra_kwargs: dict | None = Field(
        default=None,
        description="Escape hatch: extra kwargs forwarded to LLM() constructor. "
                    "Allows passing vLLM params not explicitly modelled here."
    )
```

---

## Protocol Assessment (Claude's Discretion)

The current `InferenceBackend` protocol in `core/backends/protocol.py` is minimal:
```python
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...
    def run(self, config: ExperimentConfig) -> ExperimentResult: ...
```

**Assessment**: This is the right level of strictness for Phase 19. The single-method protocol (matching lm-eval's `LM` subclass pattern) gives backends maximum flexibility to implement their measurement lifecycle. The CONTEXT.md explicitly listed a backend protocol refactor as deferred. **Do not tighten the protocol in Phase 19.** Write VLLMBackend to satisfy the two-method contract and defer any protocol changes.

---

## Docker Flags (Claude's Discretion)

Current `DockerRunner` flags: `docker run --rm --gpus all -v {dir}:/run/llem -e LLEM_CONFIG_PATH=... --shm-size 8g [image] python -m ...`

Assessment of additional flags:
- `--ipc=host`: Sometimes recommended for vLLM's multi-process communication. However, `--shm-size 8g` provides a sufficiently large shared memory namespace. `--ipc=host` shares the host's IPC namespace — less isolated, not needed with `--shm-size 8g`. **Do not add** unless testing reveals issues.
- `--ulimit memlock=-1`: Required for some RDMA/NVLink configurations. Not needed for single A100. **Do not add** unless needed.
- `HF_HOME`, `TRANSFORMERS_CACHE`: Useful for model caching. Already handled by HF_TOKEN propagation. Could add `HF_HOME=/run/llem/hf-cache` or propagate the host's HF_HOME env var. **Nice-to-have, not required** for Phase 19.
- `--cap-add SYS_PTRACE`: Sometimes needed for profiling. **Do not add.**

**Recommendation**: Keep current flags (`--gpus all`, `--shm-size 8g`). Add HF_HOME propagation (analogous to HF_TOKEN pattern) if model download fails inside container.

---

## Model Loading Strategy (Claude's Discretion)

vLLM model loading options:
- `enforce_eager=False` (default): vLLM uses CUDA graphs after warmup — better throughput
- `enforce_eager=True`: Eager PyTorch only — needed for some models (Mamba, etc.), lower throughput

**Recommendation**: Default to `enforce_eager=False` (vLLM's default) for best performance. Expose via `VLLMConfig.enforce_eager`.

Model download cache: vLLM uses the same `~/.cache/huggingface/` as standard HuggingFace. Inside containers, this is ephemeral unless a volume is mounted. For testing, the model must either be downloaded fresh each run (slow) or the HF cache directory must be mounted.

**Recommendation for Phase 19 testing**: Mount the host's HF cache into the container via `-v ${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface`. This is optional in DockerRunner (only if the env var is set). Not required for correctness, but needed for practical development speed.

---

## Open Questions

1. **`config.save_outputs` field existence**
   - What we know: PyTorchBackend references `config.save_outputs` in comments but current `ExperimentConfig` doesn't show this field (removed in v2.0 schema removal list)
   - What's unclear: Does `config.save_outputs` exist on current `ExperimentConfig`? The legacy vllm.py references it but the v2.0 schema removed it.
   - Recommendation: Check `config/models.py` full contents — if absent, skip output text collection entirely in Phase 19 (token counts only)

2. **Multi-GPU energy measurement for Phase 19**
   - What we know: Current `NVMLBackend` only monitors device 0. Multi-GPU runs (tensor_parallel_size > 1) under-measure energy.
   - What's unclear: Is multi-GPU mandatory for Phase 19 or can single-GPU be the first target?
   - Recommendation: Treat single-GPU (tensor_parallel_size=1) as Phase 19 target. Multi-GPU energy summing is a Phase 19.1/follow-up item. Document limitation in measurement warnings.

3. **`_prepare_prompts()` dataset loading**
   - What we know: PyTorchBackend has a placeholder `_prepare_prompts()` that generates synthetic prompts. Real dataset loading (`aienergyscore.jsonl`) is carried to Phase 21.
   - What's unclear: Should vLLM backend use the same M1 placeholder or the actual dataset?
   - Recommendation: Reuse the same placeholder pattern for Phase 19 — prompts are prompts regardless of backend. Dataset loading is a separate concern.

4. **`config.n` vs `config.dataset.n` for prompt count**
   - What we know: `ExperimentConfig.n = 100` is the number of prompts. `SyntheticDatasetConfig.n` is for the synthetic dataset sub-config.
   - What's unclear: Which field drives prompt count in the new backend?
   - Recommendation: Use `config.n` directly — same as PyTorchBackend's placeholder.

---

## Testing Strategy

Tests must run **without GPU** (unit tests) and **with GPU inside Docker** (integration/smoke).

### Unit Tests (GPU-free)
All vLLM imports guarded by `TYPE_CHECKING` or lazy imports. Tests mock `vllm.LLM` and verify:
- `get_backend("vllm")` returns a `VLLMBackend` instance
- `VLLMBackend.name == "vllm"`
- `_build_llm_kwargs()` maps `VLLMConfig` fields correctly
- `VLLMConfig` validation: `enforce_eager=True` is valid, unknown quantization method is rejected

File: `tests/unit/test_vllm_backend.py`

### Integration Tests (Docker required)
Cannot run in CI without GPU. Manual test using a small model:
```yaml
# test-vllm.yaml
backend: vllm
model: facebook/opt-125m
n: 5
max_input_tokens: 64
max_output_tokens: 32
warmup:
  enabled: false
vllm:
  enforce_eager: true  # faster startup for testing
```
```bash
llem run test-vllm.yaml --runner docker
```
Expected: `ExperimentResult` with non-zero `total_tokens`, `total_energy_j`, `avg_tokens_per_second`.

### Test for VLLM-03 (shm-size already fixed)
Verify `DockerRunner._build_docker_cmd()` output contains `--shm-size 8g`:
```python
def test_docker_cmd_includes_shm_size():
    runner = DockerRunner(image="test-image")
    cmd = runner._build_docker_cmd("abc123", "/tmp/llem-test")
    shm_idx = cmd.index("--shm-size")
    assert cmd[shm_idx + 1] == "8g"
```
This test likely already exists or is trivially addable to `tests/unit/test_docker_runner.py`.

---

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection — `core/backends/pytorch.py`, `infra/docker_runner.py`, `infra/container_entrypoint.py`, `core/backends/__init__.py`, `config/backend_configs.py`, `config/models.py`
- [vLLM Engine Arguments v0.7.1](https://docs.vllm.ai/en/v0.7.1/serving/engine_args.html) — parameter defaults verified
- [ML.Energy GPU energy best practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/) — `torch.cuda.synchronize()` + `nvmlDeviceGetTotalEnergyConsumption` pattern
- [lm-eval `vllm_causallms.py`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/vllm_causallms.py) — escape hatch pattern (`model_args.update(kwargs)`), offline generate() usage

### Secondary (MEDIUM confidence)
- [vLLM offline inference docs](https://docs.vllm.ai/en/stable/serving/offline_inference/) — general structure confirmed
- vLLM issue tracker patterns (verified `--shm-size` requirement for container runs from community)

---

## Metadata

**Confidence breakdown:**
- Standard stack (vLLM LLM/SamplingParams API): HIGH — cross-referenced docs + lm-eval implementation
- Architecture (offline batch pattern): HIGH — confirmed single `generate()` call from docs + lm-eval
- Energy measurement integration: HIGH — existing codebase has working NVMLBackend; pattern from ML.Energy blog
- CM-07 fix (streaming removal): HIGH — confirmed `config.streaming` no longer exists in v2.0 schema
- CM-09 fix (shm-size): HIGH — directly verified in docker_runner.py source
- VLLMConfig expansion params: HIGH — verified against vLLM v0.7.1 docs
- Multi-GPU energy: MEDIUM — deferred, single-GPU is Phase 19 target

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (vLLM API evolves fast; re-verify if vLLM version changes)
