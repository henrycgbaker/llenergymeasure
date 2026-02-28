# Phase 19 + 19.1: vLLM Backend Activation & Parameter Audit - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

**Phase 19:** Get vLLM running end-to-end via Docker using offline batch inference. Fix P0 bugs (streaming broken CM-07, shm-size CM-09). Produce a valid ExperimentResult. Ground-up rewrite of the vLLM backend — do not patch existing code.

**Phase 19.1:** Audit upstream vLLM API (`LLM()` + `SamplingParams`), expand VLLMConfig with energy-relevant parameters, wire sweep grammar with universal dotted notation.

**Important:** Several cross-cutting architectural decisions emerged during discussion. These affect ALL backends, not just vLLM. They should be captured as ADRs in `.product/decisions/` and may require their own phases or a refactor pass.

</domain>

<decisions>
## Implementation Decisions

### Inference Mode
- **Offline batch only** for Phase 19 — `vllm.LLM()` with `llm.generate(prompts)`
- Single `generate()` call with all prompts (not one-at-a-time) — matches PyTorch pattern
- Online/streaming server mode is a separate future phase
- Strip existing streaming and traffic simulation code entirely — rewrite from scratch in the server phase
- Per-token latency metrics (TTFT, ITL) skipped for offline mode — defer to online server phase

### Output Collection
- Token counts by default, full output text only when `save_outputs=True`
- Consistent with PyTorch backend pattern

### Container Lifecycle
- **Self-contained container** — receives config via volume mount, runs everything, writes result JSON
- Energy measurement happens **inside the container** (in-process NVML), following Zeus/CodeCarbon peer consensus
- Host just does `docker run` and collects result files
- Container entrypoint should be rewritten to include NVML energy measurement wrapping inference
- `torch.cuda.synchronize()` before energy reads is mandatory (Zeus best practice) — prevents under-measurement from async GPU kernels

### Error Handling
- **Generic error passthrough** for vLLM-specific errors — no special translation
- Existing harness patterns apply (skip-and-continue, error hierarchy)
- Container writes error JSON on failure (current entrypoint pattern, keep this)

### Parameter Surface (Phase 19.1)
- **Research-driven param tiers** — researcher identifies which vLLM params have biggest energy impact, expose those first
- Goal: eventually expose all vLLM params, but stagger across phases (energy-relevant first, completeness later)
- **Escape hatch (extra kwargs)** — researcher to investigate how peers (lm-eval) handle pass-through kwargs to vLLM
- **Universal dotted notation** in sweep grammar: `vllm.gpu_memory_utilization: [0.8, 0.9]` — applies to ALL backends (`pytorch.torch_dtype`, `tensorrt.precision`, etc.)

### Multi-GPU Support
- **tensor_parallel_size is configurable** — some models require multi-GPU
- Multi-GPU energy measurement (sum NVML across all devices) — vital, applies to ALL backends
- Researcher must investigate how Zeus/CodeCarbon handle multi-device energy measurement

### Cross-Cutting Harness Decisions (all backends, not vLLM-specific)
These emerged during discussion and need ADRs in `.product/decisions/`:

1. **Backend protocol/plugin architecture** — harness controls measurement lifecycle (load → warmup → measure energy → infer → stop energy → collect → cleanup). Backends are plugins implementing specific hooks. "Strict on measurement contract, flexible on inference strategy."
2. **Per-prompt token counting** — ExperimentResult schema change. Store per-prompt `input_tokens` and `output_tokens` arrays. Breaking change, acceptable pre-v2.0. Aligns with lm-eval and GenAI-Perf.
3. **Chat templates as harness-level feature** — harness applies `apply_chat_template()` before passing prompts to any backend. User specifies template in config. Matches lm-eval pattern.
4. **Batching strategy constant across backends** — single generate/forward call with all prompts.
5. **Energy measurement inside container** — in-process NVML (Zeus/CodeCarbon pattern), not host-side.
6. **Multi-GPU energy measurement** — sum across all NVML devices.

### Claude's Discretion
- Whether to tighten the InferenceBackend protocol in Phase 19 or defer to a separate refactor phase — assess current protocol strictness and recommend
- Container runtime flags beyond `--shm-size 8g` and `--gpus all`
- vLLM model loading strategy (lazy vs eager, download cache location)

</decisions>

<specifics>
## Specific Ideas

- **Ground-up rewrite**: Do not treat current `vllm.py` (1007 lines) as gospel. Design the vLLM backend from scratch based on peer research. The current code was written pre-M1 and hasn't been validated against real Docker execution.
- **Peer codebase research is mandatory**: Study how lm-eval-harness, vLLM's own benchmarks (`vllm bench throughput`), GenAI-Perf, Zeus, and CodeCarbon integrate vLLM / handle energy measurement. Design decisions should be grounded in peer patterns.
- **Zeus's approach**: `nvmlDeviceGetTotalEnergyConsumption()` with `torch.cuda.synchronize()` before reads. Energy counter API has <10ms overhead. For older GPUs, separate `PowerMonitor` process with AUC integration.
- **lm-eval's vLLM integration**: Uses `vllm.LLM()` directly (offline), supports `TokensPrompt(prompt_token_ids=...)`, handles chat templates via `apply_chat_template()`.
- **GenAI-Perf's energy approach**: DCGM Exporter as separate container, 33ms collection interval, scraped via HTTP — relevant for the future online server phase, not Phase 19.

</specifics>

<research_questions>
## Research Questions (for Phase 19/19.1 researcher)

1. **Energy-relevant vLLM params**: Which `LLM()` and `SamplingParams` parameters have the biggest energy impact? (enforce_eager, gpu_memory_utilization, quantization, block_size, kv_cache_dtype, dtype, speculative decoding, etc.)
2. **Pass-through kwargs**: How does lm-eval handle arbitrary kwargs to vLLM's `LLM()` constructor? Do they allow an escape hatch?
3. **Sampling param split**: Which sampling params (temperature, top_p, max_tokens) use the same terminology across vLLM/PyTorch/TensorRT? Where does terminology diverge?
4. **Multi-GPU energy**: How do Zeus and CodeCarbon handle energy measurement across multiple GPUs? Sum per-device? Weighted?
5. **cuda.synchronize() enforcement**: How should the protocol enforce sync before energy reads? Is this a backend responsibility or harness responsibility?
6. **Container NVML access**: What Docker flags are needed for NVML energy measurement inside the container? (`--gpus all` sufficient, or `--cap-add` needed?)

</research_questions>

<deferred>
## Deferred Ideas

- **vLLM online/streaming server mode** — own phase. GenAI-Perf pattern (client against running server). Enables per-request TTFT/ITL metrics.
- **Backend protocol refactor** — tighten plugin architecture across all backends. Enforce measurement lifecycle via protocol. May be Phase 19 or its own phase (Claude's discretion).
- **Sampling param translation layer** — abstract shared params (temperature, top_p) with backend-specific mapping where terminology diverges. Future phase.
- **Complete vLLM param exposure** — beyond energy-relevant tier. Expose all `LLM()` and `SamplingParams` options. Staggered after 19.1.
- **ADRs for cross-cutting decisions** — write `.product/decisions/` documents for: backend protocol, per-prompt tokens, chat templates, energy location, multi-GPU. Follow-up action after this context.
- **Milestone restructuring** — the scope of cross-cutting changes may warrant splitting across multiple version bumps. Review after planning.

</deferred>

---

*Phase: 19-vllm-backend-activation*
*Context gathered: 2026-02-28*
