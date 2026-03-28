# Design: Sweep Constraints and Dependent Parameters

**Date:** 2026-03-28
**Status:** RFC (Request for Comments) - all proposals are open to change
**Scope:** Config grid expansion (`config/grid.py`), study YAML schema, `study-full-suite.yaml`
**Problem:** The sweep system's Cartesian product cannot express parameter dependencies, forcing users to duplicate experiments as explicit entries.

> **Note:** This document is exploratory. The recommendation (Option A) represents the current best thinking after researching peer frameworks and analysing our constraint catalogue, but the design is not settled. All options, trade-offs, and syntax choices are open for discussion. The goal is to find the optimal solution - not to lock in prematurely.

## Context

The current sweep system (`_expand_sweep()` in `grid.py`) generates the Cartesian product of all sweep dimensions. This works for independent axes but breaks down when parameters have dependencies:

- `torch_compile_mode` only makes sense when `torch_compile: true`
- `bnb_4bit_*` sub-params only apply when `load_in_4bit: true`
- `vllm.beam_search` and `vllm.sampling` are mutually exclusive sections
- beam search params require `decoder.do_sample: false`
- `tensorrt.calib.*` only applies when `tensorrt.quant.quant_algo` is set

Today, dependent parameters must be written as explicit experiments - verbose, repetitive blocks that restate shared defaults. In `study-full-suite.yaml`, roughly half the `experiments:` section could be eliminated with a constraint mechanism.

Meanwhile, many explicit experiments have **no real dependencies** and are just unnecessarily verbose (vLLM prefix caching, block sizes, TensorRT scheduler policies, etc.). These should move into `sweep:` immediately regardless of the constraint design.

## Complete Constraint Catalogue

Every parameter dependency in the codebase, derived from Pydantic validators in `backend_configs.py` and `models.py`:

### Mutual Exclusions (A and B cannot both be set)

| Parameter A | Parameter B | Validator |
|---|---|---|
| `pytorch.load_in_4bit` | `pytorch.load_in_8bit` | `validate_quantization()` |
| `pytorch.tp_plan` | `pytorch.device_map` | `validate_tp_device_map_exclusive()` |
| `vllm.engine.kv_cache_memory_bytes` | `vllm.engine.gpu_memory_utilization` | `validate_kv_cache_memory()` |
| `vllm.beam_search` (section) | `vllm.sampling` (section) | `validate_beam_search_exclusive()` |

### Conditional Requirements (B requires A to be set) - Depth 1

| Dependent param (B) | Requires (A) | Validator |
|---|---|---|
| `pytorch.torch_compile_mode` | `pytorch.torch_compile: true` | `validate_torch_compile_options()` |
| `pytorch.torch_compile_backend` | `pytorch.torch_compile: true` | `validate_torch_compile_options()` |
| `pytorch.bnb_4bit_compute_dtype` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.bnb_4bit_quant_type` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.bnb_4bit_use_double_quant` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.cache_implementation` | `pytorch.use_cache: true\|null` | `validate_cache_options()` |
| `vllm.engine.speculative_model` | `vllm.engine.num_speculative_tokens` | `validate_speculative()` |

### Depth-2+ Chains (semantic, not all validator-enforced today)

```
pytorch.load_in_4bit: true
  ├── pytorch.bnb_4bit_compute_dtype        [validated]
  ├── pytorch.bnb_4bit_quant_type           [validated]
  │     └── pytorch.bnb_4bit_use_double_quant  [NOT validated - only meaningful with specific quant_type]
  └── pytorch.bnb_4bit_use_double_quant     [validated against load_in_4bit only]

tensorrt.quant.quant_algo (e.g. INT8)
  ├── tensorrt.quant.kv_cache_quant_algo    [NOT validated - no validator requiring quant_algo]
  └── tensorrt.calib.*                      [NOT validated - calib without quant is silently meaningless]
        ├── calib_batches
        ├── calib_dataset
        └── calib_max_seq_length
```

### Cross-Section Dependencies (not expressed as validators today)

| Scenario | Constraint |
|---|---|
| Beam search (pytorch) | Requires `decoder.do_sample: false`, `decoder.temperature: 0.0` |
| Beam search (vllm) | Requires `decoder.do_sample: false`, `decoder.temperature: 0.0` |
| Speculative decoding (pytorch) | Works only with `pytorch.batch_size: 1` |

### Missing Validators (separate hardening work)

These depth-2 dependencies should be enforced by Pydantic validators regardless of the sweep design:

| Dependency | Proposed validator |
|---|---|
| `tensorrt.calib.*` requires `tensorrt.quant.quant_algo` | `TensorRTConfig.validate_calib_requires_quant()` |
| `tensorrt.quant.kv_cache_quant_algo` requires `tensorrt.quant.quant_algo` | `TensorRTQuantConfig.validate_kv_cache_quant_requires_quant()` |

---

## Peer Research

### How other frameworks solve this

| Framework | Approach | Format | Invalid combos | Expressiveness |
|---|---|---|---|---|
| **Hydra** (native) | Cartesian only | YAML | Not prevented | Low |
| **hydra-filter-sweeper** | Post-filter expressions | YAML | Excluded after generation | Medium |
| **Optuna** | Programmatic define-by-run | Python | Structurally impossible | High |
| **Ray Tune** | Lambda deps + Optuna delegation | Python | Via Optuna | High |
| **W&B Sweeps** | Flat grid only | YAML | Not prevented | Low |
| **lm-eval-harness** | Separation of concerns | YAML + CLI | Avoided by design | N/A |
| **NNI** | Declarative hierarchical tree | JSON | Structurally impossible | Medium |
| **Ax** | `HierarchicalSearchSpace` + constraints | Python | Structurally impossible | High |

**Key patterns observed:**

1. **Programmatic conditionals** (Optuna, Ax): Most expressive but requires code, not config. Not suitable for a YAML-driven research tool.

2. **Declarative hierarchical tree** (NNI, Ax): A discriminator choice determines which sub-parameters exist. NNI uses `_type: "choice"` with `_name` discriminators in JSON. Ax uses `ChoiceParameter.dependents` mapping in Python. Both model the parameter space as a tree where choices at level N determine which parameters exist at level N+1. Structurally prevents invalid combos.

3. **Post-filtering** (hydra-filter-sweeper): Generate full Cartesian product, then exclude invalid combos via expressions. Simple but wasteful for large grids.

4. **Separation of concerns** (lm-eval): Backend params never enter a shared grid. We already do this via dotted-key routing, but dependencies *within* a backend remain unresolved.

---

## Options

### Option A: Inline Sweep Groups (Recommended)

Extend the existing `sweep:` section to support **groups** alongside independent axes, using the same dot notation. The type of the value disambiguates: a list of scalars is an independent axis; a list of dicts is a dependent group.

```yaml
sweep:
  # ── Independent axes (list of scalars → Cartesian product) ──
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, eager]

  # ── Dependent groups (list of dicts → union of variants) ────
  pytorch.compile:
    - torch_compile: false
    - torch_compile: true
      torch_compile_mode: [default, reduce-overhead, max-autotune]
      torch_compile_backend: inductor

  pytorch.quantization:
    - {}                                    # no quantisation
    - load_in_8bit: true
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: [nf4, fp4]
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4
      bnb_4bit_use_double_quant: true
```

**Disambiguation rule:** If a sweep value is a **list of dicts** (or contains `{}`), it's a group. If it's a **list of scalars**, it's an independent axis. The key path provides the group name (for display) and the backend scope (for routing).

**Key resolution within groups:** Keys within group entries are relative to the backend prefix. `torch_compile` under `pytorch.compile` resolves to `pytorch.torch_compile`. Fully-qualified keys (containing a dot whose prefix isn't the group's backend) override any field - enabling cross-section overrides like `decoder.do_sample: false` in a `pytorch.decoding` group.

**Semantics:**
- Each group entry generates its own mini-grid (Cartesian product of list-valued fields within the entry)
- Group entries are **unioned** (not crossed with each other)
- The union of all groups **is crossed** with each other and with the independent axes
- `{}` means "no overrides" (use fixed/sweep defaults) - the baseline variant
- Group names are virtual paths (not real config fields) used for display and scoping

**Expansion example:**

```
Independent: 2 dtype × 5 batch_size × 3 attn = 30 base combos

Groups:
  pytorch.compile:      1 (false) + 3 (true × modes) = 4 variants
  pytorch.quantization: 1 (none) + 1 (8bit) + 2 (4bit × types) + 1 (double) = 5 variants

Groups crossed: 4 × 5 = 20 group combos
Total: 30 × 20 = 600 pytorch experiments
```

**Pros:**
- No new top-level key - everything lives under `sweep:`
- Dot notation signals hierarchy consistently (independent axes and groups alike)
- Type-based disambiguation (list-of-scalars vs list-of-dicts) is unambiguous
- Group name doubles as backend scope (auto-routing)
- Relative keys within groups reduce repetition
- Backwards compatible - existing `sweep:` configs with only scalar lists work unchanged
- Dependencies expressed structurally (child params live under their parent's group)

**Cons:**
- Two syntaxes under one key (could confuse new users)
- Virtual group names (e.g. `pytorch.compile`) must not collide with real config paths
- Relative key resolution adds a layer of indirection

---

### Option B: Constraint Expressions (`sweep.exclude` / `sweep.require`)

Keep flat Cartesian product but add post-filtering via constraint expressions. Inspired by hydra-filter-sweeper.

```yaml
sweep:
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.torch_compile: [true, false]
  pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune, null]

  exclude:
    - pytorch.torch_compile == false and pytorch.torch_compile_mode != null
    - pytorch.load_in_4bit == true and pytorch.load_in_8bit == true

  require:
    - pytorch.num_beams > 1 implies decoder.do_sample == false
```

**Pros:**
- Maximum flexibility - any boolean condition
- No structural changes to sweep layout
- Familiar to users of SQL WHERE clauses or Hydra filters

**Cons:**
- Requires implementing an expression parser/evaluator (security risk, maintenance burden)
- `null` handling is fiddly (`!= null` vs `is not None`)
- Wasteful: generates all combos first, then filters - potentially 10x overgeneration
- Expression language is a new DSL to learn, document, and maintain
- Harder to read: the constraint is separated from the params it governs
- Does not scale: as constraints grow, the exclude list becomes the documentation

---

### Option C: Nested Hierarchical Sweep (NNI-style)

Replace flat sweep keys with a tree structure where a discriminator choice determines available sub-parameters.

```yaml
sweep:
  dtype: [float16, bfloat16]
  pytorch:
    batch_size: [1, 4, 8, 16, 32]
    quantization:
      - variant: none
      - variant: bnb_8bit
        load_in_8bit: true
      - variant: bnb_4bit
        load_in_4bit: true
        bnb_4bit_compute_dtype: float16
        bnb_4bit_quant_type: [nf4, fp4]
```

**Pros:**
- Invalid combos structurally impossible (like NNI/Ax)
- Rich, self-documenting structure
- Natural for parameters that form a tree

**Cons:**
- **Major breaking change** to sweep YAML schema
- Deeply nested YAML is hard to read and write
- `variant:` discriminator is artificial - not a real config field
- Significantly more complex expansion logic
- Doesn't map cleanly to dotted-key routing (which is flat by design)
- Over-engineers the problem: most sweep axes are genuinely independent

---

### Option D: Full Cartesian + Pydantic Auto-Filter

Generate the full Cartesian product of all parameters (including dependent ones), then rely entirely on Pydantic validators to reject invalid combinations as `SkippedConfig`.

```yaml
sweep:
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.torch_compile: [true, false]
  pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune, null]
  pytorch.load_in_4bit: [true, null]
  pytorch.load_in_8bit: [true, null]
  pytorch.bnb_4bit_compute_dtype: [float16, null]
  pytorch.bnb_4bit_quant_type: [nf4, fp4, null]
```

Enhancements:
- `null` sentinel in sweep values means "omit this field" (not set to None)
- Preflight panel shows expected skip count before execution
- Skipped configs logged with clear per-constraint reasoning

**Pros:**
- Zero new syntax or concepts
- Pydantic validators are the SSOT - constraints defined once
- Already partially implemented (SkippedConfig exists)
- Simplest implementation

**Cons - and why this is genuinely worse for dependent parameters:**

**1. Massive combinatorial waste.** In a realistic full-suite config, the numbers are stark:

| Dimension | Group approach (valid configs) | Cartesian approach (total generated) |
|---|---|---|
| torch.compile | 4 | 16 (2 × 4 × 2) |
| quantisation | 5 | 48 (2 × 2 × 3 × 2 × 2) |
| cache | 2 | 4 (2 × 2) |
| decoding | 3 | complex multi-field product |

Groups: 4 × 5 × 2 × 3 = **120 configs**, all valid by construction.
Cartesian: 16 × 48 × 4 × ... = **3,000+ raw combos**, ~120 survive validation.

Crossed with independent axes (3 dtype × 5 batch × 4 attn = 60):
- Groups: 120 × 60 = **7,200 experiments**
- Cartesian: 3,000+ × 60 = **180,000+ raw combos** → ~96% skip rate

Each `ExperimentConfig` construction with nested sub-models and 6+ validators costs ~0.5ms. 180k constructions = ~90 seconds of config expansion vs ~3.6 seconds for groups.

**2. The `null`/`_omit` sentinel problem.** To make Cartesian work for optional fields, you need a way to say "don't set this parameter". YAML `null` is ambiguous - does it mean "omit the field" or "set the value to None"? For most fields they're the same (default is None), but it's a conceptual trap that requires a new sentinel like `_omit` - a new concept users must learn.

**3. Only as correct as validators.** Depth-2 chains like `tensorrt.quant.quant_algo → tensorrt.calib.*` have no validators today. Cartesian + filter would produce configs where `calib` is set but `quant` is null - technically valid Pydantic models but scientifically meaningless. The approach fails silently wherever validators are missing. Groups prevent this structurally because the author only includes `calib.*` in entries that also set `quant.quant_algo`.

**4. Preflight becomes noise.** "180,000 configs generated, 172,800 skipped (96%)" tells the user nothing about what's being tested. The skip rate drowns the signal.

**5. Every new parameter multiplies the space.** Adding a single boolean to the Cartesian product doubles the total combos. Groups grow linearly - adding a variant to a group adds +1.

---

## Recommendation: Option A (Inline Sweep Groups)

### Rationale

1. **Unified syntax.** Groups live inside `sweep:` using the same dot notation. No new top-level key. The type of the value (list-of-scalars vs list-of-dicts) is the only new concept.

2. **Right level of abstraction.** Groups express "these parameters travel together" - which is exactly what the constraint catalogue shows. Every dependency is a group relationship: torch_compile + its sub-params, load_in_4bit + its sub-params, beam_search + decoder settings.

3. **Groups are isomorphic to dependency trees.** Each group entry is a valid root-to-leaf path through the dependency tree. NNI and Ax express the same information as nested tree structures; a flat list of valid combinations is equivalent in expressiveness but far simpler in YAML. Listing the valid paths *is* the graph - no explicit graph machinery needed.

4. **YAML-native, no DSL.** Unlike Option B, no expression parser. Unlike Option C, no artificial discriminators. Standard YAML lists and maps.

5. **Efficient.** Only valid combinations generated (vs 96% waste in Option D). Config expansion stays under 5 seconds even for large studies.

6. **Pydantic stays the safety net.** Any group that accidentally produces an invalid config still gets caught by Pydantic validators and reported as `SkippedConfig`. Belt and braces.

7. **Backwards compatible.** Existing `sweep:` configs with only scalar lists work unchanged. Groups are opt-in.

### No graph validation needed at sweep level

The dependency structure across all backends is shallow (depth 1-2 trees, no deep chains). Groups express valid paths through these trees structurally - each entry is a valid combination by construction. Pydantic validators are the authoritative constraint graph; groups just avoid generating obviously-invalid configs.

Where this could break: a user puts `tensorrt.calib.calib_batches` as an independent axis without any quant group. This produces configs where calib is set but quant is null - semantically wrong. The fix is **missing validators** (below), not sweep-level graph machinery.

### Missing validators to add (separate PR)

| Dependency | Proposed validator |
|---|---|
| `tensorrt.calib.*` requires `tensorrt.quant.quant_algo` | `TensorRTConfig.validate_calib_requires_quant()` |
| `tensorrt.quant.kv_cache_quant_algo` requires `quant_algo` | `TensorRTQuantConfig.validate_kv_cache_quant_requires_quant()` |

These strengthen Pydantic as the safety net independent of the sweep design.

### What remains as explicit experiments

After groups, the `experiments:` section is reserved for truly unique one-off experiments that:
- Override sweep axes for specific configs (e.g. `batch_size: 1` for speculative decoding)
- Use parameters not in any sweep axis (e.g. `lora.adapter_id` pointing to a specific model)
- Combine settings that span multiple unrelated groups in unusual ways

This should reduce `experiments:` from ~40 entries to ~5-10.

---

## Detailed Design

### YAML Schema

```yaml
model: Qwen/Qwen2.5-0.5B

sweep:
  # ── Independent axes (list of scalars → Cartesian product) ──────────
  dtype: [float32, float16, bfloat16]

  # ── PyTorch: independent axes ───────────────────────────────────────
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

  # ── PyTorch: dependent groups (list of dicts → union of variants) ───
  pytorch.compile:
    - torch_compile: false
    - torch_compile: true
      torch_compile_mode: [default, reduce-overhead, max-autotune]
      torch_compile_backend: inductor

  pytorch.quantization:
    - {}                                    # baseline: no quantisation
    - load_in_8bit: true
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: [nf4, fp4]
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4
      bnb_4bit_use_double_quant: true

  pytorch.cache:
    - {}
    - use_cache: true
      cache_implementation: static

  pytorch.decoding:
    - {}                                    # baseline: use shared decoder settings
    - decoder.do_sample: false              # fully-qualified → cross-section override
      decoder.temperature: 0.0
      num_beams: [2, 4]                     # relative → pytorch.num_beams
      early_stopping: true
      length_penalty: 1.0
      no_repeat_ngram_size: 3

  # ── vLLM: independent axes ─────────────────────────────────────────
  vllm.engine.gpu_memory_utilization: [0.7, 0.85, 0.95]
  vllm.engine.enforce_eager: [true, false]
  vllm.engine.max_num_seqs: [64, 128, 256]
  vllm.engine.enable_chunked_prefill: [true, false]
  vllm.engine.enable_prefix_caching: [true, false]
  vllm.engine.block_size: [8, 16, 32]
  vllm.engine.kv_cache_dtype: [auto, fp8]

  # ── vLLM: dependent groups ─────────────────────────────────────────
  vllm.decoding:
    - {}
    - decoder.do_sample: false
      decoder.temperature: 0.0
      beam_search:
        beam_width: 4
        early_stopping: true
    - sampling:
        presence_penalty: [0.0, 0.6]
        frequency_penalty: [0.0, 0.6]

  # ── TensorRT: independent axes ─────────────────────────────────────
  tensorrt.max_batch_size: [1, 4, 8, 16, 32]
  tensorrt.max_seq_len: [1536, 2048]

  # ── TensorRT: dependent groups ─────────────────────────────────────
  tensorrt.quantization:
    - {}                                    # baseline: no quantisation
    - quant.quant_algo: INT8
    - quant.quant_algo: INT8
      quant.kv_cache_quant_algo: INT8
    - quant.quant_algo: INT8
      calib.calib_batches: [256, 512]
      calib.calib_dataset: cnn_dailymail
      calib.calib_max_seq_length: [256, 512]
    - quant.quant_algo: W4A16_AWQ
    - quant.quant_algo: W8A16

  tensorrt.scheduling:
    - {}
    - scheduler.capacity_scheduling_policy: MAX_UTILIZATION
    - scheduler.capacity_scheduling_policy: STATIC_BATCH

  tensorrt.kv_cache_config:
    - {}
    - kv_cache.enable_block_reuse: true
      kv_cache.free_gpu_memory_fraction: 0.9
    - kv_cache.host_cache_size: 1073741824
```

### Type-Based Disambiguation

The expansion logic distinguishes groups from independent axes by inspecting the value type:

```python
def _is_group(value: Any) -> bool:
    """A sweep entry is a group if its value is a list containing dicts."""
    return isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict)
```

- `dtype: [float16, bfloat16]` → list of scalars → **independent axis**
- `pytorch.compile: [{torch_compile: false}, {torch_compile: true, ...}]` → list of dicts → **group**
- `pytorch.batch_size: [1, 4, 8]` → list of scalars → **independent axis**

Edge case: `pytorch.something: [1]` (single-element list of scalar) is an independent axis with one value (effectively a constant). `pytorch.something: [{}]` is a group with one variant (also effectively a constant, but processed through group expansion).

### Key Resolution Within Groups

Keys within group entries are resolved relative to the group's backend prefix:

```
Group key: pytorch.compile
Entry key: torch_compile_mode

Resolution: pytorch.torch_compile_mode
```

**Exception - cross-section overrides:** If a key within a group entry contains a dot and the prefix is not the group's backend, it's treated as a fully-qualified path:

```
Group key: pytorch.decoding
Entry key: decoder.do_sample       → decoder.do_sample (fully-qualified, not pytorch.decoder.do_sample)
Entry key: num_beams               → pytorch.num_beams (relative to pytorch)
Entry key: early_stopping          → pytorch.early_stopping (relative to pytorch)
```

Detection rule:

```python
def _resolve_group_key(group_backend: str, entry_key: str) -> str:
    """Resolve a key within a group entry to a fully-qualified config path."""
    if "." in entry_key:
        prefix = entry_key.split(".", 1)[0]
        if prefix != group_backend and prefix not in _BACKEND_SECTION_KEYS:
            # Cross-section override: decoder.do_sample, dataset.n_prompts
            return entry_key
    # Relative to group's backend
    return f"{group_backend}.{entry_key}"
```

### Backend Auto-Scoping for Multi-Backend Sweeps

Groups are automatically scoped to a backend based on their key prefix:

```
pytorch.compile       → scoped to pytorch
vllm.decoding         → scoped to vllm
tensorrt.quantization → scoped to tensorrt
```

In a multi-backend study (`backend: [pytorch, vllm, tensorrt]`):
- `pytorch.*` groups only apply to pytorch experiments
- `vllm.*` groups only apply to vllm experiments
- Universal groups (no backend prefix) apply to all backends

This mirrors the existing dotted-key routing in `_expand_sweep()` (grid.py:740-744). A group that mixes keys from multiple backends (e.g. `pytorch.*` and `vllm.*` in the same group) raises a `ConfigError` - that's never valid.

```python
def _group_backend_scope(group_key: str) -> str | None:
    """Return backend name if group is scoped, else None (universal)."""
    if "." in group_key:
        prefix = group_key.split(".", 1)[0]
        if prefix in _BACKEND_SECTION_KEYS:
            return prefix
    return None
```

### Virtual Group Names

Group keys like `pytorch.compile` and `tensorrt.quantization` are **virtual paths** - they don't correspond to real config fields. They exist only for:
- **Display:** preflight panel shows "pytorch.compile (4 variants)"
- **Scoping:** the prefix determines which backend the group applies to
- **Logging:** skipped config messages include group provenance

The expansion logic must distinguish virtual group keys from real config paths. The type-based check (`_is_group()`) handles this: if the value is a list-of-dicts, the key is a group name, not a config path.

**Collision avoidance:** Group names must not collide with real dotted config paths. In practice this is unlikely - `pytorch.compile` is not a real field (the real field is `pytorch.torch_compile`), and `pytorch.quantization` is not a real field either. If a collision occurs, the type-based check resolves it: a list-of-dicts is always a group.

### Expansion Algorithm

```
1. Separate sweep entries into independent axes and groups:
   - _is_group(value) → group (list of dicts)
   - else → independent axis (list of scalars)

2. For each group, expand entries into mini-grids:
   - Each entry's list-valued fields produce a Cartesian product within the entry
   - Single-valued fields are constants for that entry
   - {} means "no overrides" (baseline variant)
   - Resolve keys relative to group's backend prefix

3. Determine backend scope for each group:
   - pytorch.compile → scoped to pytorch
   - vllm.decoding → scoped to vllm
   - (no prefix) → universal

4. For each backend in the study:
   a. Collect applicable groups (scoped to this backend + universal groups)
   b. Cross all applicable group variants with each other:
      groups_product = product(group_A_variants, group_B_variants, ...)
   c. Collect applicable independent axes (universal + this backend's scoped axes)
   d. Cross independent axes:
      axes_product = product(axis_A_values, axis_B_values, ...)
   e. Cross groups with axes:
      backend_configs = product(groups_product, axes_product)
   f. For each config, merge with fixed defaults

5. Validate each raw config via Pydantic:
   - Valid → ExperimentConfig
   - Invalid → SkippedConfig (Pydantic safety net)
```

### Preflight Display

The Rich preflight panel shows the maths transparently:

```
┌─ Study: full-suite-all-backends ────────────────────────────────┐
│                                                                 │
│  ── Sweep Axes ──────────────────────────────────────────────── │
│  dtype:                  float32, float16, bfloat16      (×3)   │
│  pytorch.batch_size:     1, 4, 8, 16, 32                (×5)   │
│  pytorch.attn:           sdpa, fa2, fa3, eager           (×4)   │
│                                                                 │
│  ── Sweep Groups ────────────────────────────────────────────── │
│  pytorch.compile (4 variants):                           (×4)   │
│    · torch_compile=false                                        │
│    · torch_compile=true, mode=default                           │
│    · torch_compile=true, mode=reduce-overhead                   │
│    · torch_compile=true, mode=max-autotune                      │
│  pytorch.quantization (5 variants):                      (×5)   │
│    · (none)                                                     │
│    · load_in_8bit=true                                          │
│    · load_in_4bit, quant_type=nf4                               │
│    · load_in_4bit, quant_type=fp4                               │
│    · load_in_4bit, quant_type=nf4, double_quant                 │
│                                                                 │
│  ── Totals ──────────────────────────────────────────────────── │
│  Sweep:    3 dtype × 5 batch × 4 attn               = 60       │
│  Groups:   4 compile × 5 quant × 2 cache × 3 decode = 120      │
│  PyTorch:  60 × 120 = 7,200 experiments                        │
│  All backends: 7,200 + 1,944 vllm + 360 trt = 9,504            │
│  × 3 cycles = 28,512 total runs                                │
│                                                                 │
│  Skipped:  23/9,527 (0.2%)                                     │
│                                                                 │
│  Design hash: a1b2c3d4e5f6g7h8                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Combinatorial Explosion Warnings

Tiered warnings protect against accidentally-huge grids:

```python
total_before_cycles = len(valid_experiments)
total_after_cycles = total_before_cycles * n_cycles
min_runtime_hours = total_after_cycles * gap_seconds / 3600  # lower bound

if total_before_cycles > 100:
    log.info(f"Large study: {total_before_cycles} experiments")

if total_before_cycles > 500:
    log.warning(
        f"Very large study: {total_before_cycles} experiments "
        f"({total_after_cycles} total runs with {n_cycles} cycles). "
        f"Minimum runtime: ~{min_runtime_hours:.0f}h (gap time only). "
        f"Consider reducing sweep dimensions or groups."
    )

if total_before_cycles > 2000:
    log.warning(
        f"Extremely large study: {total_before_cycles} experiments "
        f"({total_after_cycles} total runs). "
        f"Pass --yes to confirm or reduce dimensions."
    )
    # CLI: require --yes flag or interactive confirmation
    # API: log.warning only (caller controls execution)
```

The existing >50% skip warning is orthogonal (catches misconfigured sweep axes). The new warning catches accidentally-huge *valid* grids.

Runtime estimate uses `study_execution.experiment_gap_seconds` as a lower bound - we can't predict experiment duration, but gap time alone gives a floor.

### Implementation Scope

1. **`config/grid.py` (`_expand_sweep`)**: Add `_is_group()` check. Separate groups from independent axes. Add `_expand_group()` for mini-grid expansion within groups. Cross groups with axes per backend. ~100 lines new code.

2. **`config/grid.py` (`_resolve_group_key`)**: New helper for relative key resolution within group entries. ~15 lines.

3. **`config/grid.py` (`build_preflight_panel`)**: Add "Sweep Groups" section showing variants per group. Add "Totals" section with maths breakdown. ~40 lines.

4. **`config/grid.py` (`expand_grid`)**: Add combinatorial explosion warnings. ~15 lines.

5. **`study-full-suite.yaml`**: Rewrite to use groups for dependent params. Move independent explicit experiments into `sweep:`.

6. **Tests**: New test class `TestExpandGridSweepGroups` covering: group detection, group expansion, group × sweep crossing, backend scoping, cross-section overrides, relative key resolution, empty groups, conflict resolution, group + explicit experiments, combinatorial warnings.

### What does NOT change

- `ExperimentConfig` and all Pydantic models: unchanged
- `_extract_fixed()`: unchanged
- Explicit `experiments:` section: still supported, still merged last
- `SkippedConfig`: still the safety net for invalid combos
- `apply_cycles()`, `compute_study_design_hash()`: unchanged
- Backend configs, validators, introspection: unchanged
- `config/loader.py`: unchanged (groups are inside `sweep:`, not a new top-level key)

---

## Worked Example: PyTorch Section of `study-full-suite.yaml`

### Before (current, 9 explicit experiments + limited sweep)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

experiments:
  # torch.compile: default — 12 lines
  - model: Qwen/Qwen2.5-0.5B
    backend: pytorch
    dtype: bfloat16
    dataset: { n_prompts: 50 }
    pytorch:
      batch_size: 4
      torch_compile: true
      torch_compile_mode: default
      torch_compile_backend: inductor
      device_map: auto
      trust_remote_code: true

  # torch.compile: reduce-overhead — 12 lines (copy-paste)
  # torch.compile: max-autotune — 12 lines (copy-paste)
  # BnB 8bit — 12 lines
  # BnB 4bit nf4 — 14 lines
  # BnB 4bit nf4 double-quant — 15 lines
  # Beam search — 15 lines
  # Speculative decoding — 12 lines
  # Static KV cache — 12 lines
  # Total: ~115 lines of explicit experiments
```

### After (groups, 1 explicit experiment)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

  pytorch.compile:
    - torch_compile: false
    - torch_compile: true
      torch_compile_mode: [default, reduce-overhead, max-autotune]
      torch_compile_backend: inductor

  pytorch.quantization:
    - {}
    - load_in_8bit: true
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: [nf4, fp4]
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4
      bnb_4bit_use_double_quant: true

  pytorch.cache:
    - {}
    - use_cache: true
      cache_implementation: static

  pytorch.decoding:
    - {}
    - decoder.do_sample: false
      decoder.temperature: 0.0
      num_beams: [2, 4]
      early_stopping: true
      length_penalty: 1.0
      no_repeat_ngram_size: 3

experiments:
  # Speculative decoding: requires batch_size=1, overriding the sweep axis
  - model: Qwen/Qwen2.5-0.5B
    backend: pytorch
    dtype: bfloat16
    dataset: { n_prompts: 50 }
    pytorch:
      batch_size: 1
      prompt_lookup_num_tokens: 3
      device_map: auto
```

**Result:** 9 explicit experiments (~115 lines) reduced to 1 (~8 lines). All dependent params properly constrained. Groups are crossed with independent sweep axes, producing a complete combinatorial grid without invalid configs.

## Alternatives Considered

See Options B (constraint expressions), C (hierarchical tree), and D (full Cartesian + auto-filter) above. Option A was chosen for its balance of expressiveness, efficiency, and backwards compatibility.

Option D (full Cartesian + filter) was specifically rejected due to ~96% waste in realistic configs, the `null`/`_omit` sentinel problem, silent failures wherever validators are missing, and unusable preflight output. See the "Option D" section for detailed analysis.
