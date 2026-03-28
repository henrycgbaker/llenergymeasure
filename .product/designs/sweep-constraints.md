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

### Phased approach

This work is split into two phases:

**Phase 1 (immediate, separate PR):** Move all genuinely independent explicit experiments into `sweep:`. Zero new code in `grid.py`, zero risk. See "Phase 1: Immediate Cleanup" section below.

**Phase 2 (future PR):** Implement sweep groups for truly dependent parameters. This is the main design in this document.

### Impact analysis: what's independent vs truly dependent?

| Category | Current explicit experiments | Independent (move to `sweep:` now) | Truly dependent (need groups) |
|---|---|---|---|
| PyTorch | 9 | 1 (static KV cache) | 8 (compile, quantisation, beam search, speculative) |
| vLLM | ~10 | ~6 (prefix caching, block sizes, KV dtype, TP) | ~4 (beam search, sampling penalties) |
| TensorRT | ~15 | ~6 (scheduler, KV cache, fast build, dtype) | ~9 (quantisation variants with calib) |
| **Total** | **~34** | **~13** | **~21** |

Phase 1 alone eliminates ~40% of the verbosity. The remaining ~21 truly dependent experiments are what this design addresses.

---

## Phase 1: Immediate Cleanup

Move all genuinely independent explicit experiments into `sweep:` axes. No new code, no new concepts - just YAML restructuring.

### What moves into `sweep:`

These explicit experiments have **no parameter dependencies** - they're single-axis variations that the existing Cartesian product handles perfectly:

**vLLM:**
- Prefix caching on/off → `vllm.engine.enable_prefix_caching: [true, false]`
- Block sizes → `vllm.engine.block_size: [8, 32]` (add to existing or create axis)
- KV cache dtype fp8 → `vllm.engine.kv_cache_dtype: [auto, fp8]`
- Tensor parallel → `vllm.engine.tensor_parallel_size: [1, 2]` (if we want to sweep it)

**TensorRT:**
- Scheduler policies → `tensorrt.scheduler.capacity_scheduling_policy: [MAX_UTILIZATION, STATIC_BATCH]` (new axis, or keep as explicit if only 2 values)
- KV cache block reuse → could become axis but has compound settings (block_reuse + free_gpu_memory_fraction) - **keep as explicit** for now
- Fast build mode → single-value flag, not a sweep axis - **keep as explicit**
- Explicit dtype float16 → covered by top-level `dtype` sweep axis already - **remove duplicate**

**PyTorch:**
- Static KV cache → `pytorch.use_cache: [true, false]` + `pytorch.cache_implementation: [static, null]` - but these are **dependent** (`cache_implementation` requires `use_cache: true`). **Keep as explicit** or defer to groups.

### What stays as explicit experiments (Phase 1)

Everything with parameter dependencies:
- torch.compile variants (mode requires compile: true)
- BitsAndBytes quantisation (4bit sub-params require load_in_4bit: true)
- Beam search (requires decoder.do_sample: false)
- Speculative decoding (requires batch_size: 1)
- vLLM beam search / sampling penalties (mutually exclusive sections)
- TensorRT quantisation + calibration (calib requires quant_algo)
- TensorRT KV cache compound settings (block_reuse + memory fraction travel together)

### Phase 1 scope

- Edit `study-full-suite.yaml` only
- Move independent params into existing or new `sweep:` axes
- Remove the now-redundant explicit experiment entries
- No code changes to `grid.py` or any Python files

---

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
| `tensorrt.quant.kv_cache_quant_algo` requires `quant_algo` | `TensorRTQuantConfig.validate_kv_cache_quant_requires_quant()` |

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

Where this could break: a user puts `tensorrt.calib.calib_batches` as an independent axis without any quant group. This produces configs where calib is set but quant is null - semantically wrong. The fix is **missing validators** (above), not sweep-level graph machinery.

### What remains as explicit experiments

After groups, the `experiments:` section is reserved for truly unique one-off experiments that:
- Override sweep axes for specific configs (e.g. `batch_size: 1` for speculative decoding)
- Use parameters not in any sweep axis (e.g. `lora.adapter_id` pointing to a specific model)
- Combine settings that span multiple unrelated groups in unusual ways

This should reduce `experiments:` from ~40 entries to ~5-10.

---

## Open Design Decisions

The core concept (groups = unions of variants, crossed with independent axes) is solid. The remaining decisions are about syntax and how much implicit behaviour to allow. Each decision is independent.

### Decision 1: Where do groups live? (`sweep:` inline vs `sweep_groups:` top-level key)

#### Option 1A: Inline under `sweep:` (original proposal)

Groups and independent axes share the same `sweep:` block. Type-based disambiguation tells them apart.

```yaml
sweep:
  # Independent axis (list of scalars)
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]

  # Group (list of dicts) - same block, different value type
  pytorch.compile:
    - torch_compile: false
    - torch_compile: true
      torch_compile_mode: default
    - torch_compile: true
      torch_compile_mode: reduce-overhead
    - torch_compile: true
      torch_compile_mode: max-autotune
```

**Implications:**
- No schema change to `_STUDY_ONLY_KEYS` - groups are inside `sweep:`, which is already excluded
- `config/loader.py` unchanged - no new top-level key to parse or validate
- `_expand_sweep()` must partition its input: `{k: v for k, v in sweep.items() if _is_group(v)}` vs not
- Detection function: `_is_group(value) = isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict)`
- Edge case: `pytorch.something: [{"a": 1}]` - is this a group with one variant or a scalar axis whose value is a dict? The rule says group. This is fine in practice (no sweep axis would have dict values) but it's an implicit contract
- User must mentally parse "is this a list-of-scalars or list-of-dicts?" to understand behaviour
- Slightly cleaner YAML (one less top-level key)

#### Option 1B: Separate `sweep_groups:` top-level key (proposed alternative)

Groups live in their own top-level section. No type-based disambiguation needed - everything under `sweep:` is an independent axis, everything under `sweep_groups:` is a group.

```yaml
sweep:
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]

sweep_groups:
  pytorch.compile:
    - torch_compile: false
    - torch_compile: true
      torch_compile_mode: default
    - torch_compile: true
      torch_compile_mode: reduce-overhead
    - torch_compile: true
      torch_compile_mode: max-autotune
```

**Implications:**
- Must add `"sweep_groups"` to `_STUDY_ONLY_KEYS` (trivial)
- `config/loader.py` must extract `raw_study.get("sweep_groups", {})` and pass it to the expansion logic
- `_expand_sweep()` gains a second parameter `groups: dict[str, list[dict]]` (or a new `_expand_groups()`)
- No ambiguity: `sweep:` values are always list-of-scalars; `sweep_groups:` values are always list-of-dicts
- Self-documenting: a new user scanning the YAML immediately sees "these are independent axes, those are dependent groups"
- The YAML is slightly longer (one extra top-level key) but the intent is clearer
- `compute_study_design_hash()` must include `sweep_groups` in the hash (currently only hashes `sweep:`)
- Preflight panel naturally separates into "Sweep Axes" and "Sweep Groups" sections (the display already proposed this separation regardless of where groups live)

**Recommendation:** Option 1B (`sweep_groups:` top-level key). The self-documenting nature outweighs the minimal schema change. Type-based disambiguation works but adds cognitive load. The YAML should make intent explicit rather than relying on value-type inspection.

---

### Decision 2: Mini-grids within group entries? (list-valued fields in entries)

#### Option 2A: Allow mini-grids (original proposal)

List-valued fields within a group entry produce a Cartesian product *within that entry*:

```yaml
sweep_groups:
  pytorch.quantization:
    - {}                                    # 1 variant: baseline
    - load_in_8bit: true                    # 1 variant: 8-bit
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: [nf4, fp4]      # 2 variants (mini-grid: nf4, fp4)
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4
      bnb_4bit_use_double_quant: true       # 1 variant: double-quant
  # Total: 1 + 1 + 2 + 1 = 5 variants
```

**Implications:**
- Two levels of Cartesian product: within-entry (mini-grid) and between-groups (group crossing)
- `_expand_group()` must detect list-valued fields, compute their product, and flatten
- The maths becomes harder to predict mentally: "this entry has 2 list fields with 3 and 4 values → 12 variants from one entry"
- More compact YAML for entries that differ by only one axis
- For the TensorRT calibration case, this is powerful:
  ```yaml
  - quant.quant_algo: INT8
    calib.calib_batches: [256, 512]            # 2
    calib.calib_max_seq_length: [256, 512]     # × 2 = 4 variants from one entry
  ```
- Risk: a user puts `bnb_4bit_quant_type: [nf4, fp4]` intending a list value, not a mini-grid. Unlikely for enum fields but possible for free-form fields

#### Option 2B: Flat entries only (no mini-grids)

Each group entry is exactly one variant. List-valued fields are literal lists (passed as-is), never expanded:

```yaml
sweep_groups:
  pytorch.quantization:
    - {}                                    # baseline
    - load_in_8bit: true                    # 8-bit
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4              # 4-bit nf4
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: fp4              # 4-bit fp4
    - load_in_4bit: true
      bnb_4bit_compute_dtype: float16
      bnb_4bit_quant_type: nf4
      bnb_4bit_use_double_quant: true       # double-quant
  # Total: 5 variants (same count, one more line)
```

**Implications:**
- One level of Cartesian product only (between-groups). Each entry = exactly one config overlay
- `_expand_group()` is trivial: each entry becomes one variant dict, no product logic
- Combinatorial maths is dead simple: count the list items, that's your variant count
- Slightly more verbose for the TensorRT calibration case:
  ```yaml
  - quant.quant_algo: INT8
    calib.calib_batches: 256
    calib.calib_max_seq_length: 256
  - quant.quant_algo: INT8
    calib.calib_batches: 256
    calib.calib_max_seq_length: 512
  - quant.quant_algo: INT8
    calib.calib_batches: 512
    calib.calib_max_seq_length: 256
  - quant.quant_algo: INT8
    calib.calib_batches: 512
    calib.calib_max_seq_length: 512
  # 4 entries instead of 1 with mini-grid
  ```
- No ambiguity about whether a list is "expand this" or "this is literally a list value"
- If a group needs many sub-combinations, the user can always put those axes into `sweep:` as independent axes instead (if they're genuinely independent within the group)

**Recommendation:** Option 2B (flat entries only). The simpler mental model is worth the slight verbosity. Mini-grids add a second layer of implicit Cartesian expansion that makes the total experiment count harder to predict. The TensorRT calibration case (4 entries vs 1) is the strongest argument for mini-grids, but 4 explicit entries is still clearer than implicit expansion. If mini-grids prove necessary later, they can be added as a backwards-compatible enhancement.

---

### Decision 3: How do cross-section overrides work within groups?

Groups are scoped to a backend (e.g. `pytorch.decoding` → pytorch scope). But some group entries need to override fields outside the backend section (e.g. beam search requires `decoder.do_sample: false`).

#### Option 3A: Implicit detection via prefix matching (original proposal)

Keys within group entries are resolved relative to the group's backend prefix. If a dotted key's prefix doesn't match the backend or a known backend name, it's treated as fully-qualified:

```yaml
sweep_groups:
  pytorch.decoding:
    - {}
    - decoder.do_sample: false              # "decoder" ≠ "pytorch" → fully-qualified
      decoder.temperature: 0.0              # same rule
      num_beams: 4                          # no dot → pytorch.num_beams (relative)
      early_stopping: true                  # no dot → pytorch.early_stopping (relative)
```

**Resolution rules:**
```
Group scope: pytorch
"decoder.do_sample"    → prefix "decoder" ≠ "pytorch" AND ∉ {pytorch, vllm, tensorrt}
                       → fully-qualified: decoder.do_sample ✓
"num_beams"            → no dot → relative: pytorch.num_beams ✓
"early_stopping"       → no dot → relative: pytorch.early_stopping ✓
"engine.max_num_seqs"  → prefix "engine" ... but what if inside vllm.decoding group?
                       → "engine" ≠ "vllm" AND ∉ {pytorch, vllm, tensorrt}
                       → treated as fully-qualified: engine.max_num_seqs ✗ (wrong! should be vllm.engine.max_num_seqs)
```

**Implications:**
- Works correctly for the common case (`decoder.*` cross-section overrides)
- Breaks for sub-sections within the same backend: `engine.max_num_seqs` inside a `vllm.*` group is relative but the prefix check doesn't know `engine` is a vLLM sub-section
- The resolution function needs a list of "known sub-section prefixes per backend" to work correctly - or it must only treat a set of known top-level sections (decoder, dataset, warmup, baseline) as cross-section
- Fragile: adding a new top-level config section (e.g. `traffic:`) requires updating the resolution logic

#### Option 3B: Explicit `@` prefix for cross-section overrides

Any key that should be treated as fully-qualified (not relative to the group's backend) must be prefixed with `@`:

```yaml
sweep_groups:
  pytorch.decoding:
    - {}
    - "@decoder.do_sample": false           # @ = fully-qualified
      "@decoder.temperature": 0.0           # @ = fully-qualified
      num_beams: 4                          # no @ = relative to pytorch
      early_stopping: true                  # no @ = relative to pytorch
```

**Implications:**
- Unambiguous: `@` means "this is a fully-qualified path, don't prepend the backend"
- No prefix-matching logic needed - just check for `@` prefix and strip it
- Slightly noisier YAML (extra `@` characters, YAML requires quoting keys starting with `@`)
- Actually - YAML keys with `@` must be quoted: `"@decoder.do_sample": false`. This is ugly but explicit
- Users must remember to use `@` for cross-section overrides. Forgetting it produces `pytorch.decoder.do_sample` which fails Pydantic validation with a clear error ("unknown field `decoder` in PyTorchConfig") - so the failure mode is good

#### Option 3C: Fully-qualified keys always (no relative resolution)

All keys within group entries are always fully-qualified. No relative resolution, no prefix magic:

```yaml
sweep_groups:
  pytorch.decoding:
    - {}
    - decoder.do_sample: false              # fully-qualified
      decoder.temperature: 0.0              # fully-qualified
      pytorch.num_beams: 4                  # fully-qualified (must include "pytorch.")
      pytorch.early_stopping: true          # fully-qualified
```

**Implications:**
- Zero ambiguity, zero magic - every key means exactly what it says
- More verbose: backend-scoped keys must repeat the backend prefix (`pytorch.num_beams` not `num_beams`)
- The group key prefix (`pytorch.decoding`) is used only for backend scoping and display, never for key resolution
- Implementation is simplest: keys are used as-is, routed through the existing `_unflatten()` + `deep_merge()` logic
- No edge cases to handle
- The group key prefix (`pytorch.`) already tells you the backend scope. Repeating `pytorch.` in keys is redundant but consistent with how `sweep:` axes work today (`pytorch.batch_size: [1, 4, 8]`)

**Recommendation:** Option 3C (fully-qualified always). The current `sweep:` section already uses fully-qualified paths everywhere (`pytorch.batch_size`, `vllm.engine.gpu_memory_utilization`). Groups should follow the same convention. The slight redundancy (`pytorch.` prefix on keys within a `pytorch.*` group) is consistent with how the rest of the YAML works and eliminates an entire class of edge cases. The implementation is also the simplest - no `_resolve_group_key()` function needed at all.

---

### Decision 4: Virtual group name collision avoidance

Group keys like `pytorch.compile` and `tensorrt.quantization` are virtual paths that don't correspond to real config fields. How do we prevent collisions with real paths?

#### Current state: rely on `_is_group()` type check

If a group name happens to collide with a real config path, the type check (`list-of-dicts` = group) resolves it. But this creates a confusing situation where the same key has different semantics depending on its value type.

**Known near-misses:**
- `pytorch.cache` (group name) vs `pytorch.use_cache` (real field) - different enough
- `pytorch.compile` (group name) vs `pytorch.torch_compile` (real field) - different enough
- `tensorrt.quant` (group name) vs `tensorrt.quant.quant_algo` (real nested path) - **potential collision** if someone writes `tensorrt.quant: [INT8, W4A16_AWQ]` as an independent axis

#### Mitigation: naming convention

Group names should be **abstract nouns** that describe the concern, not concrete parameter names:
- `pytorch.compilation` not `pytorch.torch_compile`
- `pytorch.quantization` not `pytorch.load_in_4bit`
- `tensorrt.quant_config` not `tensorrt.quant`

With Option 1B (`sweep_groups:` top-level key), this is less of a concern because groups and axes live in separate sections. A key in `sweep:` is always an axis; a key in `sweep_groups:` is always a group name. Collision is only possible if the same key appears in both sections, which can be detected and rejected at parse time.

**Recommendation:** Use abstract nouns for group names, and with Option 1B the collision risk is minimal. If both sections contain the same key, raise a `ConfigError`.

---

## Detailed Design

> **Note:** The detailed design below reflects the recommended decisions: **1B** (separate `sweep_groups:`), **2B** (flat entries), **3C** (fully-qualified keys), plus abstract group names.

### YAML Schema

```yaml
model: Qwen/Qwen2.5-0.5B

sweep:
  # ── Independent axes (list of scalars → Cartesian product) ──────────
  dtype: [float32, float16, bfloat16]

  # ── PyTorch: independent axes ───────────────────────────────────────
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

  # ── vLLM: independent axes ─────────────────────────────────────────
  vllm.engine.gpu_memory_utilization: [0.7, 0.85, 0.95]
  vllm.engine.enforce_eager: [true, false]
  vllm.engine.max_num_seqs: [64, 128, 256]
  vllm.engine.enable_chunked_prefill: [true, false]
  vllm.engine.enable_prefix_caching: [true, false]
  vllm.engine.block_size: [8, 16, 32]
  vllm.engine.kv_cache_dtype: [auto, fp8]

  # ── TensorRT: independent axes ─────────────────────────────────────
  tensorrt.max_batch_size: [1, 4, 8, 16, 32]
  tensorrt.max_seq_len: [1536, 2048]

sweep_groups:
  # ── PyTorch: dependent groups (list of dicts → union of variants) ───
  pytorch.compilation:
    - pytorch.torch_compile: false
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: default
      pytorch.torch_compile_backend: inductor
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: reduce-overhead
      pytorch.torch_compile_backend: inductor
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: max-autotune
      pytorch.torch_compile_backend: inductor

  pytorch.quantization:
    - {}                                          # baseline: no quantisation
    - pytorch.load_in_8bit: true
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: fp4
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
      pytorch.bnb_4bit_use_double_quant: true

  pytorch.caching:
    - {}
    - pytorch.use_cache: true
      pytorch.cache_implementation: static

  pytorch.decoding:
    - {}                                          # baseline: use shared decoder settings
    - decoder.do_sample: false
      decoder.temperature: 0.0
      pytorch.num_beams: 4
      pytorch.early_stopping: true
      pytorch.length_penalty: 1.0
      pytorch.no_repeat_ngram_size: 3

  # ── vLLM: dependent groups ─────────────────────────────────────────
  vllm.decoding:
    - {}
    - decoder.do_sample: false
      decoder.temperature: 0.0
      vllm.beam_search.beam_width: 4
      vllm.beam_search.early_stopping: true
    - vllm.sampling.presence_penalty: 0.6
      vllm.sampling.frequency_penalty: 0.6

  # ── TensorRT: dependent groups ─────────────────────────────────────
  tensorrt.quant_config:
    - {}                                          # baseline: no quantisation
    - tensorrt.quant.quant_algo: INT8
    - tensorrt.quant.quant_algo: INT8
      tensorrt.quant.kv_cache_quant_algo: INT8
    - tensorrt.quant.quant_algo: INT8
      tensorrt.calib.calib_batches: 512
      tensorrt.calib.calib_dataset: cnn_dailymail
      tensorrt.calib.calib_max_seq_length: 512
    - tensorrt.quant.quant_algo: W4A16_AWQ
    - tensorrt.quant.quant_algo: W8A16

  tensorrt.scheduling:
    - {}
    - tensorrt.scheduler.capacity_scheduling_policy: MAX_UTILIZATION
    - tensorrt.scheduler.capacity_scheduling_policy: STATIC_BATCH

  tensorrt.kv_cache_config:
    - {}
    - tensorrt.kv_cache.enable_block_reuse: true
      tensorrt.kv_cache.free_gpu_memory_fraction: 0.9
    - tensorrt.kv_cache.host_cache_size: 1073741824
```

### Group Detection and Backend Scoping

With `sweep_groups:` as a separate top-level key, no type-based disambiguation is needed. Every value under `sweep_groups:` is a group (list of dicts). Every value under `sweep:` is an independent axis (list of scalars).

Backend scoping uses the same prefix extraction as `_expand_sweep()`:

```python
def _group_backend_scope(group_key: str) -> str | None:
    """Return backend name if group is scoped, else None (universal)."""
    if "." in group_key:
        prefix = group_key.split(".", 1)[0]
        if prefix in _BACKEND_SECTION_KEYS:
            return prefix
    return None
```

### Key Resolution Within Groups (fully-qualified, Decision 3C)

No resolution function needed. Keys within group entries are used as-is, routed through the existing `_unflatten()` + `deep_merge()` logic. The same code path that handles `pytorch.batch_size` in `_expand_sweep()` handles `pytorch.torch_compile` in group entries.

```python
# For each group entry (a dict of fully-qualified keys):
for fq_key, value in entry.items():
    if "." in fq_key:
        prefix, param = fq_key.split(".", 1)
        if prefix in _BACKEND_SECTION_KEYS:
            # Backend-scoped: pytorch.num_beams → config["pytorch"]["num_beams"]
            backend_dict = config_dict.get(prefix, {})
            nested_update = _unflatten({param: value})
            config_dict[prefix] = deep_merge(backend_dict, nested_update)
        else:
            # Non-backend dotted key: decoder.do_sample → config["decoder"]["do_sample"]
            nested_update = _unflatten({fq_key: value})
            config_dict = deep_merge(config_dict, nested_update)
    else:
        # Simple top-level key
        config_dict[fq_key] = value
```

This is nearly identical to the existing loop in `_expand_sweep()` (lines 790-803 of `grid.py`). The code can be extracted into a shared helper.

### Expansion Algorithm

```
1. Parse sweep: and sweep_groups: from study YAML
   - sweep → independent axes (dict[str, list[scalar]])
   - sweep_groups → groups (dict[str, list[dict]])

2. Separate independent axes into universal and backend-scoped:
   - (identical to current _expand_sweep() logic)

3. For each group, determine backend scope:
   - pytorch.compilation → scoped to pytorch
   - vllm.decoding → scoped to vllm
   - (no backend prefix) → universal

4. For each backend in the study:
   a. Collect applicable groups (scoped to this backend + universal)
   b. Each group is a list of variant dicts (no mini-grid expansion needed)
   c. Cross all applicable group variants with each other:
      groups_product = product(group_A_variants, group_B_variants, ...)
   d. Collect applicable independent axes (universal + this backend's scoped)
   e. Cross independent axes:
      axes_product = product(axis_A_values, axis_B_values, ...)
   f. Cross groups with axes:
      backend_configs = product(groups_product, axes_product)
   g. For each config, merge: fixed defaults ← axes overlay ← group overlay

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
│  pytorch.compilation (4 variants):                       (×4)   │
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
│  Groups:   4 compile × 5 quant × 2 cache × 2 decode = 80       │
│  PyTorch:  60 × 80 = 4,800 experiments                         │
│  All backends: 4,800 + 1,944 vllm + 360 trt = 7,104            │
│  × 3 cycles = 21,312 total runs                                │
│                                                                 │
│  Skipped:  12/7,116 (0.2%)                                     │
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

1. **`config/grid.py` (`_expand_sweep`)**: Refactor to accept optional `groups` parameter. Add `_expand_groups()` for group variant expansion (trivial with flat entries). Cross groups with axes per backend. ~80 lines new code.

2. **`config/grid.py` (`_group_backend_scope`)**: New helper for group backend scoping. ~8 lines.

3. **`config/grid.py` (`build_preflight_panel`)**: Add "Sweep Groups" section showing variants per group. Add "Totals" section with maths breakdown. ~40 lines.

4. **`config/grid.py` (`expand_grid`)**: Extract `sweep_groups` from raw study, pass to expansion. Add combinatorial explosion warnings. ~20 lines.

5. **`_STUDY_ONLY_KEYS`**: Add `"sweep_groups"`.

6. **`compute_study_design_hash()`**: Include `sweep_groups` in the hash input.

7. **`study-full-suite.yaml`**: Rewrite to use `sweep_groups:` for dependent params. Move independent explicit experiments into `sweep:`.

8. **Tests**: New test class `TestExpandGridSweepGroups` covering: group expansion, group × sweep crossing, backend scoping, fully-qualified key routing, empty groups (`{}`), group + explicit experiments, combinatorial warnings, hash stability.

### What does NOT change

- `ExperimentConfig` and all Pydantic models: unchanged
- `_extract_fixed()`: unchanged
- Explicit `experiments:` section: still supported, still merged last
- `SkippedConfig`: still the safety net for invalid combos
- `apply_cycles()`: unchanged
- Backend configs, validators, introspection: unchanged
- `config/loader.py`: minimal change (extract `sweep_groups` key)

---

## Worked Example: PyTorch Section of `study-full-suite.yaml`

### Before (current, 9 explicit experiments + limited sweep)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

experiments:
  # torch.compile: default - 12 lines
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

  # torch.compile: reduce-overhead - 12 lines (copy-paste)
  # torch.compile: max-autotune - 12 lines (copy-paste)
  # BnB 8bit - 12 lines
  # BnB 4bit nf4 - 14 lines
  # BnB 4bit nf4 double-quant - 15 lines
  # Beam search - 15 lines
  # Speculative decoding - 12 lines
  # Static KV cache - 12 lines
  # Total: ~115 lines of explicit experiments
```

### After (sweep_groups, 1 explicit experiment)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

sweep_groups:
  pytorch.compilation:
    - pytorch.torch_compile: false
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: default
      pytorch.torch_compile_backend: inductor
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: reduce-overhead
      pytorch.torch_compile_backend: inductor
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: max-autotune
      pytorch.torch_compile_backend: inductor

  pytorch.quantization:
    - {}
    - pytorch.load_in_8bit: true
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: fp4
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
      pytorch.bnb_4bit_use_double_quant: true

  pytorch.caching:
    - {}
    - pytorch.use_cache: true
      pytorch.cache_implementation: static

  pytorch.decoding:
    - {}
    - decoder.do_sample: false
      decoder.temperature: 0.0
      pytorch.num_beams: 4
      pytorch.early_stopping: true
      pytorch.length_penalty: 1.0
      pytorch.no_repeat_ngram_size: 3

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

### Maths walkthrough

```
Independent axes (pytorch):
  3 dtype × 5 batch_size × 4 attn_implementation = 60

Groups (pytorch):
  pytorch.compilation:  4 variants (1 false + 3 true×mode)
  pytorch.quantization: 5 variants (1 none + 1 8bit + 2 4bit + 1 double)
  pytorch.caching:      2 variants (1 none + 1 static)
  pytorch.decoding:     2 variants (1 none + 1 beam_search)
  Groups crossed: 4 × 5 × 2 × 2 = 80

Total pytorch: 60 × 80 = 4,800 experiments
× 3 cycles = 14,400 runs
```

## Missing validators to add (separate PR)

| Dependency | Proposed validator |
|---|---|
| `tensorrt.calib.*` requires `tensorrt.quant.quant_algo` | `TensorRTConfig.validate_calib_requires_quant()` |
| `tensorrt.quant.kv_cache_quant_algo` requires `quant_algo` | `TensorRTQuantConfig.validate_kv_cache_quant_requires_quant()` |

These strengthen Pydantic as the safety net independent of the sweep design.

## Alternatives Considered

See Options B (constraint expressions), C (hierarchical tree), and D (full Cartesian + auto-filter) above. Option A was chosen for its balance of expressiveness, efficiency, and backwards compatibility.

Option D (full Cartesian + filter) was specifically rejected due to ~96% waste in realistic configs, the `null`/`_omit` sentinel problem, silent failures wherever validators are missing, and unusable preflight output. See the "Option D" section for detailed analysis.
