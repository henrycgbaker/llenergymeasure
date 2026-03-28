# Design: Sweep Constraints and Dependent Parameters

**Date:** 2026-03-28
**Status:** Draft
**Scope:** Config grid expansion (`config/grid.py`), study YAML schema, `study-full-suite.yaml`
**Problem:** The sweep system's Cartesian product cannot express parameter dependencies, forcing users to duplicate experiments as explicit entries.

## Context

The current sweep system (`_expand_sweep()` in `grid.py`) generates the Cartesian product of all sweep dimensions. This works for independent axes but breaks down when parameters have dependencies:

- `torch_compile_mode` only makes sense when `torch_compile: true`
- `bnb_4bit_*` sub-params only apply when `load_in_4bit: true`
- `vllm.beam_search` and `vllm.sampling` are mutually exclusive sections
- beam search params require `decoder.do_sample: false`

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

### Conditional Requirements (B requires A to be set)

| Dependent param (B) | Requires (A) | Validator |
|---|---|---|
| `pytorch.torch_compile_mode` | `pytorch.torch_compile: true` | `validate_torch_compile_options()` |
| `pytorch.torch_compile_backend` | `pytorch.torch_compile: true` | `validate_torch_compile_options()` |
| `pytorch.bnb_4bit_compute_dtype` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.bnb_4bit_quant_type` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.bnb_4bit_use_double_quant` | `pytorch.load_in_4bit: true` | `validate_bnb_4bit_options()` |
| `pytorch.cache_implementation` | `pytorch.use_cache: true\|null` | `validate_cache_options()` |
| `vllm.engine.speculative_model` | `vllm.engine.num_speculative_tokens` | `validate_speculative()` |

### Cross-Section Dependencies (not expressed as validators today)

| Scenario | Constraint |
|---|---|
| Beam search (pytorch) | Requires `decoder.do_sample: false`, `decoder.temperature: 0.0` |
| Beam search (vllm) | Requires `decoder.do_sample: false`, `decoder.temperature: 0.0` |
| Speculative decoding (pytorch) | Works only with `pytorch.batch_size: 1` |

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

2. **Declarative hierarchical tree** (NNI): A discriminator choice determines which sub-parameters exist. Structurally prevents invalid combos. Closest to what we need.

3. **Post-filtering** (hydra-filter-sweeper): Generate full Cartesian product, then exclude invalid combos via expressions. Simple but wasteful for large grids.

4. **Separation of concerns** (lm-eval): Backend params never enter a shared grid. We already do this via dotted-key routing, but dependencies *within* a backend remain unresolved.

---

## Options

### Option A: Sweep Groups (`sweep_groups`)

Add a `sweep_groups:` section for sets of parameters that must be swept as a unit (not crossed independently). The existing `sweep:` section continues to handle independent axes.

```yaml
sweep:
  # Independent axes - Cartesian product as today
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, eager]

sweep_groups:
  # Each group produces N experiments (one per list entry)
  # NOT crossed with each other, but crossed with independent sweep axes

  # torch.compile modes (only valid when torch_compile: true)
  - pytorch.torch_compile: true
    pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune]
    pytorch.torch_compile_backend: inductor

  # Standalone: torch_compile disabled
  - pytorch.torch_compile: false

  # BitsAndBytes 8-bit
  - pytorch.load_in_8bit: true

  # BitsAndBytes 4-bit variants
  - pytorch.load_in_4bit: true
    pytorch.bnb_4bit_compute_dtype: float16
    pytorch.bnb_4bit_quant_type: [nf4, fp4]

  # BitsAndBytes 4-bit with double quantisation
  - pytorch.load_in_4bit: true
    pytorch.bnb_4bit_compute_dtype: float16
    pytorch.bnb_4bit_quant_type: nf4
    pytorch.bnb_4bit_use_double_quant: true
```

**Semantics:**
- Each group entry generates its own mini-grid (Cartesian product of lists within the entry)
- Group entries are **unioned** (not crossed with each other)
- The union of all group entries **is crossed** with the independent `sweep:` axes
- Entries with a single value (not a list) are constants for that group

**Expansion example:**

```
sweep: 2 dtype x 5 batch_size x 3 attn = 30 base combos

sweep_groups (torch_compile):
  Group 1: torch_compile=true x 3 modes x backend=inductor = 3 combos
  Group 2: torch_compile=false = 1 combo
  → 4 group combos total

Total: 30 base x 4 torch_compile = 120 pytorch experiments
```

**Pros:**
- Declarative, YAML-native, no new expression language
- Dependencies expressed structurally (child params live under their parent's group)
- Existing `sweep:` unchanged - fully backwards compatible
- Eliminates ~80% of explicit experiments
- Intuitive mental model: "these params travel together"

**Cons:**
- New top-level key (`sweep_groups`)
- Users must understand group x sweep interaction
- Cannot express arbitrary boolean constraints (only structural grouping)
- Groups within the same "dimension" (e.g. all quantisation groups) need discipline to avoid overlap

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
    # Drop torch_compile_mode when torch_compile is false
    - pytorch.torch_compile == false and pytorch.torch_compile_mode != null
    # Drop 4bit + 8bit combos
    - pytorch.load_in_4bit == true and pytorch.load_in_8bit == true

  require:
    # Beam search requires greedy decoding
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
    attn_implementation: [sdpa, flash_attention_2, eager]
    quantization:
      - variant: none
      - variant: bnb_8bit
        load_in_8bit: true
      - variant: bnb_4bit
        load_in_4bit: true
        bnb_4bit_compute_dtype: float16
        bnb_4bit_quant_type: [nf4, fp4]
    compilation:
      - variant: none
        torch_compile: false
      - variant: compiled
        torch_compile: true
        torch_compile_mode: [default, reduce-overhead, max-autotune]
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

### Option D: Pydantic-Driven Auto-Skip (status quo, enhanced)

Keep the current model: Cartesian product generates all combos, Pydantic validators reject invalid ones as `SkippedConfig`. Enhance the UX around skipped configs.

```yaml
sweep:
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.torch_compile: [true, false]
  pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune, null]
  pytorch.load_in_4bit: [true, null]
  pytorch.load_in_8bit: [true, null]
```

Enhancements:
- Preflight panel shows expected skip count before execution
- `null` sentinel in sweep values means "omit this field" (not set to None)
- Skipped configs logged with clear per-constraint reasoning

**Pros:**
- Zero new syntax or concepts
- Pydantic validators are the SSOT - constraints defined once
- Already partially implemented (SkippedConfig exists)
- Simplest implementation

**Cons:**
- Generates many invalid combos in large grids (e.g. 4bit x 8bit x all batch sizes)
- Users must understand that many configs will be skipped
- `null` as "unset" vs `null` as "None value" is confusing in YAML
- Doesn't help with cross-section dependencies (beam search + decoder settings)
- `experiments:` section still needed for multi-field overrides (beam search with custom decoder)

---

## Recommendation: Option A (`sweep_groups`)

**Primary: `sweep_groups` for dependent parameter sets.**
**Secondary: Enhanced Pydantic auto-skip (Option D improvements) for edge cases.**

### Rationale

1. **Right level of abstraction.** Groups express "these parameters travel together" - which is exactly what the constraint catalogue shows. Every dependency listed above is a group relationship: torch_compile + its sub-params, load_in_4bit + its sub-params, beam_search + decoder settings.

2. **YAML-native, no DSL.** Unlike Option B, no expression parser needed. Unlike Option C, no artificial discriminators. The syntax is standard YAML lists and maps.

3. **Backwards compatible.** `sweep:` works exactly as today. `sweep_groups:` is additive. Existing configs don't change.

4. **Matches the mental model.** Researchers think "I want to test these torch.compile modes" - that's a group. They don't think "I want to exclude all combos where torch_compile is false AND torch_compile_mode is not null" - that's a constraint.

5. **Bounded complexity.** Groups are structurally simple (union of mini-grids, crossed with main sweep). No expression evaluation, no tree traversal, no new type system.

6. **Pydantic stays the safety net.** Any group that accidentally produces an invalid config still gets caught by Pydantic validators and reported as `SkippedConfig`. Belt and braces.

### What remains as explicit experiments

After sweep_groups, the `experiments:` section is reserved for truly unique one-off experiments that:
- Override shared defaults (e.g. different `dataset.n_prompts` for expensive configs)
- Combine settings from multiple unrelated dimensions (e.g. beam search + specific quantisation + custom decoder)
- Use parameters not in any sweep axis (e.g. `lora.adapter_id` pointing to a specific model)

This should reduce `experiments:` from ~40 entries to ~5-10.

---

## Detailed Design

### YAML Schema

```yaml
# Independent axes (unchanged)
sweep:
  dtype: [float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, eager]

# Dependent parameter groups
sweep_groups:
  # ── PyTorch: torch.compile ──────────────────────────
  pytorch_compile:                # Group name (for logging/display, not a config field)
    - pytorch.torch_compile: false

    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune]
      pytorch.torch_compile_backend: inductor

  # ── PyTorch: quantisation ───────────────────────────
  pytorch_quantization:
    - {}                          # No quantisation (empty = no overrides)

    - pytorch.load_in_8bit: true

    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: [nf4, fp4]

    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
      pytorch.bnb_4bit_use_double_quant: true

  # ── vLLM: decoding strategy ────────────────────────
  vllm_decoding:
    - {}                          # Default sampling (use shared decoder settings)

    - decoder.do_sample: false    # Cross-section: group can override any field
      decoder.temperature: 0.0
      vllm.beam_search:
        beam_width: 4
        early_stopping: true

    - vllm.sampling:
        presence_penalty: [0.0, 0.6]
        frequency_penalty: [0.0, 0.6]
```

### Group Naming

Groups have **names** (the keys under `sweep_groups:`). These are used for:
- Preflight display: "torch_compile: 4 variants x quantisation: 4 variants = 16 group combos"
- Skipped config messages: "Skipped: pytorch_compile[2] x pytorch_quantization[3]"
- Debugging: clear provenance for each generated config

Group names must be unique within `sweep_groups:`. They are not config fields and are stripped before experiment construction.

### Expansion Algorithm

```
1. Parse sweep: → independent_dims (unchanged)
2. Parse sweep_groups: → named_groups: dict[str, list[dict]]
3. For each group, expand list entries into mini-grids:
   - Each entry's list-valued fields produce a Cartesian product
   - Single-valued fields are constants
   - {} means "no overrides" (use fixed/sweep defaults)
4. Cross all named groups with each other:
   - groups_product = product(group_A_entries, group_B_entries, ...)
   - Each combo is a merged dict (later groups override earlier on conflict)
5. Cross groups_product with independent_dims product:
   - final_configs = product(independent_combos, group_combos)
6. For each final config, merge with fixed defaults and validate via Pydantic
```

**Conflict resolution:** If two groups set the same field, the later group (by document order) wins. This is intentional: a `vllm_decoding` group can override `decoder.temperature` set by `sweep:`.

### Preflight Display

The Rich preflight panel gains a new "Sweep Groups" section:

```
┌─ Study: full-suite-all-backends ────────────────────────┐
│                                                         │
│  Execution:  360 experiments × 3 cycles = 1,080 runs    │
│  Order:      shuffle (seed: auto)                       │
│  Gaps:       60s between experiments, 300s between cycles│
│                                                         │
│  ── Sweep Axes ──────────────────────────────────────── │
│  dtype:          float16, bfloat16                       │
│  pytorch.batch_size: 1, 4, 8, 16, 32                   │
│  pytorch.attn:   sdpa, flash_attention_2, eager          │
│                                                         │
│  ── Sweep Groups ────────────────────────────────────── │
│  pytorch_compile (4 variants):                          │
│    · torch_compile=false                                │
│    · torch_compile=true, mode=default                   │
│    · torch_compile=true, mode=reduce-overhead           │
│    · torch_compile=true, mode=max-autotune              │
│  pytorch_quantization (4 variants):                     │
│    · (none)                                             │
│    · load_in_8bit=true                                  │
│    · load_in_4bit=true, bnb_4bit_quant_type=nf4         │
│    · load_in_4bit=true, bnb_4bit_quant_type=nf4, double │
│                                                         │
│  ── Skipped ─────────────────────────────────────────── │
│  12/372 configs skipped (3.2%)                          │
│    · pytorch, float32: ...                              │
│                                                         │
│  Design hash: a1b2c3d4e5f6g7h8                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Scope

1. **`config/grid.py`**: Add `_expand_sweep_groups()` function. Modify `_expand_sweep()` to accept and cross with group combos. ~80 lines new code.

2. **`config/grid.py` (`expand_grid`)**: Extract `sweep_groups:` from raw study dict alongside `sweep:`. Pass to `_expand_sweep()`.

3. **`config/grid.py` (`build_preflight_panel`)**: Add "Sweep Groups" section to Rich display.

4. **`config/loader.py`**: Add `sweep_groups` to `_STUDY_ONLY_KEYS` so it's stripped from fixed dict.

5. **`study-full-suite.yaml`**: Rewrite to use `sweep_groups:` for dependent params. Move independent explicit experiments into `sweep:`.

6. **Tests**: New test class `TestExpandGridSweepGroups` covering group expansion, group x sweep crossing, empty groups, conflict resolution, group + explicit experiments.

### What does NOT change

- `ExperimentConfig` and all Pydantic models: unchanged
- `_extract_fixed()`: unchanged
- Explicit `experiments:` section: still supported, still merged last
- `SkippedConfig`: still the safety net for invalid combos
- `apply_cycles()`, `compute_study_design_hash()`: unchanged
- Backend configs, validators, introspection: unchanged

---

## Worked Example: PyTorch Section of `study-full-suite.yaml`

### Before (current, 9 explicit experiments + limited sweep)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

experiments:
  # torch.compile: default (explicit)
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

  # torch.compile: reduce-overhead (explicit)
  - model: Qwen/Qwen2.5-0.5B
    backend: pytorch
    dtype: bfloat16
    # ... 10 more lines ...

  # torch.compile: max-autotune (explicit)
  - ...

  # BnB 8bit (explicit)
  - ...

  # BnB 4bit nf4 (explicit)
  - ...

  # BnB 4bit nf4 double-quant (explicit)
  - ...

  # Beam search (explicit)
  - ...

  # Speculative decoding (explicit)
  - ...

  # Static KV cache (explicit)
  - ...
```

### After (sweep_groups, 1 explicit experiment)

```yaml
sweep:
  dtype: [float32, float16, bfloat16]
  pytorch.batch_size: [1, 4, 8, 16, 32]
  pytorch.attn_implementation: [sdpa, flash_attention_2, flash_attention_3, eager]

sweep_groups:
  pytorch_compile:
    - pytorch.torch_compile: false
    - pytorch.torch_compile: true
      pytorch.torch_compile_mode: [default, reduce-overhead, max-autotune]
      pytorch.torch_compile_backend: inductor

  pytorch_quantization:
    - {}
    - pytorch.load_in_8bit: true
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: [nf4, fp4]
    - pytorch.load_in_4bit: true
      pytorch.bnb_4bit_compute_dtype: float16
      pytorch.bnb_4bit_quant_type: nf4
      pytorch.bnb_4bit_use_double_quant: true

  pytorch_cache:
    - {}
    - pytorch.use_cache: true
      pytorch.cache_implementation: static

  pytorch_decoding:
    - {}
    - decoder.do_sample: false
      decoder.temperature: 0.0
      pytorch.num_beams: [2, 4]
      pytorch.early_stopping: true
      pytorch.length_penalty: 1.0
      pytorch.no_repeat_ngram_size: 3

experiments:
  # Speculative decoding: only valid with batch_size=1, so override the sweep axis
  - model: Qwen/Qwen2.5-0.5B
    backend: pytorch
    dtype: bfloat16
    dataset: { n_prompts: 50 }
    pytorch:
      batch_size: 1
      prompt_lookup_num_tokens: 3
      device_map: auto
```

**Result:** 9 explicit experiments reduced to 1. All dependent params properly constrained. The groups are crossed with independent sweep axes, producing a complete combinatorial grid without invalid configs.

---

## Open Questions

1. **Group interaction with multi-backend sweeps.** When `backend: [pytorch, vllm]`, should groups scoped to `pytorch.*` only apply to pytorch experiments? **Proposed:** Yes - groups with backend-prefixed keys are automatically scoped to that backend, matching existing dotted-key routing behaviour.

2. **Naming `sweep_groups` vs alternatives.** Considered: `sweep_bundles`, `sweep_variants`, `parameter_groups`, `sweep_axes_grouped`. `sweep_groups` chosen for clarity and consistency with `sweep:`.

3. **Should groups be required to have names?** **Proposed:** Yes - names are required for logging/display. Anonymous groups (list instead of dict) could be supported later if needed.

4. **Max group combinatorial size.** Should we warn if groups produce > N combos? **Proposed:** Yes - same >50% skip warning logic, plus a new warning if total experiments > 1000 before cycles.

## Alternatives Considered

See Options B (constraint expressions), C (hierarchical tree), and D (enhanced auto-skip) above. Option A was chosen for its balance of expressiveness, simplicity, and backwards compatibility.
