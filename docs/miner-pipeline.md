# Invariant Miner Pipeline

This document is the deep-dive reference for the invariant miner pipeline: how it extracts validation rules from engine library source code and packages them into a versioned corpus.

**Audience:** engine extenders, CI maintainers, and research readers interested in the technical approach.

For the runtime side of the pipeline (how the corpus is consumed at validation time), see [parameter-discovery.md](parameter-discovery.md).

---

## What the miner pipeline does

The miner pipeline derives structured constraint rules from ML engine config classes by combining:

1. **Static analysis** - walking the AST of validator methods to extract conditional predicates.
2. **Dynamic analysis** - instantiating config classes with combinatorial probe values and observing raise/no-raise patterns.
3. **Type-system lifting** - extracting constraints directly from Pydantic `FieldInfo`, msgspec `Meta`, and stdlib `Literal[...]` annotations.

The output is a corpus of rules, each describing one constraint on a config field or combination of fields. Corpus rules are then validated against the live library (the vendor-CI gate) before shipping.

---

## Component overview

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  INVARIANT MINER PIPELINE                                           │
  │                                                                     │
  │  ┌─────────────────────────┐   ┌─────────────────────────┐         │
  │  │    STATIC MINER         │   │    DYNAMIC MINER        │         │
  │  │                         │   │                         │         │
  │  │  inspect.getsource()    │   │  class constructors     │         │
  │  │     + ast.parse()       │   │  + validate() calls     │         │
  │  │         │               │   │         │               │         │
  │  │  ConditionalRaiseDetector   │  Cartesian probe grid   │         │
  │  │  ConditionalSelfAssign  │   │         │               │         │
  │  │  ConditionalWarnDetector│   │  predicate inference    │         │
  │  │  (etc.)                 │   │                         │         │
  │  └──────────┬──────────────┘   └──────────┬──────────────┘         │
  │             │                              │                        │
  │             └──────────────┬───────────────┘                        │
  │                            │                                        │
  │  ┌─────────────────────────▼─────────────────────────┐             │
  │  │              LIFT MODULES                          │             │
  │  │                                                    │             │
  │  │  _pydantic_lift.py   model_json_schema()           │             │
  │  │                      FieldInfo.metadata            │             │
  │  │                                                    │             │
  │  │  _msgspec_lift.py    msgspec.inspect.type_info()   │             │
  │  │                      Meta(ge=, le=, ...)           │             │
  │  │                                                    │             │
  │  │  _dataclass_lift.py  dataclasses.fields()          │             │
  │  │                      Literal[...] annotations      │             │
  │  └─────────────────────────┬─────────────────────────┘             │
  │                            │                                        │
  │                            ▼                                        │
  │                   staging files                                     │
  │              configs/validation_rules/_staging/                     │
  │                            │                                        │
  │                            ▼                                        │
  │                    build_corpus.py                                  │
  │                (merge + dedup + fingerprint)                        │
  │                            │                                        │
  │                            ▼                                        │
  │                    vendor_rules.py                                  │
  │              (replay against live library)                          │
  │                            │                                        │
  │           ┌────────────────┴──────────────────┐                    │
  │           ▼                                   ▼                    │
  │  confirmed rules                   quarantined rules               │
  │  configs/validation_rules/         configs/validation_rules/        │
  │  {engine}.yaml                     _staging/_failed_*.yaml         │
  │  src/.../vendored_rules/                                            │
  │  {engine}.json                                                      │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Static miner

The static miner reads engine library source via `inspect.getsource()` + `ast.parse()` and walks the AST of known validator methods. It does not call constructors or run the validator methods.

This is "static" in the sense that it reads source without executing the methods under analysis. The library is still imported (to get source file paths), but no config classes are instantiated.

### Why AST walking

Pure introspection (running the constructor and observing errors) cannot recover the shape of cross-field predicates. The dynamic miner sees the message `"num_beams should be divisible by num_beam_groups"` but cannot determine that the underlying check is `num_beams % num_beam_groups != 0`. The static miner reads the predicate structure directly from the AST.

Example: the rule `not_divisible_by` can only be expressed in the corpus because the static miner found `if num_beams % num_beam_groups != 0: raise` in the AST.

### AST primitives (in `_base.py`)

```
  ast.parse(source)
       │
       ▼
  find_class(module, "GenerationConfig")
       │
       ▼
  find_method(cls, "validate")
       │
       ▼
  for stmt in if_body:
       │
       ├── ConditionalRaiseDetector      → severity: "error"
       │   "if X: raise SomeException(msg)"
       │
       ├── ConditionalSelfAssignDetector → severity: "dormant"
       │   "if X: self.A = B" (silent normalisation)
       │
       ├── ConditionalWarningsWarnDetector → severity: "warn"
       │   "if X: warnings.warn(msg)"
       │
       ├── ConditionalLoggerWarningDetector → severity: "warn"
       │   "if X: logger.warning(msg)"
       │
       └── MinorIssuesDictAssignDetector → severity: "dormant"
           HF-specific: "if X: minor_issues[key] = msg"
```

### Filters (false-positive guards)

Before emitting a candidate, the static miner applies three filters:

1. `filter_condition_references_self` - the predicate must reference at least one public field via `self.<field>`. Drops argument-gated rules (`if strict: raise`) and private-state rules (`if self._initialized: ...`).

2. `filter_target_is_public_field` - for self-assign patterns, the affected field must be a public field.

3. `filter_kwargs_positive_derivable` - a representative `kwargs_positive` dict must be synthetically derivable from the predicate. Rejects predicates whose truth depends on opaque external calls.

### Miner depth

Static miner depth is fixed at 1: it walks one level of helper calls (`WatermarkingConfig.validate`, `SynthIDTextWatermarkingConfig.validate`) but does not trace through general function calls in the validator body. This avoids an unbounded call-graph traversal while capturing the most common engine validation patterns.

---

## Dynamic miner

The dynamic miner instantiates config classes with combinatorial probe values and observes raise/no-raise patterns. It then runs predicate inference on the resulting table of `(kwargs, error_message)` rows.

### Probe strategy: Cartesian primary, Hypothesis supplement

```
  cluster definition
  (e.g. beam-search: num_beams, num_beam_groups, diversity_penalty)
               │
               ▼
  representative values per field
  (e.g. num_beams=[1, 2, 4], num_beam_groups=[1, 2, 3])
               │
               ▼
  Cartesian product of values
               │
               ├── cluster size ≤ threshold
               │   Cartesian probe runs every combination
               │
               └── cluster size > threshold (e.g. 8 fields × 5 values)
                   Hypothesis from_type generates values deterministically
                   (fixed seed, no randomness; Hypothesis as value generator only)
               │
               ▼
  for each combination:
    try:
      ClassName(**kwargs)
      .validate(strict=True) if applicable
      → record (kwargs, None)
    except Exception as e:
      → record (kwargs, str(e))
               │
               ▼
  probe-row table: list[(kwargs, error_message | None)]
```

**Important:** Hypothesis is used here only as a deterministic value generator with a fixed seed, not as a property-based test runner. The miner pipeline must be deterministic: the same library version + miner code must produce the same corpus. Randomness would break Renovate-driven library bump diffs.

### Predicate inference

Given the probe-row table, the dynamic miner infers one rule per error message class using seven predicate templates (in order of preference):

| Template | Example | Fires when |
|----------|---------|-----------|
| Cross-field divisibility | `a % b != 0` | error rows align with divisibility failure |
| Cross-field comparison | `a > b` | error rows align with comparison |
| Cross-field equality gate | `a == V AND b == W` | error rows correlate with combined field values |
| Type allowlist | `type(a) not in {T1, T2}` | error rows correlate with field type |
| Single-field range | `a < 0` | error rows correlate with one field crossing a threshold |
| Single-field equality | `a == V` | error rows correlate with one field having a specific value |
| Value allowlist | `a not in {v1, v2, ...}` | error rows correlate with field value not in a set |

The dynamic miner errs toward recall: when multiple templates fit the evidence, it emits all plausible candidates. The vendor-CI gate prunes false positives downstream.

---

## Lift modules

The three lift modules extract constraints from type-system metadata without requiring probe rounds. They are independent stages that run alongside AST walking and probing.

```
  Type-system axis         Lift module              Engines using it
  ──────────────────────   ─────────────────────    ────────────────────────────
  pydantic.BaseModel       _pydantic_lift.py         vLLM (27 pydantic-dataclasses)
  pydantic.dataclasses                               TRT-LLM (TrtLlmArgs)
                                                     (Literal-typed enum fields)
  msgspec.Struct           _msgspec_lift.py          vLLM (SamplingParams)
  stdlib @dataclass        _dataclass_lift.py        transformers (GenerationConfig,
                                                     BitsAndBytesConfig)
                                                     vLLM (EngineArgs, 175 fields)
                                                     TRT-LLM (BuildConfig, QuantConfig)
```

### Pydantic lift (`_pydantic_lift.py`)

Walks `model_json_schema()` and `FieldInfo.metadata` (Pydantic v2). Emits one rule per `annotated-types` constraint or `Literal[...]` allowlist found on a field.

Operator vocabulary aligns with the `annotated-types` standard:

```
  annotated-types predicate   corpus operator key
  ─────────────────────────   ───────────────────
  Gt(value)                   ">"
  Ge(value)                   ">="
  Lt(value)                   "<"
  Le(value)                   "<="
  MultipleOf(value)           "multiple_of"
  MinLen(value)               "min_len"
  MaxLen(value)               "max_len"
  Literal[a, b, c]            "in": [a, b, c]
```

### msgspec lift (`_msgspec_lift.py`)

Walks `msgspec.inspect.type_info()` and the `Constraints` object per field. Maps `Meta(ge=, le=, ...)` constraints to corpus operator keys using the same vocabulary as the Pydantic lift.

Note: vLLM's `SamplingParams` currently ships zero `Meta` annotations - the msgspec lift returns `[]` for it. The lift exists so that if vLLM (or another msgspec user) adds `Meta(ge=...)` in a future version, the constraints are captured for free.

### Dataclass lift (`_dataclass_lift.py`)

Walks `dataclasses.fields()` and extracts `Literal[a, b, c]` annotations. Plain stdlib dataclasses carry no numeric-bound metadata, so the lift is limited to value-allowlist rules.

---

## Per-engine miner comparison

The three engines have structurally different config surfaces, which determines which components each miner uses.

```
  ┌──────────────┬─────────────────────┬──────────────┬─────────────────────────┐
  │ Engine       │ Static miner        │ Dynamic miner│ Lift modules            │
  ├──────────────┼─────────────────────┼──────────────┼─────────────────────────┤
  │ transformers │ GenerationConfig     │ Cartesian    │ dataclass_lift          │
  │              │ .validate(), BNB    │ cluster      │ (GenerationConfig,      │
  │              │ .post_init()        │ probing      │ BitsAndBytesConfig)     │
  │              │ ~1700 LoC walked    │              │                         │
  ├──────────────┼─────────────────────┼──────────────┼─────────────────────────┤
  │ vLLM         │ SamplingParams      │ Cartesian    │ pydantic_lift (27       │
  │              │ ._verify_args()     │ + Hypothesis │ vllm.config.* classes)  │
  │              │ ~20 validator       │ supplement   │ msgspec_lift            │
  │              │ methods             │              │ (SamplingParams)        │
  │              │                     │              │ dataclass_lift          │
  │              │                     │              │ (EngineArgs)            │
  ├──────────────┼─────────────────────┼──────────────┼─────────────────────────┤
  │ TRT-LLM      │ BaseLlmArgs         │ SKIPPED      │ pydantic_lift           │
  │              │ .validate_*()       │ (constructor │ (TrtLlmArgs)            │
  │              │ ~11 validator       │ yields zero  │ dataclass_lift          │
  │              │ methods             │ raises)      │ (BuildConfig,           │
  │              │                     │              │  QuantConfig)           │
  └──────────────┴─────────────────────┴──────────────┴─────────────────────────┘

  Target rule count after vendor-CI gate:
    transformers:  46 rules (shipped)
    vLLM:          80-110 rules (target)
    TRT-LLM:       20-28 rules (target)
```

**Why no dynamic miner for TRT-LLM:** empirical probing of `TrtLlmArgs(**kwargs)` constructors produced zero raises. TRT-LLM performs construction-time validation in a much more permissive way than transformers or vLLM; its constraints are primarily enforced in validator methods (covered by the static miner) and at engine build time (hardware-gated, not corpus rules).

---

## Fail-loud import contract

Every miner module must declare its version envelope and validate it at import time. This is a structural contract, not a guideline.

```python
# Every *_miner.py must declare this:
TESTED_AGAINST_VERSIONS = SpecifierSet(">=4.50,<4.60")

# And call this at import time:
check_installed_version(
    "transformers",
    importlib.metadata.version("transformers"),
    TESTED_AGAINST_VERSIONS,
)
```

If the installed library version falls outside the envelope, the miner raises `MinerVersionMismatchError` - a hard CI failure.

If an expected class or method is missing from the library source (e.g. a class was renamed in a library refactor), the miner raises `MinerLandmarkMissingError` - also a hard CI failure.

```
  check_installed_version()
       │
       ├── version in TESTED_AGAINST_VERSIONS → continue
       │
       └── version out of range → MinerVersionMismatchError (CI fatal)

  find_class(module, "GenerationConfig")
       │
       ├── class found → continue
       │
       └── None → MinerLandmarkMissingError (CI fatal)
```

**Why this matters:** the Haiku-era TRT-LLM extractor (PRs #415-#417, reverted in #423) silently degraded when it encountered an import error - it caught `ImportError` and returned `[]` instead of failing. The silent degradation was indistinguishable from "no rules found for this engine", which masked a broken extractor. The fail-loud contract makes that impossible.

### Structural fixpoint: ensuring the contract is enforced

`_fixpoint_test.py` includes a structural test that synthesises one malformed rule per gate-soundness check and asserts the vendor-CI gate records a divergence for each. This pins the three checks in place:

1. `positive_raises` - `kwargs_positive` must cause the library to raise.
2. `message_template_match` - the raised message must contain the template's static fragment.
3. `negative_does_not_raise` - `kwargs_negative` must construct without raising.

If any of the three checks is removed from `vendor_rules.compute_gate_soundness_divergences`, the corresponding case in `_fixpoint_test.py` fails loudly.

---

## Build corpus: merge and dedup

`build_corpus.py` is the orchestration entrypoint. It runs all miners, collects staging files, merges them, deduplicates, and calls the vendor-CI gate.

### Fingerprinting

The deduplication key is:

```python
canonical_serialise({
    "engine": rule.engine,
    "severity": rule.severity,
    "match_fields": rule.match["fields"],
})
```

Two rules with the same fingerprint are treated as the same constraint discovered by two independent paths (cross-validation). The merger keeps one rule with the primary `added_by` source and records the secondary source in `cross_validated_by`.

### Per-field merge precedence

When static and dynamic miners both emit a rule with the same fingerprint, the fields are merged by source preference:

| Field | Source that wins |
|-------|-----------------|
| `match.fields` predicate | static miner (more specific operators) |
| `message_template` | dynamic miner (real library text) |
| `observed_messages` | dynamic miner (real captured emissions) |
| `kwargs_positive` / `kwargs_negative` | static miner (derived from conditional) |
| `miner_source.line_at_scan` | static miner (real source line) |
| `references` | union (all evidence preserved) |
| `id` | first source's id is canonical |

---

## Vendor-CI gate

The vendor-CI gate runs after merge. It replays every rule's `kwargs_positive` and `kwargs_negative` against the live library inside the engine's Docker container and compares observed behaviour against the declared `expected_outcome`.

```
  for each rule in merged corpus:
       │
       ▼
  run_case(kwargs_positive, native_type) → CaptureBuffers
       │
       ├── CHECK positive_raises
       │   CaptureBuffers.exception_type must not be None
       │
       ├── CHECK message_template_match
       │   CaptureBuffers.exception_message must contain
       │   rule.message_template (static fragment)
       │
       └── CHECK negative_does_not_raise
           run_case(kwargs_negative, native_type)
           CaptureBuffers.exception_type must be None
       │
       ├── all checks pass → rule confirmed → write to corpus
       │
       └── any check fails → rule quarantined to _failed_validation_*.yaml
```

The gate runs inside the Docker container for each engine so that the live library version used for validation matches the version the miner was built against.

**Exit codes from `vendor_rules.py`:**
- `0` - all rules confirmed.
- `1` - one or more divergences; vendored JSON still written (for diagnostic purposes).
- `2` - hard error (corpus malformed, engine not importable).

---

## Renovate-driven refresh loop

Library version bumps trigger corpus regeneration automatically.

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                    RENOVATE REFRESH LOOP                          │
  │                                                                   │
  │  Upstream library releases new version                            │
  │  (e.g. transformers 4.56.0 → 4.57.0)                             │
  │               │                                                   │
  │               ▼                                                   │
  │  Renovate detects version bump                                    │
  │  (weekly schedule, 3-day stability window)                        │
  │               │                                                   │
  │               ▼                                                   │
  │  Renovate opens PR bumping Dockerfile ARG                         │
  │  or PyPI version pin in requirements file                         │
  │               │                                                   │
  │               ▼                                                   │
  │  invariant-miner.yml fires                                        │
  │  (guarded: only Renovate PRs touching engine version files)       │
  │               │                                                   │
  │         ┌─────┴─────────────────────────────┐                    │
  │         ▼                                   ▼                    │
  │  GH-hosted runner                   Self-hosted GPU runner       │
  │  - transformers static miner        - TRT-LLM static miner       │
  │  - transformers dynamic miner       (CUDA-aware import required)  │
  │  - vLLM static miner                                              │
  │  - vLLM dynamic miner                                             │
  │  (CPU-safe imports confirmed)                                     │
  │         │                                   │                    │
  │         └─────────────┬─────────────────────┘                    │
  │                       ▼                                           │
  │         build_corpus.py + vendor_rules.py run                    │
  │         (inside Docker container for engine)                      │
  │                       │                                           │
  │                       ▼                                           │
  │         Bot writes updated vendored JSON to PR branch             │
  │         (llem-ci-bot GitHub App; see reference_llem_ci_bot.md)   │
  │                       │                                           │
  │                       ▼                                           │
  │         CI green required before merge                            │
  │         Divergences are P0 incidents (block merge)                │
  │                       │                                           │
  │                       ▼                                           │
  │         Maintainer reviews corpus diff in PR                      │
  │         (gate-breaking = action required before merge)            │
  └───────────────────────────────────────────────────────────────────┘
```

### Version mismatch as CI signal

When `TESTED_AGAINST_VERSIONS` in a miner module does not cover the newly bumped library version, `MinerVersionMismatchError` is raised and CI fails. This is intentional: it forces a maintainer to update the miner against the new library version before the corpus is regenerated.

The update workflow:

1. Renovate opens PR bumping library version.
2. CI fires, `MinerVersionMismatchError` raised for the affected miner.
3. Maintainer checks the library's release notes for validator changes.
4. Maintainer updates `TESTED_AGAINST_VERSIONS` and any landmark names that changed.
5. CI re-runs with updated miner; vendor-CI gate runs.
6. If any rules now diverge, they are quarantined; maintainer updates the corpus.

---

## Two-tier CI

Miners run on two runner tiers based on their import requirements.

| Tier | Runner | What runs |
|------|--------|-----------|
| GH-hosted | `ubuntu-latest` | All static miners (pure file I/O); transformers + vLLM dynamic miners (CPU-safe imports confirmed) |
| Self-hosted | GPU runner (closes issue #389) | TRT-LLM static miner (requires CUDA-aware `import tensorrt_llm`); the TRT-LLM Docker image `llenergymeasure:tensorrt` at pin v0.21.0 is the runtime |

TRT-LLM is pinned at v0.21.0 (CUDA 12.6.x) because v1.x requires CUDA 13.x, which is not available on the current A100 (SM80) runner fleet.

---

## `_base.py` shared infrastructure

All miners import from `scripts/miners/_base.py`. It provides:

- `RuleCandidate` - the output type; fields mirror the corpus YAML schema exactly (no translation step needed).
- `MinerSource` - `{path, method, line_at_scan}` provenance record.
- `MinerError`, `MinerVersionMismatchError`, `MinerLandmarkMissingError` - fail-loud error hierarchy.
- `check_installed_version` - version envelope guard.
- `find_class`, `find_method` - AST navigation helpers.
- `call_func_path`, `first_string_arg`, `extract_condition_fields`, `resolve_local_assign`, `extract_loop_literal_iterable` - AST extraction primitives.
- `ConditionalRaiseDetector`, `ConditionalSelfAssignDetector`, `ConditionalWarningsWarnDetector`, `ConditionalLoggerWarningDetector`, `MinorIssuesDictAssignDetector` - pattern detectors.
- `filter_condition_references_self`, `filter_target_is_public_field`, `filter_kwargs_positive_derivable` - false-positive guards.
- `candidate_to_dict` - serialises `RuleCandidate` to the corpus YAML dict shape.

---

## Predicate-inference template coverage

The seven templates were derived empirically from the transformers corpus. When the static miner encounters an AST predicate it cannot translate, it logs the dropped sub-clause (without failing). A monthly audit of the unparsed-predicate log drives empirical template expansion - templates are only added when a real rule shape appears three or more times.

The templates NOT adopted from Daikon's full library: linear arithmetic ternary (`z = ax + by + c`), sortedness, sequence-equality. These cover scientific-computing trace patterns not seen in engine config classes.

---

## See also

- [architecture-overview.md](architecture-overview.md) - system overview and data-flow
- [validation-rule-corpus.md](validation-rule-corpus.md) - corpus YAML format reference
- [extending-miners.md](extending-miners.md) - how to add a new engine miner
- [parameter-discovery.md](parameter-discovery.md) - runtime validation pipeline
- [research-context.md](research-context.md) - academic positioning
- [engines.md](engines.md) - engine configuration reference
- [schema-refresh.md](schema-refresh.md) - Renovate-driven schema refresh
