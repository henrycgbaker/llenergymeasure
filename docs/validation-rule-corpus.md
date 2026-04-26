# Validation Rule Corpus Format

This document is the reference for the YAML corpus format: what each field means, valid values, and how to read an existing rule.

**Audience:** maintainers writing or reviewing corpus rules; extenders adding rules for new engines; anyone debugging a rule that behaves unexpectedly.

---

## File locations

```
  configs/validation_rules/
  ├── transformers.yaml          Authoritative corpus - human-reviewable, git-tracked
  └── _staging/
      ├── transformers_static_miner.yaml   Per-miner staging output (not committed)
      ├── transformers_dynamic_miner.yaml
      └── _failed_validation_transformers.yaml  Quarantined rules

  src/llenergymeasure/config/vendored_rules/
  ├── transformers.json          Vendored observations - ships with package
  └── loader.py                  Runtime consumer
```

---

## Top-level corpus envelope

```yaml
schema_version: 1.0.0          # Major version must match loader's SUPPORTED_MAJOR_VERSION
engine: transformers            # Engine name; must match a known engine key
engine_version: 4.56.0         # Library version the corpus was mined and validated against
mined_at: '2026-04-25T18:01:18Z'  # ISO 8601 timestamp of last full mine run
rules:
  - ...                          # List of rule entries
```

---

## Rule schema: annotated example

Below is a complete rule from the transformers corpus with every field annotated.

```yaml
- id: transformers_beam_search_num_beams_not_divisible_by_num_beam_groups
  # Unique identifier for this rule.
  # Convention: {engine}_{subject_field}_{condition_slug}
  # Used as the key in divergence reports and error messages.

  engine: transformers
  # Engine this rule applies to.
  # Must match the corpus envelope's engine field.

  library: transformers
  # Python package name (importlib.metadata.version uses this).
  # Matches engine for single-library engines;
  # may differ for engines that alias a library.

  rule_under_test: "GenerationConfig.__init__ flags `num_beams` (num beams not divisible by num beam groups)"
  # Human-readable description of what library behaviour this rule captures.
  # Format: {NativeType}.{method} flags {field} ({condition})

  severity: error
  # One of: error | warn | dormant
  # error    - engine raises; loader rejects before initialisation
  # warn     - engine announces; loader warns the user
  # dormant  - engine silently normalises; loader annotates the config

  native_type: transformers.GenerationConfig
  # The fully-qualified class name the rule's predicate applies to.
  # Used by the vendor-CI gate to know which class to instantiate.

  miner_source:
    path: transformers/generation/configuration_utils.py
    # Relative path within the library's source tree.
    method: __init__
    # Method name where the AST detector found this rule.
    line_at_scan: 361
    # Line number in the source at mined_at time.
    # Will drift when the library is updated; used for human inspection only.

  match:
    engine: transformers
    # Must match rule.engine (redundant, for grep-ability).
    fields:
      transformers.sampling.num_beams:
        not_divisible_by: "@num_beam_groups"
      # Field paths are dotted, resolved against ExperimentConfig.
      # Operator: not_divisible_by - fires when a % b != 0.
      # @num_beam_groups is a @field_ref: resolved as a sibling field.

  kwargs_positive:
    num_beams: 2
    num_beam_groups: 3
    # kwargs that trigger the rule (should cause the engine to raise/warn/normalise).
    # Passed directly to the native_type constructor in the vendor-CI gate.
    # 2 is not divisible by 3, so the rule fires.

  kwargs_negative:
    num_beams: 4
    num_beam_groups: 2
    # kwargs that do NOT trigger the rule (should pass cleanly).
    # 4 is divisible by 2, so the rule does not fire.

  expected_outcome:
    outcome: error
    # One of: dormant_silent | dormant_announced | warn | error | pass
    emission_channel: none
    # How the engine signals the issue.
    # One of: warnings_warn | logger_warning | logger_warning_once |
    #         minor_issues_dict | none | runtime_exception
    normalised_fields: []
    # For dormant rules: which fields the engine silently normalises.
    # Empty for error rules.

  message_template: >
    `num_beams` has to be divisible by `num_beam_groups`, but got
    `num_beams`={declared_value} and `num_beam_groups`={declared_value}.
  # The static fragment of the library's error message.
  # Used by the vendor-CI gate's message_template_match check:
  # the gate asserts this fragment appears in the live library's exception message.
  # Template variables ({declared_value}, {effective_value}, etc.) are
  # substituted when the rule fires at validation time.

  references:
    - "transformers.GenerationConfig.__init__() - observed via construction-time ValueError"
  # Human-readable provenance citations. Free-form strings.
  # Useful for tracking down the library source line that motivated the rule.

  added_by: dynamic_miner
  # Provenance: which pipeline component produced this rule.
  # See AddedBy values below.

  added_at: '2026-04-25'
  # Date (YYYY-MM-DD) when this rule was added to the corpus.

  cross_validated_by: []
  # Optional. Other miner sources that independently emitted a rule with the
  # same fingerprint (engine + severity + match.fields).
  # Set by build_corpus.py when two miners agree; empty for single-source rules.
```

---

## Field reference

### `id`

Unique identifier for the rule. Used in error messages, divergence reports, and logs. Convention:

```
{engine}_{native_type_slug}_{condition_slug}
```

Examples:
- `transformers_beam_search_num_beams_not_divisible_by_num_beam_groups`
- `vllm_sampling_temperature_out_of_range`
- `tensorrt_quantization_fp8_not_supported_on_sm80`

### `severity`

| Value | When it fires | User sees |
|-------|--------------|-----------|
| `error` | Engine would raise at construction / validate time | `ValueError` (surfaced as Pydantic `ValidationError`) before initialisation |
| `warn` | Engine announces a suboptimal setting but proceeds | Warning message |
| `dormant` | Engine silently normalises or ignores the field | Annotation: "field X will be coerced to Y" |

### `expected_outcome.outcome`

| Value | Meaning |
|-------|---------|
| `dormant_silent` | Engine silently normalises; no user-visible emission |
| `dormant_announced` | Engine writes to `minor_issues` dict / logger, but config runs |
| `warn` | Engine calls `warnings.warn(...)` or equivalent |
| `error` | Engine raises at construct / validate time |
| `pass` | Predicate matched but engine handles it cleanly (positive-reference rules) |

### `expected_outcome.emission_channel`

| Value | Meaning |
|-------|---------|
| `warnings_warn` | Python `warnings.warn(...)` |
| `logger_warning` | stdlib logger `.warning(...)` |
| `logger_warning_once` | stdlib logger `.warning_once(...)` |
| `minor_issues_dict` | HF's internal `minor_issues` dict (user-observable via strict-mode raise or log) |
| `none` | No user-visible emission (silent coercion or bare raise with no warning prefix) |
| `runtime_exception` | Exception raised at engine construct / runtime |

### `added_by`

The provenance of the rule - which pipeline component produced it.

```
  AddedBy values and their sources:
  ─────────────────────────────────────────────────────────────────────
  static_miner       AST walking of validator methods
  dynamic_miner      combinatorial probing (raise/no-raise observation)
  pydantic_lift      model_json_schema() + FieldInfo.metadata
  msgspec_lift       msgspec.inspect.type_info() + Meta constraints
  dataclass_lift     dataclasses.fields() + Literal[...] annotations
  manual_seed        hand-written by a maintainer (pipeline-failure debt;
                     use sparingly, add justification comment)
  runtime_warning    proposed by feedback loop from observed logger.warning_once
                     emissions (needs human generalisation before landing)
  observed_collision proposed by feedback loop from config-hash collision
                     detection (needs human generalisation before landing)
```

```
  Provenance flow diagram:

  Engine library source
         │
         ├──► AST walk ──────────────────────────────────► static_miner
         │
         ├──► Combinatorial probes ──────────────────────► dynamic_miner
         │
         ├──► pydantic.BaseModel.model_json_schema() ────► pydantic_lift
         │
         ├──► msgspec.inspect.type_info() ───────────────► msgspec_lift
         │
         ├──► dataclasses.fields() + Literal[...] ───────► dataclass_lift
         │
         ├──► Maintainer (coverage gap, justified) ──────► manual_seed
         │
         ├──► Runtime warning feedback loop ─────────────► runtime_warning
         │
         └──► Observed-collision feedback loop ───────────► observed_collision
```

### `miner_source`

The `{path, method, line_at_scan}` record pointing back to the library source.

- `path`: relative path within the library's source tree (e.g. `transformers/generation/configuration_utils.py`).
- `method`: the method name where the detector found this rule.
- `line_at_scan`: line number at the time of mining. This will drift when the library is updated; it is for human inspection only, not machine comparison.

### `cross_validated_by`

When two or more miners independently emit a rule with the same fingerprint (same `engine + severity + match.fields`), `build_corpus.py` keeps one rule as primary (`added_by` = primary source) and records the secondary source in `cross_validated_by`. Cross-validation is evidence that the rule is real: independent paths agree.

---

## Match predicate operators

Full reference for the `match.fields` operator keys:

| Operator key | Fires when | Notes |
|---|---|---|
| `"=="`/ `"equals"` | `field == value` | Word and symbol forms are aliases |
| `"!="` / `"not_equal"` | `field != value` | None-safe (does not fire if field is None) |
| `"<"` | `field < value` | None-safe |
| `"<="` | `field <= value` | None-safe |
| `">"` | `field > value` | None-safe |
| `">="` | `field >= value` | None-safe |
| `"in"` | `field in [v1, v2, ...]` | Spec must be list/tuple/set |
| `"not_in"` | `field not in [v1, v2, ...]` | None-safe; spec must be list/tuple/set |
| `"present"` | `field is not None` | |
| `"absent"` | `field is None` | |
| `"type_is"` | `type(field).__name__ in name_set` | Accepts string or list of strings |
| `"type_is_not"` | `type(field).__name__ not in name_set` | None-safe |
| `"divisible_by"` | `field % divisor == 0` | Both operands must be non-bool ints; `b=0` → False |
| `"not_divisible_by"` | `field % divisor != 0` | Both operands must be non-bool ints; `b=0` → False |

### Bare value shorthand

A bare value (not a dict) in the `match.fields` spec is shorthand for equality:

```yaml
# These two are equivalent:
transformers.sampling.num_beams: 1
transformers.sampling.num_beams:
  "==": 1
```

### `@field_ref` cross-field references

Any operator value that starts with `@` is resolved as a field reference:

```yaml
transformers.sampling.num_beams:
  not_divisible_by: "@num_beam_groups"
  # "@num_beam_groups" resolves as a sibling:
  # config.transformers.sampling.num_beam_groups

transformers.sampling.num_beams:
  not_divisible_by: "@transformers.sampling.num_beam_groups"
  # Dotted ref resolves from the config root.
  # Equivalent to the sibling form when the parent namespace is the same.
```

---

## `manual_seed` rules

`manual_seed` rules are hand-written by a maintainer. They exist for coverage gaps where the miner pipeline cannot mechanically derive the constraint (e.g. a type-check rule in a library method the static miner does not walk, or a constraint that requires understanding semantics the AST cannot express).

`manual_seed` is **pipeline-failure debt**: the right long-term fix is extending the miner to cover the gap. Each `manual_seed` entry should carry a justification comment explaining why the miner cannot cover it and what would be needed to close the gap.

```yaml
- id: bitsandbytes_load_in_4bit_and_8bit_mutually_exclusive
  ...
  added_by: manual_seed
  # Justification: BitsAndBytesConfig.__init__ checks load_in_4bit AND load_in_8bit
  # in the same branch, but the dynamic miner's BNB cluster was not added in the
  # refactor (scope decision). Extend the dynamic miner with a bitsandbytes_quant
  # cluster to close this gap.
```

---

## Schema version history

| Version | Changes |
|---------|---------|
| `1.0.0` | Initial release. `added_by` as single string, `cross_validated_by` optional list, `mined_at` top-level field. Replaces pre-1.0 `walked_at` field name. |

---

## Corpus invariants

The CI pipeline enforces these invariants on every corpus file:

1. `schema_version` major must equal `SUPPORTED_MAJOR_VERSION` in `loader.py`.
2. Every `added_by` value must be in the `AddedBy` Literal.
3. Every `severity` value must be in `{"error", "warn", "dormant"}`.
4. Every `expected_outcome.outcome` must be in `Outcome`.
5. Every `expected_outcome.emission_channel` must be in `EmissionChannel`.
6. All `id` values within one corpus file must be unique.
7. Dormant rules must converge to a stable fixpoint (verified by `_fixpoint_test.py`).
8. Vendor-CI gate: `kwargs_positive` must trigger the rule, `message_template` must match, `kwargs_negative` must not trigger.

---

## See also

- [parameter-discovery.md](parameter-discovery.md) - how the corpus is consumed at runtime
- [miner-pipeline.md](miner-pipeline.md) - how the corpus is built
- [extending-miners.md](extending-miners.md) - adding rules for new engines
- [architecture-overview.md](architecture-overview.md) - system overview
