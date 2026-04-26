# Parameter Discovery and Config Validation

This document covers the runtime config-validation pipeline: how a user's `ExperimentConfig` is evaluated against the vendored rule corpus, what the loader grammar means, and what happens when a rule fires.

**Audience:** end users debugging a rejected config; extenders writing new corpus rules; anyone wanting to understand the error messages llem produces.

For the compile-time side (how the corpus is built), see [miner-pipeline.md](miner-pipeline.md).

---

## Why configs are rejected before engine initialisation

Engine initialisation is expensive: model weights load from disk, CUDA contexts initialise, and TensorRT-LLM may need to compile an engine (minutes). A rejected config discovered after two minutes of initialisation wastes GPU time.

llem evaluates each submitted `ExperimentConfig` against a pre-computed corpus of validation rules before the engine starts. Invalid combinations are caught at config-parse time - milliseconds, not minutes.

---

## Data flow: user config to validation result

```
  User submits config
  (YAML file, CLI flags, or Python API)
               │
               ▼
  ExperimentConfig (Pydantic model)
  parsed and validated by Pydantic
               │
               ▼
  _apply_vendored_rules in config/models.py
  calls VendoredRulesLoader().load_rules(engine)
               │
               ▼
  loader.py: parse vendored JSON
  ┌─────────────────────────────────────────┐
  │  VendoredRules                          │
  │  engine: "transformers"                 │
  │  schema_version: "1.0.0"               │
  │  rules: [Rule, Rule, Rule, ...]         │
  └─────────────────────────────────────────┘
               │
               ▼
  for each Rule in rules:
    Rule.try_match(config)
               │
               ├── None (predicate did not fire) → skip
               │
               └── RuleMatch (predicate fired)
                       │
                       ├── severity: "error"
                       │   → raise ValueError (Pydantic surfaces it
                       │     as ValidationError) with message
                       │
                       ├── severity: "warn"
                       │   → emit warning to user
                       │
                       └── severity: "dormant"
                           → annotate config
                           → log "field X will be silently
                              ignored or normalised by the engine"
```

---

## The loader grammar

The `match.fields` section of each corpus rule contains predicates expressed in a small domain-specific grammar. The loader's `evaluate_predicate()` function implements it.

### Grammar tree

```
  match.fields value types
  ─────────────────────────
  bare value                   field == value  (equality shorthand)
  dict with operator keys:
  ├── comparison
  │   ├── "=="  / "equals"     equality
  │   ├── "!="  / "not_equal"  inequality
  │   ├── "<"                  less-than
  │   ├── "<="                 less-than-or-equal
  │   ├── ">"                  greater-than
  │   └── ">="                 greater-than-or-equal
  │
  ├── membership
  │   ├── "in"                 value in [v1, v2, ...]
  │   └── "not_in"             value not in [v1, v2, ...]
  │
  ├── presence
  │   ├── "present"            field is not None
  │   └── "absent"             field is None
  │
  ├── type check
  │   ├── "type_is"            type(field).__name__ in name_set
  │   └── "type_is_not"        type(field).__name__ not in name_set
  │
  └── cross-field divisibility
      ├── "divisible_by"       a % b == 0
      └── "not_divisible_by"   a % b != 0  (b=0 → False, no rule fires)
```

### Field path resolution

Field paths are dotted strings resolved against the config model attribute by attribute:

```yaml
match:
  fields:
    transformers.sampling.num_beams:
      ">": 1
    transformers.sampling.num_beam_groups:
      ">": 1
```

- `transformers.sampling.num_beams` resolves as `config.transformers.sampling.num_beams`.
- Pydantic models, dataclasses, and plain dicts are all supported.
- A missing attribute at any path segment yields `None` - the predicate does not fire (rules do not produce false positives on configs that simply lack a field).

### `@field_ref` cross-field references

Operator values may be `@field_path` strings, which are resolved against the same config before evaluation. This is how cross-field constraints are expressed:

```yaml
match:
  fields:
    transformers.sampling.num_beams:
      not_divisible_by: "@num_beam_groups"
```

`@num_beam_groups` resolves as a sibling field (relative to `transformers.sampling.num_beams`'s parent namespace). Dotted refs (e.g. `@transformers.sampling.num_beam_groups`) resolve from the config root.

### Loader grammar examples

```yaml
# Single-field range: temperature must be positive
match:
  fields:
    vllm.sampling.temperature:
      ">": 0.0

# Value allowlist: cache_implementation must be one of these
match:
  fields:
    transformers.sampling.cache_implementation:
      in: ["static", "sliding_window", "hybrid"]

# Cross-field divisibility: num_beams must divide evenly by num_beam_groups
match:
  fields:
    transformers.sampling.num_beams:
      not_divisible_by: "@num_beam_groups"

# Multi-field gate: rule fires only when both conditions hold
match:
  fields:
    transformers.sampling.num_beams:
      ">": 1
    transformers.sampling.diversity_penalty:
      "==": 0.0

# Type check: field must be a float, not an int
match:
  fields:
    transformers.sampling.temperature:
      type_is_not: "int"
```

---

## Severity levels

Each rule has a severity that determines how the loader responds when the predicate fires.

```
  severity: "error"
  ├── The engine raises if the config is submitted as-is.
  └── Loader raises ValueError before engine initialisation.
      Pydantic surfaces it to the user as a ValidationError.
      Message template is rendered with declared_value substituted.
      Example: "num_beams (2) is not divisible by num_beam_groups (3)"

  severity: "warn"
  ├── The engine announces a suboptimal setting but still proceeds.
  └── Loader emits a warning to the user.
      Example: "temperature=0 with do_sample=True; engine will warn"

  severity: "dormant"
  ├── The engine silently normalises or ignores the field.
  │   The user's declared value is not the effective value.
  └── Loader annotates the config:
      "field X will be silently coerced by the engine to Y"
      Example: "seed=-1 will be normalised to None by the engine"
```

### Dormant rules: the "silent surprise" class

`dormant` rules are the most subtle. They describe configurations where the engine accepts the value but silently normalises it to something else. Without the corpus, the user would submit `seed=-1`, not see any error, and later discover the seed was ignored.

The `expected_outcome.normalised_fields` list in a dormant rule tells the loader which fields are affected. The fixpoint contract (`_fixpoint_test.py`) asserts that applying dormant rules to a config converges to a stable state - no two dormant rules should conflict by normalising the same field to different values under the same conditions.

---

## Error messages

When a rule fires at `error` severity, the loader renders the rule's `message_template` using field values from the matched config.

Template substitution variables:
- `{declared_value}` - the value of the triggering field.
- `{effective_value}` - the normalised value (dormant rules only).
- `{rule_id}` - the rule's identifier.
- Any `match.fields` key - the actual field value.

Example error message from the corpus:

```
ValidationError: `diversity_penalty` is not 0.0 or `num_beam_groups` is
not 1, triggering group beam search. In this generation mode,
`diversity_penalty` should be greater than `0.0`, otherwise your groups will
be identical.
```

If no template is available, the loader falls back to `[{rule_id}] <no message template>` rather than raising silently.

---

## Library version resolution

The vendored JSON carries the engine version the corpus was validated against. When the loader loads the vendored JSON:

1. It checks `schema_version` major against `SUPPORTED_MAJOR_VERSION`. A major-version mismatch raises `UnsupportedSchemaVersionError` (the package is incompatible with the installed corpus version).
2. It parses all rules with strict enum validation: unknown `added_by` values raise `UnknownAddedByError`; unknown severity values raise `UnknownSeverityError`.

The loader does not check whether the currently installed engine library version matches the corpus version. That alignment is enforced at corpus-build time (the miner's `TESTED_AGAINST_VERSIONS` + `check_installed_version`) and at runtime via the engine's own constructor validation.

---

## Gap reporting

When a user submits a config combination that no rule in the corpus addresses, no validation fires - the combination passes through. This is by design (the corpus is recall-first, not exhaustive), but it means some invalid combinations are caught only by the engine constructor.

Gap reporting surfaces these at the `dormant` level or via a separate gap-detection pipeline. When a `gap_detected: true` group appears in experiment results, it indicates the config combination triggered a library-side normalisation that the corpus did not yet describe.

Extending the corpus to cover new gap classes is done by adding a miner cluster or a `manual_seed` rule. See [extending-miners.md](extending-miners.md).

---

## Corpus vs vendored JSON: the two-file structure

The corpus YAML (`configs/validation_rules/{engine}.yaml`) is the authoritative human-reviewable format. The vendored JSON (`src/llenergymeasure/config/vendored_rules/{engine}.json`) is the CI-validated version that ships with the package.

```
  YAML corpus                         Vendored JSON
  ─────────────────────────────────   ─────────────────────────────────
  Human-reviewable                    Machine-parsed only
  Source of truth (git-tracked)       Output of vendor_rules.py
  Carries declared expected_outcome   Carries observed outcomes (CI run)
  Read by vendor_rules.py             Read by loader.py at runtime
  Regenerated by miners               Regenerated by vendor_rules.py
  In configs/validation_rules/        In src/.../vendored_rules/
```

The loader overlays vendored observations onto the corpus so downstream consumers see CI-validated truth. When the vendored JSON is absent (e.g. in a local development environment without a vendor run), the loader falls back to the YAML corpus.

---

## Loader API

The loader is in `src/llenergymeasure/config/vendored_rules/loader.py`.

```python
from llenergymeasure.config.vendored_rules.loader import (
    VendoredRules,
    VendoredRulesLoader,
    Rule,
    RuleMatch,
)

# Load corpus for an engine
loader = VendoredRulesLoader()
rules = loader.load_rules("transformers")

# Match against a config
for rule in rules.rules:
    match = rule.try_match(config)
    if match is not None:
        print(rule.severity, rule.render_message(match))
```

For higher-level use cases, see `llenergymeasure.api.report_gaps.load_rules_corpus`, which loads all configured engines via a shared loader. Per-instance caching in `VendoredRulesLoader` ensures the corpus JSON is parsed once per engine per process; tests can construct a fresh loader for isolation.

---

## Troubleshooting: common error messages

### "ValidationError: `num_beams` is not divisible by `num_beam_groups`"

Rule: `transformers_beam_search_num_beams_not_divisible_by_num_beam_groups`

The transformers engine requires `num_beams` to be an exact multiple of `num_beam_groups` for group beam search. Set both to compatible values: e.g. `num_beams=4, num_beam_groups=2`.

### "ValidationError: `diversity_penalty` is not 0.0 or `num_beam_groups` is not 1..."

Rule: `transformers_beam_search_diversity_penalty_eq_0p0`

When `num_beams > 1` and `num_beam_groups > 1` (group beam search mode), `diversity_penalty` must be greater than 0.0. Set `diversity_penalty` to a positive value, or disable group beam search.

### "Warning: field `seed` will be silently normalised to None by the engine"

Rule: a dormant rule matching `seed=-1`. The engine treats -1 as "no seed" and normalises it to `None`. Set an explicit non-negative seed, or leave the field unset.

### "UnsupportedSchemaVersionError"

The vendored JSON in the installed package was built with a schema major version the current loader does not understand. This indicates a library/package version mismatch. Update llenergymeasure to the version that matches your installed engines.

---

## See also

- [architecture-overview.md](architecture-overview.md) - system overview
- [validation-rule-corpus.md](validation-rule-corpus.md) - corpus YAML format reference
- [miner-pipeline.md](miner-pipeline.md) - how the corpus is built
- [extending-miners.md](extending-miners.md) - adding new rules
- [engines.md](engines.md) - engine configuration reference
- [troubleshooting.md](troubleshooting.md) - general troubleshooting guide
