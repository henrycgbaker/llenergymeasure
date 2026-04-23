# Validation rules corpus

This directory is the SSOT for per-engine configuration validation rules — the
data that tells users "this combination will error", "this field will be
silently ignored", or "this will trigger a library warning". Rules live as
**data**, not code; a single generic Pydantic validator
(landing in phase 50.2c) consumes the corpus and emits error / warn / dormant
annotations per rule.

Design doc: [`.product/designs/config-deduplication-dormancy/runtime-config-validation.md`](../../.product/designs/config-deduplication-dormancy/runtime-config-validation.md)
(primary).

## Layout

- `transformers.yaml` — transformers + BitsAndBytesConfig rules
- `vllm.yaml` — vLLM rules (landing in a later per-engine phase)
- `tensorrt.yaml` — TensorRT-LLM rules (landing in a later per-engine phase)

## File envelope

```yaml
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
walker_pinned_range: ">=4.50,<5.0"
walked_at: "2026-04-23T00:00:00Z"
rules:
  - id: ...
  - id: ...
```

- `schema_version` — semver. Loader supports major 1; mismatches raise
  `UnsupportedSchemaVersionError` (see
  `src/llenergymeasure/engines/vendored_rules/loader.py`).
- `engine_version` — the library version the corpus was seeded against.
  Informational; the vendor CI pipeline (50.2b) will revalidate against each
  Dockerfile-pinned version.
- `walker_pinned_range` — echoes the walker module's `TESTED_AGAINST_VERSIONS`
  constant. CI fails if the installed library is outside range.
- `walked_at` — ISO-8601 UTC timestamp of the walker run. Byte-reproducibility
  via `LLENERGY_WALKER_FROZEN_AT=<timestamp>` env variable.

## Rule schema

Every entry under `rules:` must populate the following fields.

```yaml
- id: transformers_greedy_strips_temperature
  engine: transformers
  library: transformers
  rule_under_test: >
    GenerationConfig.validate() records dormant `temperature` when
    do_sample=False and `temperature` is set to a non-default value
  severity: dormant                      # dormant | warn | error
  native_type: transformers.GenerationConfig
  walker_source:
    path: transformers/generation/configuration_utils.py
    method: validate
    line_at_scan: 598
    walker_confidence: high              # high | medium | low
  match:
    engine: transformers
    fields:
      transformers.sampling.do_sample: false
      transformers.sampling.temperature: {present: true, not_equal: 1.0}
  kwargs_positive:
    do_sample: false
    temperature: 0.9
  kwargs_negative:
    do_sample: true
    temperature: 0.9
  expected_outcome:
    outcome: dormant_announced           # pass | warn | error | dormant_silent | dormant_announced
    emission_channel: minor_issues_dict  # warnings_warn | logger_warning | logger_warning_once | minor_issues_dict | runtime_exception | none
    normalised_fields: []                # populated for dormant_silent outcomes
  message_template: |
    `do_sample=False` is set, so `temperature` ({declared_value}) has no effect.
    Remove it or set do_sample=True.
  references:
    - "transformers.GenerationConfig.validate() (line ~598)"
  added_by: ast_walker                   # ast_walker | manual_seed | runtime_warning_pr | h3_collision_pr
  added_at: "2026-04-23"
```

### Match predicate operators

`match.fields` maps dotted field paths (e.g.
`transformers.sampling.temperature`) to a predicate. Shapes:

| Shape | Meaning |
|---|---|
| Bare value (`0.9`, `false`, `"greedy"`) | Equality. |
| `{">": 1}`, `{">=": 1}`, `{"<": 1}`, `{"<=": 1}` | Comparison. None-safe. |
| `{"==": x}`, `{"equals": x}` | Equality (long form). |
| `{"!=": x}`, `{"not_equal": x}` | Inequality, None-safe. |
| `{"in": [a, b]}`, `{"not_in": [a, b]}` | Membership. |
| `{"present": true}`, `{"absent": true}` | None-presence check. |
| Multi-key dict | All predicates AND-combined. |

Multi-key example — "field is set AND isn't default":

```yaml
transformers.sampling.temperature: {present: true, not_equal: 1.0}
```

Every predicate in `match.fields` must hold for `rule.try_match(config)` to
return a `RuleMatch`.

### ID convention

`{engine}_{rule_summary_snake}` — unique within engine. Walker-authored IDs
encode the pattern (`greedy_strips_X`, `single_beam_strips_X`,
`bnb_X_type`); manual seeds use a descriptive snake-case tail.

### Corpus invariants

These are enforced today via `tests/unit/engines/vendored_rules/test_corpus_invariants.py`.
Phase 50.2b's vendor CI gate extends them.

1. Every rule has a unique `id` within the engine.
2. `match.fields` is non-empty.
3. Both `kwargs_positive` and `kwargs_negative` are populated.
4. `severity` ↔ `expected_outcome.outcome` consistency:
   - `error` ↔ `error`
   - `warn` ↔ `warn`
   - `dormant` ↔ `dormant_silent | dormant_announced`
5. `dormant_silent` outcomes imply non-empty `expected_outcome.normalised_fields`.

## Adding a rule

### Via walker (preferred)

1. Rerun the walker for the engine:
   `python -m scripts.walkers.{engine} --out configs/validation_rules/{engine}.yaml`
   (optionally with `LLENERGY_WALKER_FROZEN_AT=<iso-utc>` for reproducibility).
2. Inspect the diff against the previous corpus file. Walker-marked `high`
   entries are usually safe; `medium`/`low` need explicit review.
3. Open a draft PR. Phase 50.2b's CI gate will verify each rule fires on
   `kwargs_positive` and doesn't fire on `kwargs_negative`.

### Via manual seed

Some rules live outside walker scope (e.g., vLLM's greedy-strip in
`EngineArgs.create_engine_config()` rather than `__post_init__`). Author the
YAML entry directly:

- Pick a descriptive `id`.
- Set `added_by: manual_seed` and today's `added_at`.
- Cite the source location in `references`.
- Populate both `kwargs_positive` and `kwargs_negative`; the vendor CI step
  verifies both.

### Via feedback loop

Phase 50.3 introduces two automated discovery channels: runtime warning
capture (`runtime_warning_pr`) and H3-collision detection (`h3_collision_pr`).
Both open draft PRs with `added_by` set accordingly.

## PR review checklist

When reviewing a corpus PR:

- [ ] Invariants pass (`pytest tests/unit/engines/vendored_rules/test_corpus_invariants.py`).
- [ ] Each new rule's `message_template` reads correctly when substituted.
- [ ] `kwargs_positive` genuinely triggers the rule in the target library.
- [ ] `kwargs_negative` genuinely does NOT trigger it.
- [ ] `walker_source.path` is a stable relative path (site-packages-rooted).
- [ ] Confidence downgrades (`high` → `medium`/`low`) carry a justifying comment.

## Non-goals

- This directory does not drive sweep generation — that's
  `configs/example-study-full.yaml` and related. Corpus YAMLs are data, not
  executable configs.
- The corpus is not consumed at runtime yet (phase 50.2c wires the generic
  `@model_validator`). Today the loader parses the corpus so tests and 50.2b
  tooling have a stable entry point.
