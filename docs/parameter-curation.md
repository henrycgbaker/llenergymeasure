# Parameter Curation

> **Note:** This document covers engine-API-parameter introspection and Pydantic-model curation (what fields each engine accepts, type information, drift detection). For the runtime validation of parameter *values* (how invalid combinations are caught before engine initialisation), see [parameter-discovery.md](parameter-discovery.md).

---

llem exposes engine parameters to users through hand-authored Pydantic models. This document explains how those models stay in sync with the underlying engines.

---

## Overview

```
  programmatic discovery                  Pydantic curation
  (scripts/discover_*.py)                 (config/engine_configs.py)
  introspect engine APIs                  hand-authored sub-config models
  → discovered_schemas/*.json             expose typed, documented fields
            │                                         │
            └──────────────┬──────────────────────────┘
                           ▼
                     drift checker
              (scripts/check_pydantic_matches_discovered.py)
                           │
               flags Pydantic fields with no
               corresponding discovered entry
                           │
                    LLEM_NATIVE_FIELDS
              the "yes, this divergence is intentional"
              allowlist — suppresses known-good exceptions
```

---

## Programmatic discovery

`scripts/discover_*.py` introspects each engine's public Python API (e.g. `inspect.signature(vllm.LLM.__init__)`, `inspect.signature(AutoModelForCausalLM.from_pretrained)`) and writes the result to `src/llenergymeasure/config/discovered_schemas/{engine}.json`.

These JSON files are the ground truth for "what parameters does this engine version accept". They are vendored into the repo and regenerated via the schema-refresh pipeline when an engine version bumps (see [schema-refresh.md](schema-refresh.md)).

---

## Pydantic curation

`src/llenergymeasure/config/engine_configs.py` contains hand-authored Pydantic models that llem exposes to users. Curation decisions:

- **Field names match native engine names.** A field called `quant_config` maps directly to the engine kwarg `quant_config`. No translation layer, no llem aliases.
- **Sub-configs group related parameters.** e.g. `TensorRTKvCacheConfig` groups all kv-cache knobs under `tensorrt.kv_cache_config.*`. The sub-config name matches the native engine kwarg name.
- **Types may be narrowed.** A field typed `str` in discovery might become `Literal["bfloat16", "float16", "float32"]` in curation — this is intentional and allowed by the drift checker.
- **Descriptions are added.** Pydantic `Field(description=...)` docs are user-facing; discovery has none.

---

## Drift checker

`scripts/check_pydantic_matches_discovered.py` compares the set of leaf field names in the Pydantic models against the discovered schemas and reports two kinds of drift:

| Kind | Meaning |
|------|---------|
| `pydantic_only` | Pydantic has a field that discovery doesn't — likely a stale field that was removed from the engine, or a kwargs-passed field invisible to signature inspection |
| `type_mismatch` | Both sides have the field but with different types (beyond intentional narrowing) |

Run it locally:

```bash
python scripts/check_pydantic_matches_discovered.py
```

CI runs it automatically on every PR.

---

## LLEM_NATIVE_FIELDS

Some Pydantic fields legitimately have no discovered counterpart. Common reasons:

| Reason | Example |
|--------|---------|
| Passed via `**kwargs`, invisible to `inspect.signature` | `transformers.dtype` — `from_pretrained` accepts it as a kwarg alias |
| llem surfaces a sub-config field that the engine accepts as a flat kwarg at a different nesting level | `tensorrt.quant_algo` inside `TensorRTQuantConfig` |
| Beam-search or speculative-decoding params from a separate params class | `vllm.beam_width` (from `BeamSearchParams`, not `LLM.__init__`) |

These are listed in `LLEM_NATIVE_FIELDS` in the drift checker. Each entry suppresses one `pydantic_only` warning for a named `(engine, field_name)` pair.

**When to add an entry:** when the drift checker flags a `pydantic_only` field and you have confirmed it is intentionally in the Pydantic model but unreachable by signature-based discovery. Add a comment explaining why.

**When to remove an entry:** when the corresponding Pydantic field is deleted. Stale entries are harmless but misleading — remove them during the same PR that removes the field.

**Never add an entry to paper over a naming divergence.** If a Pydantic field is named differently from the engine kwarg, rename the field instead.

---

## See also

- [parameter-discovery.md](parameter-discovery.md) - config validation pipeline (how invalid combinations are caught)
- [schema-refresh.md](schema-refresh.md) - Renovate-driven schema refresh
- [engines.md](engines.md) - engine configuration reference
