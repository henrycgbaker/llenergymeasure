# Architecture Overview

This document is the entry point to the LLenergyMeasure architecture documentation suite. It introduces the two major subsystems - the invariant miner pipeline and the runtime config-validation pipeline - and shows how they connect to the broader measurement framework.

**Start here.** Deep-dive docs for each subsystem are linked throughout.

---

## Who this is for

- **End users** running experiments: read the "Why configs are rejected" section and the [parameter-discovery](parameter-discovery.md) guide.
- **Engine extenders** adding a new backend: read this overview, then [miner-pipeline](miner-pipeline.md) and [extending-miners](extending-miners.md).
- **Researchers and paper readers**: read this overview, then [research-context](research-context.md) for academic positioning.

---

## System overview

LLenergyMeasure has two pipelines that work together to give users early, actionable feedback when their configs are invalid before an expensive engine initialisation takes place.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  COMPILE-TIME (CI / Renovate-driven library bumps)                  │
  │                                                                     │
  │   Engine library source                                             │
  │   (transformers, vLLM, TRT-LLM)                                    │
  │              │                                                      │
  │              ▼                                                      │
  │   ┌─────────────────────────┐                                       │
  │   │   Invariant Miner       │  scripts/miners/                     │
  │   │   Pipeline              │                                       │
  │   │  ┌──────────────┐       │                                       │
  │   │  │ static miner │       │  AST walking of validator methods    │
  │   │  └──────────────┘       │                                       │
  │   │  ┌──────────────┐       │                                       │
  │   │  │dynamic miner │       │  combinatorial probing               │
  │   │  └──────────────┘       │                                       │
  │   │  ┌──────────────┐       │                                       │
  │   │  │  lift modules│       │  pydantic / msgspec / dataclass      │
  │   │  └──────────────┘       │                                       │
  │   │         │               │                                       │
  │   │    staging files        │                                       │
  │   │         │               │                                       │
  │   │    build_corpus.py      │  merge + dedup + fingerprint         │
  │   │         │               │                                       │
  │   │    vendor_rules.py      │  replay against live library         │
  │   │         │               │                                       │
  │   │  corpus YAML            │  configs/validation_rules/{e}.yaml   │
  │   │  vendored JSON          │  src/.../vendored_rules/{e}.json     │
  │   └─────────────────────────┘                                       │
  └─────────────────────────────────────────────────────────────────────┘
                      │
                      │ vendored JSON ships with package
                      │
  ┌─────────────────────────────────────────────────────────────────────┐
  │  RUNTIME (user submits ExperimentConfig)                            │
  │                                                                     │
  │   User YAML / Python API                                            │
  │              │                                                      │
  │              ▼                                                      │
  │   ┌─────────────────────────┐                                       │
  │   │  Config Validation      │  src/.../config/vendored_rules/      │
  │   │  Pipeline               │  loader.py                           │
  │   │                         │                                       │
  │   │  ┌───────────────┐      │                                       │
  │   │  │ loader.py     │      │  parse corpus + evaluate predicates  │
  │   │  └───────────────┘      │                                       │
  │   │  ┌───────────────┐      │                                       │
  │   │  │ rule match    │      │  try_match() per rule per engine     │
  │   │  └───────────────┘      │                                       │
  │   │         │               │                                       │
  │   │    error / warn /       │                                       │
  │   │    dormant annotation   │                                       │
  │   └─────────────────────────┘                                       │
  │              │                                                      │
  │              ▼                                                      │
  │   User sees rejection BEFORE engine initialisation                  │
  │   (engine initialisation is expensive; this saves GPU time)         │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## The two pipelines

### 1. The invariant miner pipeline

**What it does:** Extracts validation invariants from ML engine library source code and packages them into a versioned corpus of structured rules. Runs in CI whenever a library version bumps (Renovate-driven).

**Inputs:** Engine library source code (at a pinned version).

**Outputs:** `configs/validation_rules/{engine}.yaml` (authoritative corpus) and `src/llenergymeasure/config/vendored_rules/{engine}.json` (ship-ready vendored observations).

**Three components:**
- Static miner - walks Python AST of validator methods; no constructor calls.
- Dynamic miner - instantiates config classes with combinatorial probe values; observes raise/no-raise patterns.
- Lift modules (`_pydantic_lift.py`, `_msgspec_lift.py`, `_dataclass_lift.py`) - extract constraints directly from type-system metadata (Pydantic `FieldInfo`, msgspec `Meta`, stdlib `Literal[...]`).

Deep-dive: [miner-pipeline.md](miner-pipeline.md)

### 2. The parameter-discovery / config-validation pipeline

**What it does:** At runtime, when a user submits an `ExperimentConfig`, evaluates each rule in the vendored corpus against the config and rejects invalid combinations before engine initialisation begins.

**Inputs:** User's `ExperimentConfig`; vendored corpus JSON.

**Outputs:** Error / warning / dormant annotations surfaced to the user via the CLI or the Python API.

**Key components:**
- `loader.py` - parses the corpus and exposes `Rule.try_match()`.
- Loader grammar - the predicate DSL (`type_is`, `@field_ref`, `not_divisible_by`, etc.).
- Gap reporting - flags when a config combination the corpus has no rule for is encountered.

Deep-dive: [parameter-discovery.md](parameter-discovery.md)

---

## Broader framework context

Both pipelines sit inside the larger LLenergyMeasure architecture. The config-validation pipeline plugs into Layer 0 (`config/`), which the rest of the stack builds on.

```
  Layer 6  cli/            llem run, llem config
               │
  Layer 5  api/            run_experiment(), run_study()
               │
  Layer 4  study/          StudyRunner, sweep expansion
               │
  Layer 3  harness/        MeasurementHarness, energy sampling
               │
  Layer 2  engines/        PyTorch, vLLM, TensorRT-LLM plugins
               │
  Layer 1  infra/          Docker runner, container entrypoint
               │
  Layer 0  config/  ◄──── config validation pipeline lives here
           domain/         vendored_rules/loader.py
           device/
           utils/
```

The invariant miner pipeline lives in `scripts/miners/` - it is a build-time tool, not a library module. Its output is the vendored corpus that ships with the package.

---

## Data flow: end-to-end

```
  Library version bump (e.g. transformers 4.56.0 → 4.57.0)
               │
               ▼
  Renovate opens PR bumping Dockerfile ARG
               │
               ▼
  CI fires config-rules-refresh.yml
               │
               ├──► static miner runs (GH-hosted runner, CPU only)
               │
               ├──► dynamic miner runs (GH-hosted runner)
               │
               ├──► lift modules run (pydantic / msgspec / dataclass)
               │
               ▼
  build_corpus.py merges staging files
  (dedup by fingerprint; static miner wins on match.fields,
   dynamic miner wins on message_template)
               │
               ▼
  vendor_rules.py replays every rule against the live library
  (checks: kwargs_positive raises, message matches template,
            kwargs_negative does NOT raise)
               │
               ├──► divergent rules quarantined to _failed_validation_*.yaml
               │
               └──► confirmed rules written to:
                    configs/validation_rules/{engine}.yaml   (YAML corpus)
                    src/.../vendored_rules/{engine}.json     (vendored JSON)
               │
               ▼
  Bot writes vendored JSON back to PR branch
  CI must be green before merge
               │
               ▼
  Package ships with updated corpus
               │
               ▼
  User submits ExperimentConfig
               │
               ▼
  loader.py evaluates rules against config
               │
               ▼
  Invalid combination caught BEFORE engine initialisation
  User sees: "config rejected: num_beams must be divisible by num_beam_groups"
```

---

## Why validate before engine initialisation?

Engine initialisation is expensive: model weights load from disk, CUDA contexts initialise, and for TensorRT-LLM the engine may need compilation. A rejected config discovered after two minutes of initialisation wastes GPU time and researcher patience.

Pre-construction validation from a corpus catches the most common mistakes at config-parse time - a few milliseconds rather than several minutes.

The corpus complements, rather than replaces, engine-side validation: it captures invariants that fire only in specific combinations (cross-field constraints), silent normalisations (`dormant` rules), and invariants from methods that run at build time rather than construction time.

---

## Why a versioned corpus instead of live introspection?

Live introspection at runtime would require importing each engine at startup - which on vLLM and TRT-LLM means initialising CUDA contexts. The corpus is pre-computed and ships as a JSON file that loads in a few milliseconds with no GPU dependency.

The trade-off is staleness risk: the corpus must be regenerated when the engine library changes. The Renovate-driven refresh loop and the vendor-CI gate together enforce this discipline. See [miner-pipeline.md - Renovate refresh loop](miner-pipeline.md#renovate-driven-refresh-loop).

---

## Key concepts

| Term | Meaning |
|------|---------|
| **Invariant miner** | The umbrella for the mining pipeline; extracts constraints from library source |
| **Static miner** | The AST-walking component; reads source, no constructor calls |
| **Dynamic miner** | The probing component; constructs config objects, observes raises |
| **Lift module** | Type-system adapter; extracts constraints from Pydantic / msgspec / dataclass metadata |
| **Corpus** | The YAML file of extracted, vendor-validated rules for one engine |
| **Vendored JSON** | The CI-observed version of the corpus that ships with the package |
| **Vendor-CI gate** | The step that replays every rule against the live library; divergences fail CI |
| **Fixpoint contract** | `_fixpoint_test.py` - asserts dormant rules converge to a stable state under repeated application |
| **AddedBy** | Provenance field on each rule: `static_miner`, `dynamic_miner`, `pydantic_lift`, `msgspec_lift`, `dataclass_lift`, `manual_seed`, `runtime_warning`, `observed_collision` (full reference in [validation-rule-corpus.md](validation-rule-corpus.md#added_by)) |
| **MinerSource** | The `{path, method, line_at_scan}` record pointing back to the library source line that produced a rule |
| **Loader grammar** | The predicate DSL used in `match.fields`: `in`, `not_in`, `@field_ref`, `not_divisible_by`, `type_is`, etc. |

---

## File and package map

```
  scripts/
  └── miners/                     Invariant miner pipeline (build-time)
      ├── _base.py                Shared infrastructure: RuleCandidate, MinerError types,
      │                           AST primitives, pattern detectors
      ├── _pydantic_lift.py       Pydantic v2 sub-library lift
      ├── _msgspec_lift.py        msgspec sub-library lift
      ├── _dataclass_lift.py      stdlib dataclass sub-library lift
      ├── _fixpoint_test.py       Gate-soundness + corpus fixpoint contract
      ├── transformers_miner.py   Transformers orchestration entry
      ├── transformers_static_miner.py
      ├── transformers_dynamic_miner.py
      ├── vllm_static_miner.py    (in flight)
      ├── vllm_dynamic_miner.py   (in flight)
      ├── tensorrt_static_miner.py  (in flight)
      └── build_corpus.py         Merge + dedup + vendor-validate orchestration

  scripts/
  ├── vendor_rules.py             Replay rules against live library; write vendored JSON
  └── _vendor_common.py           Shared capture + comparison utilities

  configs/
  └── validation_rules/
      ├── transformers.yaml       Authoritative corpus (transformers, 46 rules)
      └── _staging/               Per-miner staging output (not committed)

  src/llenergymeasure/config/
  └── vendored_rules/
      ├── loader.py               Runtime corpus consumer + predicate engine
      ├── transformers.json       Vendored observations (ships with package)
      └── __init__.py
```

---

## See also

- [miner-pipeline.md](miner-pipeline.md) - invariant miner deep-dive
- [parameter-discovery.md](parameter-discovery.md) - runtime validation pipeline
- [validation-rule-corpus.md](validation-rule-corpus.md) - corpus YAML format reference
- [extending-miners.md](extending-miners.md) - how to add a new engine miner
- [research-context.md](research-context.md) - academic positioning
- [engines.md](engines.md) - engine configuration reference
- [methodology.md](methodology.md) - energy measurement methodology
- [schema-refresh.md](schema-refresh.md) - Renovate-driven schema refresh pipeline
