# Schema Refresh Guide

`llenergymeasure` ships **vendored JSON schemas** that describe every parameter each supported
framework (vLLM, TensorRT-LLM, HuggingFace Transformers) exposes at its pinned Docker image
version. These schemas live at `src/llenergymeasure/config/discovered_schemas/{vllm,tensorrt,transformers}.json`
and are the single source of truth for *"what CAN I configure per framework"*. The Pydantic
models in `config/backend_configs.py` remain the source of truth for *"what SHOULD I configure"* —
the small, curated, measurement-relevant subset.

This guide explains how the vendored schemas stay in sync with upstream releases, what to do
when a framework bump introduces a breaking change, and how to refresh a schema manually.

> **TL;DR.** You almost never have to touch schemas by hand. Renovate opens a Dockerfile
> bump PR, GitHub Actions re-runs discovery inside the new image, commits the regenerated
> schema back to the PR branch, computes a semantic diff, and labels the PR `schema-safe`
> or `schema-breaking`. Safe PRs auto-merge. Breaking PRs are gated on human review and
> blocked until a contributor updates Pydantic, YAMLs, and CHANGELOG.

---

## Why vendor schemas?

Frameworks expose hundreds of parameters across their `EngineArgs`, `SamplingParams`,
`BitsAndBytesConfig`, `GenerationConfig`, `LlmArgs`, and `BuildConfig` classes. These surfaces
change between releases — fields are added, defaults shift, parameters are deprecated or
renamed. Historically the codebase tracked a small hand-curated subset in
`config/backend_configs.py` plus several manually maintained "support" dicts in `ssot.py`.
That approach had two failure modes:

1. **Drift.** When upstream added a useful new parameter, users had no way to pass it unless
   someone manually added it to the Pydantic model. The hand-curated subset was always
   behind reality.
2. **Duplication.** The same parameter description, default, and type lived in the framework's
   source, our Pydantic model, our capability dicts, and our generated parameter docs — four
   places that had to agree.

Vendoring the framework's own schema at a specific pinned image version solves both. The
schema becomes the authoritative catalogue of every native parameter; the Pydantic model is a
*filtered projection* of it; capability matrices and parameter docs derive from the
vendored JSON rather than being typed by hand.

---

## How the pipeline works end-to-end

```
┌──────────────────┐     1. upstream release       ┌──────────────────────┐
│ vllm/vllm-openai │ ─────────────────────────────▶│ Renovate (continuous │
│ v0.7.3 → v0.7.4  │                               │ monitoring)          │
└──────────────────┘                               └──────────┬───────────┘
                                                              │ 2. opens PR
                                                              │    bumping
                                                              ▼    FROM tag
                    ┌───────────────────────────────────────────────────┐
                    │ PR #NNN: chore(deps): update vllm/vllm-openai     │
                    │          v0.7.3 → v0.7.4                          │
                    │ author: renovate[bot]                             │
                    │ diff: docker/Dockerfile.vllm                      │
                    └──────────────────────────────┬────────────────────┘
                                                   │ 3. triggers
                                                   ▼
                    ┌───────────────────────────────────────────────────┐
                    │ .github/workflows/schema-refresh.yml              │
                    │                                                   │
                    │  a) Detect framework from Dockerfile diff         │
                    │  b) docker build --build-arg VLLM_VERSION=v0.7.4  │
                    │  c) ./scripts/update_backend_schema.sh vllm       │
                    │     (runs inside the new image)                   │
                    │  d) git commit discovered_schemas/vllm.json       │
                    │     → pushed back to renovate branch as           │
                    │     github-actions[bot]                           │
                    │  e) scripts/diff_schemas.py                       │
                    │     → structured classification                   │
                    │  f) Post PR comment with diff                     │
                    │  g) Apply label: schema-safe | schema-breaking    │
                    └──────────────────────────────┬────────────────────┘
                                                   │
                                ┌──────────────────┴──────────────────┐
                                │                                     │
                                ▼                                     ▼
                    ┌─────────────────────┐             ┌─────────────────────┐
                    │ schema-safe         │             │ schema-breaking     │
                    │ → Renovate          │             │ → PR blocked        │
                    │   auto-merges once  │             │ → human contributor │
                    │   required checks   │             │   updates Pydantic, │
                    │   pass              │             │   fixtures,         │
                    │                     │             │   CHANGELOG         │
                    └─────────────────────┘             └─────────────────────┘
```

### Step by step

1. **Renovate watches FROM tags.** `renovate.json` has datasource rules for the three
   Dockerfiles that base from upstream images:
   - `pytorch/pytorch` in `docker/Dockerfile.pytorch`
   - `vllm/vllm-openai` in `docker/Dockerfile.vllm`
   - `nvcr.io/nvidia/tensorrt-llm/release` in `docker/Dockerfile.tensorrt`
2. **Renovate opens a bump PR** the moment an upstream release hits the registry. The PR only
   touches the `ARG ..._VERSION=...` line in a single Dockerfile. It does not touch Python code,
   vendored schemas, CHANGELOG, or tests.
3. **`schema-refresh.yml` fires on that PR.** It filters on `author: renovate[bot]` and the
   Dockerfile paths, so ad-hoc human PRs don't trigger it.
4. **The workflow rebuilds the image** with the new FROM tag and runs
   `./scripts/update_backend_schema.sh <framework>` inside a container based on the fresh image.
   That script calls `scripts/discover_backend_schemas.py` against the installed framework and
   writes the regenerated JSON to `src/llenergymeasure/config/discovered_schemas/<framework>.json`.
5. **If the schema changed, the workflow commits it back** to the Renovate PR branch as
   `github-actions[bot]` with message `chore(schemas): refresh <framework> schema for <version>`.
   Renovate treats contributions on top of its own branches as normal commits and leaves them
   alone on subsequent rebases.
6. **`scripts/diff_schemas.py` computes a semantic diff** between the previous vendored schema
   and the newly regenerated one. It classifies every change into one of five categories:
   *added*, *removed*, *renamed*, *retyped*, *default-changed*.
7. **The diff is posted as a PR comment.** Contributors see the full change set without having
   to rebuild the image themselves.
8. **A label is applied.** If every change is in the `safe` category, the workflow applies
   `schema-safe`. If any change is in the `breaking` category, it applies `schema-breaking`.
9. **Auto-merge vs human review.** Renovate's `automerge: true` is gated on the
   `schema-safe` label plus required CI checks. A breaking change pins the PR to human review
   until the contributor addresses the downstream cascade (see below).

### Drift guard

A second workflow, `.github/workflows/schema-drift.yml`, runs on every *non-Renovate* PR.
It rebuilds the backend image from the current Dockerfile, re-runs discovery, and fails the
PR if `git diff --exit-code src/llenergymeasure/config/discovered_schemas/` is non-empty.
This catches silent schema drift when a contributor edits a Dockerfile by hand without
running the refresh pipeline.

If the drift guard fails unexpectedly, somebody probably bumped a version in a Dockerfile
manually. Fix it by running the manual refresh recipe below and pushing the regenerated
schema in the same PR.

---

## Semantic diff classification rules

`scripts/diff_schemas.py` applies the following rules. This table is also the
contract between the pipeline and Renovate's auto-merge gate.

| Change | Label |
|---|---|
| Field added | `schema-safe` |
| Default value changed | `schema-safe` |
| Description updated | `schema-safe` |
| Field deprecated (added `deprecated: true`) | `schema-safe` |
| Field removed | `schema-breaking` |
| Field renamed (detected via type + description heuristics) | `schema-breaking` |
| Type narrowed (e.g. `int \| None` → `int`) | `schema-breaking` |
| Enum value removed from `Literal` | `schema-breaking` |
| Required/optional flip (optional → required) | `schema-breaking` |

### Why these choices?

- **Added fields are always safe.** They cannot break any existing YAML, config, or Pydantic
  model. They just expand what's available.
- **Default changes are safe at the schema level.** They may subtly change measurement
  results for users who rely on the default, but that's a framework-behaviour change that
  `llenergymeasure` is not responsible for preventing — the user gets the upstream semantics
  of the version they chose to upgrade to. Defaults are recorded in the
  `config.json` sidecar for every experiment run, so any default change is visible post-hoc.
- **Deprecation is safe** because the field is still present and still works. It becomes
  breaking only when the framework actually removes it in a later release.
- **Removal is breaking** because existing YAMLs may reference the field. The contributor
  must decide whether to deprecate gracefully in Pydantic, hard-break, or map to a replacement.
- **Renaming is breaking** for the same reason. The automation cannot tell whether a
  removed-then-added pair of similarly-typed fields is a rename or two independent changes —
  that decision needs a human.
- **Type narrowing is breaking** because a YAML that passes today (`None`, `0`, `"auto"`) may
  fail tomorrow.
- **Enum tightening is breaking** for the same reason.

---

## What to do when you see `schema-breaking`

A breaking PR blocks on human review. Work through this checklist in order:

1. **Read the PR comment.** The semantic diff tells you exactly which fields changed and how.
2. **Find consumers of each changed field.** Start with `backend_configs.py`:
   ```
   rg -n "<field_name>" src/llenergymeasure/config/backend_configs.py
   rg -n "<field_name>" configs/ tests/fixtures/
   ```
3. **Update the Pydantic model** in `backend_configs.py`. Decide per field:
   - **Remove the Pydantic field** if the framework removed it outright. There is no
     migration path for a field that no longer exists.
   - **Rename the Pydantic field** if the framework renamed it. Maintain the native-naming
     principle: the new Pydantic field name matches the new framework-native name.
   - **Tighten the Pydantic type** if the framework tightened it. Let the migration surface
     through a `ValidationError` on user YAMLs rather than silently coerce.
4. **Update YAML fixtures.** There are four that reference backend-specific fields:
   - `configs/example-study-full.yaml`
   - `configs/test.yaml`
   - `tests/fixtures/killpg_vllm_study.yaml`
   - `tests/fixtures/sigint_study.yaml`
5. **Add a CHANGELOG entry** under `Breaking changes`, naming each removed/renamed field and
   the upstream version that introduced the break:
   ```
   - **vLLM 0.8.0**: `EngineArgs.enforce_eager` removed; use `cuda_graph_sizes: []`
     instead (vllm#12345). Pydantic field `VLLMEngineConfig.enforce_eager` removed.
     Migrate YAMLs: delete the `enforce_eager:` key.
   ```
6. **Re-run the drift guard locally** to confirm your regenerated schema matches the new
   image state:
   ```bash
   ./scripts/update_backend_schema.sh vllm
   git diff src/llenergymeasure/config/discovered_schemas/vllm.json
   ```
7. **Push your changes** to the Renovate branch. The CI will re-run `schema-refresh.yml`,
   post a fresh diff comment, and (once Pydantic + fixtures are consistent with the new
   schema) all required checks should pass. The PR remains `schema-breaking`, so Renovate
   won't auto-merge — you merge it manually when you're satisfied.

---

## Manual refresh recipe

Sometimes you need to regenerate a schema without waiting for Renovate:

- You're testing a new framework version locally before Renovate has picked it up.
- You hand-edited a Dockerfile and the drift guard is failing.
- You need to validate a schema change against a specific image tag.

Use the convenience wrapper:

```bash
# vLLM
./scripts/update_backend_schema.sh vllm

# TensorRT-LLM
./scripts/update_backend_schema.sh tensorrt

# HuggingFace Transformers (runs inside the pytorch image)
./scripts/update_backend_schema.sh transformers
```

Under the hood the wrapper does three things:

1. Builds the backend's Docker image from the current Dockerfile.
2. Runs `python scripts/discover_backend_schemas.py <framework>` inside the container.
3. Writes the regenerated JSON to `src/llenergymeasure/config/discovered_schemas/<framework>.json`
   and prints the git diff so you can eyeball the result.

If you prefer the GitHub Actions path without waiting for Renovate, the `schema-refresh.yml`
workflow also has a `workflow_dispatch` trigger. Run it from the Actions tab with a
framework parameter, and it will perform the same rebuild-and-commit flow on a branch of
your choosing.

---

## Troubleshooting

### Renovate didn't open a PR for a release I expected

Check Renovate's dashboard comment on an existing PR, or the Renovate logs in Actions.
Common causes:
- The release is a pre-release tag (`v0.8.0rc1`) and Renovate's `ignoreUnstable: true` rule
  filtered it out. Add the tag to `renovate.json`'s `allowedVersions` list if you want it
  picked up.
- The Dockerfile ARG default isn't a valid datasource target (e.g. you used a digest instead
  of a tag). Switch back to a tag.
- Renovate is rate-limited. Wait an hour and retrigger with a `@renovatebot rebase` comment
  on any existing PR.

You can always trigger a manual refresh via `workflow_dispatch` while you wait.

### `schema-drift.yml` fails and you didn't touch the Dockerfile

This usually means the framework silently changed its schema discovery output — perhaps
because the underlying NGC image was retagged, or because Docker rebuilt a layer from
scratch and picked up a patch release. Run the manual refresh recipe locally; whatever
changed will show up in `git diff`. Treat it exactly like a Renovate bump: classify the
diff, update Pydantic/fixtures/CHANGELOG if needed, and commit the regenerated schema.

### Discovery fails inside the new image

`scripts/discover_backend_schemas.py` imports framework modules and calls their introspection
APIs (`dataclasses.fields`, `model_json_schema`, etc.). If those APIs change, discovery
breaks. The workflow surfaces the failure in the Actions log.

- **Missing attribute on native config class** (e.g. `EngineArgs` no longer has
  `max_num_batched_tokens`): this is a field removal that our discovery code isn't yet aware
  of. Update `discover_backend_schemas.py` to handle the new surface, then regenerate.
- **Import error** (e.g. `cannot import name 'SamplingParams' from 'vllm'`): the framework
  moved the symbol. Update the discovery script's import path and regenerate.
- **Framework itself fails to import** (e.g. CUDA ABI mismatch with the NGC base image):
  this is a Dockerfile or base-image problem, not a schema problem. Fix the Dockerfile first;
  the schema refresh will follow once the image builds cleanly.

### The semantic diff is enormous and I don't understand it

Run `scripts/diff_schemas.py` locally with the old and new JSONs as arguments:

```bash
python scripts/diff_schemas.py \
    path/to/old/vllm.json \
    src/llenergymeasure/config/discovered_schemas/vllm.json
```

It prints the same structured classification the workflow uses. Filter on
`--only breaking` to focus on the changes that matter for Pydantic/fixture updates.

---

## Related docs

- [`docs/docker-setup.md`](docker-setup.md) — initial Docker install and image build
- [`docs/backends.md`](backends.md) — per-backend capabilities and configuration
- [`.product/designs/parameter-discovery.md`](../.product/designs/parameter-discovery.md) — design doc with the full rationale and envelope format
- [`.product/designs/parameter-discovery-architecture.md`](../.product/designs/parameter-discovery-architecture.md) — three-layer architecture (vendored schemas + Pydantic projection + doc generation)
