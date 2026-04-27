# Schema Refresh Pipeline

Engine parameter schemas are vendored as JSON files in
`src/llenergymeasure/config/discovered_schemas/`. When an upstream engine
releases a new version, these schemas must be regenerated so that config
validation stays in sync with the engine's actual parameters.

This pipeline automates that process end-to-end.

---

## Overview

```
Upstream releases new engine version (e.g. vLLM v0.8.0)
                      |
                      v
Renovate detects new tag on Docker Hub / NGC / PyPI
(checks weekly, waits 3 days for stability)
                      |
                      v
Renovate opens PR bumping ARG in Dockerfile
e.g. ARG VLLM_VERSION=v0.7.3 -> v0.8.0
                      |
                      v
parameter-discovery.yml auto-fires
(guarded: only for renovate[bot] PRs touching Dockerfiles)
                      |
                      v
+------------------------------------------+
|  Runs on self-hosted GPU runner:         |
|  1. Pulls/builds new engine image        |
|  2. Runs discover_engine_schemas.py      |
|     inside the container                 |
|  3. Compares old vs new schema           |
|     (scripts/diff_discovered_schemas.py)            |
|  4. Commits updated schema to PR         |
|  5. Posts diff summary as PR comment     |
|  6. Labels: schema-safe / schema-breaking|
+------------------------------------------+
                      |
                      v
Maintainer reviews PR:
- schema-safe: review diff, merge
- schema-breaking: update Pydantic models / tests, then merge
```

---

## How It Works

### Automated Flow (Renovate PRs)

1. **Renovate** monitors `docker/Dockerfile.*` for upstream version bumps:
   Docker Hub image tags (vLLM), NGC image tags (TensorRT-LLM), and PyPI
   package versions via `customManagers` regex (transformers). Weekly schedule,
   3-day stability window before opening a PR.
2. When Renovate opens a PR, **parameter-discovery.yml** auto-fires on the
   self-hosted GPU runner.
3. The workflow determines which engine(s) changed by inspecting the modified
   Dockerfile paths, then runs `./scripts/refresh_discovered_schemas.sh <engine>`.
4. After discovery, `scripts/diff_discovered_schemas.py` classifies changes as safe or
   breaking, commits the updated schema to the PR branch, posts a diff comment,
   and applies a label (`schema-safe` or `schema-breaking`).

### Manual Version Bumps (CI version guard)

If a developer bumps an engine version ARG in a Dockerfile without running
discovery, the `schema-version-check` job in `ci.yml` catches it:

```
Developer bumps engine version in Dockerfile
                      |
                      v
ci.yml schema-version-check job fires
(path-filtered to docker/Dockerfile.*, skips Renovate PRs)
                      |
                      v
Compares ARG version in Dockerfile vs engine_version in schema JSON
- MATCH: pass (non-version changes like build opts are fine)
- MISMATCH: fail with actionable message
```

On failure, the developer can either:
- Run locally: `./scripts/refresh_discovered_schemas.sh <engine>`
- Trigger remotely: `gh workflow run parameter-discovery.yml --field engine=<engine> --field pr_number=<N>`

### Manual Refresh (workflow_dispatch)

For ad-hoc refreshes outside the Renovate flow:

```bash
gh workflow run parameter-discovery.yml \
  --field engine=vllm \
  --field pr_number=123
```

---

## Change Classification

`scripts/diff_discovered_schemas.py` classifies parameter changes by comparing old and new
schema JSONs:

| Change type         | Classification | Example                              |
|---------------------|----------------|--------------------------------------|
| Field added         | safe           | New `enable_chunked_prefill` param   |
| Description updated | safe           | Docstring clarification              |
| Default changed     | safe           | `gpu_memory_utilization: 0.9 -> 0.95`|
| Type widened        | safe           | `int` -> `int | None`                |
| Field removed       | BREAKING       | Deprecated param dropped             |
| Type narrowed       | BREAKING       | `int | None` -> `int`                |
| Enum value removed  | BREAKING       | Quantisation mode dropped            |

Metadata fields (`discovered_at`, `engine_commit_sha`, `image_ref`,
`base_image_ref`) are excluded from classification as they change on every run.

---

## Handling Breaking Changes

When parameter-discovery labels a PR `schema-breaking`:

1. Check which fields were removed/narrowed (see the PR comment diff)
2. Update Pydantic models in `src/llenergymeasure/config/engine_configs.py`
3. Update affected tests and YAML fixtures
4. Add CHANGELOG entry under Breaking Changes
5. Push fixes to the Renovate PR branch, re-run CI

---

## Adding a New Engine

1. Create `docker/Dockerfile.<engine>` with an `ARG` version pin
2. Add a discovery function in `scripts/discover_engine_schemas.py`
3. Add a case to `scripts/refresh_discovered_schemas.sh`
4. Run discovery: `./scripts/refresh_discovered_schemas.sh <engine>`
5. Add a Renovate `packageRule` in `renovate.json`
6. If the Dockerfile ARG maps directly to the engine version, add an entry to
   `_ENGINE_SPECS` in `scripts/check_discovered_schema_versions.py`

For engines pre-installed in their upstream Docker image (vLLM, TensorRT-LLM),
the `dockerfile` manager monitors image tag bumps automatically. For engines
installed via pip on top of a base image (transformers), add a `customManagers`
regex entry with `datasourceTemplate: "pypi"` to monitor PyPI releases against
the Dockerfile `ARG` pin.

The parameter-discovery workflow and version guard automatically cover new engines
via path-based triggers (`docker/Dockerfile.*`).

---

## Prerequisites

- [Mend Renovate](https://github.com/apps/renovate) GitHub App installed on the
  repo (free for open-source)
- Self-hosted GPU runner available for parameter-discovery jobs
- Docker + NVIDIA Container Toolkit on the runner

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Renovate not detecting bumps | `fileMatch` pattern doesn't cover the Dockerfile | Check Renovate dashboard, verify `docker/Dockerfile\\..*` matches |
| Renovate not detecting transformers bumps | `customManagers` regex not matching | Verify `ARG TRANSFORMERS_VERSION=X.Y.Z` format in Dockerfile.transformers |
| parameter-discovery fails to import engine | Needs `--gpus all` | Verify GPU runner has NVIDIA drivers + Container Toolkit |
| Version guard fails on non-version change | Won't happen - guard only compares version ARGs | If it does, check `_parse_arg` regex in `check_discovered_schema_versions.py` |
| NGC registry auth failure | Private image or rate-limited | Add `hostRules` to `renovate.json` |
| Schema unchanged after discovery | Engine version didn't change params | Expected - workflow commits nothing and exits cleanly |

---

## Related

- [Docker Setup](docker-setup.md) - building engine images locally
- [Engine Configuration](engines.md) - configuring engine parameters
- [Miner Pipeline](miner-pipeline.md) - how the validation-rule corpus is regenerated alongside parameter schemas on library bumps
- [Architecture Overview](architecture-overview.md) - full system context
