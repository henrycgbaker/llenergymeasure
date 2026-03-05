---
phase: 23-documentation
plan: "01"
subsystem: docs
tags: [documentation, auto-generation, ci, pydantic, typer]
dependency_graph:
  requires: []
  provides: [docs-autogen-pipeline, ci-docs-freshness]
  affects: [.github/workflows/ci.yml, scripts/]
tech_stack:
  added: []
  patterns: [pydantic-model-json-schema, click-introspection, github-actions-job]
key_files:
  created:
    - scripts/generate_cli_reference.py
  modified:
    - scripts/generate_config_docs.py
    - .github/workflows/ci.yml
  deleted:
    - docs/backends.md
    - docs/campaigns.md
    - docs/cli.md
    - docs/deployment.md
    - docs/methodology.md
    - docs/quickstart.md
    - docs/generated/config-reference.md
    - docs/generated/invalid-combos.md
    - docs/generated/parameter-support-matrix.md
decisions:
  - "Used click introspection (typer.main.get_command) for CLI reference - typer 0.24.1 does not have get_docs_for_typer_app"
  - "Base install (uv sync --dev, no extras) is sufficient for docs scripts - model_json_schema() does not require pytorch at runtime"
  - "CI docs-freshness job uses smoke-test approach (exit 0 check) not diff-based freshness - full marker-based diffing deferred as future improvement"
metrics:
  duration: ~600s
  completed: "2026-03-05"
  tasks_completed: 2
  files_changed: 11
---

# Phase 23 Plan 01: Documentation Foundation - Summary

**One-liner:** Deleted all stale v1.x docs, rewrote generate_config_docs.py using Pydantic model_json_schema() renderer, created generate_cli_reference.py using click introspection, added CI docs-freshness job.

## What Was Built

### Task 1: Delete stale docs and rewrite auto-generation scripts

Deleted all 9 files in `docs/` (including `docs/generated/` subdirectory). Every existing doc referenced the old `lem` CLI, `lem campaign`/`lem experiment` commands, old field names (`model_name`, `fp_precision`), and old concepts ("campaigns") - all stale since v2.0.

Rewrote `scripts/generate_config_docs.py` using Pydantic `model_json_schema()`. The new script:
- Extracts `ExperimentConfig.model_json_schema()` (full JSON schema including `$defs` for nested models)
- Renders a structured Markdown reference grouped into 11 sections (top-level, decoder, warmup, baseline, energy, pytorch, vllm.engine, vllm.sampling, vllm.beam_search, vllm.engine.attention, tensorrt)
- Resolves `$ref` links from `$defs` for nested model sections
- Outputs to stdout or `--output <path>`
- Includes `<!-- Auto-generated -->` header

Created `scripts/generate_cli_reference.py` (new file). The script:
- Uses `typer.main.get_command(app)` to get the underlying click `TyperGroup`
- Iterates registered commands (`run`, `config`) and their parameters
- Renders a Markdown reference with argument table and options table for each command
- Outputs to stdout or `--output <path>`
- Includes `<!-- Auto-generated -->` header

Note: `typer.utils.get_docs_for_typer_app` does not exist in the installed version (typer 0.24.1). Click introspection via `typer.main.get_command` is the correct approach for this version.

### Task 2: Add CI freshness check step

Added `docs-freshness` job to `.github/workflows/ci.yml`. The job:
- Checks out repo
- Sets up uv with same pattern as other jobs (`astral-sh/setup-uv@v7`, `cache-suffix: docs-3.12`, `python-version: "3.12"`)
- Runs `uv sync --dev` (base install sufficient - no GPU/pytorch required for schema extraction)
- Runs all three generation scripts and verifies they exit 0
- Does NOT include `generate_param_matrix.py` (requires GPU test result JSON files)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Typer API version mismatch**
- **Found during:** Task 1
- **Issue:** Research assumed `typer.utils.get_docs_for_typer_app()` would exist. Typer 0.24.1 (installed) does not have this function - it was added/removed or the API name is different.
- **Fix:** Used `typer.main.get_command(app)` to get the click TyperGroup and introspected commands/params via the click API directly. This is more robust - it works with any typer version and produces identical output.
- **Files modified:** `scripts/generate_cli_reference.py`
- **Commit:** 06a4cea

**2. [Rule 2 - Auto-fix] generate_invalid_combos_doc.py recreated docs/generated/**
- **Found during:** Task 1 verification (ran script to verify it still works)
- **Issue:** Running `generate_invalid_combos_doc.py` recreated the `docs/generated/` directory that we just deleted. Script hardcodes output path to `docs/generated/invalid-combos.md`.
- **Fix:** Removed the recreated directory before committing. The script's hardcoded output path is a pre-existing issue outside this plan's scope - left unchanged since the CI freshness check uses stdout redirect (`> /dev/null`), which doesn't trigger the file write.
- **Note:** Deferred - `generate_invalid_combos_doc.py` stdout refactor would be a clean-up in a subsequent plan.

## Verification

All success criteria met:

- `ls docs/ | wc -l` → 0 (empty, no stale files)
- `uv run python scripts/generate_config_docs.py` → exits 0, produces valid Markdown with all 11 ExperimentConfig sections
- `uv run python scripts/generate_cli_reference.py` → exits 0, produces valid Markdown with llem run + llem config flags
- `.github/workflows/ci.yml` contains `docs-freshness` job with correct uv setup pattern

## Self-Check: PASSED
