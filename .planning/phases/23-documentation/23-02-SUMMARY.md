---
phase: 23-documentation
plan: "02"
subsystem: docs
tags: [documentation, installation, getting-started, cli-reference, onboarding]
dependency_graph:
  requires: [23-01]
  provides: [docs-installation, docs-getting-started, docs-cli-reference]
  affects: [docs/installation.md, docs/getting-started.md, docs/cli-reference.md]
tech_stack:
  added: []
  patterns: [imperative-practical-docs, annotated-output, two-track-guide]
key_files:
  created:
    - docs/installation.md
    - docs/getting-started.md
    - docs/cli-reference.md
  modified: []
decisions:
  - "Annotated output in getting-started.md constructed from _display.py source code rather than live run - host cannot run pytorch outside containers"
  - "CLI reference combines auto-generated flag table with manually written context (effective defaults, exit codes, examples) for a complete reference"
  - "llem config stays as current command name - rename to llem init is deferred, per CONTEXT.md"
metrics:
  duration: ~106s
  completed: "2026-03-05"
  tasks_completed: 2
  files_changed: 3
---

# Phase 23 Plan 02: Core Researcher Onboarding Docs - Summary

**One-liner:** Wrote installation guide (pip extras + dev install + Docker path), two-track getting started guide (PyTorch quick start + Docker recommended, both using GPT-2 with annotated output), and complete CLI reference for llem run and llem config.

## What Was Built

### Task 1: installation.md and getting-started.md

**docs/installation.md** covers:
- System requirements table (Python 3.10+, Linux, NVIDIA GPU with CUDA 12.x, Docker + NVIDIA CT)
- macOS/Windows note (PyTorch-only, no Docker backends)
- All pip extras in a table (`[pytorch]`, `[vllm]`, `[tensorrt]`, `[zeus]`, `[codecarbon]`) with combined extras example
- Base install note (no inference backend without extras)
- Dev install via `uv sync --dev --extra pytorch` with expected version output
- Docker path as a brief pointer to `docker-setup.md` — no content duplication
- `llem config` verification step with annotated example output explaining each section (GPU, Backends, Energy, Config, Python)
- Link to getting-started.md as next step

**docs/getting-started.md** covers:
- Two-track structure: Quick Start (Local PyTorch) and Recommended Start (Docker + vLLM)
- Both tracks use GPT-2 (124M) as the example model
- Annotated terminal output constructed from `cli/_display.py` display format: each line of the result summary explained with inline comments
- Result field explanation table (Total J, Baseline W, Adjusted J, Throughput tok/s, FLOPs, Duration, Warmup)
- Output file structure (`results/experiment-id/result.json`)
- Config file intro with minimal YAML example
- Linear next-step pointer to study-config.md and docker-setup.md

### Task 2: cli-reference.md

**docs/cli-reference.md** covers:
- All 15 `llem run` flags with Flag, Short, Type, Default, Description columns
- CLI effective defaults note for study mode: `--cycles` defaults to `3`, `--order` defaults to `shuffled`
- `llem config` documentation with flags and example output
- `llem --version` with example output
- Examples section: single experiment via flags, via YAML, dry run, study with cycle override, thermal gap suppression, preflight skip, environment check
- HTML comment marking auto-generated section with regeneration command

## Deviations from Plan

### Auto-fixed Issues

None.

### Notes

- The annotated output in getting-started.md was constructed from the `cli/_display.py` source code rather than captured from a live run. The host cannot run PyTorch inference outside Docker containers (CUDA unavailable on host). The constructed output matches the exact format produced by `print_result_summary()` and is labelled accurately.
- The CLI reference uses the auto-generator output as the basis for the flag table, then supplements with manually written context (effective defaults, exit codes, examples) that the generator does not produce.

## Verification

All success criteria met:

- `docs/installation.md` exists, contains pip install instructions for all 5 extras
- `docs/getting-started.md` exists with both PyTorch quick start and Docker recommended tracks
- `docs/cli-reference.md` exists with all 15 `llem run` flags documented
- No references to old CLI name `lem`, old commands, or old field names in any file
- Cross-links between docs are valid: `installation.md` → `getting-started.md` → `study-config.md`/`docker-setup.md`

## Self-Check: PASSED
