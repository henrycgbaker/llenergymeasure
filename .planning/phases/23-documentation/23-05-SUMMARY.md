---
phase: 23-documentation
plan: "05"
subsystem: docs
tags: [documentation, policy-makers, guides, readme]
dependency_graph:
  requires: [23-02, 23-03, 23-04]
  provides: [docs-policy-maker-guides, readme-overview]
  affects:
    - docs/guide-what-we-measure.md
    - docs/guide-interpreting-results.md
    - docs/guide-getting-started.md
    - docs/guide-comparison-context.md
    - README.md
tech_stack:
  added: []
  patterns: [github-rendered-markdown, dual-audience-docs]
key_files:
  created:
    - docs/guide-what-we-measure.md
    - docs/guide-interpreting-results.md
    - docs/guide-getting-started.md
    - docs/guide-comparison-context.md
  modified:
    - README.md
  deleted: []
decisions:
  - "README rewritten as concise overview (75 lines) with links to all 13 docs — no inline content"
  - "Policy maker guides are fully self-contained — a reader can progress through all four without needing researcher docs"
  - "guide-getting-started.md links to researcher installation.md for system requirements rather than duplicating them"
  - "guide-comparison-context.md explains CodeCarbon and Zeus as measurement backends (not benchmarks) to avoid reader confusion"
metrics:
  duration: ~223s
  completed: "2026-03-05"
  tasks_completed: 2
  files_changed: 5
---

# Phase 23 Plan 05: Policy Maker Guides and README - Summary

**One-liner:** Four policy maker guides (energy/throughput/FLOPs explainer, results interpreter, getting-started walkthrough, benchmarks comparison) plus a concise 75-line README linking all 13 docs across researcher and policy maker audiences.

## What Was Built

### Task 1: Four Policy Maker Guide Docs

**guide-what-we-measure.md** — "What We Measure and Why It Matters"

- Opens with policy context: data centre energy use, AI as a growing contributor, why measurement enables procurement and sustainability decisions
- Energy section: joules explained with car fuel analogy, total vs baseline vs adjusted energy
- Throughput section: tokens/second explained with typing speed analogy, the efficiency trade-off
- FLOPs section: calculation count analogy, why it enables fair comparison across model sizes
- Closes with a table showing which combination of metrics answers which policy question

**guide-interpreting-results.md** — "How to Read llenergymeasure Output"

- Reproduces actual terminal output format with annotated walkthrough of every field
- Explains experiment ID structure, energy metrics (total/baseline/adjusted), performance metrics (throughput/FLOPs), timing (duration/warmup)
- Result file field table: `energy_joules`, `inference_energy_joules`, `throughput_tokens_per_second`, `latency_seconds`, `inference_memory_mb`, `total_prompts`, `total_output_tokens`, `effective_config`
- Meaningful comparison guidance: use energy per token, match prompt counts, note hardware and precision
- Order-of-magnitude context table for energy figures (smartphone charge, boiling water — clearly labelled as approximate)

**guide-getting-started.md** — "Running Your First Measurement (Policy Maker Guide)"

- Prerequisites table with plain explanations (not just "Python 3.10+" but what Python is and how to check)
- Step-by-step install (`pip install "llenergymeasure[pytorch]"`), verify (`llem config` with annotated output), run (`llem run --model gpt2`), and read results
- Explains warmup, model download, result file location
- Next steps: compare models, compare precisions, escalation to researcher docs for sweeps

**guide-comparison-context.md** — "Comparison with Other Benchmarks"

- Frames llenergymeasure in the broader landscape: inference energy efficiency vs throughput benchmarks
- MLPerf: industry standard for throughput/accuracy; llenergymeasure adds energy dimension and works without certified configurations
- AI Energy Score: llenergymeasure includes this dataset by default, results are directly comparable; explains tool vs benchmark distinction
- CodeCarbon: full-system + CO₂ backend in llenergymeasure; explains when to use it vs default NVML
- Zeus: kernel-level GPU measurement backend; explains polling granularity trade-off
- Summary comparison table and guidance for citing results in reports and sustainability accounting

### Task 2: README.md Rewrite

Replaced the stale 265-line v1.x README with a 75-line overview:

- Title + three existing badges (MIT, Python 3.10+, Ruff) preserved
- One-liner description + 2-sentence overview
- Key Features bullet list: multi-backend, energy measurement, study/sweep, Docker, reproducibility, built-in datasets
- Quick Install: `pip install "llenergymeasure[pytorch]"` + `llem run` with links to installation.md and getting-started.md
- Documentation table: all 13 docs split into Researcher Docs (9) and Policy Maker Guides (4)
- Contributing: link to development install in installation.md
- License: MIT with link to LICENSE file

Removed: all v1.x content (`lem` CLI, campaigns, `lem experiment`, config table, old quick start YAML, old metrics table, old development section).

## Deviations from Plan

None - plan executed exactly as written.

## Verification

All success criteria met:

- All 13 doc files exist in docs/ (9 researcher + 4 policy maker) ✓
- README.md links to all 13 docs ✓
- README.md is 75 lines (under 150 limit) ✓
- Policy maker docs use accessible non-technical language with analogies ✓
- No file references old CLI name (`lem`) or old commands ✓
- Navigation: README → installation → getting-started → study-config/backends/docker-setup ✓

## Self-Check: PASSED
