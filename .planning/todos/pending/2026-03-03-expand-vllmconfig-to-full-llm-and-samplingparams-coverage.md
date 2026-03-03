---
created: 2026-03-03T17:05:12.601Z
title: Expand VLLMConfig to full LLM() + SamplingParams coverage
area: infrastructure
files:
  - src/llenergymeasure/backends/vllm.py
  - src/llenergymeasure/config/vllm_config.py
---

## Problem

Phase 19.1 implements an energy-relevant subset of vLLM params (~15–20 params covering memory + compute effects). Full parity with vLLM's `LLM()` (~40 args) and `SamplingParams` (~20 fields) was deferred due to the mechanical overhead per field (~5–10 lines each across field definition, backend wiring, and SSOT introspection metadata).

## Solution

In a future phase, audit remaining `LLM()` constructor args and `SamplingParams` fields not covered by Phase 19.1 and add them to `VLLMConfig`. Optionally mark non-energy-affecting params with metadata so researchers can distinguish efficiency knobs from general config.

Reference the Phase 19.1 SSOT introspection pattern as the template for adding each new field.
