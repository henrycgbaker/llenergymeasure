"""Miners that extract validation rules from engine library source.

Each ``scripts/miners/{engine}_miner.py`` (added per-engine in follow-up
PRs) is version-pinned to a specific library release via the miner module's
``TESTED_AGAINST_VERSIONS`` and emits a corpus-compatible YAML document of
rule candidates.

Two extraction mechanisms are in scope:

- **Dynamic mining** — when the library exposes a structured validation method
  (e.g. HF transformers' ``GenerationConfig.validate(strict=True)``), the
  dynamic miner wraps the call and infers predicates from probe results.
- **Static mining** — when no such API exists (vLLM, TRT-LLM), the static
  miner parses the library source AST using the primitives in
  :mod:`scripts.miners._base`.

No concrete vLLM or TRT-LLM miner ships in this module today; they land as
independent PRs per engine.
"""
