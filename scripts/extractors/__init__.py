"""Walkers that extract validation rules from engine library source.

Each ``scripts/extractors/{engine}.py`` walker (added per-engine in follow-up
PRs) is version-pinned to a specific library release via the walker module's
``TESTED_AGAINST_VERSIONS`` and emits a corpus-compatible YAML document of
rule candidates.

Two extraction mechanisms are in scope:

- **Library-API introspection** — when the library exposes a structured
  validation method (e.g. HF transformers'
  ``GenerationConfig.validate(strict=True)``), the walker simply wraps the
  call and formats the results.
- **AST source parsing** — when no such API exists (vLLM, TRT-LLM), the
  walker parses the library source AST using the primitives in
  :mod:`scripts.extractors._base`.

No concrete walker ships in this module today; they land as independent PRs
per engine.
"""
