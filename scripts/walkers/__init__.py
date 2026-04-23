"""AST walkers that extract validation rules from engine library source.

Each ``scripts/walkers/{engine}.py`` walker is version-pinned to a specific
library release via :data:`_base.TESTED_AGAINST_VERSIONS` and emits a
corpus-compatible YAML document of rule candidates.

The transformers walker is an introspection wrapper around
``GenerationConfig.validate()`` rather than a full AST walker — see
:mod:`scripts.walkers.transformers` for the rationale. vLLM and TensorRT-LLM
walkers (added in later phases) use the AST-extraction primitives in
:mod:`scripts.walkers._base`.
"""
