# Extending the Invariant Miner: Adding a New Engine

This document is the practitioner's guide to adding invariant miner support for a new engine. It uses the transformers miner as the gold-standard reference throughout.

**Audience:** engine extenders. Assumes familiarity with the [miner-pipeline.md](miner-pipeline.md) concepts.

---

## Before you start

1. Read [miner-pipeline.md](miner-pipeline.md) to understand the static miner / dynamic miner / lift module split.
2. Read the [corpus format reference](validation-rule-corpus.md) to understand what rules look like.
3. Review `scripts/miners/transformers_static_miner.py` and `scripts/miners/transformers_dynamic_miner.py` as the gold standard. The comments in those files contain important design decisions.

---

## Step 0: Research the engine's validation surface

Before writing any code, answer these questions:

1. Which config classes does the engine validate? Where does validation happen - in `__init__`, in a separate `validate()` method, in validators decorated with `@model_validator`?

2. Which Python type system does each class use? (`pydantic.BaseModel` / `pydantic.dataclasses.dataclass`, `msgspec.Struct`, `@dataclasses.dataclass`, or something else?)

3. Does the engine constructor raise on invalid inputs, or silently normalise them? (Transformers and vLLM raise; TRT-LLM constructors are more permissive, so TRT-LLM has no dynamic miner.)

4. What is the CUDA / import dependency? Can you `import enginelib` on a CPU-only host? (vLLM: yes on CPU-only with a non-GPU-enabled pip install; TRT-LLM: requires CUDA-aware container.)

5. What is a realistic post-vendor-CI rule count? (Transformers: 46; vLLM: 80-110; TRT-LLM: 20-28.) This helps plan the scope.

---

## Step 1: Add the fail-loud import contract

Create `scripts/miners/{engine}_miner.py` (the orchestration entry point). The very first thing it must do:

```python
import importlib.metadata
from packaging.specifiers import SpecifierSet
from scripts.miners._base import check_installed_version, MinerLandmarkMissingError

TESTED_AGAINST_VERSIONS = SpecifierSet(">=X.Y,<X.Z")
# Set this to the specific version range you have tested against.
# Keep the upper bound tight (e.g. <4.60 not <99.0) so
# MinerVersionMismatchError fires on a library bump.

_installed = importlib.metadata.version("your-engine-library")
check_installed_version("your-engine-library", _installed, TESTED_AGAINST_VERSIONS)
# Raises MinerVersionMismatchError if installed version is outside the range.
# This is CI-fatal: the miner will not emit partial output.
```

Then declare landmark checks for every class or method the miner will walk:

```python
import ast
import inspect
from scripts.miners._base import find_class, find_method

from enginelib.config import SomeConfigClass

_source = inspect.getsource(SomeConfigClass)
_module = ast.parse(_source)
_cls = find_class(_module, "SomeConfigClass")
if _cls is None:
    raise MinerLandmarkMissingError(
        "SomeConfigClass",
        "expected in enginelib.config - check if the class was renamed"
    )
```

**Why this matters:** the Haiku-era TRT-LLM extractor imported `LlmConfig` - a class that does not exist in TRT-LLM 0.21.0. It caught the `ImportError` and silently returned `[]`. The fail-loud contract makes silent coverage loss impossible.

---

## Step 2: Apply the relevant lift module(s)

Based on your Step 0 research, apply one or more lift modules to extract constraints directly from type metadata.

All three lift modules expose a single function named `lift` with the same signature: `lift(target_type, *, namespace, today, source_path) -> list[RuleCandidate]`. The engine/library is derived automatically from `target_type.__module__`. Import each lift under an alias to keep call sites readable.

### If the engine uses Pydantic v2

```python
from datetime import date
from scripts.miners._pydantic_lift import lift as lift_pydantic
from enginelib.config import CacheConfig, SchedulerConfig

TODAY = date.today().isoformat()

def mine_pydantic_rules():
    rules = []
    for cls in [CacheConfig, SchedulerConfig]:
        rules.extend(lift_pydantic(
            cls,
            namespace="myengine.config",
            today=TODAY,
            source_path="enginelib/config.py",
        ))
    return rules
```

The lift emits one rule per `Gt`, `Ge`, `Lt`, `Le`, `MultipleOf`, `MinLen`, `MaxLen` constraint and per `Literal[...]` allowlist found on any field.

### If the engine uses msgspec

```python
from scripts.miners._msgspec_lift import lift as lift_msgspec
from enginelib.config import SamplingParams

def mine_msgspec_rules():
    return lift_msgspec(
        SamplingParams,
        namespace="myengine.sampling",
        today=TODAY,
        source_path="enginelib/sampling.py",
    )
```

Note: if the class ships zero `Meta(ge=...)` annotations (common for msgspec classes), the lift returns `[]` - that is expected and not an error.

### If the engine uses stdlib dataclasses

```python
from scripts.miners._dataclass_lift import lift as lift_dataclass
from enginelib.config import EngineArgs

def mine_dataclass_rules():
    return lift_dataclass(
        EngineArgs,
        namespace="myengine.args",
        today=TODAY,
        source_path="enginelib/args.py",
    )
```

The dataclass lift is limited to `Literal[...]` value-allowlist rules (no numeric bounds; stdlib dataclasses carry no bound metadata by default).

---

## Step 3: Write the static miner

Create `scripts/miners/{engine}_static_miner.py`. The static miner walks the AST of validator methods and emits rules for conditional raises, warnings, and silent normalisations.

### Pattern: walking a validator method

```python
import ast
import inspect
from scripts.miners._base import (
    find_class, find_method, extract_condition_fields,
    filter_condition_references_self,
    ConditionalRaiseDetector, ConditionalSelfAssignDetector,
    ConditionalWarningsWarnDetector, ConditionalLoggerWarningDetector,
    RuleCandidate, MinerSource,
)

def walk_validate_method(cls_source: str, cls_name: str) -> list[RuleCandidate]:
    module = ast.parse(cls_source)
    cls_node = find_class(module, cls_name)
    if cls_node is None:
        raise MinerLandmarkMissingError(cls_name)

    validate = find_method(cls_node, "validate")
    if validate is None:
        raise MinerLandmarkMissingError(f"{cls_name}.validate")

    public_fields = frozenset(
        # derive from the class's dataclasses.fields() or __annotations__
    )

    detectors = (
        ConditionalRaiseDetector(),
        ConditionalSelfAssignDetector(),
        ConditionalWarningsWarnDetector(),
        ConditionalLoggerWarningDetector(),
    )

    candidates = []
    for node in ast.walk(validate):
        if not isinstance(node, ast.If):
            continue
        if not filter_condition_references_self(node.test, public_fields):
            continue
        for stmt in node.body:
            for detector in detectors:
                pattern = detector.detect(stmt)
                if pattern is not None:
                    # build RuleCandidate from pattern + condition
                    candidate = _build_candidate(node.test, pattern, ...)
                    candidates.append(candidate)
    return candidates
```

### Per-engine detector customisation

The five default detectors cover the most common patterns. For engine-specific patterns, write a custom detector:

```python
# Example: engine uses self.errors.append(...) for error collection
class ErrorsAppendDetector:
    def detect(self, stmt: ast.stmt) -> DetectedPattern | None:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None
        path = call_func_path(stmt.value)
        if path != ["self", "errors", "append"]:
            return None
        return DetectedPattern(
            severity="error",
            emission_channel="none",
            affected_field=None,
            message_template=first_string_arg(stmt.value),
            detail="self.errors.append",
        )
```

### Important: the "revisit" comment

Per the transformers static miner's header, per-engine miners currently define their own `_detect_*` functions rather than using `_base.py`'s detector classes directly. This is because the `DetectedPattern` shape from `_base.py` doesn't carry the structured `FieldPredicate` data needed for cross-field corpus rules (operators like `not_divisible_by` and `@field_ref`). Once two or more engine miners exist and we can see whether the parallel detector logic is genuinely divergent or accidentally so, harmonise in a `_base.py` refactor.

---

## Step 4: Write the dynamic miner (if applicable)

Create `scripts/miners/{engine}_dynamic_miner.py` if the engine's constructors raise on invalid inputs.

**Skip this step if:** probing the engine's constructors yields zero raises. This is the case for TRT-LLM, where `TrtLlmArgs(**kwargs)` is extremely permissive at construction time; constraints are enforced in validator methods (covered by the static miner) or at build time.

### Cluster definition

Clusters group related fields for Cartesian probing:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class _Cluster:
    name: str
    fields: list[str]
    values: dict[str, list[Any]]
    constructor: type  # e.g. SamplingParams
    validate_method: str | None = None  # e.g. "_verify_args"

CLUSTERS = [
    _Cluster(
        name="sampling_temperature",
        fields=["temperature", "top_p", "top_k"],
        values={
            "temperature": [0.0, 0.5, 1.0, 2.0, -0.1],
            "top_p": [0.0, 0.5, 1.0, 1.1],
            "top_k": [0, 1, 50, -1],
        },
        constructor=SamplingParams,
    ),
]
```

Cluster size rule: if `product(len(values[f]) for f in fields) > 200`, use Hypothesis as a supplement instead of Cartesian product:

```python
import itertools
import hypothesis.strategies as st
from hypothesis import given, settings

def probe_cluster(cluster: _Cluster) -> list[tuple[dict, str | None]]:
    size = 1
    for vs in cluster.values.values():
        size *= len(vs)

    if size <= 200:
        # Cartesian probe
        rows = []
        for combo in itertools.product(*[cluster.values[f] for f in cluster.fields]):
            kwargs = dict(zip(cluster.fields, combo))
            rows.append(_run_probe(kwargs, cluster))
        return rows
    else:
        # Hypothesis supplement (deterministic, fixed seed)
        return _hypothesis_probe(cluster)
```

**Important:** Hypothesis is used here as a deterministic value generator with a fixed seed - not as a property-based test runner. The pipeline must be deterministic: the same library version + miner code must produce the same corpus.

### Predicate inference

After probing, group error rows by message class and infer predicates:

```python
def infer_predicates(rows: list[tuple[dict, str | None]]) -> list[RuleCandidate]:
    # Group by error message
    by_message: dict[str, list[dict]] = {}
    for kwargs, error in rows:
        if error is not None:
            by_message.setdefault(error, []).append(kwargs)

    candidates = []
    for message, trigger_kwargs in by_message.items():
        # Try templates in order of preference:
        # 1. cross-field divisibility: a % b != 0
        # 2. cross-field comparison: a > b
        # 3. type allowlist
        # 4. single-field range
        # 5. single-field equality
        # 6. value allowlist
        # Emit ALL plausible candidates (recall-first; vendor CI prunes false positives)
        ...
    return candidates
```

---

## Step 5: Write the corpus orchestration entry

`scripts/miners/{engine}_miner.py` is the main entry point:

```python
def mine() -> list[RuleCandidate]:
    candidates = []
    candidates.extend(mine_pydantic_rules())
    candidates.extend(mine_dataclass_rules())

    # Static miner
    from scripts.miners.myengine_static_miner import mine as static_mine
    candidates.extend(static_mine())

    # Dynamic miner (if applicable)
    from scripts.miners.myengine_dynamic_miner import mine as dynamic_mine
    candidates.extend(dynamic_mine())

    return candidates

if __name__ == "__main__":
    import yaml
    from scripts.miners._base import candidate_to_dict
    results = mine()
    staging = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "rules": [candidate_to_dict(c) for c in results],
    }
    output_path = Path("configs/validation_rules/_staging/myengine_miner.yaml")
    output_path.write_text(yaml.dump(staging, allow_unicode=True))
    print(f"Wrote {len(results)} candidates to {output_path}")
```

---

## Step 6: Write fixpoint regression tests

Each per-engine miner ships with parametrised tests:

```python
# tests/unit/scripts/miners/test_myengine_miner.py

import pytest
from scripts.miners.myengine_miner import CLUSTERS, TESTED_AGAINST_VERSIONS

@pytest.mark.parametrize("cluster", CLUSTERS, ids=lambda c: c.name)
def test_cluster_probes_without_crashing(cluster):
    """Each cluster must complete probing without an unhandled exception."""
    rows = probe_cluster(cluster)
    assert isinstance(rows, list)

def test_version_envelope_set():
    """TESTED_AGAINST_VERSIONS must be a non-empty SpecifierSet."""
    from packaging.specifiers import SpecifierSet
    assert isinstance(TESTED_AGAINST_VERSIONS, SpecifierSet)
    assert str(TESTED_AGAINST_VERSIONS) != ""

def test_landmark_checks_raise_on_missing():
    """find_class returning None must raise MinerLandmarkMissingError."""
    from scripts.miners._base import find_class, MinerLandmarkMissingError
    import ast
    module = ast.parse("class Unrelated: pass")
    cls = find_class(module, "SomeConfigClass")
    assert cls is None
    # Confirm caller raises (contract test)
    with pytest.raises(MinerLandmarkMissingError):
        if cls is None:
            raise MinerLandmarkMissingError("SomeConfigClass")
```

---

## Step 7: Add to CI

1. Add the engine to `config-rules-refresh.yml`:

```yaml
- name: Run myengine miners
  run: |
    python scripts/miners/myengine_miner.py
```

2. Set the runner tier:
   - CPU-safe imports (`import myenginelib` works without CUDA): add to the GH-hosted job.
   - CUDA-aware import required: add to the self-hosted GPU runner job.

3. Add the engine's Dockerfile to the vendor-rules step so `vendor_rules.py` can replay rules inside the engine's container.

4. Add a Renovate `packageRule` so library bumps trigger `config-rules-refresh.yml`.

---

## Step 8: Generate and review the corpus

Run the miner locally (inside the engine's Docker container if CUDA is required):

```bash
python scripts/miners/myengine_miner.py
# Writes configs/validation_rules/_staging/myengine_miner.yaml

python scripts/miners/build_corpus.py --engine myengine
# Merges staging files, runs vendor-CI gate, writes corpus

python scripts/vendor_rules.py \
  --engine myengine \
  --corpus configs/validation_rules/myengine.yaml \
  --out src/llenergymeasure/config/vendored_rules/myengine.json
# Validates all rules against live library
```

Review the corpus manually:
- Do the `kwargs_positive` examples look right?
- Are there rules that fire too broadly (false positives)?
- Are there obvious constraints the miner missed (coverage gaps)?

If coverage gaps exist, extend the miner. Only add `manual_seed` rules as a last resort, with a justification comment.

---

## Transformers as the gold standard: key patterns

The transformers miner is the reference implementation. Key patterns to follow:

### The `find_class` / `find_method` / `MinerLandmarkMissingError` contract

Every class and method the miner walks must be guarded:

```python
cls_node = find_class(module, "GenerationConfig")
if cls_node is None:
    raise MinerLandmarkMissingError("GenerationConfig")

method_node = find_method(cls_node, "validate")
if method_node is None:
    raise MinerLandmarkMissingError("GenerationConfig.validate")
```

### The `public_fields` filter

Derive public fields from the class's dataclass fields or `__annotations__`, and use `filter_condition_references_self` to drop predicates that don't reference a public field:

```python
public_fields = frozenset(
    f.name for f in dataclasses.fields(GenerationConfig)
    if not f.name.startswith("_")
)
```

### Unparseable sub-clauses: log, don't drop

When the static miner encounters a condition sub-clause it cannot translate (e.g. an opaque function call), it logs the clause and emits the surrounding rule with the parseable parts. The rule is still useful; the vendor-CI gate will confirm whether it fires correctly:

```python
# transformers_static_miner.py pattern:
if unparseable_clause:
    logger.debug(
        "static_miner: dropped sub-clause in %s.%s:%d: %s",
        cls_name, method_name, node.lineno, ast.unparse(sub_clause)
    )
    # Continue emitting the rule without the sub-clause
```

### Recall-first: emit all plausible candidates

Both static and dynamic miners err toward recall. The vendor-CI gate is the prune step. Do not add extra filters "just in case" - if a rule candidate is wrong, the vendor-CI gate will quarantine it.

---

## Common mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Not setting `TESTED_AGAINST_VERSIONS` | Miner runs against wrong library version silently | Add `TESTED_AGAINST_VERSIONS = SpecifierSet(">=X,<Y")` and call `check_installed_version` at import |
| Catching `ImportError` on landmark imports | Silent degradation (returns `[]` on failure) | Let `ImportError` propagate; or raise `MinerLandmarkMissingError` explicitly |
| Cartesian-only probing with large clusters | Exponential probe count; CI timeouts | Add Hypothesis supplement for clusters > 200 combinations |
| Adding `manual_seed` rules for automatable constraints | Pipeline-failure debt | Extend the miner instead |
| Using Hypothesis as property-based test runner (not value generator) | Non-deterministic corpus | Use `hypothesis.strategies.from_type` with a fixed seed; never `@given` |
| Not calling `find_method` before walking | `AttributeError` on `None` if method renamed | Always guard: `if method is None: raise MinerLandmarkMissingError(...)` |

---

## See also

- [miner-pipeline.md](miner-pipeline.md) - pipeline architecture reference
- [validation-rule-corpus.md](validation-rule-corpus.md) - corpus format
- [parameter-discovery.md](parameter-discovery.md) - runtime validation
- [architecture-overview.md](architecture-overview.md) - system overview
- `scripts/miners/transformers_static_miner.py` - gold-standard static miner
- `scripts/miners/transformers_dynamic_miner.py` - gold-standard dynamic miner
- `scripts/miners/_base.py` - shared infrastructure
